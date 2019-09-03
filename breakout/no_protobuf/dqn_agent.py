import os
import argparse
import datetime
from tqdm import tqdm
import numpy as np

import tensorflow as tf

from models.dqn_model import DQN_Model, ExperienceBuffer
from util.gcp_io import gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_rows
from util.logging import TimeLogger
from util.distributions import get_distribution_strategy

#SET HYPERPARAMETERS
VISUAL_OBS_SPEC = [210,160,3]
NUM_ACTIONS=2
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(512,200)
LEARNING_RATE=0.00042
GAMMA = 0.9

class DQN_Agent():
    """
    Class for controlling and managing training from a bigtable database.
      
    Attributes:
        cbt_table (google.cloud.bigtable.Table): Bigtable table object returned from [util.gcp_io.cbt_load_table].
        gcs_bucket (google.cloud.storage.Bucket): GCS bucket object returned from [util.gcp_io.gcs_load_bucket].
        gcs_bucket_id (str): Global name of the GCS bucket where the model will be saved/loaded.
        prefix (str): Prefix used for model and trajectory names.
        tmp_weights_filepath (str): Temporary local path for saving model before copying to GCS.
        buffer_size (int): Max size of the experience buffer.
        batch_size (int): Batch size for estimator.
        train_epochs (int): Number of cycles of querying bigtable and training.
        train_steps (int): Number of train steps per epoch.
        period (int): Interval for saving models.
        output_dir (str): Output directory for logs and models.
        log_time (bool): Flag for time logging.
        num_gpus (int): Number of gpu devices for estimator.
    """
    def __init__(self, **kwargs):
        """
        The constructor for DQN_Agent class.

        """
        self.cbt_table = kwargs['cbt_table']
        self.gcs_bucket = kwargs['gcs_bucket']
        self.gcs_bucket_id = kwargs['gcs_bucket_id']
        self.prefix = kwargs['prefix']
        self.tmp_weights_filepath = kwargs['tmp_weights_filepath']
        self.exp_buff = ExperienceBuffer(kwargs['buffer_size'])
        self.batch_size = kwargs['batch_size']
        self.num_trajectories = kwargs['num_trajectories']
        self.train_epochs = kwargs['train_epochs']
        self.train_steps = kwargs['train_steps']
        self.period = kwargs['period']
        self.output_dir = kwargs['output_dir']
        self.log_time = kwargs['log_time']
        self.num_gpus = kwargs['num_gpus']
        self.tpu_name = kwargs['tpu_name']

        if self.tpu_name is not None:
            self.distribution_strategy = get_distribution_strategy(distribution_strategy='tpu', tpu_address=self.tpu_name)
            self.device = '/job:worker'
        else:
            self.distribution_strategy = get_distribution_strategy(distribution_strategy='default', num_gpus=self.num_gpus)
            self.device = None
        with tf.device(self.device), self.distribution_strategy.scope():
            self.model = DQN_Model(input_shape=VISUAL_OBS_SPEC,
                                   num_actions=NUM_ACTIONS,
                                   conv_layer_params=CONV_LAYER_PARAMS,
                                   fc_layer_params=FC_LAYER_PARAMS,
                                   learning_rate=LEARNING_RATE)
        gcs_load_weights(self.model, self.gcs_bucket, self.prefix, self.tmp_weights_filepath)

    def fill_experience_buffer(self):
        """
        Method that fills the experience buffer object from CBT.

        Reads a batch of rows and parses through them until experience buffer reaches buffer_size.

        """
        self.exp_buff.reset()

        if self.log_time is True: self.time_logger.reset()

        #FETCH DATA
        global_i = cbt_global_iterator(self.cbt_table)
        rows = cbt_read_rows(self.cbt_table, self.prefix, self.num_trajectories, global_i)

        if self.log_time is True: self.time_logger.log("Fetch Data      ")
        
        for row in tqdm(rows, "Parsing trajectories {} - {}".format(global_i - self.num_trajectories, global_i - 1)):
            #DESERIALIZE DATA
            bytes_obs = row.cells['trajectory']['obs'.encode()][0].value
            bytes_actions = row.cells['trajectory']['actions'.encode()][0].value
            bytes_rewards = row.cells['trajectory']['rewards'.encode()][0].value

            if self.log_time is True: self.time_logger.log("Parse Bytes     ")

            #FORMAT DATA
            actions = np.frombuffer(bytes_actions, dtype=np.uint8).astype(np.int32)
            rewards = np.frombuffer(bytes_rewards, dtype=np.float32)
            num_steps = actions.size
            obs_shape = np.append(num_steps, VISUAL_OBS_SPEC).astype(np.int32)
            obs = np.frombuffer(bytes_obs, dtype=np.float32).reshape(obs_shape)

            if self.log_time is True: self.time_logger.log("Format Data     ")

            self.exp_buff.add_trajectory(obs, actions, rewards, num_steps)

            if self.log_time is True: self.time_logger.log("Add To Exp_Buff ")
        self.exp_buff.preprocess()

        dataset = tf.data.Dataset.from_tensor_slices(
            ((self.exp_buff.obs, self.exp_buff.next_obs),
            (self.exp_buff.actions, self.exp_buff.rewards, self.exp_buff.next_mask)))
        dataset = dataset.shuffle(self.exp_buff.max_size).repeat().batch(self.batch_size)

        # dist_dataset = self.distribution_strategy.experimental_distribute_dataset(dataset)

        if self.log_time is True: self.time_logger.log("To Dataset      ")

        return dataset

    def train(self):
        """
        Method that trains a model using using parameters defined in the constructor.

        """
        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                ((b_obs, b_next_obs), (b_actions, b_rewards, b_next_mask)) = inputs

                with tf.GradientTape() as tape:
                    q_pred, q_next = self.model(b_obs), self.model(b_next_obs)
                    one_hot_actions = tf.one_hot(b_actions, NUM_ACTIONS)
                    q_pred = tf.reduce_sum(q_pred * one_hot_actions, axis=-1)
                    q_next = tf.reduce_max(q_next, axis=-1)
                    q_target = b_rewards + (tf.constant(GAMMA, dtype=tf.float32) * q_next)
                    mse = self.model.loss(q_target, q_pred)
                    loss = tf.reduce_sum(mse)
                
                total_grads = tape.gradient(loss, self.model.trainable_weights)
                self.model.opt.apply_gradients(list(zip(total_grads, self.model.trainable_weights)))
                return mse

            per_example_losses = self.distribution_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
            # mean_loss = self.distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
            # return mean_loss

        if self.log_time is True:
            self.time_logger = TimeLogger(["Fetch Data      ",
                                           "Parse Bytes     ",
                                           "Format Data     ",
                                           "Add To Exp_Buff ",
                                           "To Dataset      ",
                                           "Train Step      ",
                                           "Save Model      "])
        print("-> Starting training...")
        for epoch in range(self.train_epochs):
            with tf.device(self.device), self.distribution_strategy.scope():
                dataset = self.fill_experience_buffer()
                exp_buff = iter(dataset)

                for step in tqdm(range(self.train_steps), "Training epoch {}".format(epoch)):
                    train_step(next(exp_buff))
                    if self.log_time is True: self.time_logger.log("Train Step      ")

            if epoch > 0 and epoch % self.period == 0:
                model_filename = self.prefix + '_model.h5'
                gcs_save_weights(self.model, self.gcs_bucket, self.tmp_weights_filepath, model_filename)

            if self.log_time is True: self.time_logger.log("Save Model      ")

            if self.log_time is True: self.time_logger.print_totaltime_logs()
        print("-> Done!")
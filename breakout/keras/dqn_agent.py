import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from protobuf.bytes_experience_replay_pb2 import Observations, Actions, Rewards, Info
from models.dqn_model import DQN_Model, ExperienceBuffer
from util.gcp_io import gcs_load_weights, gcs_save_weights, cbt_get_global_trajectory_buffer, cbt_read_trajectory
from util.logging import TimeLogger
from util.distributions import get_distribution_strategy

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
        hyperparams = kwargs['hyperparams']
        self.input_shape = hyperparams['input_shape']
        self.num_actions = hyperparams['num_actions']
        self.gamma = hyperparams['gamma']
        self.update_horizon = hyperparams['update_horizon']
        self.future_discounts = np.power(self.gamma, range(self.update_horizon))
        self.bootstrap_discount = np.power(self.gamma, self.update_horizon)
        
        self.cbt_table = kwargs['cbt_table']
        self.gcs_bucket = kwargs['gcs_bucket']
        self.gcs_bucket_id = kwargs['gcs_bucket_id']
        self.prefix = kwargs['prefix']
        self.tmp_weights_filepath = kwargs['tmp_weights_filepath']
        self.output_dir = kwargs['output_dir']

        self.buffer_size = kwargs['buffer_size']
        self.batch_size = kwargs['batch_size']
        self.train_epochs = kwargs['train_epochs']
        self.train_steps = kwargs['train_steps']
        self.period = kwargs['period']
        
        self.log_time = kwargs['log_time']
        self.num_gpus = kwargs['num_gpus']
        self.tpu_name = kwargs['tpu_name']
        self.wandb = kwargs['wandb']

        self.exp_buff = ExperienceBuffer(self.buffer_size, self.update_horizon)

        if self.tpu_name is not None:
            self.distribution_strategy = get_distribution_strategy(distribution_strategy='tpu', tpu_address=self.tpu_name)
            self.device = '/job:worker'
        else:
            self.distribution_strategy = get_distribution_strategy(distribution_strategy='default', num_gpus=self.num_gpus)
            self.device = None
        with tf.device(self.device), self.distribution_strategy.scope():
            self.model = DQN_Model(input_shape=self.input_shape,
                                   num_actions=self.num_actions,
                                   conv_layer_params=hyperparams['conv_layer_params'],
                                   fc_layer_params=hyperparams['fc_layer_params'],
                                   learning_rate=hyperparams['learning_rate'])
            self.target_model = DQN_Model(input_shape=self.input_shape,
                                          num_actions=self.num_actions,
                                          conv_layer_params=hyperparams['conv_layer_params'],
                                          fc_layer_params=hyperparams['fc_layer_params'],
                                          learning_rate=hyperparams['learning_rate'])
        gcs_load_weights(self.model, self.gcs_bucket, self.prefix, self.tmp_weights_filepath)

    def fill_experience_buffer(self):
        """
        Method that fills the experience buffer object from CBT.

        Reads a batch of rows and parses through them until experience buffer reaches buffer_size.

        """
        self.exp_buff.reset()
        total_rewards = []

        if self.log_time is True: self.time_logger.reset()

        global_traj_buff = cbt_get_global_trajectory_buffer(self.cbt_table)

        for traj_i in global_traj_buff:
            rows = cbt_read_trajectory(self.cbt_table, traj_i)
        
            observations, actions, rewards, total_rewards = [], [], [], []
            for row in rows:
                #DESERIALIZE DATA
                bytes_obs = row.cells['step']['obs'.encode()][0].value
                bytes_action = row.cells['step']['action'.encode()][0].value
                bytes_reward = row.cells['step']['reward'.encode()][0].value
                bytes_info = row.cells['step']['info'.encode()][0].value
                
                pb2_obs, pb2_actions, pb2_rewards, info = Observations(), Actions(), Rewards(), Info()
                pb2_obs.ParseFromString(bytes_obs)
                pb2_actions.ParseFromString(bytes_action)
                pb2_rewards.ParseFromString(bytes_reward)
                info.ParseFromString(bytes_info)

                if self.log_time is True: self.time_logger.log("Parse Bytes     ")

                #FORMAT DATA
                obs_shape = np.append(1, info.visual_obs_spec).astype(np.int32)
                observations.append(np.frombuffer(pb2_obs.visual_obs, dtype=np.float32).reshape(obs_shape))
                actions.append(np.frombuffer(pb2_actions.actions, dtype=np.int32))
                rewards.append(np.frombuffer(pb2_rewards.rewards, dtype=np.float32))

            obs = np.concatenate(observations, axis=0)
            actions = np.concatenate(actions, axis=0)
            rewards = np.concatenate(rewards, axis=0)

            num_steps = len(rewards)
            total_rewards.append(np.sum(rewards))
            discounted_future_rewards = []
            for i in range(num_steps):
                end_i = i + self.update_horizon
                if end_i <= num_steps:
                    horizon_rewards = rewards[i:end_i]
                else:
                    horizon_rewards = np.append(rewards[i:], np.zeros(end_i-num_steps))
                discounted_future_rewards.append(np.sum(horizon_rewards * self.future_discounts))
            discounted_future_rewards = np.asarray(discounted_future_rewards).astype(np.float32)

            if self.log_time is True: self.time_logger.log("Format Data     ")

            self.exp_buff.add_trajectory(obs, actions, discounted_future_rewards, num_steps)

            if self.exp_buff.size >= self.exp_buff.max_size: break

            if self.log_time is True: self.time_logger.log("Add To Exp_Buff ")
        self.exp_buff.preprocess()

        dataset = tf.data.Dataset.from_tensor_slices(
            ((self.exp_buff.obs, self.exp_buff.next_obs),
            (self.exp_buff.actions, self.exp_buff.rewards, self.exp_buff.next_mask)))
        dataset = dataset.shuffle(self.exp_buff.max_size).repeat().batch(self.batch_size)

        dist_dataset = self.distribution_strategy.experimental_distribute_dataset(dataset)

        if self.log_time is True: self.time_logger.log("To Dataset      ")

        return dist_dataset, np.mean(total_rewards)

    def train(self):
        """
        Method that trains a model using using parameters defined in the constructor.

        """
        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                ((b_obs, b_next_obs), (b_actions, b_rewards, b_next_mask)) = inputs

                with tf.GradientTape() as tape:
                    q_pred, q_next = self.model(b_obs), self.target_model(b_next_obs)
                    one_hot_actions = tf.one_hot(b_actions, self.num_actions)
                    q_pred = tf.reduce_sum(q_pred * one_hot_actions, axis=-1)
                    q_next = tf.reduce_max(q_next, axis=-1) * b_next_mask
                    q_target = b_rewards + (tf.constant(self.bootstrap_discount, dtype=tf.float32) * q_next)
                    mse = self.model.loss(q_target, q_pred)
                    loss = tf.reduce_sum(mse)

                total_grads = tape.gradient(loss, self.model.trainable_weights)
                self.model.opt.apply_gradients(list(zip(total_grads, self.model.trainable_weights)))
                return mse

            dist_losses = self.distribution_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
            mean_loss = self.distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, dist_losses, axis=None)
            return mean_loss

        if self.log_time is True:
            self.time_logger = TimeLogger(["Fetch Data      ",
                                           "Parse Bytes     ",
                                           "Format Data     ",
                                           "Add To Exp_Buff ",
                                           "To Dataset      ",
                                           "Train Step      ",
                                           "Save Model      "])
        print("-> Starting training...")
        with tf.device(self.device), self.distribution_strategy.scope():
            for epoch in range(self.train_epochs):
                dataset, mean_reward = self.fill_experience_buffer()
                exp_buff = iter(dataset)

                if self.log_time is True: self.time_logger.set_start()

                #UPDATE TARGET MODEL
                self.target_model.set_weights(self.model.get_weights())
                losses = []
                for step in tqdm(range(self.train_steps), "Training epoch {}".format(epoch)):
                    loss = train_step(next(exp_buff))
                    losses.append(loss)

                    if self.log_time is True: self.time_logger.log("Train Step      ")
                
                if self.wandb is not None:
                    mean_loss = np.mean(losses)
                    self.wandb.log({"Epoch": epoch,
                                    "Mean Loss": mean_loss,
                                    "Mean Reward": mean_reward})

                if self.period > 0 and epoch % self.period == 0:
                    model_filename = self.prefix + '_model.h5'
                    gcs_save_weights(self.model, self.gcs_bucket, self.tmp_weights_filepath, model_filename)

                if self.log_time is True: self.time_logger.log("Save Model      ")

                if self.log_time is True: self.time_logger.print_totaltime_logs()
        print("-> Done!")
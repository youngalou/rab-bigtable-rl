import os
import argparse
import datetime
from tqdm import tqdm
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from protobuf.experience_replay_pb2 import Trajectory, Info
from models.estimator_model import DQN_Model, ExperienceBuffer
from util.gcp_io import gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_rows
from util.logging import TimeLogger
from util.distributions import get_distribution_strategy

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
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
    def __init__(self,
                 cbt_table,
                 gcs_bucket,
                 gcs_bucket_id,
                 prefix,
                 tmp_weights_filepath,
                 buffer_size,
                 batch_size,
                 num_trajectories,
                 train_epochs,
                 train_steps,
                 period,
                 output_dir=None,
                 log_time=False,
                 num_gpus=0):
        """
        The constructor for DQN_Agent class.

        """
        self.cbt_table = cbt_table
        self.gcs_bucket = gcs_bucket
        self.gcs_bucket_id = gcs_bucket_id
        self.prefix = prefix
        self.tmp_weights_filepath = tmp_weights_filepath
        self.exp_buff = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size
        self.num_trajectories = num_trajectories
        self.train_epochs = train_epochs
        self.train_steps = train_steps
        self.period = period
        self.output_dir = output_dir
        self.log_time = log_time

        distribution_strategy = get_distribution_strategy(distribution_strategy="default", num_gpus=num_gpus)
        with distribution_strategy.scope():
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

        if self.log_time is True: self.time_logger.log(0)
        
        for row in tqdm(rows, "Trajectories {} - {}".format(global_i - self.num_trajectories, global_i - 1)):
            #DESERIALIZE DATA
            bytes_traj = row.cells['trajectory']['traj'.encode()][0].value
            bytes_info = row.cells['trajectory']['info'.encode()][0].value
            traj, info = Trajectory(), Info()
            traj.ParseFromString(bytes_traj)
            info.ParseFromString(bytes_info)

            #FORMAT DATA
            obs_shape = np.append(info.num_steps, info.visual_obs_spec).astype(int)
            obs = np.asarray(traj.visual_obs).reshape(obs_shape)

            self.exp_buff.add_trajectory(obs, traj.actions, traj.rewards, info.num_steps)

        if self.log_time is True: self.time_logger.log(1)

    def train(self):
        """
        Method that trains a model using tf.Estimator using parameters defined in the constructor.

        """
        if self.log_time is True:
            self.time_logger = TimeLogger(["Fetch Data      ",
                                           "Parse Data      ",
                                           "Compute Loss    ",
                                           "Generate Grads  ",
                                           "Save Model      "])
        print("-> Starting training...")
        for epoch in range(self.train_epochs):
            self.fill_experience_buffer()

            #COMPUTE LOSS        
            with tf.GradientTape() as tape:
                q_pred, q_next = self.model(self.exp_buff.obs), self.model(self.exp_buff.next_obs)
                one_hot_actions = tf.one_hot(self.exp_buff.actions, NUM_ACTIONS)
                q_pred = tf.reduce_sum(q_pred * one_hot_actions, axis=-1)
                q_next = tf.reduce_max(q_next, axis=-1)
                q_next = tf.cast(q_next, dtype=tf.float64) * self.exp_buff.next_mask
                q_target = self.exp_buff.rewards + tf.multiply(tf.constant(GAMMA, dtype=tf.float64), q_next)
                loss = tf.reduce_sum(self.model.loss(q_target, q_pred))
            
            if self.log_time is True: self.time_logger.log(2)

            #GENERATE GRADIENTS
            total_grads = tape.gradient(loss, self.model.trainable_weights)
            self.model.opt.apply_gradients(zip(total_grads, self.model.trainable_weights))

            if self.log_time is True: self.time_logger.log(3)

            if epoch > 0 and epoch % self.period == 0:
                model_filename = self.prefix + '_model.h5'
                gcs_save_weights(self.model, self.gcs_bucket, self.tmp_weights_filepath, model_filename)

            if self.log_time is True: self.time_logger.log(4)

            if self.log_time is True: self.time_logger.print_logs()
        print("-> Done!")
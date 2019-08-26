import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from protobuf.experience_replay_pb2 import Trajectory, Info
from models.dqn_model import DQN_Model, ExperienceBuffer
from util.gcp_io import gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_rows
from util.logging import TimeLogger
from util.distributions import get_distribution_strategy

#SET API CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
VISUAL_OBS_SPEC = [210,160,3]
NUM_ACTIONS=2
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(512,200)
LEARNING_RATE=0.00042
GAMMA = 0.9

class DQN_Agent():
    """ Initializes a Deep Queue Network(DQN) Model.
    """
    def __init__(self,
                 model,
                 cbt_table=None,
                 gcs_bucket=None,
                 prefix=None,
                 tmp_weights_filepath=None,
                 output_dir=None,
                 buffer_size=None,
                 log_time=False):
        self.cbt_table = cbt_table
        self.gcs_bucket = gcs_bucket
        self.prefix = prefix
        self.tmp_weights_filepath = tmp_weights_filepath
        self.output_dir = output_dir
        self.model = model
        self.exp_buff = ExperienceBuffer(buffer_size)
        self.log_time = log_time

        gcs_load_weights(self.model, gcs_bucket, self.prefix, self.tmp_weights_filepath)

    def train(self, train_epochs, train_steps):
        """ Performs the training loop of the DQN model.
            Trajectories are downloaded from Bigtable as Protobuf, then deserialized as Numpy arrays 
            Outputs an .h5 weights file uploaded to gcs bucket.

            train_epochs -- integer representing the number of epochs (default none)
            train_steps -- integer representing the number of trajectories to download from Bigtable
        """
        train_log_dir = os.path.join(self.output_dir, 'logs/')
        os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)
        loss_metrics = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if self.log_time is True:
            time_logger = TimeLogger(["Fetch Data", "Parse Data", "Compute Loss", "Generate Grads"], num_cycles=train_steps)

        print("-> Starting training...")
        step = 0
        for epoch in range(train_epochs):
            if self.log_time is True: time_logger.reset()

            #FETCH DATA
            global_i = cbt_global_iterator(self.cbt_table)
            rows = cbt_read_rows(self.cbt_table, self.prefix, train_steps, global_i)

            if self.log_time is True: time_logger.log(0)

            for row in tqdm(rows, "Trajectories {} - {}".format(global_i - train_steps, global_i - 1)):
                #DESERIALIZE DATA
                bytes_traj = row.cells['trajectory']['traj'.encode()][0].value
                bytes_info = row.cells['tsrajectory']['info'.encode()][0].value
                traj, info = Trajectory(), Info()
                traj.ParseFromString(bytes_traj)
                info.ParseFromString(bytes_info)

                #FORMAT DATA
                obs_shape = np.append(info.num_steps, info.visual_obs_spec).astype(int)
                obs = np.asarray(traj.visual_obs).reshape(obs_shape)

                self.exp_buff.add_trajectory(obs, traj.actions, traj.rewards, info.num_steps)

                if self.log_time is True: time_logger.log(1)

                #COMPUTE LOSS
                with tf.GradientTape() as tape:
                    q_pred, q_next = self.model(self.exp_buff.obs), self.model(self.exp_buff.next_obs)
                    one_hot_actions = tf.one_hot(self.exp_buff.actions, NUM_ACTIONS)
                    q_pred = tf.reduce_sum(q_pred * one_hot_actions, axis=-1)
                    q_next = tf.reduce_max(q_next, axis=-1)
                    q_next = q_next * self.exp_buff.next_mask
                    q_target = self.exp_buff.rewards + tf.multiply(tf.constant(GAMMA, dtype=tf.float32), q_next)
                    loss = self.model.loss(q_pred, q_target)
                
                if self.log_time is True: time_logger.log(2)

                #GENERATE GRADIENTS
                total_grads = tape.gradient(loss, self.model.trainable_weights)
                self.model.opt.apply_gradients(zip(total_grads, self.model.trainable_weights))

                if self.log_time is True: time_logger.log(3)

                #TENSORBOARD LOGGING
                loss_metrics(loss)
                total_reward = np.sum(traj.rewards)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_metrics.result(), step=step)
                    tf.summary.scalar('total reward', total_reward, step=step)
                step += 1

            if self.log_time is True: time_logger.print_logs()

            #SAVE MODEL WEIGHTS
            model_filename = self.prefix + '_model.h5'
            gcs_save_weights(model, gcs_bucket, self.tmp_weights_filepath, model_filename)
        print("-> Done!")
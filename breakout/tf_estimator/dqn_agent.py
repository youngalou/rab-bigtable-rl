import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from protobuf.experience_replay_pb2 import Trajectory, Info
from models.estimator_model import DQN_Model, ExperienceBuffer
from util.gcp_io import gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_rows
from util.logging import TimeLogger
from util.distributions import get_distribution_strategy

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
    def __init__(self,
                 model,
                 cbt_table=None,
                 gcs_bucket=None,
                 prefix=None,
                 tmp_weights_filepath=None,
                 num_trajectories=10,
                 buffer_size=10008000,
                 batch_size=32,
                 train_epochs=1000,
                 train_steps=1,
                 period=1,
                 output_dir=None,
                 log_time=False,
                 num_gpus=0):
        self.model = model
        self.cbt_table = cbt_table
        self.gcs_bucket = gcs_bucket
        self.prefix = prefix
        self.tmp_weights_filepath = tmp_weights_filepath
        self.num_trajectories = num_trajectories
        self.exp_buff = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.train_steps = train_steps
        self.period = period
        self.output_dir = output_dir
        self.log_time = log_time

        gcs_load_weights(self.model, gcs_bucket, self.prefix, self.tmp_weights_filepath)

        distribution_strategy = get_distribution_strategy(distribution_strategy="default", num_gpus=num_gpus)
        run_config = tf.estimator.RunConfig(train_distribute=distribution_strategy)
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        model_dir = os.path.join(self.output_dir, 'models/')
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=model_dir,
            config=run_config,
            params={'data_format': data_format})

        # self.model.compile(loss=self.model.loss, optimizer=self.model.opt)
        # self.estimator = tf.keras.estimator.model_to_estimator(keras_model=self.model, model_dir=model_dir)

        # train_log_dir = os.path.join(self.output_dir, 'logs/')
        # os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)
        # self.loss_metrics = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        # self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def model_fn(self, features, labels, mode, params):
        model = DQN_Model(input_shape=VISUAL_OBS_SPEC,
                          num_actions=NUM_ACTIONS,
                          conv_layer_params=CONV_LAYER_PARAMS,
                          fc_layer_params=FC_LAYER_PARAMS,
                          learning_rate=LEARNING_RATE)

        if self.log_time is True: self.time_logger.log(2)

        (obs, next_obs) = features
        (actions, rewards, next_mask) = labels
        
        q_pred, q_next = model(obs), model(next_obs)
        one_hot_actions = tf.one_hot(actions, NUM_ACTIONS)
        q_pred = tf.reduce_sum(q_pred * one_hot_actions, axis=-1)
        q_next = tf.reduce_max(q_next, axis=-1)
        q_next = tf.cast(q_next, dtype=tf.float64) * next_mask
        q_target = rewards + tf.multiply(tf.constant(GAMMA, dtype=tf.float64), q_next)
        loss = tf.reduce_sum(model.loss(q_target, q_pred))

        train_op = model.opt.minimize(loss, var_list=model.trainable_variables, global_step=tf.compat.v1.train.get_or_create_global_step())

        if self.log_time is True: self.time_logger.log(3)

        # #TENSORBOARD LOGGING
        # self.loss_metrics(loss)
        # total_reward = np.sum(rewards)
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', self.loss_metrics.result(), step=tf.train.get_or_create_global_step())
        #     tf.summary.scalar('total reward', total_reward, step=tf.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            predictions=q_pred,
            loss=loss,
            train_op=train_op)

    def train_input_fn(self):
        if self.log_time is True: self.time_logger.reset()
        #FETCH DATA
        global_i = cbt_global_iterator(self.cbt_table)
        rows = cbt_read_rows(self.cbt_table, self.prefix, self.num_trajectories, global_i)

        if self.log_time is True: self.time_logger.log(0)

        self.exp_buff.reset()
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

            # if self.exp_buff.size >= self.exp_buff.max_size: break
                
        dataset = tf.data.Dataset.from_tensor_slices(
            ((self.exp_buff.obs, self.exp_buff.next_obs),
            (self.exp_buff.actions, self.exp_buff.rewards, self.exp_buff.next_mask)))
        dataset = dataset.shuffle(1000).repeat().batch(self.batch_size)

        if self.log_time is True: self.time_logger.log(1)

        return dataset

    def train(self):
        if self.log_time is True:
            self.time_logger = TimeLogger(["Fetch Data", "Parse Data", "Build Model", "Compute Loss", "Estimator"])
        print("-> Starting training...")
        for epoch in range(self.train_epochs):
            self.estimator.train(input_fn=self.train_input_fn, steps=self.train_steps)

            if self.log_time is True: self.time_logger.log(4)
            if self.log_time is True: self.time_logger.print_logs()

            if epoch > 0 and epoch % self.period == 0:
                model_filename = self.prefix + '_model.h5'
                gcs_save_weights(self.model, self.gcs_bucket, self.tmp_weights_filepath, model_filename)
        print("-> Done!")
import os
import argparse
import datetime
from tqdm import tqdm
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from protobuf.experience_replay_pb2 import Trajectory, Info
from models.estimator_model import DQN_Model, ExperienceBuffer
from util.gcp_io import gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_row
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
        self.train_epochs = train_epochs
        self.train_steps = train_steps
        self.period = period
        self.output_dir = output_dir
        self.log_time = log_time

        distribution_strategy = get_distribution_strategy(distribution_strategy="default", num_gpus=num_gpus)
        run_config = tf.estimator.RunConfig(train_distribute=distribution_strategy)
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        model_dir = os.path.join(self.output_dir, 'models/')
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=model_dir,
            config=run_config,
            params={'data_format': data_format})

    def model_fn(self, features, labels, mode, params):
        """
        Function to be passed as argument to tf.Estimator.
  
        Parameters: 
           features (tuple): (obs, next_obs) S and S' of a (S,A,R,S') transition.
           labels (tuple): (actions, rewards, next_mask) A and R of a transition, plus a mask for bootstrapping.
           mode (tf.estimator.ModeKeys): Estimator object that defines which op is called. (currently always .TRAIN)
           params (dict): Optional dictionary of parameters for building custom models. (not currently implemented)
        """
        #BUILD MODEL
        model = DQN_Model(input_shape=VISUAL_OBS_SPEC,
                          num_actions=NUM_ACTIONS,
                          conv_layer_params=CONV_LAYER_PARAMS,
                          fc_layer_params=FC_LAYER_PARAMS,
                          learning_rate=LEARNING_RATE)

        if self.log_time is True: self.time_logger.log(3)

        (obs, next_obs) = features
        (actions, rewards, next_mask) = labels

        #COMPUT LOSS        
        with tf.GradientTape() as tape:
            q_pred, q_next = model(obs), model(next_obs)
            one_hot_actions = tf.one_hot(actions, NUM_ACTIONS)
            q_pred = tf.reduce_sum(q_pred * one_hot_actions, axis=-1)
            q_next = tf.reduce_max(q_next, axis=-1)
            q_next = tf.cast(q_next, dtype=tf.float64) * next_mask
            q_target = rewards + tf.multiply(tf.constant(GAMMA, dtype=tf.float64), q_next)
            loss = tf.reduce_sum(model.loss(q_target, q_pred))

        #GENERATE GRADIENTS
        total_grads = tape.gradient(loss, model.trainable_variables)
        grads_op = model.opt.apply_gradients(zip(total_grads, model.trainable_variables), tf.compat.v1.train.get_global_step())
        train_op = grads_op

        if self.log_time is True: self.time_logger.log(4)

        #RUN ESTIMATOR IN TRAIN MODE
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            predictions=q_pred,
            loss=loss,
            train_op=train_op)

    def train_input_fn(self):
        """
        Input function to be passed to estimator.train().

        Reads a single row from bigtable at a time until an experience buffer is filled to a specified buffer_size.

        """
        if self.log_time is True: self.time_logger.reset()

        global_i = cbt_global_iterator(self.cbt_table) - 1
        i = 0
        self.exp_buff.reset()
        while True:
            #FETCH ROW FROM CBT
            row_i = global_i - i
            row = cbt_read_row(self.cbt_table, self.prefix, row_i)

            if self.log_time is True: self.time_logger.log(0)

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

            if self.exp_buff.size >= self.exp_buff.max_size: break
            i += 1
        print("-> Fetched trajectories {} - {}".format(global_i - i, global_i))
                
        dataset = tf.data.Dataset.from_tensor_slices(
            ((self.exp_buff.obs, self.exp_buff.next_obs),
            (self.exp_buff.actions, self.exp_buff.rewards, self.exp_buff.next_mask)))
        dataset = dataset.shuffle(self.exp_buff.max_size).repeat().batch(self.batch_size)

        if self.log_time is True: self.time_logger.log(2)

        return dataset

    def export_model(self):
        """
        Method that saves the latest checkpoint to gcs_bucket.

        """
        model_path = 'gs://' + self.gcs_bucket_id + '/' +  self.prefix + '_model'
        latest_checkpoint = self.estimator.latest_checkpoint()
        all_checkpoint_files = tf.io.gfile.glob(latest_checkpoint + '*')
        for filename in all_checkpoint_files:
            suffix = filename.partition(latest_checkpoint)[2]
            destination_path = model_path + suffix
            print('Copying {} to {}'.format(filename, destination_path))
            tf.io.gfile.copy(filename, destination_path)

    def train(self):
        """ 
        Method that trains a model using tf.Estimator using parameters defined in the constructor.

        """
        if self.log_time is True:
            self.time_logger = TimeLogger(["Fetch Data    ",
                                           "Parse Data    ",
                                           "To Dataset    ",
                                           "Build Model   ",
                                           "Compute Loss  ",
                                           "Estimator     ",
                                           "Save Model    "])
        print("-> Starting training...")
        for epoch in range(self.train_epochs):
            self.estimator.train(input_fn=self.train_input_fn, steps=self.train_steps)

            if self.log_time is True: self.time_logger.log(5)

            if epoch > 0 and epoch % self.period == 0:
                self.export_model()

            if self.log_time is True: self.time_logger.log(6)

            if self.log_time is True: self.time_logger.print_logs()
        print("-> Done!")
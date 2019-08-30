import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
#from models.dqn_model import DQN_Model
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_rows
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from util.logging import TimeLogger

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
NUM_ACTIONS=2
FC_LAYER_PARAMS=(200,)
LEARNING_RATE=0.00042
GAMMA = 0.9

class DQN_Model(tf.keras.Model):
    def __init__(self,
                 input_shape=None,
                 num_actions=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 learning_rate=0.00042):
        super().__init__()
        if conv_layer_params is not None:
            self.convs = Custom_Convs(conv_layer_params)
        if fc_layer_params is not None:
            self.fc_layers = [Dense(neurons, activation="relu", name="fc_layer_{}".format(i)) for i,(neurons) in enumerate(fc_layer_params)]
        self.q_layer = Dense(num_actions, name='output')

        self.step(np.zeros(input_shape, dtype=np.float32))
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        self.opt = tf.optimizers.Adam(learning_rate)

        self.num_actions = num_actions
        self.public_url = None
    @tf.function(
        input_signature=[tf.TensorSpec(
                            shape=[None,4],
                            dtype=tf.float32)])

    def call(self, inputs):
        if self.convs is not None:
            inputs = self.convs(inputs)
        for layer in self.fc_layers:
            inputs = layer(inputs)
        logits = self.q_layer(inputs)
        return tf.cast(logits, dtype=tf.float32)

    def step(self, inputs):
        inputs = np.expand_dims(inputs, 0)
        q_values = tf.squeeze(self(inputs))
        action = tf.argmax(q_values).numpy()
        return action

    def step_stochastic(self, inputs):
        inputs = np.expand_dims(inputs, 0)
        logits = self(inputs)
        action = tf.squeeze(tf.random.categorical(logits, 1)).numpy()
        return action

    def step_epsilon_greedy(self, inputs, epsilon):
        sample = np.random.random()
        if sample > 1 - epsilon:
            return np.random.randint(self.num_actions)
        return self.step(inputs)

def connect_to_tpu(tpu=None):
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)
    tf.config.experimental_connect_to_host(cluster_resolver.get_master())
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    return strategy, "/job:worker"

""" Gathers Trajectory rows from Bigtable. Trains model. Uploades model weights to GCS bucket
        In each gathering step `global_i` fetches the most recent written row

        A Row consist of a Trajectory protobuf object, which contains all the Trajectories (Observation,Action,Reward)
        of a single played game.

        train-epochs -- Max number of rows to gather before training loop stops 
        train-steps --  Number of rows to gather before starting a training step, equivilent to batch_size
        period -- Not used in training agent. Max number of State, Action pairs to collect before stopping.
"""

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Train-From-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='cartpole-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='cartpole')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--train-epochs', type=int, default=10000)
    parser.add_argument('--train-steps', type=int, default=1000)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    parser.add_argument('--log-time', default=False, action='store_true')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)

    #LOAD MODEL
    def make_model():
        model = DQN_Model(input_shape=VECTOR_OBS_SPEC,
                            num_actions=NUM_ACTIONS,
                            fc_layer_params=FC_LAYER_PARAMS,
                            learning_rate=LEARNING_RATE)
        return(model)

    def collect_from_bigtable():
        for epoch in range(args.train_epochs):
            #FETCH DATA
            global_i = cbt_global_iterator(cbt_table)

            rows = cbt_read_rows(cbt_table, args.prefix, args.train_steps, global_i)

            for row in tqdm(rows, "Trajectories {} - {}".format(global_i - args.train_steps, global_i - 1)):
                #DESERIALIZE DATA
                bytes_traj = row.cells['trajectory']['traj'.encode()][0].value
                bytes_info = row.cells['trajectory']['info'.encode()][0].value
                traj, info = Trajectory(), Info()
                traj.ParseFromString(bytes_traj)
                info.ParseFromString(bytes_info)

                #print(info.num_steps)
                #print(info.vector_obs_spec)
                #FORMAT DATA
                traj_shape = np.append(info.num_steps, info.vector_obs_spec)
                obs = np.asarray(traj.vector_obs).reshape(traj_shape)
                actions = tf.convert_to_tensor(np.asarray(traj.actions))
                rewards = tf.convert_to_tensor(np.asarray(traj.rewards), dtype=tf.float32)
                next_obs = np.roll(obs, shift=-1, axis=0)
                next_mask = np.ones(info.num_steps)
                next_mask[-1] = 0

        dataset = tf.data.Dataset.from_tensor_slices(obs, actions, rewards, next_obs, next_mask)
        return dataset

    def train_and_export():
        strategy, device = connect_to_tpu('junwong-ny')
        with tf.device(device), strategy.scope():
            summary_writer = tf.summary.create_file_writer("gs://junwong-ny/logdir")
            dataset = iter(strategy.experimental_distribute_dataset(collect_from_bigtable()))
            model = make_model()
            loss_metric = tf.keras.metrics.Mean()
            optimizer = model.opt

            def gradient_tape(model, q_pred, q_target, obs, next_obs, rewards):
                with tf.GradientTape() as tape:
                    q_pred, q_next = tf.convert_to_tensor(model(obs), dtype=tf.float32), tf.convert_to_tensor(model(next_obs), dtype=tf.float32)
                    one_hot_actions = tf.one_hot(actions, NUM_ACTIONS)
                    q_pred = tf.reduce_sum(q_pred * one_hot_actions, axis=-1)
                    q_next = tf.reduce_max(q_next, axis=-1)
                    q_next = tf.multiply(q_next,next_mask)
                    q_target = tf.math.add(rewards, tf.multiply(tf.constant(GAMMA, dtype=tf.float32), q_next))

                    mse = tf.keras.losses.MeanSquaredError()
                    loss = mse(q_pred, q_target)
                    return loss, tape

                logging.info("Calculating gradients")
                #GENERATE GRADIENTS
                total_grads = tape.gradient(loss, model.trainable_weights)
                logging.info("Applying gradients")
                train_op = model.opt.apply_gradients(zip(total_grads, model.trainable_weights))
                with tf.control_dependencies([train_op]):
                    return tf.cast(optimizer.iterations, tf.float32)

            @tf.function
            def train_step():
                distributed_metric = strategy.experimental_run_v2(gradient_tape, args=[model, q_pred, q_target, obs, next_obs, rewards])
                step = strategy.reduce(
                    tf.distrubte.ReduceOp.MEAN, distributed_metric, axis = None)
                return step

            model, q_pred, q_target, obs, next_obs, rewards = next(dataset)
            step = tf.cast(train_step(model, q_pred, q_target, obs, next_obs, rewards), tf.float32)
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss_metric.result(), step=optimizer.iterations)
    steps_1 = 0
    if steps_1 >= args.train_epochs:
        #break
        model(tf.random.normal([1,4]))
        gcs_save_weights(model, gcs_bucket, args.tmp_weights_filepath, model_filename)
        steps_1 += 1
    train_and_export()
    
    '''
    #SETUP TENSORBOARD/LOGGING
    train_log_dir = os.path.join(args.output_dir, 'logs/')
    os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)
    loss_metrics = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    if args.log_time is True:
        time_logger = TimeLogger(["Fetch Data", "Parse Data", "Compute Loss", "Generate Grads"])
    '''
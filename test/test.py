import os
import argparse
import datetime
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from train.dqn_model import DQN_Model
from train.gcp_io import gcp_load_pipeline, gcs_load_weights, gcs_save_weights, cbt_global_iterator

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
NUM_ACTIONS=2
FC_LAYER_PARAMS=(200,)
LEARNING_RATE=0.00042
GAMMA = 0.9

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Train-From-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='cartpole-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='cartpole')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--train-epochs', type=int, default=1000000)
    parser.add_argument('--train-steps', type=int, default=100)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)

    #LOAD MODEL
    model = DQN_Model(input_shape=VECTOR_OBS_SPEC,
                      num_actions=NUM_ACTIONS,
                      fc_layer_params=FC_LAYER_PARAMS,
                      learning_rate=LEARNING_RATE)
    gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)

    for epoch in range(1):
        global_i = cbt_global_iterator(cbt_table)
        for i in tqdm(range(args.train_steps), "Trajectories {} - {}".format(global_i - args.train_steps, global_i)):
            row_key_i = global_i - args.train_steps + i
            row_key = args.prefix + '_trajectory_' + str(row_key_i)
            row = cbt_table.read_row(row_key)
            if row is None:
                print("Row_key [{}] not found.")
                exit()
            bytes_traj = row.cells['trajectory']['traj'.encode()][0].value
            bytes_info = row.cells['trajectory']['info'.encode()][0].value
            traj, info = Trajectory(), Info()
            traj.ParseFromString(bytes_traj)
            info.ParseFromString(bytes_info)

            print("row_key_i: ", row_key_i)
            print("row_key:", row_key)
            print("row: ", row)
            print(range(len(traj.vector_obs)))
            #FORMAT DATA
            traj_shape = np.append(info.num_steps, info.vector_obs_spec)
            obs = np.asarray(traj.vector_obs).reshape(traj_shape)
            actions = np.asarray(traj.actions)
            next_obs = np.roll(obs, shift=-1, axis=0)

            print("traj shape: ", traj_shape)
            print("obs : ", obs)
            print("actions: ", actions)
            print("next_obs: ", next_obs)

            with tf.GradientTape() as tape:
                q_pred = model(obs)
                q_pred = [q[a] for q, a in zip(q_pred, traj.actions)]
                q_next = model(next_obs)
                q_next = [q[tf.argmax(q)] for q in q_next]
                q_next[-1] = 0
                q_target = traj.rewards + tf.multiply(tf.constant(GAMMA, dtype=tf.float32), q_next)
                print ("q_target")
                print (q_target)
                print ("q_next")
                print (q_next)
                print ("tf.multiply")
                print (tf.multiply(tf.constant(GAMMA, dtype=tf.float32), q_next))
                print ("q_pred")
                print (q_pred)
                mse = tf.keras.losses.MeanSquaredError()
                loss = mse(q_pred, q_target)
                print ("loss")
                print (loss)

            total_grads = tape.gradient(loss, model.trainable_weights)
            model.opt.apply_gradients(zip(total_grads, model.trainable_weights))


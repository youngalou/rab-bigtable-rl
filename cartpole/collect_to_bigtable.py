import os
import argparse
import datetime
import time
import struct
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from cartpole.dqn_model import DQN_Model
from cartpole.gcp_io import gcp_load_pipeline, gcs_load_weights, cbt_global_iterator

import gym

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
NUM_ACTIONS=2
FC_LAYER_PARAMS=(200,)
LEARNING_RATE=0.00042
EPSILON = 0.5

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='cartpole-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='cartpole')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=1000000)
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)

    #LOAD MODEL
    model = DQN_Model(input_shape=VECTOR_OBS_SPEC,
                      num_actions=NUM_ACTIONS,
                      fc_layer_params=FC_LAYER_PARAMS,
                      learning_rate=LEARNING_RATE)

    #INITIALIZE ENVIRONMENT
    print("-> Initializing Gym environement...")
    env = gym.make('CartPole-v0')
    print("-> Environment intialized.")

    #GLOBAL ITERATOR
    global_i = cbt_global_iterator(cbt_table)
    print("global_i = {}".format(global_i))

    #COLLECT DATA FOR CBT
    print("-> Starting data collection...")
    rows = []
    for cycle in range(args.num_cycles):
        gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)
        for i in tqdm(range(args.num_episodes), "Cycle {}".format(cycle)):
            #RL LOOP GENERATES A TRAJECTORY
            observations, actions, rewards = [], [], []
            obs = np.asarray(env.reset())
            reward = 0
            done = False
            
            for _ in range(args.max_steps):
                action = model.step_epsilon_greedy(obs, EPSILON)
                new_obs, reward, done, info = env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)

                if done: break
                obs = np.asarray(new_obs)

            #BUILD PB2 OBJECTS
            traj, info = Trajectory(), Info()
            traj.vector_obs.extend(np.asarray(observations).flatten())
            traj.actions.extend(actions)
            traj.rewards.extend(rewards)
            info.vector_obs_spec.extend(observations[0].shape)
            info.num_steps = len(actions)

            #WRITE TO AND APPEND ROW
            row_key_i = i + global_i + (cycle * args.num_episodes)
            row_key = '{}_trajectory_{}'.format(args.prefix,row_key_i).encode()
            row = cbt_table.row(row_key)
            row.set_cell(column_family_id='trajectory',
                        column='traj'.encode(),
                        value=traj.SerializeToString())
            row.set_cell(column_family_id='trajectory',
                        column='info'.encode(),
                        value=info.SerializeToString())
            rows.append(row)
        gi_row = cbt_table.row('global_iterator'.encode())
        gi_row.set_cell(column_family_id='global',
                        column='i'.encode(),
                        value=struct.pack('i',row_key_i+1),
                        timestamp=datetime.datetime.utcnow())
        rows.append(gi_row)
        cbt_table.mutate_rows(rows)
        rows = []
        print("-> Saved trajectories {} - {}.".format(row_key_i - (args.num_episodes-1), row_key_i))
    env.close()
    print("-> Done!")
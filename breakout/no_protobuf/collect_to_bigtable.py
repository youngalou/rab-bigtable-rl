import os
import argparse
import struct
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from models.dqn_model import DQN_Model
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, cbt_global_iterator

import gym

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
EPSILON = 0.5

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='noprotobuf-breakout-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='breakout')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=1000000)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--log-time', default=False, action='store_true')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)
    cbt_batcher = cbt_table.mutations_batcher(flush_count=args.num_episodes, max_row_bytes=500000000)
                                                                                           #104857600
    #INITIALIZE ENVIRONMENT
    print("-> Initializing Gym environement...")
    env = gym.make('Breakout-v0')
    print("-> Environment intialized.")

    #LOAD MODEL
    model = DQN_Model(input_shape=env.observation_space.shape,
                      num_actions=env.action_space.n,
                      conv_layer_params=CONV_LAYER_PARAMS,
                      fc_layer_params=FC_LAYER_PARAMS,
                      learning_rate=LEARNING_RATE)

    #GLOBAL ITERATOR
    global_i = cbt_global_iterator(cbt_table)
    print("global_i = {}".format(global_i))

    #COLLECT DATA FOR CBT
    print("-> Starting data collection...")
    rows = []
    for cycle in range(args.num_cycles):
        # gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)
        for i in tqdm(range(args.num_episodes), "Cycle {}".format(cycle)):

            #RL LOOP GENERATES A TRAJECTORY
            observations, actions, rewards = [], [], []
            obs = np.asarray(env.reset() / 255).astype(np.float32)
            reward = 0
            done = False
            
            for _ in range(args.max_steps):
                action = model.step_epsilon_greedy(obs, EPSILON)
                new_obs, reward, done, info = env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)

                if done: break
                obs = np.asarray(new_obs / 255).astype(np.float32)

            observations = np.asarray(observations).flatten().tobytes()
            actions = np.asarray(actions).astype(np.uint8).tobytes()
            rewards = np.asarray(rewards).astype(np.float32).tobytes()

            # SET CELLS WITH DEFAULT PYTHON ENCODING
            row_key_i = i + global_i + (cycle * args.num_episodes)
            row_key = '{}_trajectory_{}'.format(args.prefix,row_key_i).encode()
            row = cbt_table.row(row_key)
            row.set_cell(column_family_id='trajectory',
                         column='obs'.encode(),
                         value=observations)
            row.set_cell(column_family_id='trajectory',
                         column='actions'.encode(),
                         value=actions)
            row.set_cell(column_family_id='trajectory',
                         column='rewards',
                         value=rewards)
            rows.append(row)
        
        #UPDATE GLOBAL ITERATOR
        gi_row = cbt_table.row('global_iterator'.encode())
        gi_row.set_cell(column_family_id='global',
                        column='i'.encode(),
                        value=struct.pack('i',row_key_i+1),
                        timestamp=datetime.datetime.utcnow())
        
        #ADD TRAJECTORIES AS ROWS TO BIGTABLE
        rows.append(gi_row)
        cbt_batcher.mutate_rows(rows)
        cbt_batcher.flush()
        rows = []
        print("-> Saved trajectories {} - {}.".format(row_key_i - (args.num_episodes-1), row_key_i))
    env.close()
    print("-> Done!")
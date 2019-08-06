import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from train.dqn_model import DQN_Model
from train.gcp_io import cbt_load_table, gcs_load_weights

import gym

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

VECTOR_OBS_SPEC = [4]

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='cartpole-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--model-prefix', type=str, default='cartpole_model')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-episodes', type=int, default=5000)
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND LOAD WEIGHTS FROM GCS
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    table = cbt_load_table(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, credentials)
    bucket, model_found = gcs_load_weights(args.gcp_project_id, args.bucket_id, credentials, args.model_prefix, args.tmp_weights_filepath)

    #LOAD MODEL
    model = DQN_Model(input_shape=VECTOR_OBS_SPEC,
                      num_actions=2,
                      fc_layer_params=(200,),
                      learning_rate=.00042)
    if model_found:
        model.load_weights(args.tmp_weights_filepath)

    #INITIALIZE ENVIRONMENT
    print("-> Initializing Gym environement...")
    env = gym.make('CartPole-v0')
    print("-> Environment intialized.")

    #COLLECT DATA FOR CBT
    print("-> Starting data collection...")
    rows = []
    for i in tqdm(range(args.num_episodes), "Generating trajectories"):
        #RL LOOP GENERATES A TRAJECTORY
        observations, actions, rewards = [], [], []
        obs = np.array(env.reset())
        reward = 0
        done = False
        
        for _ in range(args.max_steps):
            action = model.step(obs)
            new_obs, reward, done, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

            if done: break
            obs = np.array(new_obs)

        #BUILD PB2 OBJECTS
        traj, info = Trajectory(), Info()
        traj.vector_obs.extend(np.array(observations).flatten())
        traj.actions.extend(actions)
        traj.rewards.extend(rewards)
        info.vector_obs_spec.extend(observations[0].shape)
        info.num_steps = len(actions)

        #WRITE TO AND APPEND ROW
        row_key = "cartpole_trajectory_{}".format(i).encode()
        row = table.row(row_key)
        row.set_cell(column_family_id='trajectory',
                        column='traj'.encode(),
                        value=traj.SerializeToString(),
                        timestamp=datetime.datetime.utcnow())
        row.set_cell(column_family_id='trajectory',
                        column='info'.encode(),
                        value=info.SerializeToString(),
                        timestamp=datetime.datetime.utcnow())
        rows.append(row)
    table.mutate_rows(rows)
    env.close()
    print("-> Done!")
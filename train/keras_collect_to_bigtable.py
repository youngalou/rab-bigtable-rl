import argparse
import datetime

import struct
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from google.oauth2 import service_account
from google.cloud import bigtable

from protobuf.experience_replay_pb2 import Trajectory, Info
from train.dqn_model import DQN_Model

import gym

SCOPES = ['https://www.googleapis.com/auth/bigtable.admin']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

VISUAL_OBS_SPEC = [84, 84, 3]
VECTOR_OBS_SPEC = [4]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='cartpole-experience-replay')
    parser.add_argument('--env-filepath', type=str, default='envs/ObstacleTower/obstacletower.app')
    parser.add_argument('--num-episodes', type=int, default=5000)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--restore', type=str, default=None)
    args = parser.parse_args()

    model = DQN_Model(num_actions=2,
                      fc_layer_params=(200,))
    if args.restore:
        model.load_weights(args.restore)

    print('Looking for the [{}] table.'.format(args.cbt_table_name))
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = bigtable.Client(args.gcp_project_id, admin=True, credentials=credentials)
    instance = client.instance(args.cbt_instance_id)
    table = instance.table(args.cbt_table_name)
    if not table.exists():
        print("Table doesn't exist. Creating [{}] table...".format(args.cbt_table_name))
        max_versions_rule = bigtable.column_family.MaxVersionsGCRule(1)
        column_families = {'trajectory': max_versions_rule}
        table.create(column_families=column_families)
        print('Table created. Give it ~60 seconds to initialize before loading data.')
        exit()
    else:
        print("Table found.")

    print("Initializing Unity environement...")
    env = gym.make('CartPole-v0')
    print("Environment intialized.")

    print("Starting RL loop...")
    rows = []
    for i in tqdm(range(args.num_episodes), "Generating trajectories"):
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

        traj, info = Trajectory(), Info()
        traj.vector_obs.extend(np.array(observations).flatten())
        traj.actions.extend(actions)
        traj.rewards.extend(rewards)
        info.vector_obs_spec.extend(observations[0].shape)
        info.num_steps = len(actions)

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
        # rows.append(row)
    # table.mutate_rows(rows)
    env.close()
    print("Done!")
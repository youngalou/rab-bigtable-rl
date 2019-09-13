import argparse
import time
from tqdm import tqdm

import numpy as np

from google.oauth2 import service_account

from models.dqn_model import DQN_Model
from util.gcp_io import gcs_load_bucket, gcs_load_weights, cbt_global_iterator

import gym

#SET API CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#MODEL HYPERPARAMETERS
VISUAL_OBS_SPEC = [210,160,3]
NUM_ACTIONS=2
FC_LAYER_PARAMS=(512,)
LEARNING_RATE=0.00042

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Breakout Evaluation Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='breakout')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=1000)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gcs_bucket = gcs_load_bucket(args.gcp_project_id, args.bucket_id, credentials)

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

    #COLLECT DATA FOR CBT
    print("-> Starting evaluation...")
    for cycle in range(args.num_cycles):
        gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)
        total_reward = 0
        for i in tqdm(range(args.num_episodes), "Cycle {}".format(cycle)):
            observations, actions, rewards = [], [], []
            obs = np.array(env.reset())
            reward = 0
            done = False
            
            for _ in range(args.max_steps):
                env.render()
                time.sleep(.05)

                action = model.step(obs)
                new_obs, reward, done, info = env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)

                total_reward += reward
                if done: break
                obs = np.array(new_obs)
        print("Average reward: {}".format(total_reward / args.num_episodes))
    env.close()
    print("-> Done!")
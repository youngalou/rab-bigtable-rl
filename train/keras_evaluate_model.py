import argparse
import time
from tqdm import tqdm

import numpy as np

from google.oauth2 import service_account

from train.dqn_model import DQN_Model
from train.gcp_io import gcs_load_bucket, gcs_load_weights, cbt_global_iterator

import gym

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

VECTOR_OBS_SPEC = [4]

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='cartpole')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--update-interval', type=int, default=100)
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gcs_bucket = gcs_load_bucket(args.gcp_project_id, args.bucket_id, credentials)

    #LOAD MODEL
    model = DQN_Model(input_shape=VECTOR_OBS_SPEC,
                      num_actions=2,
                      fc_layer_params=(200,),
                      learning_rate=.00042)
    gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)

    #INITIALIZE ENVIRONMENT
    print("-> Initializing Gym environement...")
    env = gym.make('CartPole-v0')
    print("-> Environment intialized.")

    #COLLECT DATA FOR CBT
    print("-> Starting evaluation...")
    for i in tqdm(range(args.num_episodes), "Evaluating"):
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

            if done: break
            obs = np.array(new_obs)

        if i > 0 and i % args.update_interval == 0:
            gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)

    env.close()
    print("-> Done!")
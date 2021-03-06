import argparse
import time
from tqdm import tqdm
import numpy as np

from google.oauth2 import service_account

from models.dqn_model import DQN_Model
from util.gcp_io import gcs_load_bucket, gcs_load_weights
from util.unity_env import UnityEnvironmentWrapper

#SET API CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#MODEL HYPERPARAMETERS
VISUAL_OBS_SPEC = [224,224,2]
VECTOR_OBS_SPEC = [29]
NUM_ACTIONS=8
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(512,)

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('RAB Evaluation Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--bucket-id', type=str, default='youngalou')
    parser.add_argument('--prefix', type=str, default='crane-dualobs')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=1000)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--env-filename', type=str, default='envs/RAB-Arms/RabRobotArm_0930_V1/RabRobotArm_0930_V1.app')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gcs_bucket = gcs_load_bucket(args.gcp_project_id, args.bucket_id, credentials)

    #LOAD MODEL
    model = DQN_Model(input_shape=VECTOR_OBS_SPEC,
                      num_actions=NUM_ACTIONS,
                      fc_layer_params=FC_LAYER_PARAMS)

    #INITIALIZE ENVIRONMENT
    print("-> Initializing Crane environement...")
    env = UnityEnvironmentWrapper(environment_filename=args.env_filename,
                                  use_visual=True,
                                  use_vector=True,
                                  docker_training=False)
    print("-> Environment intialized.")

    #COLLECT DATA FOR CBT
    print("-> Starting evaluation...")
    for cycle in range(args.num_cycles):
        gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)
        total_reward = 0
        for i in tqdm(range(args.num_episodes), "Cycle {}".format(cycle)):
            obs = env.reset()
            reward = 0
            done = False
            
            for _ in range(args.max_steps):
                env.render(mode='human')
                time.sleep(.05)

                action = model.step(obs)
                new_obs, reward, done = env.step(action)

                total_reward += reward
                if done: break
                obs = new_obs
        print("Average reward: {}".format(total_reward / args.num_episodes))
    env.close()
    print("-> Done!")
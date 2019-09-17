import os
import argparse
import struct
import datetime
import logging

from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.bytes_experience_replay_pb2 import Observations, Actions, Rewards, Info
from models.dqn_model import DQN_Model
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, cbt_global_iterator
from util.logging import TimeLogger
from util.unity_env import UnityEnvironmentWrapper

# Retrieve environment variables
POD_NAME = os.environ.get('HOSTNAME')

#SET API CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [8]
VISUAL_OBS_SPEC = [224,224,3]
NUM_ACTIONS=6
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(512,200)
LEARNING_RATE=0.00042
EPSILON = 0.5

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='bytes-crane-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='youngalou')
    parser.add_argument('--env-filename', type=str, default='envs/CraneML/OSX/CraneML_OSX.app')
    parser.add_argument('--prefix', type=str, default='crane')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=1000000)
    parser.add_argument('--num-episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--update-interval', type=int, default=10)
    parser.add_argument('--log-time', default=False, action='store_true')
    parser.add_argument('--docker-training', type=bool, default=False)
    args = parser.parse_args()

    # logging for debugger
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger("crane")

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)
    max_row_bytes = (32 * np.prod(VISUAL_OBS_SPEC) + 64) * args.max_steps * args.num_episodes
    cbt_batcher = cbt_table.mutations_batcher(flush_count=args.num_episodes, max_row_bytes=max_row_bytes)

    #INITIALIZE ENVIRONMENT
    print("-> Initializing Crane environement...")
    env = UnityEnvironmentWrapper(environment_filename=args.env_filename,
                                  use_visual=True,
                                  docker_training=args.docker_training)
    print("-> Environment intialized.")

    #LOAD MODEL
    model = DQN_Model(input_shape=env._observation_space.shape,
                      num_actions=env._action_space.n,
                      conv_layer_params=CONV_LAYER_PARAMS,
                      fc_layer_params=FC_LAYER_PARAMS,
                      learning_rate=LEARNING_RATE)
                      
    #GLOBAL ITERATOR
    global_i = cbt_global_iterator(cbt_table)
    print("global_i = {}".format(global_i))

    #INITIALIZE EXECUTION TIME LOGGER
    if args.log_time is True:
        time_logger = TimeLogger(["Load Weights     ",
                                  "Run Environment  ",
                                  "Data To Bytes    ",
                                  "Write Cells      ",
                                  "Mutate Rows      "])

    #COLLECT DATA FOR CBT
    print("-> Starting data collection...")
    rows = []
    for cycle in range(args.num_cycles):
        if args.log_time is True: time_logger.reset()

        if cycle % args.update_interval == 0:
            gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)

        if args.log_time is True: time_logger.log("Load Weights     ")

        print("Collecting cycle {}:".format(cycle))
        for episode in range(args.num_episodes):
            #RL LOOP GENERATES A TRAJECTORY
            obs = np.asarray(env.reset() / 255).astype(np.float32)
            reward = 0
            done = False
            
            for step in tqdm(range(args.max_steps), "Episode {}".format(episode)):
                action = model.step_epsilon_greedy(obs, EPSILON)
                new_obs, reward, done, info = env.step(action)

                if done: break
                obs = np.asarray(new_obs / 255).astype(np.float32)
        
                if args.log_time is True: time_logger.log("Run Environment  ")

                observation = obs.flatten().tobytes()
                action = np.asarray(action).astype(np.uint8).tobytes()
                reward = np.asarray(reward).astype(np.float32).tobytes()

                #BUILD PB2 OBJECTS
                pb2_obs, pb2_actions, pb2_rewards, pb2_info = Observations(), Actions(), Rewards(), Info()
                pb2_obs.visual_obs = observation
                pb2_actions.actions = action
                pb2_rewards.rewards = reward
                pb2_info.visual_obs_spec.extend(VISUAL_OBS_SPEC)

                if args.log_time is True: time_logger.log("Data To Bytes    ")

                #WRITE TO AND APPEND ROW
                row_key_i = episode + global_i + (cycle * args.num_episodes)
                row_key = 'traj_pod_{}_{:05d}_step_{:05d}'.format(POD_NAME ,row_key_i, step).encode()
                # logger.info("==> row_key: {}".format(row_key))
                row = cbt_table.row(row_key)
                row.set_cell(column_family_id='trajectory',
                            column='obs'.encode(),
                            value=pb2_obs.SerializeToString())
                row.set_cell(column_family_id='trajectory',
                            column='actions'.encode(),
                            value=pb2_actions.SerializeToString())
                row.set_cell(column_family_id='trajectory',
                            column='rewards'.encode(),
                            value=pb2_rewards.SerializeToString())
                row.set_cell(column_family_id='trajectory',
                            column='info'.encode(),
                            value=pb2_info.SerializeToString())
                rows.append(row)

                if args.log_time is True: time_logger.log("Write Cells      ")
        
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

        if args.log_time is True: time_logger.log("Mutate Rows      ")

        print("-> Saved trajectories {} - {}.".format(row_key_i - (args.num_episodes-1), row_key_i))

        if args.log_time is True: time_logger.print_totaltime_logs()
    env.close()
    print("-> Done!")

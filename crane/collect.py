import os
import argparse
import struct
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.bytes_experience_replay_pb2 import Observations, Actions, Rewards, Info
from models.dual_obs_dqn_model import DQN_Model
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, \
                        cbt_global_iterator, cbt_global_trajectory_buffer
from util.logging import TimeLogger
from util.unity_env import UnityEnvironmentWrapper

#SET API CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VISUAL_OBS_SPEC = [224,224,2]
VECTOR_OBS_SPEC = [29]
NUM_ACTIONS=8
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(512,)
EPS_START = 0.8
EPS_FINAL = 0.2
EPS_STEPS = 10000

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='rab-arms-frameskip-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='youngalou')
    parser.add_argument('--env-filename', type=str, default='envs/RabRobotArm_frameskip/RabRobotArm_0927.x86_64')
    parser.add_argument('--prefix', type=str, default='rab-arms-frameskip')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=1000000)
    parser.add_argument('--num-episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--global-traj-buff-size', type=int, default=10)
    parser.add_argument('--frame-skip', type=int, default=5)
    parser.add_argument('--log-time', default=False, action='store_true')
    parser.add_argument('--docker-training', type=bool, default=False)
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)
    max_row_bytes = (4*np.prod(VISUAL_OBS_SPEC) + 64)
    cbt_batcher = cbt_table.mutations_batcher(flush_count=args.num_episodes, max_row_bytes=max_row_bytes)

    #INITIALIZE ENVIRONMENT
    print("-> Initializing Crane environement...")
    env = UnityEnvironmentWrapper(environment_filename=args.env_filename,
                                  use_visual=True,
                                  use_vector=True,
                                  allow_multiple_visual_obs=True,
                                  docker_training=args.docker_training)
    print("-> Environment intialized.")

    #LOAD MODEL
    model = DQN_Model(visual_obs_shape=VISUAL_OBS_SPEC,
                      vector_obs_shape=VECTOR_OBS_SPEC,
                      num_actions=NUM_ACTIONS,
                      conv_layer_params=CONV_LAYER_PARAMS,
                      fc_layer_params=FC_LAYER_PARAMS)

    #INITIALIZE EXECUTION TIME LOGGER
    if args.log_time is True:
        time_logger = TimeLogger(["Load Weights     ",
                                  "Global Iterator  ",
                                  "Run Environment  ",
                                  "Data To Bytes    ",
                                  "Write Cells      ",
                                  "Mutate Rows      "])

    #COLLECT DATA FOR CBT
    print("-> Starting data collection...")
    for cycle in range(args.num_cycles):
        if args.log_time is True: time_logger.reset()

        if cycle % args.update_interval == 0:
            gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)

        if args.log_time is True: time_logger.log("Load Weights     ")

        rows = []
        local_traj_buff = []
        print("Collecting cycle {}:".format(cycle))
        for episode in range(args.num_episodes):
            #UPDATE GLOBAL ITERATOR
            global_i = cbt_global_iterator(cbt_table)
            local_traj_buff.append(global_i)

            if global_i < EPS_STEPS:
                epsilon = EPS_START - (((EPS_START - EPS_FINAL) / EPS_STEPS) * global_i)
            else:
                epsilon = EPS_FINAL

            if args.log_time is True: time_logger.log("Global Iterator  ")

            #RL LOOP GENERATES A TRAJECTORY
            obs = env.reset()
            reward = 0
            done = False
            
            for step in tqdm(range(args.max_steps), "Episode {}".format(episode)):
                action = model.step_epsilon_greedy(obs, epsilon)
                new_obs, reward, done = env.step(action)
        
                if args.log_time is True: time_logger.log("Run Environment  ")

                (visual_obs, vector_obs) = obs
                visual_obs = np.expand_dims(visual_obs, axis=0).flatten().tobytes()
                vector_obs = np.expand_dims(vector_obs, axis=0).flatten().tobytes()
                action = np.asarray(action).astype(np.int32).tobytes()
                reward = np.asarray(reward).astype(np.float32).tobytes()

                #BUILD PB2 OBJECTS
                pb2_obs, pb2_actions, pb2_rewards, pb2_info = Observations(), Actions(), Rewards(), Info()
                pb2_obs.visual_obs = visual_obs
                pb2_obs.vector_obs = vector_obs
                pb2_actions.actions = action
                pb2_rewards.rewards = reward
                pb2_info.visual_obs_spec.extend(VISUAL_OBS_SPEC)
                pb2_info.vector_obs_spec.extend(VECTOR_OBS_SPEC)

                if args.log_time is True: time_logger.log("Data To Bytes    ")

                #WRITE TO AND APPEND ROW
                row_key = 'traj_{:05d}_step_{:05d}'.format(global_i, step).encode()
                row = cbt_table.row(row_key)
                row.set_cell(column_family_id='step',
                            column='obs'.encode(),
                            value=pb2_obs.SerializeToString())
                row.set_cell(column_family_id='step',
                            column='action'.encode(),
                            value=pb2_actions.SerializeToString())
                row.set_cell(column_family_id='step',
                            column='reward'.encode(),
                            value=pb2_rewards.SerializeToString())
                row.set_cell(column_family_id='step',
                            column='info'.encode(),
                            value=pb2_info.SerializeToString())
                rows.append(row)

                if args.log_time is True: time_logger.log("Write Cells      ")

                if done: break
                obs = new_obs
        
        #ADD ROWS TO BIGTABLE
        cbt_global_trajectory_buffer(cbt_table, np.asarray(local_traj_buff).astype(np.int32), args.global_traj_buff_size)
        cbt_batcher.mutate_rows(rows)
        cbt_batcher.flush()

        if args.log_time is True: time_logger.log("Mutate Rows      ")

        print("-> Saved trajectories {}.".format(local_traj_buff))

        if args.log_time is True: time_logger.print_totaltime_logs()
    env.close()
    print("-> Done!")

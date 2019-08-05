import os
import argparse
import datetime

import struct
import numpy as np
import tensorflow as tf

from google.oauth2 import service_account
from google.cloud import bigtable
from google.cloud.bigtable import row_filters
from protobuf.experience_replay_pb2 import Trajectory, Info

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read-From-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='cartpole-experience-replay')
    args = parser.parse_args()

    print('Looking for the [{}] table.'.format(args.cbt_table_name))
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = bigtable.Client(args.gcp_project_id, admin=True, credentials=credentials)
    instance = client.instance(args.cbt_instance_id)
    table = instance.table(args.cbt_table_name)
    if not table.exists():
        print("Table doesn't exist.")
        exit()
    else:
        print("Table found.")

    for i in range(10):
        #QUERY TABLE FOR PARTIAL ROWS
        regex_filter = '^cartpole_trajectory_{}$'.format(i)
        row_filter = row_filters.RowKeyRegexFilter(regex_filter)
        filtered_rows = table.read_rows(filter_=row_filter)

        for row in filtered_rows:
            bytes_traj = row.cells['trajectory']['traj'.encode()][0].value
            bytes_info = row.cells['trajectory']['info'.encode()][0].value
            traj, info = Trajectory(), Info()
            traj.ParseFromString(bytes_traj)
            info.ParseFromString(bytes_info)

            print("num_steps: {}".format(info.num_steps))
            traj_shape = np.append(np.array(info.num_steps), np.array(info.vector_obs_spec))
            print("trajectory shape: {}".format(traj_shape))
            observations = np.array(traj.vector_obs).reshape(traj_shape)
            
            for i in range(info.num_steps):
                print("obs: {}".format(observations[i]))
                print("action: {}".format(traj.actions[i]))
                print("reward: {}".format(traj.rewards[i]))
                print("-----------------------------")
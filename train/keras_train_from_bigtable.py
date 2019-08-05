import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.cloud import bigtable
from google.cloud.bigtable import row_filters

from protobuf.experience_replay_pb2 import Trajectory, Info
from train.dqn_model import DQN_Model

SCOPES = ['https://www.googleapis.com/auth/bigtable.admin']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#MODEL HYPERPARAMETERS
fc_layer_params = (200,)
learning_rate = 1e-3

#ENV PARAMETERS
gamma = 1.0 
epsilon = 0.1
max_nb_actions = 2
min_array_obs = [-4.8000002e+00, -3.4028234663852886e+38, -4.1887903e-01, -3.4028234663852886e+38]
max_array_obs = [4.8000002e+00, 3.4028234663852886e+38, 4.1887903e-01, 3.4028234663852886e+38]

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Read-From-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='cartpole-experience-replay')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--train-steps', type=int, default=10000)
    parser.add_argument('--period', type=int, default=100)
    args = parser.parse_args()

    model = DQN_Model(num_actions=2,
                      fc_layer_params=(200,))
    if args.restore:
        model.load_weights(args.restore)

    #LOAD/CREATE CBT TABLE
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

    #TRAINING LOOP
    for i in tqdm(range(5000), "Training"):
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

            traj_shape = np.append(np.array(info.num_steps), np.array(info.vector_obs_spec))
            obs = np.array(traj.vector_obs).reshape(traj_shape)
            next_obs = np.roll(obs, 1)
            input()

            with tf.GradientTape() as tape:
                q_pred = model(obs)
                q_next = model(next_obs)
                np.apply_along_axis(np.argmax())
                q_target = traj.rewards + model(next_obs)



            # traj_obs = np.rollaxis(np.array([obs, np.roll(obs, 1)]), 0 , 2)
            # traj_actions = np.rollaxis(np.array([traj.actions, np.roll(traj.actions, 1)]), 0 , 2)
            # traj_rewards = np.rollaxis(np.array([traj.rewards, np.roll(traj.rewards, 1)]), 0 , 2)
            # traj_discounts = np.ones((info.num_steps,2))
            

            # traj_obs = tf.constant(traj_obs, dtype=tf.float32)
            # traj_actions = tf.constant(traj_actions, dtype=tf.int32)
            # policy_info = ()
            # traj_rewards = tf.constant(traj_rewards, dtype=tf.float32)
            # traj_discounts = tf.constant(traj_discounts, dtype=tf.float32)

            # traj = trajectory.boundary(traj_obs, traj_actions, policy_info, traj_rewards, traj_discounts)
            # train_loss = tf_agent.train(traj)
        
        # tf_agent._q_network.save_weights("cartpole_model.h5")
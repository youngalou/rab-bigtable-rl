import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.cloud import bigtable
from google.cloud.bigtable import row_filters
from protobuf.experience_replay_pb2 import Trajectory, Info

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.replay_buffers import py_uniform_replay_buffer

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
    args = parser.parse_args()

    #INITIALIZE RL AGENT
    observation_spec = tensor_spec.BoundedTensorSpec(             # Make observation spec manually
            shape=(len(min_array_obs),), dtype=np.float32, minimum=min_array_obs, maximum=max_array_obs)

    action_spec = tensor_spec.BoundedTensorSpec(                  # Make action spec manually
            shape=(), dtype=np.int32, minimum=0, maximum=max_nb_actions - 1)

    time_step_spec = ts.time_step_spec(tensor_spec.from_spec(observation_spec))

    q_net = q_network.QNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.compat.v2.Variable(0, dtype='int64')
    
    tf_agent = dqn_agent.DqnAgent(
        time_step_spec,
        action_spec,
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=tf.compat.v2.keras.losses.MSE,
        train_step_counter=train_step_counter,
        gamma=gamma,
        epsilon_greedy=epsilon)
    tf_agent.initialize()

    #LOAD/CREATE CBT TABLE
    print('Looking for the [{}] table.'.format(args.cbt_table_name))
    client = bigtable.Client(args.gcp_project_id, admin=True)
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
            observations = np.array(traj.vector_obs).reshape(traj_shape)
            traj_obs = np.rollaxis(np.array([observations, np.roll(observations, 1)]), 0 , 2)
            traj_actions = np.rollaxis(np.array([traj.actions, np.roll(traj.actions, 1)]), 0 , 2)
            traj_rewards = np.rollaxis(np.array([traj.rewards, np.roll(traj.rewards, 1)]), 0 , 2)
            traj_discounts = np.ones((info.num_steps,2))

            traj_obs = tf.constant(traj_obs, dtype=tf.float32)
            traj_actions = tf.constant(traj_actions, dtype=tf.int32)
            policy_info = ()
            traj_rewards = tf.constant(traj_rewards, dtype=tf.float32)
            traj_discounts = tf.constant(traj_discounts, dtype=tf.float32)

            traj = trajectory.boundary(traj_obs, traj_actions, policy_info, traj_rewards, traj_discounts)
            train_loss = tf_agent.train(traj)
        
        tf_agent._q_network.save_weights("cartpole_model.h5")
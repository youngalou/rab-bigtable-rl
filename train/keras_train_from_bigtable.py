import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account
from google.cloud import bigtable
from google.cloud.bigtable import row_filters

from protobuf.experience_replay_pb2 import Trajectory, Info
from train.dqn_model import DQN_Model

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#ENV PARAMETERS
GAMMA = 0.9
EPSILON = 0.1
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
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    args = parser.parse_args()

    #LOAD/RESTORE MODEL
    model = DQN_Model(num_actions=2,
                      fc_layer_params=(200,),
                      learning_rate=.00042)
    if args.restore:
        model.load_weights(args.restore)

    #LOAD/CREATE CBT TABLE
    print("Looking for the [{}] table.".format(args.cbt_table_name))
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = bigtable.Client(args.gcp_project_id, admin=True, credentials=credentials)
    instance = client.instance(args.cbt_instance_id)
    table = instance.table(args.cbt_table_name)
    if not table.exists():
        print("Table doesn't exist.")
        exit()
    else:
        print("Table found.")

    os.makedirs(os.path.dirname(os.path.join(args.output_dir, 'models/')), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(args.output_dir, 'logs/')), exist_ok=True)

    #SETUP TENSORBOARD/METRICS
    loss_metrics = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_log_dir =  os.path.join(args.output_dir, "logs/")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    #TRAINING LOOP
    for i in tqdm(range(5000), "Training"):
        #QUERY TABLE FOR PARTIAL ROWS
        regex_filter = '^cartpole_trajectory_{}$'.format(i)
        row_filter = row_filters.RowKeyRegexFilter(regex_filter)
        filtered_rows = table.read_rows(filter_=row_filter)

        for row in filtered_rows:
            #PARSE ROWS
            bytes_traj = row.cells['trajectory']['traj'.encode()][0].value
            bytes_info = row.cells['trajectory']['info'.encode()][0].value
            traj, info = Trajectory(), Info()
            traj.ParseFromString(bytes_traj)
            info.ParseFromString(bytes_info)

            #FORMAT DATA
            traj_shape = np.append(np.array(info.num_steps), np.array(info.vector_obs_spec))
            obs = np.array(traj.vector_obs).reshape(traj_shape)
            next_obs = np.roll(obs, 1)

            #COMPUTE GRADIENTS
            with tf.GradientTape() as tape:
                q_pred = model(obs)
                q_pred = [q[a] for q, a in zip(q_pred, traj.actions)]
                q_next = model(next_obs)
                q_next = [tf.argmax(q) for q in q_next]
                q_target = traj.rewards + tf.multiply(tf.constant(GAMMA, dtype=tf.float32), q_next)

                mse = tf.keras.losses.MeanSquaredError()
                loss = mse(q_pred, q_target)

            #APPLY GRADIENTS
            total_grads = tape.gradient(loss, model.trainable_weights)
            model.opt.apply_gradients(zip(total_grads, model.trainable_weights))

            #TENSORBOARD LOGGING
            loss_metrics(loss)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_metrics.result(), step=i)

        #SAVE MODEL WEIGHTS
        if i > 0 and i % args.period == 0:
            output_filepath = os.path.join(args.output_dir, 'models/', 'cartpole_model_{}.h5'.format(i))
            model.save_weights(output_filepath)
            print("Saved model to [{}].".format(output_filepath))
            # model_json = model.to_json()
            # print(model_json)

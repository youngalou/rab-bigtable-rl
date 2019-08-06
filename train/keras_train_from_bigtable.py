import os
import argparse
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account
from google.cloud.bigtable import row_filters

from protobuf.experience_replay_pb2 import Trajectory, Info
from train.dqn_model import DQN_Model
from train.gcp_io import cbt_load_table, gcs_load_weights, gcs_save_model

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#ENV PARAMETERS
VECTOR_OBS_SPEC = [4]
GAMMA = 0.9
max_nb_actions = 2
min_array_obs = [-4.8000002e+00, -3.4028234663852886e+38, -4.1887903e-01, -3.4028234663852886e+38]
max_array_obs = [4.8000002e+00, 3.4028234663852886e+38, 4.1887903e-01, 3.4028234663852886e+38]

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Read-From-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='cartpole-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--model-prefix', type=str, default='cartpole_model')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--train-steps', type=int, default=10000)
    parser.add_argument('--period', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND LOAD MODEL FROM GCS
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    table = cbt_load_table(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, credentials)
    bucket, model_found = gcs_load_weights(args.gcp_project_id, args.bucket_id, credentials, args.model_prefix, args.tmp_weights_filepath)

    #LOAD MODEL
    model = DQN_Model(input_shape=VECTOR_OBS_SPEC,
                      num_actions=2,
                      fc_layer_params=(200,),
                      learning_rate=.00042)
    if model_found:
        model.load_weights(args.tmp_weights_filepath)

    #SETUP TENSORBOARD/METRICS
    os.makedirs(os.path.dirname(os.path.join(args.output_dir, 'models/')), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(args.output_dir, 'logs/')), exist_ok=True)
    loss_metrics = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_log_dir =  os.path.join(args.output_dir, "logs/")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    #TRAINING LOOP
    print("-> Starting training...")
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
            model_filename = args.model_prefix + '_{}.h5'.format(i)
            gcs_save_model(model, bucket, args.tmp_weights_filepath, model_filename)
    print("-> Done!")
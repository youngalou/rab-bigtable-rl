import argparse
from google.oauth2 import service_account

from util.gcp_io import gcp_load_pipeline
from crane.dqn_agent import DQN_Agent

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

hyperparams = dict([
    ('visual_obs_shape', [224,224,2]),
    ('vector_obs_shape', [29]),
    ('num_actions', 8),
    ('conv_layer_params', ((8,4,32),(4,2,64),(3,1,64))),
    ('fc_layer_params', (512,)),
    ('gamma', 0.9),
    ('update_horizon', 5),
    ('learning_rate', 0.00042)
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='rab-arms-dualobs-frameskip-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='youngalou')
    parser.add_argument('--prefix', type=str, default='rab-arms-dualobs-frameskip')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--buffer-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-trajectories', type=int, default=10)
    parser.add_argument('--train-epochs', type=int, default=1000000)
    parser.add_argument('--train-steps', type=int, default=100)
    parser.add_argument('--period', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    parser.add_argument('--log-time', default=False, action='store_true')
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--tpu-name', type=str, default='grpc://10.240.1.2:8470')
    parser.add_argument('--wandb', type=str, default=None)
    args = parser.parse_args()

    if args.wandb is not None:
        import wandb
        wandb.init(name=args.wandb,
                   project="rab-bigtable-rl",
                   entity="42 Robolab")
        wandb.config.update({"visual_obs_shape": hyperparams['visual_obs_shape'],
                             "vector_obs_shape": hyperparams['vector_obs_shape'],
                             "num_actions": hyperparams['num_actions'],
                             "conv_layer_params": hyperparams['conv_layer_params'],
                             "fc_layer_params": hyperparams['fc_layer_params'],
                             "gamma": hyperparams['gamma'],
                             "learning_rate": hyperparams['learning_rate'],
                             "buffer_size": args.buffer_size,
                             "batch_size": args.batch_size,
                             "num_trajectories": args.num_trajectories,
                             "train_epochs": args.train_epochs,
                             "train_steps": args.train_steps,
                             "period": args.period,
                             "num_gpus": args.num_gpus,
                             "tpu_name": args.tpu_name})
    else: wandb = None

    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)

    agent = DQN_Agent(hyperparams=hyperparams,
                      cbt_table=cbt_table,
                      gcs_bucket=gcs_bucket,
                      gcs_bucket_id=args.bucket_id,
                      prefix=args.prefix,
                      tmp_weights_filepath=args.tmp_weights_filepath,
                      buffer_size=args.buffer_size,
                      batch_size=args.batch_size,
                      num_trajectories=args.num_trajectories,
                      train_epochs=args.train_epochs,
                      train_steps=args.train_steps,
                      period=args.period,
                      output_dir=args.output_dir,
                      log_time=args.log_time,
                      num_gpus=args.num_gpus,
                      tpu_name=args.tpu_name,
                      wandb=wandb)
    agent.train()
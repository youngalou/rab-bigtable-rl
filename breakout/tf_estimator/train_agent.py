import argparse
from google.oauth2 import service_account

from util.gcp_io import gcp_load_pipeline
from util.logging import TimeLogger
from models.estimator_model import DQN_Model
from breakout.tf_estimator.dqn_agent import DQN_Agent

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='breakout-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='breakout')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-trajectories', type=int, default=10)
    parser.add_argument('--buffer-size', type=int, default=10008000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-epochs', type=int, default=1000000)
    parser.add_argument('--train-steps', type=int, default=100)
    parser.add_argument('--period', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    parser.add_argument('--log-time', default=False, action='store_true')
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--wandb', type=str, default=None)
    args = parser.parse_args()

    if args.wandb is not None:
        import wandb
        wandb.init(name=args.wandb,
                   project="rab-bigtable-rl",
                   entity="42 Robolab")

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)

    agent = DQN_Agent(cbt_table=cbt_table,
                      gcs_bucket=gcs_bucket,
                      prefix=args.prefix,
                      tmp_weights_filepath=args.tmp_weights_filepath,
                      num_trajectories=args.num_trajectories,
                      buffer_size=args.buffer_size,
                      batch_size=args.batch_size,
                      train_epochs=args.train_epochs,
                      train_steps=args.train_steps,
                      period=args.period,
                      output_dir=args.output_dir,
                      log_time=args.log_time,
                      num_gpus=args.num_gpus)
    agent.train()
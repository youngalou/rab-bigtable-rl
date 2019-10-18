# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Train and Eval DQN.

To run DQN on CartPole:

```bash
tensorboard --logdir $HOME/tmp/dqn/gym/CartPole-v0/ --port 2223 &

python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn/gym/CartPole-v0/ \
  --alsologtostderr
```

To run DQN-RNNs on MaskedCartPole:

```bash
python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn_rnn/gym/MaskedCartPole-v0/ \
  --gin_param='train_eval.env_name="MaskedCartPole-v0"' \
  --gin_param='train_eval.train_sequence_length=10' \
  --alsologtostderr
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole  # pylint: disable=unused-import
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from tfa.bigtable_replay_buffer import BigtableReplayBuffer
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, \
                        cbt_global_iterator, cbt_global_trajectory_buffer

#SET API CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

def bigtable_collect(
    root_dir,
    env_name='CartPole-v0',
    num_iterations=100000,
    # Params for QNetwork
    fc_layer_params=(100,),
    # Params for QRnnNetwork
    input_fc_layer_params=(50,),
    lstm_size=(20,),
    output_fc_layer_params=(20,),

    # Params for collect
    num_episodes=1,
    epsilon_greedy=0.1,
    replay_buffer_capacity=100000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    learning_rate=1e-3,
    n_step_update=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=1000,
    # Params for checkpoints
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=20000,
    # Params for summaries and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=fc_layer_params)
    train_sequence_length = n_step_update

    # TODO(b/127301657): Decay epsilon based on global step, cf. cl/188907839
    tf_agent = dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        epsilon_greedy=epsilon_greedy,
        n_step_update=n_step_update,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)
    max_row_bytes = (4*np.prod(VISUAL_OBS_SPEC) + 64)
    cbt_batcher = cbt_table.mutations_batcher(flush_count=args.num_episodes, max_row_bytes=max_row_bytes)

    bigtable_replay_buffer = BigtableReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        max_size=replay_buffer_capacity)

    collect_driver = dynamic_episode_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[bigtable_replay_buffer.add_batch] + train_metrics,
        num_episodes=num_episodes)

    # train_checkpointer = common.Checkpointer(
    #     ckpt_dir=train_dir,
    #     agent=tf_agent,
    #     global_step=global_step,
    #     metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    # policy_checkpointer = common.Checkpointer(
    #     ckpt_dir=os.path.join(train_dir, 'policy'),
    #     policy=eval_policy,
    #     global_step=global_step)
    # rb_checkpointer = common.Checkpointer(
    #     ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
    #     max_to_keep=1,
    #     replay_buffer=bigreplay_buffer)

    # train_checkpointer.initialize_or_restore()
    # rb_checkpointer.initialize_or_restore()

    if use_tf_functions:
      # To speed up collect use common.function.
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    for _ in range(num_iterations):
        collect_driver.run()

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  bigtable_collect(root_dir='tfa_output')

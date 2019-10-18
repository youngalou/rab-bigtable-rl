import numpy as np
import tensorflow as tf

import pickle

from protobuf.tf_agents_trajectory_pb2 import CurrentTimeStep, ActionStep, NextTimeStep

from tf_agents.replay_buffers import replay_buffer
from tf_agents.specs import array_spec
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage

class BigtableReplayBuffer():
    def __init__(self, cbt_table, data_spec, max_size):
        self.cbt_table = cbt_table
        self.data_spec = data_spec
        self.max_size = max_size
        self.rows = []

    def add_batch(self, traj):
        current_timestep, actionstep, next_timestep = CurrentTimeStep(), ActionStep(), NextTimeStep()
        current_timestep.step_type = traj.step_type
        current_timestep.observation = traj.observation.flatten().tobytes()
        actionstep.action = traj.action.astype(np.int32).tobytes()
        actionstep.policy_info = traj.policy_info
        next_timestep.step_type = traj.next_step_type
        next_timestep.reward = traj.reward
        next_timestep.discount = traj.discount

        row_key = 'traj_{:05d}_step_{:05d}'.format(global_i, step).encode()
        row = self.cbt_table.row(row_key)
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
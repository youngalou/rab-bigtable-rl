import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

class CustomConvs(tf.keras.Model):
    def __init__(self,
                 conv_layer_params,
                 activation='relu'):
        super().__init__()
        self.conv_layers = [Conv2D(padding='same',
                                    kernel_size=k,
                                    strides=s,
                                    filters=f,
                                    activation=activation,
                                    name='conv_{}'.format(i))
                      for i,(k,s,f) in enumerate(conv_layer_params)]
        self.flatten = Flatten()
    
    def call(self, inputs):
        for conv_layer in self.conv_layers:
            inputs = conv_layer(inputs)
        embedding = self.flatten(inputs)
        return embedding

class ValueHead(tf.keras.Model):
    def __init__(self,
                 fc_layer_params,
                 activation='relu'):
        super().__init__()
        self.fc_layers = [Dense(neurons, activation='relu', name='value_fc_layer_{}'.format(i)) \
                            for i,(neurons) in enumerate(fc_layer_params)]
        self.value_output = Dense(1, name='value_output')

    def call(self, inputs):
        for layer in self.fc_layers:
            inputs = layer(inputs)
        return self.value_output(inputs)

class AdvantageHead(tf.keras.Model):
    def __init__(self,
                 fc_layer_params,
                 num_actions=None,
                 activation='relu'):
        super().__init__()
        self.fc_layers = [Dense(neurons, activation='relu', name='advantage_fc_layer_{}'.format(i)) \
                            for i,(neurons) in enumerate(fc_layer_params)]
        self.advantage_output = Dense(num_actions, name='advantage_output')

    def call(self, inputs):
        for layer in self.fc_layers:
            inputs = layer(inputs)
        return self.advantage_output(inputs)

class DQN_Model(tf.keras.Model):
    def __init__(self,
                 visual_obs_shape=None,
                 vector_obs_shape=None,
                 num_actions=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 learning_rate=0.00042):
        super().__init__()
        if visual_obs_shape and conv_layer_params:
            self.convs = CustomConvs(conv_layer_params)
        else:
            self.convs = None
        if fc_layer_params is not None:
            self.value_head = ValueHead(fc_layer_params)
            self.advantage_head = AdvantageHead(fc_layer_params, num_actions)

        init_visual_obs = np.zeros(visual_obs_shape, dtype=np.float32) if visual_obs_shape else None
        init_vector_obs = np.zeros(vector_obs_shape, dtype=np.float32) if vector_obs_shape else None
        self.step((init_visual_obs, init_vector_obs))
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        self.opt = tf.optimizers.Adam(learning_rate)

        self.num_actions = num_actions
        self.public_url = None

    def call(self, inputs):
        (visual_obs, vector_obs) = inputs
        if self.convs is not None:
            embedding = self.convs(visual_obs)
            fc_inputs = tf.concat([embedding, vector_obs], axis=1)
        else:
            fc_inputs = vector_obs
        value = self.value_head(fc_inputs)
        advantage = self.advantage_head(fc_inputs)
        output = value + (advantage - tf.reduce_mean(advantage))
        return tf.cast(output, dtype=tf.float32)
    
    def step(self, inputs):
        (visual_obs, vector_obs) = inputs
        visual_obs = np.expand_dims(visual_obs, 0) if visual_obs is not None else None
        vector_obs = np.expand_dims(vector_obs, 0) if vector_obs is not None else None
        inputs = (visual_obs, vector_obs)
        q_values = tf.squeeze(self(inputs))
        action = tf.argmax(q_values).numpy()
        return action

    def step_stochastic(self, inputs):
        (visual_obs, vector_obs) = inputs
        visual_obs = np.expand_dims(visual_obs, 0) if visual_obs is not None else None
        vector_obs = np.expand_dims(vector_obs, 0) if vector_obs is not None else None
        inputs = (visual_obs, vector_obs)
        logits = self(inputs)
        action = tf.squeeze(tf.random.categorical(logits, 1)).numpy()
        return action

    def step_epsilon_greedy(self, inputs, epsilon=0):
        sample = np.random.random()
        if sample > 1 - epsilon:
            return np.random.randint(self.num_actions)
        return self.step(inputs)

class ExperienceBuffer():
    def __init__(self, max_size, update_horizon):
        self.vis_obs,  self.vec_obs, self.actions, self.rewards, \
            self.next_vis_obs, self.next_vec_obs, self.next_mask = \
            [], [], [], [], [], [], []
        
        self.max_size = max_size
        self.update_horizon = update_horizon
        self.size = 0
    
    def reset(self):
        self.__init__(self.max_size, self.update_horizon)

    def add_trajectory(self, obs, actions, rewards, num_steps):
        (vis_obs, vec_obs) = obs
        shift = self.update_horizon * -1
        next_vis_obs = np.roll(vis_obs, shift=shift, axis=0)
        next_vec_obs = np.roll(vec_obs, shift=shift, axis=0)
        next_mask = np.append(np.ones(num_steps-self.update_horizon), np.zeros(self.update_horizon))
        next_mask = next_mask.astype(np.float32)

        if self.size >= self.max_size:
            self.reset()

        new_size = self.size + num_steps
        if new_size > self.max_size:
            vis_obs, vec_obs, actions, rewards, next_vis_obs, next_vec_obs, next_mask = \
                self.truncate(vis_obs, vec_obs, actions, rewards, next_vis_obs, next_vec_obs, next_mask)
            new_size = self.max_size

        self.append(vis_obs, vec_obs, actions, rewards, next_vis_obs, next_vec_obs, next_mask)
        self.size = new_size

    def append(self, vis_obs, vec_obs, actions, rewards, next_vis_obs, next_vec_obs, next_mask):
        self.vis_obs.append(vis_obs)
        self.vec_obs.append(vec_obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_vis_obs.append(next_vis_obs)
        self.next_vec_obs.append(next_vec_obs)
        self.next_mask.append(next_mask)
    
    def preprocess(self):
        self.vis_obs = np.concatenate(self.vis_obs, axis=0)
        self.vec_obs = np.concatenate(self.vec_obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.next_vis_obs = np.concatenate(self.next_vis_obs, axis=0)
        self.next_vec_obs = np.concatenate(self.next_vec_obs, axis=0)
        self.next_mask = np.concatenate(self.next_mask, axis=0)

    def truncate(self, vis_obs, vec_obs, actions, rewards, next_vis_obs, next_vec_obs, next_mask):
        split = self.max_size - self.size
        return vis_obs[:split], vec_obs[:split], actions[:split], rewards[:split],\
            next_vis_obs[:split], next_vec_obs[:split], next_mask[:split]
        
    def serve_to_dataset(self):
        return (((self.vis_obs, self.vec_obs), (self.next_vis_obs, self.next_vec_obs)), \
            (self.actions, self.rewards, self.next_mask))
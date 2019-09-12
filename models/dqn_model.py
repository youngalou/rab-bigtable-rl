import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

class Custom_Convs(tf.keras.Model):
    def __init__(self,
                 conv_layer_params,
                 activation="relu"):
        super().__init__(name='')
        tf.keras.backend.set_floatx('float32')

        self.conv_layers = [Conv2D(padding="same",
                                    kernel_size=k,
                                    strides=s,
                                    filters=f,
                                    activation=activation,
                                    name="conv_{}".format(i))
                      for i,(k,s,f) in enumerate(conv_layer_params)]
        self.flatten = Flatten()
    
    def call(self, inputs):
        for conv_layer in self.conv_layers:
            inputs = conv_layer(inputs)
        embedding = self.flatten(inputs)
        return embedding

class DQN_Model(tf.keras.Model):
    def __init__(self,
                 input_shape=None,
                 num_actions=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 learning_rate=0.00042):
        super().__init__()
        if conv_layer_params is not None:
            self.convs = Custom_Convs(conv_layer_params)
        else: self.convs = None
        if fc_layer_params is not None:
            self.fc_layers = [Dense(neurons, activation="relu", name="fc_layer_{}".format(i)) for i,(neurons) in enumerate(fc_layer_params)]
        self.q_layer = Dense(num_actions, name='output')

        self.step(np.zeros(input_shape, dtype=np.float32))
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        self.opt = tf.optimizers.Adam(learning_rate)

        self.num_actions = num_actions
        self.public_url = None

    def call(self, inputs):
        if self.convs is not None:
            inputs = self.convs(inputs)
        for layer in self.fc_layers:
            inputs = layer(inputs)
        logits = self.q_layer(inputs)
        return tf.cast(logits, dtype=tf.float32)
    
    def step(self, inputs):
        inputs = np.expand_dims(inputs, 0)
        q_values = tf.squeeze(self(inputs))
        action = tf.argmax(q_values).numpy()
        return action

    def step_stochastic(self, inputs):
        inputs = np.expand_dims(inputs, 0)
        logits = self(inputs)
        action = tf.squeeze(tf.random.categorical(logits, 1)).numpy()
        return action

    def step_epsilon_greedy(self, inputs, epsilon):
        sample = np.random.random()
        if sample > 1 - epsilon:
            return np.random.randint(self.num_actions)
        return self.step(inputs)

class ExperienceBuffer():
    def __init__(self, max_size, update_horizon):
        self.obs, self.actions, self.rewards, self.next_obs, self.next_mask = \
            [], [], [], [], []
        
        self._obs, self._actions, self._rewards, self._next_obs, self._next_mask = \
            [], [], [], [], []
        
        self.max_size = max_size
        self.update_horizon = update_horizon
        self.size = 0

    def add_trajectory(self, obs, actions, rewards, num_steps):
        shift = self.update_horizon * -1
        next_obs = np.roll(obs, shift=shift, axis=0)
        next_mask = np.append(np.ones(num_steps-self.update_horizon), np.zeros(self.update_horizon))
        next_mask = next_mask.astype(np.float32)

        self.append(obs, actions, rewards, next_obs, next_mask)
        self.size = self.size + num_steps

    def append(self, obs, actions, rewards, next_obs, next_mask):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_obs.append(next_obs)
        self.next_mask.append(next_mask)
    
    def preprocess(self):
        self.obs = np.concatenate(self.obs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.next_obs = np.concatenate(self.next_obs, axis=0)
        self.next_mask = np.concatenate(self.next_mask, axis=0)

    def reset(self):
        self.__init__(self.max_size, self.update_horizon)
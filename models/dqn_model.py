import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

class Custom_Convs(tf.keras.Model):
    def __init__(self,
                 conv_layer_params,
                 activation="relu"):
        super().__init__(name='')

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
        if fc_layer_params is not None:
            self.fc_layers = [Dense(neurons, activation="relu", name="fc_layer_{}".format(i)) for i,(neurons) in enumerate(fc_layer_params)]
        self.q_layer = Dense(num_actions, name='output')

        self.step(np.zeros(input_shape))
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
    def __init__(self, max_size):
        self.obs, self.actions, self.rewards, self.next_obs, self.next_mask = \
            None, None, None, None, None
        
        self._obs, self._actions, self._rewards, self._next_obs, self._next_mask = \
            None, None, None, None, None
        
        self.max_size = max_size
        self.size = 0
        self.remainder = 0

    def add_trajectory(self, obs, actions, rewards, num_steps):
        next_obs = np.roll(obs, shift=-1, axis=0)
        next_mask = np.ones(num_steps)
        next_mask[-1] = 0

        # if self.size >= self.max_size:
        #     self.reset()

        new_size = self.size + obs.size
        # if new_size > self.max_size:
        #     obs, actions, rewards, next_obs, next_mask = \
        #         self.split_remainder(obs, actions, rewards, next_obs, next_mask)
        #     new_size = self.max_size

        self.append(obs, actions, rewards, next_obs, next_mask)
        self.size = new_size

    def append(self, obs, actions, rewards, next_obs, next_mask):
        if self.size == 0:
            self.obs, self.actions, self.rewards, self.next_obs, self.next_mask = \
                obs, actions, rewards, next_obs, next_mask
        else:
            self.obs = np.append(self.obs, obs, axis=0)
            self.actions = np.append(self.actions, actions, axis=0)
            self.rewards = np.append(self.rewards, rewards, axis=0)
            self.next_obs = np.append(self.next_obs, next_obs, axis=0)
            self.next_mask = np.append(self.next_mask, next_mask, axis=0)
    
    def split_remainder(self, obs, actions, rewards, next_obs, next_mask):
        split = self.max_size - self.size
        obs, self._obs = obs[:split], obs[split:]
        actions, self._actions = actions[:split], actions[split:]
        rewards, self._rewards = rewards[:split], rewards[split:]
        next_obs, self._next_obs = next_obs[:split], next_obs[split:]
        next_mask, self._next_mask = next_mask[:split], next_mask[split:]
        self.remainder = self._obs.shape[0]
        return obs, actions, rewards, next_obs, next_mask

    def reset(self):
        if self.remainder == 0:
            self.__init__(self.max_size)
        else:
            self.size = self.remainder
            self.remainder = 0
            self.obs, self.actions, self.rewards, self.next_obs, self.next_mask = \
                self._obs, self._actions, self._rewards, self._next_obs, self._next_mask
            self._obs, self._actions, self._rewards, self._next_obs, self._next_mask = \
                None, None, None, None, None
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

class Custom_Convs(tf.keras.Model):
    def __init__(self, conv_layer_params, actv="relu"):
        super().__init__(name='')

        self.convs = [Conv2D(padding="same",
                                    kernel_size=k,
                                    strides=s,
                                    filters=f,
                                    activation=actv,
                                    name="conv_{}".format(i))
                      for i,(k,s,f) in enumerate(conv_layer_params)]
        self.flatten = Flatten()
    
    def call(self, inputs):
        for conv in self.convs:
            inputs = conv(inputs)
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
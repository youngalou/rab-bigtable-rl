import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DQN_Model(tf.keras.models.Model):
    def __init__(self,
                 input_shape=None,
                 num_actions=None,
                 fc_layer_params=None,
                 learning_rate=0.00042):
        super().__init__()
        self.fc_layers = [layers.Dense(neurons, activation="relu", name="fc_layer_{}".format(i)) for i,(neurons) in enumerate(fc_layer_params)]
        self.q_layer = layers.Dense(num_actions, name='output')

        self.step(np.zeros(input_shape))
        self.opt = tf.optimizers.Adam(learning_rate)

        self.public_url = None

    def call(self, inputs):
        for layer in self.fc_layers:
            inputs = layer(inputs)
        q_values = self.q_layer(inputs)
        return tf.squeeze(q_values)
    
    def step(self, inputs):
        inputs = np.expand_dims(inputs, 0)
        q_values = self(inputs)
        action = tf.argmax(q_values).numpy()
        return action
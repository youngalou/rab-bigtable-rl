import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class DQN_Model(tf.keras.models.Model):
    def __init__(self,
                 num_actions=None,
                 fc_layer_params=None):
        super().__init__()
        self.fc_layers = [layers.Dense(neurons, activation="relu", name="fc_layer_{}".format(i)) for i,(neurons) in enumerate(fc_layer_params)]
        self.q_layer = layers.Dense(num_actions, name='output')

    def call(self, inputs):
        for layer in self.fc_layers:
            inputs = layer(inputs)
        q_values = self.q_layer(inputs)
        return tf.squeeze(q_values)
    
    def step(self, inputs):
        inputs = np.expand_dims(inputs, 0)
        q_values = self(inputs)
        t = tf.argmax(q_values)
        action = K.eval(tf.argmax(q_values))
        return action

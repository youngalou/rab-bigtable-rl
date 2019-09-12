import numpy as np
import tensorflow as tf

tpu_address = 'youngalou'
device = '/job:worker'
train_epochs = 100
train_steps = 100
dataset_size = 1000
batch_size = 256

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
tf.config.experimental_connect_to_host(cluster_resolver.master())
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

def get_dataset():
    dataset = tf.data.Dataset.from_tensor_slices((np.zeros((dataset_size,128),dtype=np.float32), np.zeros((dataset_size,1),dtype=np.float32)))
    dataset = dataset.shuffle(dataset_size).repeat().batch(batch_size)
    return tpu_strategy.experimental_distribute_dataset(dataset)

with tf.device(device), tpu_strategy.scope():
    for _ in range(train_epochs):
        dataset = get_dataset()
        exp_buff = iter(dataset)

        for _ in range(train_steps):
            train_batch = next(exp_buff)
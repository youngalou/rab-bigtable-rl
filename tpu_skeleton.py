import numpy as np
import tensorflow as tf

tpu_address = 'youngalou'
device = '/job:worker'
train_epochs = 100
train_steps = 100
dataset_size = 1000
batch_size = 256
learning_rate = 0.00042

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
tf.config.experimental_connect_to_host(cluster_resolver.master())
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

def create_model():
    #DEFINE YOUR MODEL HERE
    pass

with tf.device(device), tpu_strategy.scope():
    model = create_model()
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    opt = tf.optimizers.Adam(learning_rate)

@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        (features, labels) = inputs

        with tf.GradientTape() as tape:
            predictions = model(features)

            dist_losses = mse(predictions, labels)
            loss = tf.reduce_sum(dist_losses)
        
        total_grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(list(zip(total_grads, model.trainable_weights)))
        return dist_losses

    per_example_losses = tpu_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = tpu_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=None)
    return mean_loss

def get_dataset():
    features = None #LOAD FEATURES HERE
    labels = None #LOAD LABELS HERE
    dataset = tf.data.Dataset.from_tensor_slices(features, labels)
    dataset = dataset.shuffle(dataset_size).repeat().batch(batch_size)
    return tpu_strategy.experimental_distribute_dataset(dataset)

for _ in range(train_epochs):
    with tf.device(device), tpu_strategy.scope():
        dataset = get_dataset()
        exp_buff = iter(dataset)

        for _ in range(train_steps):
            train_batch = next(exp_buff)
            mean_loss = train_step(train_batch)
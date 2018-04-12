import tensorflow as tf
import numpy as np
from DataCollectorClass import DataCollector
from FeatureExtractorClass import FeatureExtractor
from sklearn import metrics
from matplotlib import pyplot as plt
import os
import glob
import h5py

def collect_h5_files(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".hdf5"):
                files_list.append(os.path.join(root, file))
    return files_list



path_to_train_h5 = "D:\\h5dataset\\train"
path_to_validation_h5 = "D:\\h5dataset\\validation"
train_files = collect_h5_files(path_to_train_h5)
validation_files = collect_h5_files(path_to_validation_h5)


def _read_train_py_function(filename):
    with h5py.File(filename, 'r') as f:
        spectrogram = f['train_data']
        spectrogram.astype(np.float32)
        label = f['train_data_labels']
        label.astype(np.float32)
    return spectrogram, label


def _read_validation_py_function(filename):
    with h5py.File(filename, 'r') as f:
        spectrogram = f['validation_data']
        spectrogram.astype(np.float32)
        label = f['validation_data_labels']
        label.astype(np.float32)
    return spectrogram, label


batch_size = 10
num_epochs = 10
train_dataset = tf.data.Dataset.from_tensors(train_files)
train_dataset = train_dataset.map(
    lambda filename: tuple(tf.py_func(
        _read_train_py_function, [filename], [tf.float32, tf.float32])))
train_dataset = train_dataset.repeat(num_epochs)
#train_dataset = train_dataset.padded_batch(batch_size)



validation_dataset = tf.data.Dataset.from_tensors(validation_files)
validation_dataset = validation_dataset.map(
    lambda filename: tuple(tf.py_func(
        _read_validation_py_function, [filename], [tf.float32, tf.float32])))

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(train_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)



writer = tf.summary.FileWriter('./events')
writer.add_graph(tf.get_default_graph())
n_classes = 2
learning_rate = 0.001

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, shape=(21, 256))
    x = tf.reshape(x, [-1,  21, 256, 1], name='input_tensor')
    y = tf.placeholder(tf.float32, shape=(None, 1))

# Convolution Layer with 50 filters and a kernel size of 5
conv1 = tf.layers.conv2d(x, 20, 5, activation=tf.nn.relu)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

# Convolution Layer with 50 filters and a kernel size of 5
conv2 = tf.layers.conv2d(conv1, 50, 5, activation=tf.nn.relu)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

# Flatten the data to a 1-D vector for the fully connected layer
fc1 = tf.contrib.layers.flatten(conv2)

# Fully connected layer (in tf contrib folder for now)
fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)

# Output layer, class prediction
out = tf.layers.dense(fc1, n_classes)


# Predictions
y_pred = tf.argmax(out, axis=1)
#y_pred = tf.expand_dims(y_pred, 0)
pred_probas = tf.nn.softmax(out)

# Define loss and optimizer
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=out, labels=tf.cast(y, tf.int32)))
loss_summary = tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

validation_op = loss
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
print('Start training...\n')

for i in range(4800):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  if  i % 100 == 0:
      # Initialize an iterator over the validation dataset.
      sess.run(validation_init_op)
      for _ in range(50):
         val_loss = sess.run(next_element)

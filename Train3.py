import tensorflow as tf
import numpy as np
from DataCollectorClass import DataCollector
from FeatureExtractorClass import FeatureExtractor
from sklearn import metrics
from matplotlib import pyplot as plt
import os
import glob
import h5py
import tqdm

def collect_h5_files(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".hdf5"):
                files_list.append(os.path.join(root, file))
    return files_list


#path_to_train_h5 = "D:\\h5dataset\\train"
path_to_train_h5 = "D:\\h5dataset\\validation"
path_to_validation_h5 = "D:\\h5dataset\\validation"
train_files = collect_h5_files(path_to_train_h5)
validation_files = collect_h5_files(path_to_validation_h5)


def _read_py_function(filename):
    with h5py.File(filename, 'r') as f:
        spectrogram = f['data'].value
        spectrogram = np.float32(spectrogram)
        label = f['data_labels'].value
        label = np.float32(label)
    return spectrogram, [label, np.float32(abs(label - 1))]


batch_size_const = 10
num_epochs = 10

n_classes = 2
learning_rate = 0.0001

batch_size = tf.placeholder(tf.int64, name='batch_size')

input_file = tf.placeholder(tf.string, shape=[1], name='input_file')
dataset = tf.data.Dataset.from_tensor_slices(input_file)
dataset = dataset.map(
    lambda filename: tuple(tf.py_func(
        _read_py_function, [filename], [tf.float32, tf.float32])))
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()

iter = dataset.make_initializable_iterator()
features_, labels_ = iter.get_next()



with tf.name_scope('inputs'):
    x_reshaped = tf.reshape(features_, [-1,  21, 256, 1], name='input_tensor')

# Convolution Layer with 50 filters and a kernel size of 5
conv1 = tf.layers.conv2d(x_reshaped, 20, 5, activation=tf.nn.relu)
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
            logits=out, labels=tf.cast(labels_, dtype=tf.int32)))
loss_summary = tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./events')
writer.add_graph(tf.get_default_graph())

n_batches = len(train_files) // batch_size_const

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('./events')
writer.add_graph(tf.get_default_graph())

validation_op = loss
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict={input_file: [train_files[0]], batch_size: batch_size_const})
    print('Training...')
    for i in range(num_epochs):
        print("Epoch {}".format(i))
        tot_loss = 0
        for j in tqdm.tqdm(range(n_batches)):
            try:
                _, loss_value, summary = sess.run([train_op, loss, merged])
                writer.add_summary(summary, j)
                tot_loss += loss_value
            except tf.errors.OutOfRangeError:
                break

            if j % 100 == 0:
                sess.run(iter.initializer,
                         feed_dict={input_file: [validation_files[0]], batch_size: len(validation_files)})
                print('Test Loss: {:6e}'.format(sess.run(loss)))
                print(', Accuracy: {:3f}'.format(sess.run(accuracy)))

        print("\nEpoch: {}, Loss: {:.6e}".format(i, tot_loss / n_batches))

    # initialise iterator with test data
    sess.run(iter.initializer, feed_dict={input_file: [validation_files[0]], batch_size: len(validation_files)})
    print('\nTest Loss: {:6e}'.format(sess.run(loss)))
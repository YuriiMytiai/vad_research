import tensorflow as tf
import numpy as np
import os
import h5py
import tqdm

from tensorflow.python import debug as tf_debug





class Train:

    def __init__(self, path_to_train_h5="D:\\h5dataset\\train", path_to_validation_h5="D:\\h5dataset\\validation",
                 batch_size=10, num_epochs=10, num_classes=2, learning_rate=0.0001):

        self.path_to_train_h5 = path_to_train_h5
        self.path_to_validation_h5 = path_to_validation_h5
        self.train_files = self.collect_h5_files(path_to_train_h5)
        self.validation_files = self.collect_h5_files(path_to_validation_h5)


        self.train_files_idx = range(0, len(self.train_files) - 1)
        self.validation_files_idx = range(0, len(self.validation_files) - 1)

        self.batch_size_const = batch_size
        self.num_epochs = num_epochs
        self.n_classes = num_classes
        self.learning_rate = learning_rate

        self.input_file_idx = tf.placeholder(tf.int64, shape=[1, None], name='input_file_idx')
        self.batch_size = tf.placeholder(tf.int64, name='batch_size')
        self.dataset_switcher = 'train'

    @staticmethod
    def collect_h5_files(path):
        files_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".hdf5"):
                    files_list.append(os.path.join(root, file))
        return files_list

    def _read_py_function(self, file):
        with h5py.File(file, 'r') as f:
            spectrogram = f['data'].value
            spectrogram = np.float32(spectrogram)
            label = f['data_labels'].value
            label = np.float32(label)
        return spectrogram, [label, np.float32(abs(label - 1))]

    def build_datasets(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.input_file_idx)
        dataset = dataset.map(
            lambda file_idx: tuple(tf.py_func(
                self._read_py_function, [file_idx], [tf.float32, tf.float32])))
        #dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()

        iter = dataset.make_initializable_iterator()
        features_, labels_ = iter.get_next()

        return dataset, features_, labels_, iter

    def build_classifier(self, input_size=[-1, 21, 256, 1]):
        dataset, features_, labels_, iter = self.build_dataset()

        with tf.name_scope('inputs'):
            x_reshaped = tf.reshape(features_, input_size, name='input_tensor')

        # Convolution Layer with 50 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x_reshaped, 20, 5, activation=tf.nn.relu, name='conv1')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 50 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 50, 5, activation=tf.nn.relu, name='conv2')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc1_activ')

        # Output layer, class prediction
        out = tf.layers.dense(fc1, self.n_classes, name='out')

        # Predictions
        y_pred = tf.argmax(out, axis=1)
        # y_pred = tf.expand_dims(y_pred, 0)
        pred_probas = tf.nn.softmax(out)

        # Define loss and optimizer
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=out, labels=tf.cast(labels_, dtype=tf.int32)), name='loss')
        loss_summary = tf.summary.scalar('loss', loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(labels_, 1), name='correct_prediction')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam_opt')
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), name='opt_min')

        writer = tf.summary.FileWriter('./events')
        writer.add_graph(tf.get_default_graph())

        n_batches = len(self.train_files) // self.batch_size_const

        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter('./events')
        writer.add_graph(tf.get_default_graph())

        return n_batches, train_op, loss, accuracy, merged, writer, iter

    def run_training(self, input_size=[-1, 21, 256, 1]):
        n_batches, train_op, loss, accuracy, merged, writer, iter = self.build_classifier(input_size)

        with tf.Session() as sess:

            #uncomment to enable TensorFlow debugger
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess) #debugging mode

            sess.run(tf.global_variables_initializer())
            # initialise iterator with train data
            self.dataset_switcher = 'train'
            sess.run(iter.initializer, feed_dict={self.input_file_idx: [self.train_files_idx], self.batch_size: self.batch_size_const})
            print('Training...')
            for i in range(self.num_epochs):
                print("Epoch {}".format(i))
                tot_loss = 0
                for j in tqdm.tqdm(range(n_batches)):
                    try:
                        _, loss_value, summary = sess.run([train_op, loss, merged])
                        writer.add_summary(summary, j)
                        tot_loss += loss_value
                    except tf.errors.OutOfRangeError:
                        break

                    if j % 500 == 0:
                        self.dataset_switcher = 'validation'
                        sess.run(iter.initializer,
                                 feed_dict={self.input_file_idx: [self.validation_files_idx], self.batch_size: len(self.validation_files)})
                        print('Validation Loss: {:6e}'.format(sess.run(loss)))
                        print(', Accuracy: {:3f}'.format(sess.run(accuracy)))
                        self.dataset_switcher = 'train'

                print("\nEpoch: {}, Loss: {:.6e}".format(i, tot_loss / n_batches))

            # initialise iterator with test data
            self.dataset_switcher = 'validation'
            sess.run(iter.initializer, feed_dict={self.input_file_idx: [self.validation_files_idx], self.batch_size: len(self.validation_files)})
            print('\nValidation Loss: {:6e}'.format(sess.run(loss)))
            self.dataset_switcher = 'train'

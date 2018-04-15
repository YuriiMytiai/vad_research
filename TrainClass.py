import tensorflow as tf
import numpy as np
import os
import h5py
import tqdm

#from tensorflow.python import debug as tf_debug





class Train:

    def __init__(self, path_to_train_h5="D:\\h5dataset\\train", path_to_validation_h5="D:\\h5dataset\\validation",
                 batch_size=10, num_epochs=10, num_classes=2, learning_rate=0.0001):

        self.path_to_train_h5 = path_to_train_h5
        self.path_to_validation_h5 = path_to_validation_h5
        self.train_files = self.collect_h5_files(path_to_train_h5)
        self.validation_files = self.collect_h5_files(path_to_validation_h5)
        self.x = tf.placeholder(tf.float32, shape=(None, 21, 256), name='raw_input')
        self.y = tf.placeholder(tf.float32, shape=(None, 2), name='features')

        self.batch_size_const = batch_size
        self.num_epochs = num_epochs
        self.n_classes = num_classes
        self.learning_rate = learning_rate


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
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_files)
        train_dataset = train_dataset.map(
            lambda file_idx: tuple(tf.py_func(
                self._read_py_function, [file_idx], [tf.float32, tf.float32])))
        train_dataset = train_dataset.batch(self.batch_size_const)
        train_dataset = train_dataset.repeat()

        train_iter = train_dataset.make_one_shot_iterator()


        validation_dataset = tf.data.Dataset.from_tensor_slices(self.validation_files)
        validation_dataset = validation_dataset.map(
            lambda file_idx: tuple(tf.py_func(
                self._read_py_function, [file_idx], [tf.float32, tf.float32])))
        validation_dataset = validation_dataset.batch(len(self.validation_files))
        validation_dataset = validation_dataset.repeat()

        validation_iter = validation_dataset.make_one_shot_iterator()


        return train_dataset, train_iter, \
               validation_dataset, validation_iter

    def build_classifier(self, input_size=[-1, 21, 256, 1]):

        with tf.name_scope('inputs'):
            x_reshaped = tf.reshape(self.x, input_size, name='input_tensor')

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
                logits=out, labels=tf.cast(self.y, dtype=tf.int32)), name='loss')
        loss_summary = tf.summary.scalar('loss', loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(self.y, 1), name='correct_prediction')
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

        return n_batches, train_op, loss, accuracy, merged, writer

    def run_training(self, input_size=[-1, 21, 256, 1]):
        n_batches, train_op, loss, accuracy, merged, writer = self.build_classifier(input_size)
        _, train_iter, _, validation_iter = self.build_datasets()

        print('Configuring session...\n')
        #config = tf.ConfigProto(log_device_placement=True) # to use GPU + CPU
        with tf.Session() as sess:

            #uncomment to enable TensorFlow debugger
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess) #debugging mode

            sess.run(tf.global_variables_initializer())
            # initialise iterator with train data
            print('Training...')
            for i in range(self.num_epochs):
                print("Epoch {}".format(i))
                tot_loss = 0
                train_features_, train_labels_ = sess.run(train_iter.get_next())
                validation_features_, validation_labels_ = sess.run(validation_iter.get_next())
                for j in tqdm.tqdm(range(n_batches)):
                    try:
                        
                        if j % 35 == 0:
                            _, loss_value, summary = sess.run([train_op, loss, merged], feed_dict={self.x: train_features_, self.y: train_labels_})
                            writer.add_summary(summary, j)
                            tot_loss += loss_value
                        else:
                            _ = sess.run(train_op, feed_dict={self.x: train_features_,
                                                                         self.y: train_labels_})

                    except tf.errors.OutOfRangeError:
                        break

                    if (j % 500 == 0) & (j > 1):
                        
                        loss_v, acc_v = sess.run([loss, accuracy], feed_dict={self.x: validation_features_, self.y: validation_labels_})
                        print('Validation Loss: {:6e}'.format(loss_v))
                        print(', Accuracy: {:3f}'.format(acc_v))

                print("\nEpoch: {}, Loss: {:.6e}".format(i, tot_loss / n_batches))

            # initialise iterator with test data
            validation_features_, validation_labels_ = sess.run(validation_iter.get_next())
            print('\nValidation Loss: {:6e}'.format(sess.run(loss, feed_dict={self.x: validation_features_, self.y: validation_labels_})))

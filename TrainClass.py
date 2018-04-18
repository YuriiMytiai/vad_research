import tensorflow as tf
import numpy as np
import os
import h5py
import tqdm
from tensorflow.python import debug as tf_debug


class Train:

    def __init__(self, path_to_train_h5="D:\\h5dataset\\train",
                 path_to_validation_h5="D:\\h5dataset\\validation",
                 batch_size=10, validation_batch_size=500, num_epochs=10, num_classes=2,
                 learning_rate=0.0001, regularization=0.01, enable_debug_mode=False,
                 checkpoint_dir='C:\\Users\\User\\Desktop\\vad_research\\src\\checkpoints',
                 events_log_dir='C:\\Users\\User\\Desktop\\vad_research\\src\\events',
                 validation_cache_dir='D:\\h5dataset\\cache\\'):
        self.path_to_train_h5 = path_to_train_h5
        self.path_to_validation_h5 = path_to_validation_h5
        self.checkpoint_dir = checkpoint_dir
        self.events_log_dir = events_log_dir
        self.validation_cache_dir = validation_cache_dir
        self.check_paths()

        self.train_files = self.collect_h5_files(path_to_train_h5)
        self.validation_files = self.collect_h5_files(path_to_validation_h5)
        self.enable_debug_mode = enable_debug_mode

        self.validation_batch_size = validation_batch_size
        self.batch_size_const = batch_size
        self.num_epochs = num_epochs
        self.n_classes = num_classes
        self.learning_rate = learning_rate
        self.regularization = regularization

        self.x = tf.placeholder(tf.float32, shape=(None, 21, 256), name='raw_input')
        self.y = tf.placeholder(tf.float32, shape=(None, 2), name='features')

    def check_paths(self):
        if not os.path.isdir(self.path_to_train_h5):
            raise FileExistsError("Train path: {} do not exist!".format(self.path_to_train_h5))
        if not os.path.isdir(self.path_to_validation_h5):
            raise FileExistsError("Validation path: {} do not exist!".format(self.path_to_train_h5))
        if not os.path.isdir(self.checkpoint_dir):
            try:
                os.makedirs(self.checkpoint_dir)
            except Exception as e:
                    print(e)
            print('Directory for checkpoints was made: {}'.format(self.checkpoint_dir))
        if not os.path.isdir(self.events_log_dir):
            try:
                os.makedirs(self.events_log_dir + '\\train')
            except Exception as e:
                    print(e)
            print('Directory for train events logging was made: {}'.format(self.checkpoint_dir + '\\train'))
            try:
                os.makedirs(self.events_log_dir + '\\validation')
            except Exception as e:
                    print(e)
            print('Directory for validation events logging was made: {}'.format(self.checkpoint_dir + '\\validation'))
        if not os.path.isdir(self.events_log_dir + '\\train'):
            try:
                os.makedirs(self.events_log_dir + '\\train')
            except Exception as e:
                    print(e)
            print('Directory for train events logging was made: {}'.format(self.checkpoint_dir + '\\train'))
        if not os.path.isdir(self.events_log_dir + '\\validation'):
            try:
                os.makedirs(self.events_log_dir + '\\validation')
            except Exception as e:
                    print(e)
            print('Directory for validation events logging was made: {}'.format(self.checkpoint_dir + '\\validation'))
        if not os.path.isdir(self.validation_cache_dir):
            try:
                os.makedirs(self.validation_cache_dir)
            except Exception as e:
                    print(e)
            print('Directory for validation cache was made: {}'.format(self.validation_cache_dir))
        for _, _, files in os.walk(self.validation_cache_dir):
            if files:
                print('All files from {} will be deleted!'.format(self.validation_cache_dir))
                for file in files:
                    file_path = os.path.join(self.validation_cache_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)



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
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_files)\
            .map(lambda file_idx: tuple(tf.py_func(
                self._read_py_function, [file_idx], [tf.float32, tf.float32])),
                 num_parallel_calls=2)\
            .batch(self.batch_size_const)\
            .repeat()
        train_iter = train_dataset.make_one_shot_iterator()

        validation_dataset = tf.data.Dataset.from_tensor_slices(self.validation_files)\
            .cache(filename=self.validation_cache_dir)\
            .map(lambda file_idx: tuple(tf.py_func(
                self._read_py_function, [file_idx], [tf.float32, tf.float32])),
                 num_parallel_calls=2)\
            .batch(self.validation_batch_size)
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
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularization)
        fc1 = tf.contrib.layers.fully_connected(fc1, num_outputs=500,
                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                weights_regularizer=regularizer,
                                                activation_fn=tf.nn.relu)

        # Fully connected layer (in tf contrib folder for now)
        # = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc1_activ')

        # Output layer, class prediction
        #out = tf.layers.dense(fc1, self.n_classes, name='out')
        out = tf.contrib.layers.fully_connected(fc1, num_outputs=self.n_classes,
                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                weights_regularizer=regularizer,
                                                activation_fn=None)

        # Predictions
        y_pred = tf.argmax(out, axis=1)
        # y_pred = tf.expand_dims(y_pred, 0)
        pred_probas = tf.nn.softmax(out)

        # Define loss and optimizer
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=out, labels=tf.cast(self.y, dtype=tf.int32)), name='loss')
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            loss += reg_term
        tf.summary.scalar('loss', loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(self.y, 1), name='correct_prediction')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam_opt')
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), name='opt_min')

        writer = tf.summary.FileWriter('./events')
        writer.add_graph(tf.get_default_graph())
        merged = tf.summary.merge_all()  # merge accuracy & loss

        num_train_batches = len(self.train_files) // self.batch_size_const
        num_validation_batches = len(self.validation_files) // self.validation_batch_size

        train_writer = tf.summary.FileWriter(self.events_log_dir + '\\train')
        validation_writer = tf.summary.FileWriter(self.events_log_dir + '\\validation')
        train_writer.add_graph(tf.get_default_graph())

        return num_train_batches, num_validation_batches, train_op, loss, accuracy, merged, train_writer, validation_writer

    def run_training(self, input_size=[-1, 21, 256, 1]):
        num_train_batches, num_validation_batches, train_op, loss, accuracy,\
            merged, train_writer, validation_writer = self.build_classifier(input_size)
        _, train_iter, _, validation_iter = self.build_datasets()


        print('Configuring session...\n')
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        with tf.Session(config=config) as sess:

            if self.enable_debug_mode:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)  # debugging mode

            sess.run(tf.global_variables_initializer())

            # init saver
            saver = tf.train.Saver()

            print('Training...')
            for i in range(self.num_epochs):
                print("Epoch {}".format(i))
                tot_loss = 0
                train_features_, train_labels_ = sess.run(train_iter.get_next())
                validation_features_, validation_labels_ = sess.run(validation_iter.get_next())
                for j in tqdm.tqdm(range(num_train_batches)):
                    try:
                        
                        if j % 35 == 0:
                            _, loss_value, summary = sess.run([train_op, loss, merged],
                                                              feed_dict={self.x: train_features_,
                                                                         self.y: train_labels_})
                            train_writer.add_summary(summary, j)
                            tot_loss += loss_value
                        else:
                            _ = sess.run(train_op, feed_dict={self.x: train_features_,
                                                              self.y: train_labels_})

                    except tf.errors.OutOfRangeError:
                        break

                    if j % 100 == 0:
                        num_correct = 0
                        sum_loss = 0
                        for _ in range(num_validation_batches):
                            loss_v, acc_v, summary = sess.run([loss, accuracy, merged],
                                                              feed_dict={self.x: validation_features_,
                                                                         self.y: validation_labels_})
                            num_correct += acc_v * self.validation_batch_size
                            sum_loss += loss_v
                        validation_writer.add_summary(summary, j)
                        validation_loss = sum_loss / num_validation_batches
                        validation_accuracy = num_correct / (self.validation_batch_size * num_validation_batches)
                        print('Validation Loss: {:6e}'.format(validation_loss))
                        print('Validation Accuracy: {:.3f}'.format(validation_accuracy))

                print("\nEpoch: {}, Train Loss: {:.6e}, Validation Loss: {:.3f}"
                      .format(i, tot_loss / num_train_batches, validation_loss))
                checkpoint_path = os.path.join(self.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)

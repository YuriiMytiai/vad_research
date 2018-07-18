import tensorflow as tf
import os
import tqdm
from tensorflow.python import debug as tf_debug
from datetime import datetime


class Train:

    def __init__(self, path_to_train_dataset="D:\\h5dataset\\train",
                 path_to_validation_dataset="D:\\h5dataset\\validation",
                 batch_size=10, validation_batch_size=500, num_epochs=10, num_classes=2,
                 learning_rate=0.0001, regularization=0.01, enable_debug_mode=False,
                 enable_regularization=False, weights_init=tf.initializers.random_normal,
                 dropout_keep_prob=0.5, enable_dropout=True,
                 checkpoint_dir='/home/yurii/Documents/vad_research/vad_research/checkpoints',
                 events_log_dir='/home/yurii/Documents/vad_research/vad_research/events',
                 model_name="my_model",
                 train_valid_freq=50,
                 use_just_amplitude_spec=False, num_train_examples=None, num_validation_examples=None):

        self.path_to_train_dataset = path_to_train_dataset
        self.path_to_validation_dataset = path_to_validation_dataset
        self.checkpoint_dir = checkpoint_dir
        self.events_log_dir = events_log_dir
        self.model_name = model_name
        self.just_ampl = use_just_amplitude_spec

        self.check_paths()
        self.train_file = self.collect_tfrecords_file(self.path_to_train_dataset)
        self.validation_file = self.collect_tfrecords_file(self.path_to_validation_dataset)

        self.enable_debug_mode = enable_debug_mode

        self.validation_batch_size = validation_batch_size
        self.batch_size_const = batch_size
        self.num_epochs = num_epochs
        self.n_classes = num_classes
        self.learning_rate = learning_rate
        self.enable_regularization = enable_regularization
        self.regularization = regularization
        self.weights_initializer = weights_init
        self.enable_dropout = enable_dropout
        self.keep_prob = dropout_keep_prob

        self.train_valid_freq = train_valid_freq

        self.num_train_batches = num_train_examples // self.batch_size_const
        self.num_validation_batches = num_validation_examples // self.validation_batch_size

        # need it to run sessions in the loop
        tf.reset_default_graph()
        self.is_training = tf.placeholder(tf.bool)

    def check_paths(self):
        if not os.path.isdir(self.path_to_train_dataset):
            raise FileExistsError("Train path: {} do not exist!".format(self.path_to_train_dataset))
        if not os.path.isdir(self.path_to_validation_dataset):
            raise FileExistsError("Validation path: {} do not exist!".format(self.path_to_train_dataset))
        if not os.path.isdir(self.checkpoint_dir):
            try:
                os.makedirs(self.checkpoint_dir)
            except Exception as e:
                    print(e)
            print('Directory for checkpoints was made: {}'.format(self.checkpoint_dir))
        if not os.path.isdir(self.events_log_dir + '/train' + "/" + self.model_name):
            try:
                os.makedirs(self.events_log_dir + '/train' + "/" + self.model_name)
            except Exception as e:
                    print(e)
            print('Directory for train events logging was made: {}'.format(self.checkpoint_dir + '/train'))
            try:
                os.makedirs(self.events_log_dir + '/validation' + "/" + self.model_name)
            except Exception as e:
                    print(e)
            print('Directory for validation events logging was made: {}'.format(self.checkpoint_dir + '/validation'))
        if not os.path.isdir(self.events_log_dir + '/train'):
            try:
                os.makedirs(self.events_log_dir + '/train')
            except Exception as e:
                    print(e)
            print('Directory for train events logging was made: {}'.format(self.checkpoint_dir + '/train'))
        if not os.path.isdir(self.events_log_dir + '/validation'):
            try:
                os.makedirs(self.events_log_dir + '/validation')
            except Exception as e:
                    print(e)
            print('Directory for validation events logging was made: {}'.format(self.checkpoint_dir + '/validation'))

    @staticmethod
    def collect_h5_file(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".hdf5"):
                    return os.path.join(root, file)

    @staticmethod
    def collect_tfrecords_file(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".tfrecords"):
                    return os.path.join(root, file)

    @staticmethod
    def count_number_of_examples(tfrecords_file):
        c = 0
        for _ in tf.python_io.tf_record_iterator(tfrecords_file):
            c += 1
        return c

    def close_files(self):
        self.train_file.close()
        self.validation_file.close()

    def _read_py_function(self, example):
        feature = {"label": tf.FixedLenFeature([], tf.int64),
                   "spectrogram": tf.FixedLenFeature([], tf.string)}
        # Decode the record read by the reader
        features = tf.parse_single_example(example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features["spectrogram"], tf.float32)
        # Cast label data into int32
        label = tf.cast(features["label"], tf.int32)
        label_rev = label - 1
        # Reshape image data into the original shape
        image = tf.reshape(image, [21, 256, 2])
        if self.just_ampl:
            image = image[:, :, 0]
        return image, tf.stack([label, tf.abs(label_rev)], axis=0)

    def build_datasets(self):
        train_dataset = tf.data.TFRecordDataset([self.train_file])\
            .map(self._read_py_function)\
            .batch(self.batch_size_const)\
            .prefetch(self.batch_size_const * 2) \
            .repeat()
        train_iter = train_dataset.make_one_shot_iterator()

        validation_dataset = tf.data.TFRecordDataset([self.validation_file]) \
            .map(self._read_py_function) \
            .batch(self.validation_batch_size) \
            .prefetch(self.validation_batch_size * 2) \
            .cache() \
            .repeat()
        validation_iter = validation_dataset.make_one_shot_iterator()

        return train_dataset, train_iter, validation_dataset, validation_iter

    def build_classifier(self, input_size, train_iter, validation_iter):

        x, y = tf.cond(tf.equal(self.is_training, tf.constant(True)),
                       lambda: train_iter.get_next(),
                       lambda: validation_iter.get_next())

        with tf.name_scope('inputs'):
            x_reshaped = tf.reshape(x, input_size, name='input_tensor')
            y_reshaped = tf.reshape(y, [-1, 2], name="labels")

        # Regularization:
        if self.enable_regularization:
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularization)
        else:
            regularizer = None
        # Convolution Layer with 50 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x_reshaped, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, name='conv1',
                                 kernel_initializer=self.weights_initializer,
                                 kernel_regularizer=regularizer)

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 50 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, filters=50, kernel_size=[5, 5], activation=tf.nn.relu, name='conv2',
                                 kernel_initializer=self.weights_initializer,
                                 kernel_regularizer=regularizer)

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(fc1, num_outputs=500,
                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                weights_initializer=self.weights_initializer,
                                                weights_regularizer=regularizer,
                                                activation_fn=tf.nn.relu)

        # Fully connected layer (in tf contrib folder for now)
        # = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc1_activ')

        # Output layer, class prediction
        # out = tf.layers.dense(fc1, self.n_classes, name='out')

        # Let's add dropout here
        if self.enable_dropout:
            drop_out = tf.contrib.layers.dropout(fc1, keep_prob=self.keep_prob, is_training=self.is_training)
        else:
            drop_out = fc1

        # One more FC layer
        out = tf.contrib.layers.fully_connected(drop_out, num_outputs=self.n_classes,
                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                weights_initializer=self.weights_initializer,
                                                weights_regularizer=regularizer,
                                                activation_fn=None)

        # Predictions
        # y_pred = tf.argmax(out, axis=1)
        # y_pred = tf.expand_dims(y_pred, 0)
        # pred_probas = tf.nn.softmax(out)
        logits_out_layer = tf.layers.dense(inputs=out, units=self.n_classes)

        # Define loss and optimizer
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits_out_layer, labels=tf.cast(y_reshaped, dtype=tf.int32)), name='loss')

            # loss = tf.reduce_mean(tf.losses.huber_loss(tf.cast(self.y, dtype=tf.int32), pred_probas), name='loss')
            if self.enable_regularization:
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
                loss += reg_term
        tf.summary.scalar('loss', loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_reshaped, 1), name='correct_prediction')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam_opt')
        # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step(), name='opt_min')
        train_op = optimizer.minimize(loss, name='opt_min')

        writer = tf.summary.FileWriter('./events')
        writer.add_graph(tf.get_default_graph())
        merged = tf.summary.merge_all()  # merge accuracy & loss

        train_writer = tf.summary.FileWriter(self.events_log_dir + '/train' + "/" + self.model_name,
                                             filename_suffix=self.model_name)
        validation_writer = tf.summary.FileWriter(self.events_log_dir + '/validation' + "/" + self.model_name,
                                                  filename_suffix=self.model_name)
        train_writer.add_graph(tf.get_default_graph())

        return train_op, loss, accuracy, merged, train_writer, validation_writer

    def validation_loop(self, sess, loss, accuracy, merged, epoch, validation_writer):
        num_correct = 0
        sum_loss = 0
        for _ in tqdm.tqdm(range(self.num_validation_batches)):
            loss_v, acc_v, summary = sess.run([loss, accuracy, merged],
                                              feed_dict={self.is_training: False})
            num_correct += acc_v * self.validation_batch_size
            sum_loss += loss_v
        # validation_writer.add_summary(summary, epoch * num_train_batches + batch)
        validation_writer.add_summary(summary, (epoch + 1) * self.num_train_batches)
        validation_loss = sum_loss / self.num_validation_batches
        validation_accuracy = num_correct / (self.validation_batch_size * self.num_validation_batches)
        print('Validation Loss: {:6e}'.format(validation_loss))
        print('Validation Accuracy: {:.3f}'.format(validation_accuracy))
        return sess, validation_loss

    def run_training(self, **kwargs):
        if self.just_ampl:
            size_param = [-1, 21, 256, 1]
        else:
            size_param = [-1, 21, 256, 2]
        input_size = kwargs.get('input_size', size_param)
        _, train_iter, _, validation_iter = self.build_datasets()
        train_op, loss, accuracy,\
            merged, train_writer, validation_writer = self.build_classifier(input_size, train_iter, validation_iter)

        print('Configuring session...\n')
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        # tf.logging.set_verbosity(tf.logging.ERROR)
        
        with tf.Session(config=config) as sess:
            if self.enable_debug_mode:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)  # debugging mode

            sess.run(tf.global_variables_initializer())

            # init saver
            saver = tf.train.Saver()

            print("Validation before training:\n")
            sess, validation_loss = self.validation_loop(sess, loss, accuracy, merged,
                                                         -1, validation_writer)

            print('Training...\n')
            for epoch in range(self.num_epochs):
                print("Epoch {}".format(epoch))
                tot_loss = 0
                for batch in tqdm.tqdm(range(self.num_train_batches)):
                    try:
                        if batch % self.train_valid_freq == 0:
                            _, loss_value, summary = sess.run([train_op, loss, merged],
                                                              feed_dict={self.is_training: True})
                            train_writer.add_summary(summary, epoch * self.num_train_batches + batch)
                            tot_loss += loss_value
                        else:
                            _ = sess.run(train_op, feed_dict={self.is_training: True})

                    except tf.errors.OutOfRangeError:
                        break

                sess, validation_loss = self.validation_loop(sess, loss, accuracy, merged,
                                                             epoch, validation_writer)
                print("Epoch: {}, Train Loss: {:.6e}, Validation Loss: {:.3f}\n"
                      .format(epoch, tot_loss / self.num_train_batches, validation_loss))
                saver.save(sess, self.checkpoint_dir + "/" + self.model_name)

            # self.close_files()
            print('The end of the training at {}'.format(str(datetime.now())))

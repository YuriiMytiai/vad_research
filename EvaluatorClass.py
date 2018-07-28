import tensorflow as tf
import os
import tqdm
import numpy as np
from datetime import datetime


class Evaluator:

    def __init__(self, evaluation_file, checkpoint_dir, use_just_amplitude_spec=True,
                 num_examples=None, num_classes=2):
        self.checkpoint_dir = checkpoint_dir
        self.just_ampl = use_just_amplitude_spec
        self.num_examples = num_examples
        self.n_classes = num_classes
        self.evaluation_file = evaluation_file
        self.check_paths()

        self.is_training = False
        self.weights_initializer = None
        self.labels = []

    def check_paths(self):
        if not os.path.isfile(self.evaluation_file):
            raise FileExistsError("Evaluation file: {} does not exist!".format(self.evaluation_file))

    def _read_py_function(self, example):
        feature = {"spectrogram": tf.FixedLenFeature([], tf.string)}
        # Decode the record read by the reader
        features = tf.parse_single_example(example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features["spectrogram"], tf.float32)
        # Reshape image data into the original shape
        image = tf.reshape(image, [21, 256, 2])
        if self.just_ampl:
            image = image[:, :, 0]
        return image

    def build_datasets(self):
        dataset = tf.data.TFRecordDataset([self.evaluation_file])\
            .map(self._read_py_function)\
            .prefetch(10)
        dataset_iter = dataset.make_one_shot_iterator()

        return dataset, dataset_iter

    def build_classifier(self, input_size, file_iter):

        x = file_iter.get_next()

        with tf.name_scope('inputs'):
            x_reshaped = tf.reshape(x, input_size, name='input_tensor')

        # Regularization:
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

        # One more FC layer
        out = tf.contrib.layers.fully_connected(fc1, num_outputs=self.n_classes,
                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                weights_initializer=self.weights_initializer,
                                                weights_regularizer=regularizer,
                                                activation_fn=None)

        # Predictions
        # y_pred = tf.argmax(out, axis=1)
        # y_pred = tf.expand_dims(y_pred, 0)
        pred_probas = tf.nn.softmax(out)
        # logits_out_layer = tf.layers.dense(inputs=out, units=self.n_classes)

        return pred_probas

    def evaluation_loop(self, sess, pred_probas):
        for _ in tqdm.tqdm(range(self.num_examples)):
            pred = sess.run([pred_probas])
            self.labels.append(pred[0][0])

    def run_evaluation(self, **kwargs):
        if self.just_ampl:
            size_param = [-1, 21, 256, 1]
        else:
            size_param = [-1, 21, 256, 2]
        input_size = kwargs.get('input_size', size_param)
        _, dataset_iter = self.build_datasets()
        pred_probas = self.build_classifier(input_size, dataset_iter)

        print('Configuring session...\n')
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        # tf.logging.set_verbosity(tf.logging.ERROR)

        # init saver
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            # restore variables
            saver.restore(sess, self.checkpoint_dir)

            print("Evaluation:\n")
            self.evaluation_loop(sess, pred_probas)

            print('The end of the evaluation at {}'.format(str(datetime.now())))

        return np.array(self.labels)

import tensorflow as tf
import numpy as np
from DataCollectorClass import DataCollector
from FeatureExtractorClass import FeatureExtractor
from sklearn import metrics
from matplotlib import pyplot as plt
import os
import glob

# Data Location
path = "C:\\Users\\User\\Desktop\\vad_research\\datasets\\qut-noise-timit\\qutnoise\\QUT-NOISE\\QUT-NOISE-TIMIT"

# Training Parameters
learning_rate = 0.001
num_steps = 6000 * 0.8
batch_size = 1

# Network Parameters
num_input = 2048 # data input (img shape: 64*32)
num_classes = 2 # total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit


def collect_data(path_to_data):
    data = DataCollector(path_to_data)
    data.load_data()
    data.preprocess_files()
    return data

data = collect_data(path)
feature_extractor = FeatureExtractor()

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['spectrogram']

        # data input is a 1-D vector of 2048 features (64*32 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 64, 32, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 20, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 1)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 50, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 1)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    print('Accuracy: {}'.format(acc_op))
    #print('LogLoss: {}'.format(metrics.log_loss(y_true=labels, y_pred=pred_classes)))

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Define the input function for training




def trainset_input_fn():

    num_epochs = 10

    def _read_py_function(wav, label):
        spectrogram = feature_extractor.extract_features_from_wav(data.data_set_files_list["train_wavs"][wav])
        label = feature_extractor.extract_features_from_wav(data.data_set_files_list["train_labels"][label])
        return spectrogram, label

    files_wavs = list(range(0, len(data.data_set_files_list["train_wavs"])))
    files_labels = list(range(0, len(data.data_set_files_list["train_labels"])))
    dataset = tf.data.Dataset.from_tensor_slices((files_wavs, files_labels))
    dataset = dataset.map(lambda file, label: tuple(tf.py_func(
        _read_py_function, [file, label], [tf.float32, tf.int64])))

    dataset = dataset.batch(1)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return {'spectrogram': features}, labels

# Train the Model


def validationset_input_fn():

    def _read_py_function(wav, label):
        spectrogram = feature_extractor.extract_features_from_wav(data.data_set_files_list["validation_wavs"][wav])
        label = feature_extractor.extract_features_from_wav(data.data_set_files_list["validation_labels"][label])
        return spectrogram, label

    files_wavs = list(range(0, len(data.data_set_files_list["validation_wavs"])))
    files_labels = list(range(0, len(data.data_set_files_list["validation_labels"])))
    dataset = tf.data.Dataset.from_tensor_slices((files_wavs, files_labels))
    dataset = dataset.map(lambda file, label: tuple(tf.py_func(
        _read_py_function, [file, label], [tf.float32, tf.int64])))

    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()
    return {'spectrogram': features}, labels

# Train the Model



def train_model(
        learning_rate,
        steps,
        batch_size,
        #training_examples,
        #training_targets,
        #validation_examples,
        #validation_targets
        ):
    """Trains a linear classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, and a confusion
    matrix.

    Args:
      learning_rate: An `int`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.

    Returns:
      The trained `LinearClassifier` object.
    """

    periods = int(6000*0.8)

    steps_per_period = steps / periods
    # Create the input functions.
    predict_training_input_fn = trainset_input_fn()
    predict_validation_input_fn = lambda: validationset_input_fn()
    training_input_fn = lambda: trainset_input_fn()

    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.

        model.train(input_fn=training_input_fn, steps=300)
        """
        # Take a break and compute probabilities.
        training_predictions = list(model.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 2)

        validation_predictions = list(model.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 2)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
        """
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(model.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = model.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

   # accuracy = metrics.accuracy_score(validation_targets, final_predictions)
   # print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    return model



classifier = train_model(
             learning_rate=0.002,
             steps=6000*0.8,
             batch_size=1)
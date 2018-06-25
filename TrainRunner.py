from TrainClass import Train
import tensorflow as tf


my_hdd = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/"

path_to_h5 = my_hdd + "datasets/h5-full_spec"

train = Train(path_to_train_h5=path_to_h5 + "/train",
              path_to_validation_h5=path_to_h5 + "/validation",
              batch_size=64, learning_rate=1e-10, num_epochs=100,
              enable_regularization=True, regularization=5e-11,
              weights_init=tf.initializers.variance_scaling, validation_batch_size=1000,
              train_valid_freq=1000, valid_valid_freq=5000,
              validation_cache_dir=path_to_h5 + '/cache/',
              model_name="CNN_full_spec")

train.run_training()

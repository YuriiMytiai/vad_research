from TrainClass import Train
import tensorflow as tf


my_hdd = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/"

path_to_h5 = my_hdd + "datasets/h5-full_spec"


def train_iter(_batch_size, _learning_rate, _num_epochs, _model_name):
    train = Train(path_to_train_h5=path_to_h5 + "/train",
                  path_to_validation_h5=path_to_h5 + "/validation",
                  batch_size=_batch_size, learning_rate=_learning_rate, num_epochs=_num_epochs,
                  enable_regularization=True, regularization=5e-11,
                  weights_init=tf.initializers.variance_scaling, validation_batch_size=1000,
                  train_valid_freq=1000, valid_valid_freq=5000,
                  validation_cache_dir=path_to_h5 + '/cache/',
                  model_name=_model_name)
    train.run_training()


lr = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]

for cur_lr in lr:
    print("Iteration learning rate = {}".format(cur_lr))
    name = "CNN_full_spec" + "_lr-" + str(cur_lr)
    train_iter(64, cur_lr, 1, name)

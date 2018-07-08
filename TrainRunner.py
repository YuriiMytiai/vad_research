from TrainClass import Train
import tensorflow as tf


my_hdd = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/"
#my_datasets_on_ssd = "/home/yurii/Documents/datasets/"

path_to_h5 = my_hdd + "datasets/h5-new_data"
#path_to_h5 = my_datasets_on_ssd + "h5-new_data"


#path_to_h5 = "/home/yurii/Documents/datasets/h5-new_data"



dropout = [0.3]

for cur_kp in dropout:
    print("Iteration dropout = {}".format(cur_kp))
    name = "CNN_full_spec" + "_dropout-" + str(cur_kp) + "_best_hyperparams_new_data"

    train = Train(path_to_train_h5=path_to_h5 + "/train",
                  path_to_validation_h5=path_to_h5 + "/validation",
                  batch_size=256, learning_rate=1e-7, num_epochs=20,
                  enable_regularization=True, regularization=0.1,
                  enable_dropout=True, dropout_keep_prob=cur_kp,
                  weights_init=tf.initializers.variance_scaling, validation_batch_size=1000,
                  train_valid_freq=1000, valid_valid_freq=5000,
                  validation_cache_dir=path_to_h5 + '/cache/',
                  model_name=name,
                  use_just_amplitude_spec=True, enable_debug_mode=False,
                  num_train_examples=1652884, num_validation_examples=205581)
    train.run_training()

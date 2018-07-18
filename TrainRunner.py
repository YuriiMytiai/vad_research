from TrainClass import Train
import tensorflow as tf


my_hdd = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/"
path_to_dataset_old = my_hdd + "datasets/h5-full_spec"

my_datasets_on_ssd = "/home/yurii/Documents/datasets/"
path_to_dataset_new = my_datasets_on_ssd + "h5-new_data"


datasets_paths = [path_to_dataset_new]
lr = [1e-7]
dropout = [0.1]
reg = [1e-5]

for path_to_dataset in datasets_paths:
    for cur_lr in lr:
        for cur_reg in reg:
            for cur_kp in dropout:
                print("Iteration data: {}, lr={}, reg={}, dropout={}".format(path_to_dataset,
                                                                             cur_lr, cur_reg, cur_kp))
                if path_to_dataset == path_to_dataset_old:
                    dataset = "old_data"
                    num_valid_ex = 255209
                    num_train_ex = 2064391
                else:
                    dataset = "new_data"
                    num_valid_ex = 205581
                    num_train_ex = 1652884

                cur_name = "CNN_ampl_spec_" + dataset + "_lr-" + str(cur_lr) +\
                           "_reg-" + str(cur_reg) + "_dropout-" + str(cur_kp)

                train = Train(path_to_train_dataset=path_to_dataset + "/train",
                              path_to_validation_dataset=path_to_dataset + "/validation",
                              batch_size=256, learning_rate=cur_lr, num_epochs=40,
                              enable_regularization=True, regularization=cur_reg,
                              enable_dropout=True, dropout_keep_prob=cur_kp,
                              weights_init=tf.initializers.variance_scaling(seed=10), validation_batch_size=1000,
                              train_valid_freq=1000,
                              model_name=cur_name,
                              use_just_amplitude_spec=True, enable_debug_mode=False,
                              num_train_examples=num_train_ex, num_validation_examples=num_valid_ex)
                train.run_training()

                cur_name = "CNN_ampl_spec_" + dataset + "_lr-" + str(cur_lr) + \
                           "_reg-" + "disable" + "_dropout-" + "disable"
                train = Train(path_to_train_dataset=path_to_dataset + "/train",
                              path_to_validation_dataset=path_to_dataset + "/validation",
                              batch_size=256, learning_rate=cur_lr, num_epochs=40,
                              enable_regularization=False, regularization=cur_reg,
                              enable_dropout=False, dropout_keep_prob=cur_kp,
                              weights_init=tf.initializers.variance_scaling(seed=10), validation_batch_size=1000,
                              train_valid_freq=1000,
                              model_name=cur_name,
                              use_just_amplitude_spec=True, enable_debug_mode=False,
                              num_train_examples=num_train_ex, num_validation_examples=num_valid_ex)
                train.run_training()

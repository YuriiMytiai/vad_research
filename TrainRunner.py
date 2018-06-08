from TrainClass import Train
import tensorflow as tf

path_to_h5 = "/media/hdd/datasets/qut-noise-timit-specgrams-high_snr/"

train = Train(path_to_train_h5=path_to_h5 + "/train",
              path_to_validation_h5=path_to_h5 + "/validation",
              batch_size=128, learning_rate=0.0000005, num_epochs=20,
              enable_regularization=True, regularization=0.1,
              weights_init=tf.initializers.variance_scaling, validation_batch_size=500,
              train_valid_freq=100, valid_valid_freq=5000,
              checkpoint_dir='/home/nv1050ti/Documents/vad_research/checkpoints',
              events_log_dir='/home/nv1050ti/Documents/vad_research/events',
              validation_cache_dir=path_to_h5 + 'cache/',
              )

train.run_training()

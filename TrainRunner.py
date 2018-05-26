from TrainClass import Train


path_to_h5 = "/media/yuriy/Data/VAD_research/qut-noise-timit-features"

train = Train(path_to_train_h5=path_to_h5 + "/train",
              path_to_validation_h5=path_to_h5 + "/validation",
              batch_size=50, learning_rate=0.00001, num_epochs=10,
              regularization=0.1, validation_batch_size=500,
              train_valid_freq=50, valid_valid_freq=150,
              checkpoint_dir='/home/yuriy/Desktop/vad_research/vad_research/checkpoints',
              events_log_dir='/home/yuriy/Desktop/vad_research/vad_research/events',
              validation_cache_dir='/media/yuriy/Data/VAD_research/qut-noise-timit-features/cache',
              )

train.run_training()

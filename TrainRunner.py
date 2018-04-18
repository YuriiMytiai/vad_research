from TrainClass import Train

train = Train(batch_size=50, learning_rate=0.00001, num_epochs=10,
              regularization=0.05, validation_batch_size=500,
              train_valid_freq=50, valid_valid_freq=150)

train.run_training()

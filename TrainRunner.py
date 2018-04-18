from TrainClass import Train

train = Train(batch_size=50, learning_rate=0.000005, num_epochs=10,
              regularization=0.05, validation_batch_size=500)

train.run_training()
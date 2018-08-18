from EvaluationFileProcessing import EvaluationProcessing
from EvaluatorClass import Evaluator

audio_filename = "test_audio.flac"

# run preprocessing that means to create h5, then tfrecords file
proc = EvaluationProcessing(audio_filename)
tfrec_filename, num_exs = proc.run()

# run evaluation
checkpoint_dir = "./checkpoints/CNN_ampl_spec_old_data_lr-1e-05_reg-1e-06_dropout-0.3_epoch_8.ckpt"
evaluator = Evaluator(tfrec_filename, checkpoint_dir, num_examples=num_exs)
overlapped_predicts = evaluator.run_evaluation()

# delete tfrec file
proc.delete_tfrec()

# change probas to labels and write csv file:
csv_filename = audio_filename.split('.')[0] + ".csv"
proc.write_labels_to_csv(csv_filename, overlapped_predicts, threshold=0.3)

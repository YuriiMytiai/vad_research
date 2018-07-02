import os
import random


class DataCollector2:
    audio_list = []
    labels_list = []
    data_set_files_list = {}

    def __init__(self, speech_path, noise_path):
        if not os.path.isdir(speech_path):
            raise ValueError("The speech path is not a directory or does not exist")
        if not os.path.isdir(noise_path):
            raise ValueError("The noise path is not a directory or does not exist")
        self.path_to_speech = speech_path
        self.path_to_noise = noise_path

    def load_data(self):
        for root, dirs, files in os.walk(self.path_to_speech):
            for file in files:
                if file.endswith(".flac"):
                    self.audio_list.append(os.path.join(root, file))
        for root, dirs, files in os.walk(self.path_to_noise):
            for file in files:
                if file.endswith(".wav"):
                    self.audio_list.append(os.path.join(root, file))

    def preprocess_files(self, part_of_train_data=0.8):
        random.shuffle(self.audio_list)

        for audio in self.audio_list:
            self.labels_list.append(audio.split('.')[0] + ".csv")

        num_train = int(len(self.audio_list) * part_of_train_data // 1)
        num_test = int((len(self.audio_list) - num_train) / 2)
        train_audio = self.audio_list[0:num_train]
        train_labels = self.labels_list[0:num_train]
        test_audio = self.audio_list[num_train + 1:num_train + num_test]
        test_labels = self.labels_list[num_train + 1:num_train + num_test]
        validation_audio = self.audio_list[num_train + num_test + 1:len(self.audio_list) - 1]
        validation_labels = self.labels_list[num_train + num_test + 1:len(self.audio_list) - 1]

        self.data_set_files_list = {"train_wavs": train_audio, "train_labels": train_labels, "test_wavs": test_audio,
                                    "test_labels": test_labels, "validation_wavs": validation_audio,
                                    "validation_labels": validation_labels}

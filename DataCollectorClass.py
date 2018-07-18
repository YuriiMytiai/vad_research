import os
import random


class DataCollector:
    path_to_data = ''
    wavs_list = []
    labels_list = []
    data_set_files_list = {}

    def __init__(self, path, filter_negative_SNR=True):
        if not os.path.isdir(path):
            raise ValueError("The path is not a directory or does not exist")
        self.path_to_data = path
        self.filt=True

    def load_data(self):
        for root, dirs, files in os.walk(self.path_to_data):
            for file in files:
                if file.endswith(".wav"):

                    # here we want to filter wavs with SNR lower than 0:
                    if self.filt:
                        if "_n-05_" in file:
                            continue
                        elif "_n-10_" in file:
                            continue
                        else:
                            self.wavs_list.append(os.path.join(root, file))
                    else:
                        self.wavs_list.append(os.path.join(root, file))

    def preprocess_files(self, part_of_train_data=0.8):
        random.shuffle(self.wavs_list)

        for wav in self.wavs_list:
            self.labels_list.append(wav.split('.')[0] + ".eventlab")

        num_train = int(len(self.wavs_list) * part_of_train_data // 1)
        num_test = int((len(self.wavs_list) - num_train) / 2)
        train_wavs = self.wavs_list[0:num_train]
        train_labels = self.labels_list[0:num_train]
        test_wavs = self.wavs_list[num_train + 1:num_train + num_test]
        test_labels = self.labels_list[num_train + 1:num_train + num_test]
        validation_wavs = self.wavs_list[num_train + num_test + 1:len(self.wavs_list) - 1]
        validation_labels = self.labels_list[num_train + num_test + 1:len(self.wavs_list) - 1]

        self.data_set_files_list = {"train_wavs": train_wavs, "train_labels": train_labels, "test_wavs": test_wavs,
                                    "test_labels": test_labels, "validation_wavs": validation_wavs,
                                    "validation_labels": validation_labels}

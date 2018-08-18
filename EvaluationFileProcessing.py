import tensorflow as tf
import numpy as np
import h5py
import tqdm
import os
import csv
from Feature_Extractor_2 import FeatureExtractor2


class EvaluationProcessing:

    def __init__(self, filename, fs=16000, fft_size=512, fft_overlap=128, segment_len=8000):
        self.filename = filename
        self.h5_filename = self.filename.split('.')[0] + ".hdf5"
        self.tfrec_filename = self.filename.split('.')[0] + ".tfrecords"

        self.fs = fs
        self.fft_size = fft_size
        self.fft_overlap = fft_overlap
        self.segment_len = segment_len
        self.feature_extractor = FeatureExtractor2(self.h5_filename, dataset_name='data',
                                                   fs=self.fs, fft_size=self.fft_size, fft_overlap=self.fft_overlap,
                                                   segment_len=self.segment_len,
                                                   window=True,
                                                   norm_target_rms=0.1, full_spec=True, segment_normalization=False)

        self.create_h5_file()
        self.h5_file = h5py.File(self.h5_filename, 'r')
        self.labels = []

    def delete_tfrec(self):
        os.remove(self.tfrec_filename)

    def create_h5_file(self):
        self.feature_extractor.extract_features_from_audio_to_h5_evaluation(self.filename)
        self.feature_extractor.close_files()

    def extract_example(self, idx):
        spectrogram = self.h5_file['data'][idx, :]
        spectrogram = np.float32(spectrogram)
        return spectrogram

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _floats_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def run(self):
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(self.tfrec_filename)

        num_examples = self.h5_file['data'].shape[0]
        for example_idx in tqdm.tqdm(range(0, num_examples)):
            spectrogram = self.extract_example(example_idx)
            # Create a feature
            feature = {'spectrogram': self._bytes_feature(spectrogram.tostring())}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        writer.close()

        # delete h5 file
        os.remove(self.h5_filename)

        return self.tfrec_filename, num_examples

    def probas_to_labels(self, probas, threshold):
        for i in range(len(probas)):
            cur_prob = probas[i]
            if cur_prob[0] >= threshold:
                cur_label = 1
            else:
                cur_label = 0
            self.labels.append(cur_label)
        self.labels = np.array(self.labels)

    def write_csv(self, sig_len, csv_filename):
        labels = self.labels
        labels = np.append(labels, [-1])
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter="\t")
            if np.sum(labels) == 0:
                writer.writerow(['0', str((sig_len - 1) / float(self.fs)), 'nonspeech'])
            iter_ctr = 0
            for cur_idx in range(labels.shape[0] - 1):
                if labels[cur_idx] == labels[cur_idx+1]:
                    iter_ctr += 1
                else:
                    start_time = ((cur_idx - iter_ctr) * (self.segment_len - self.fft_overlap)) / float(self.fs)
                    end_time = ((cur_idx + 1) * (self.segment_len - self.fft_overlap)) / float(self.fs)
                    if end_time > ((sig_len - 1) / float(self.fs)):
                        end_time = (sig_len - 1) / float(self.fs)
                    iter_ctr = 0
                    if labels[cur_idx] == 0:
                        label = "nonspeech"
                    elif labels[cur_idx] == 1:
                        label = "speech"
                    writer.writerow([start_time, end_time, label])

    def write_labels_to_csv(self, filename, probas, threshold=0.5):
        self.probas_to_labels(probas, threshold)
        self.write_csv(self.feature_extractor.audio_len, filename)

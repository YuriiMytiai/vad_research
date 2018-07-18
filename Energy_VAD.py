import numpy as np
import soundfile as sf
import librosa
import csv


class EnergyVAD:

    def __init__(self, threshold_ratio=2, buffer_size=2048, buffer_overlap=256, fs=16000):
        self.threshold_ratio = threshold_ratio
        self.buffer_size = buffer_size
        self.buffer_overlap = buffer_overlap
        self.fs = fs

    def create_csv(self, input_filename):
        audio = self.read_audio(input_filename)
        buffered_audio, sig_len = self.buffer_signal(audio)
        buffered_rms = self.calc_rms(buffered_audio)
        labels = self.rms_to_labels(buffered_rms)
        self.write_csv(labels, sig_len, input_filename)

    def read_audio(self, filename):
        data, samplerate = sf.read(filename)
        if samplerate != self.fs:
            data = librosa.resample(data, samplerate, self.fs)
        data = data - np.mean(data)
        k = 1.0 / np.max(np.abs(data))
        data = data * k
        data = np.asarray(data)
        return data

    def buffer_signal(self, sig):
        buffered_sig = []
        sig_len = len(sig)
        start_idx = 0
        end_idx = self.buffer_size
        while end_idx <= sig_len + self.buffer_size - self.buffer_overlap:
            if end_idx > (sig_len - 1):
                sig_segment = sig[start_idx:sig_len - 1]
                sig_segment = np.append(sig_segment, np.zeros(end_idx - (sig_len - 1)))
            else:
                sig_segment = sig[start_idx:end_idx]
            buffered_sig.append(sig_segment)
            start_idx = end_idx - self.buffer_overlap
            end_idx = start_idx + self.buffer_size
        return np.asarray(buffered_sig), sig_len

    @staticmethod
    def calc_rms(buffered_audio):
        rms = []
        for cur_frame in range(buffered_audio.shape[0]):
            cur_rms = np.sqrt(np.mean(buffered_audio[cur_frame, :]**2))
            rms.append(cur_rms)
        return np.asarray(rms)

    def rms_to_labels(self, buffered_rms):
        labels = np.zeros(np.shape(buffered_rms))
        max_rms = np.max(buffered_rms)
        min_rms = np.min(buffered_rms)
        if (max_rms / min_rms) < self.threshold_ratio:
            if max_rms < 1e-6:
                return labels
        for cur_rms in range(buffered_rms.shape[0]):
            if buffered_rms[cur_rms] > (self.threshold_ratio * min_rms):
                labels[cur_rms] = 1
        return labels

    def write_csv(self, labels, sig_len, input_filename):
        csv_filename = input_filename[:-4] + "csv"
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter="\t")
            if np.sum(labels) == 0:
                writer.writerow(['0', str((sig_len - 1) / float(self.fs)), 'nonspeech'])
            iter_ctr = 0
            for cur_idx in range(labels.shape[0] - 1):
                if labels[cur_idx] == labels[cur_idx+1]:
                    iter_ctr += 1
                else:
                    start_time = ((cur_idx - iter_ctr) * (self.buffer_size - self.buffer_overlap)) / float(self.fs)
                    end_time = ((cur_idx + 1) * (self.buffer_size - self.buffer_overlap)) / float(self.fs)
                    if end_time > ((sig_len - 1) / float(self.fs)):
                        end_time = (sig_len - 1) / float(self.fs)
                    iter_ctr = 0
                    if labels[cur_idx] == 0:
                        label = "nonspeech"
                    elif labels[cur_idx] == 1:
                        label = "speech"
                    writer.writerow([start_time, end_time, label])

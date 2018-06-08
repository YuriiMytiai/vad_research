import scipy.io.wavfile
import scipy.signal
import numpy as np
import librosa
#import librosa.display
import h5py
import math
import soundfile as sf


class FeatureExtractor:
    def __init__(self, h5_filename, dataset_name='data',
                 fs=16000, fft_size=512, fft_overlap=128, segment_len=8000, window=True,
                 norm_target_rms=0.1):

        self.h5_filename = h5_filename
        self.dataset_name = dataset_name
        self.label_dataset_name = self.dataset_name + '_labels'
        self.fs = fs
        self.fft_size = fft_size
        self.fft_overlap = fft_overlap
        self.window = window
        self.segment_len = segment_len
        self.target_rms = norm_target_rms
        self.spectrogram_len = math.ceil(segment_len / (self.fft_size - self.fft_overlap))
        self.spectrogram_high = math.ceil(fft_size / 2)

        self.num_speech_events = 0
        self.num_nonspeech_events = 0

        self.file = h5py.File(self.h5_filename)
        self.labels_dataset = self.file.create_dataset(self.label_dataset_name,
                                                       (1, 1),
                                                       maxshape=(None, 1),
                                                       chunks=(1, 1),
                                                       compression="gzip")
        self.features_dataset = self.file.create_dataset(self.dataset_name,
                                                         (1, self.spectrogram_len, self.spectrogram_high),
                                                         maxshape=(None, self.spectrogram_len, self.spectrogram_high),
                                                         chunks=(1, self.spectrogram_len, self.spectrogram_high),
                                                         compression="gzip")

    def add_example_to_h5(self, example, label):
        self.features_dataset.resize((self.features_dataset.shape[0] + 1, self.spectrogram_len,
                                      self.spectrogram_high))
        self.features_dataset[-1, :] = example
        self.labels_dataset.resize((self.labels_dataset.shape[0] + 1, 1))
        self.labels_dataset[-1, :] = label

    def close_files(self):
        self.file.close()

    def extract_features_from_wav_to_h5(self, wav_file):
        event_file = wav_file[:-3] + 'eventlab'
        speech_time_stamps = self.extract_labels_from_eventlab(event_file)

        # file_fs, data = scipy.io.wavfile.read(wav_file)

        # fu**ing Octave saved audio in some strange format under this version of Ubuntu,
        # so we should open files like raw
        data, file_fs = sf.read(wav_file, channels=1, samplerate=16000,
                          format='RAW', subtype='PCM_16')

        if len(data) < 1000:
            print("invalid file " + wav_file)
            return
        elif np.max(data) == 0:
            print("file with no data " + wav_file)
            return


        if file_fs != self.fs:
            data = librosa.resample(data, file_fs, self.fs)
        data = data - np.mean(data)
        k = 1.0 / np.max(data)
        data = data * k
        data = np.asarray(data)

        num_different_events = speech_time_stamps.shape[0]
        for event in range(num_different_events):
            start_event_idx = speech_time_stamps[event, 0]
            end_event_idx = speech_time_stamps[event, 1]
            event_len = (end_event_idx - start_event_idx)
            if event_len < self.segment_len:
                continue
            num_segments = event_len // self.segment_len
            label = speech_time_stamps[event, 2]

            start_idx = start_event_idx
            for segment in range(num_segments):
                end_idx = start_idx + self.segment_len

                buffered_data_segment = self.buffer_signal(data[start_idx:end_idx])
                normalized_data = self.normalization(buffered_data_segment)
                spectrogram = self.calculate_spectrum(normalized_data)

                self.add_example_to_h5(spectrogram, label)
                if label == 1:
                    self.num_speech_events += 1
                elif label == 0:
                    self.num_nonspeech_events += 1

                start_idx = (segment + 1) * self.segment_len

    def extract_labels_from_eventlab(self, event_file):
        speech_time_stamps = []
        with open(event_file, 'r') as file:
            for line in file:
                cells = line.split(' ')
                start_idx = int(float(cells[0]) * self.fs)
                end_idx = int(float(cells[1]) * self.fs)

                if cells[2] == 'speech\n':
                    speech_time_stamps += [(start_idx, end_idx, 1)]
                elif cells[2] == 'nonspeech\n':
                    speech_time_stamps += [(start_idx, end_idx, 0)]
        return np.asarray(speech_time_stamps)

    def buffer_signal(self, sig):
        buffered_sig = []
        sig_len = len(sig)
        start_idx = 0
        end_idx = self.fft_size
        while end_idx <= sig_len + self.fft_size - self.fft_overlap:
            if end_idx > (sig_len - 1):
                sig_segment = sig[start_idx:sig_len - 1]
                sig_segment = np.append(sig_segment, np.zeros(end_idx - (sig_len - 1)))
            else:
                sig_segment = sig[start_idx:end_idx]
            buffered_sig.append(sig_segment)
            start_idx = end_idx - self.fft_overlap
            end_idx = start_idx + self.fft_size
        return np.asarray(buffered_sig)

    def normalization(self, sig):
        num_chunks = sig.shape[0]
        rms = []
        for i in range(0, num_chunks - 1):
            rms.append(self.calc_rms(sig[i]))
        max_rms = np.max(rms)
        k_amp = self.target_rms / max_rms

        sig *= k_amp
        return sig

    @staticmethod
    def calc_rms(sig_vec):
        sum_el = 0
        for i in range(0, len(sig_vec)):
            sum_el += sig_vec[i] ** 2
        return np.sqrt(sum_el / len(sig_vec))

    def calculate_spectrum(self, normalized_data):
        spec = []
        for i in range(0, normalized_data.shape[0]):
            windowed_data = normalized_data[i] * np.hanning(self.fft_size)
            mat_spec = np.fft.fft(windowed_data, n=self.fft_size)
            # amp_spec = mat_spec[int(mat_spec.shape[0]/2):mat_spec.shape[0]]
            amp_spec = mat_spec[0:int(mat_spec.shape[0] / 2)]
            spec.append(np.abs(amp_spec) ** 2)
        return np.asarray(spec)

    #@staticmethod
    #def plot_spectrogram(spectrogram):
    #    import matplotlib.pyplot as plt
    #    plt.figure(figsize=(10, 4))
    #    spectrogram = np.transpose(spectrogram)
    #    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
    #                             y_axis='mel', fmax=8000, x_axis='time')
    #    plt.colorbar(format='%+2.0f dB')
    #    plt.title('Mel spectrogram')
    #    plt.tight_layout()
    #    plt.show()

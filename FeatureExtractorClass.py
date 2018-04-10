import scipy.io.wavfile
import scipy.signal
import numpy as np
import librosa
import librosa.display


class FeatureExtractor:
    fs = 16000
    fft_size = 4096
    fft_overlap = 1024
    window = True

    def __init__(self, fs=16000, fft_size=4096, fft_overlap=1024, window=True):
        self.fs = fs
        self.fft_size = fft_size
        self.fft_overlap = fft_overlap
        self.window = window

    def extract_features_from_wav(self, wav_file):
        file_fs, data = scipy.io.wavfile.read(wav_file)
        data = data - np.mean(data)
        k = 1.0 / np.max(data)
        data = data * k
        data = np.asarray(data)
        if file_fs != self.fs:
            data = librosa.resample(data, file_fs, self.fs)

        buffered_data = self.buffer_signal(data)
        normalized_data = self.normalization(buffered_data)

        spectrogram = self.calculate_spectrum(normalized_data)
        return spectrogram

    def extract_labels_from_eventlab(self, event_file):
        step = self.fft_size - self.fft_overlap
        labels = []
        label = [0]
        np.asarray(labels)
        rest = 0
        with open(event_file, 'r') as file:
            for line in file:
                cells = line.split(' ')
                start_idx = int(float(cells[0]) * self.fs)
                end_idx = int(float(cells[1]) * self.fs)

                label_len = end_idx - start_idx + rest
                num_of_chunks = label_len // step
                rest = label_len % step
                if num_of_chunks == 0:
                    continue
                if cells[2] == 'speech\n':
                    label[0] = 1
                    labels = np.concatenate([labels, np.ones(num_of_chunks)])
                elif cells[2] == 'nonspeech\n':
                    label[0] = 0
                    labels = np.concatenate([labels, np.zeros(num_of_chunks)])
                else:
                    raise ValueError("Unexpected label in file {}".format(event_file))
        if rest > 0:
            labels = np.concatenate([labels, np.asarray(label)])
        return labels

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
        target_rms = 0.01
        num_chunks = sig.shape[0]
        rms = []
        for i in range(0, num_chunks - 1):
            rms.append(self.calc_rms(sig[i]))
        max_rms = np.max(rms)
        k_amp = target_rms / max_rms

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

    @staticmethod
    def plot_spectrogram(spectrogram):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        spectrogram = np.transpose(spectrogram)
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                                 y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()

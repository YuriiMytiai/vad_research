import scipy.io.wavfile
import scipy.signal
import numpy as np
import librosa
import librosa.display
import h5py


class FeatureExtractor:
    fs = 16000
    fft_size = 4096
    fft_overlap = 1024
    window = True
    segment_len = fs * 0.5

    def __init__(self, fs=16000, fft_size=512, fft_overlap=128, segment_len=8000, window=True):
        self.fs = fs
        self.fft_size = fft_size
        self.fft_overlap = fft_overlap
        self.window = window
        self.segment_len = segment_len


    def extract_features_from_wav(self, wav_file, h5_filename, dataset_name):
        event_file = wav_file[:-3] + 'eventlab'
        speech_time_stamps = self.extract_labels_from_eventlab(event_file)
        file_fs, data = scipy.io.wavfile.read(wav_file)
        if file_fs != self.fs:
            data = librosa.resample(data, file_fs, self.fs)
        data = data - np.mean(data)
        k = 1.0 / np.max(data)
        data = data * k
        data = np.asarray(data)

        num_segments = data.shape[0] // self.segment_len
        zeros = np.zeros(data.shape[0] % self.segment_len)
        data = np.append(data, zeros)
        start_idx = 0

        for i in range(0, num_segments):
            end_idx = start_idx + self.segment_len
            buffered_data_segment = self.buffer_signal(data[start_idx:end_idx])
            normalized_data = self.normalization(buffered_data_segment)
            spectrogram = self.calculate_spectrum(normalized_data)
            label = np.asarray(self.read_label(start_idx, end_idx, speech_time_stamps))

            start_idx = (i + 1)*self.segment_len
            filename = h5_filename + '_' + str(i) + '.hdf5'
            with h5py.File(filename, "w") as h5file:
                h5file.create_dataset(dataset_name, spectrogram.shape, spectrogram.dtype, spectrogram)
                label_dataset_name = dataset_name + '_labels'
                h5file.create_dataset(label_dataset_name, label.shape, label.dtype, label)



    def extract_labels_from_eventlab(self, event_file):
        speech_time_stamps = []
        with open(event_file, 'r') as file:
            for line in file:
                cells = line.split(' ')
                start_idx = int(float(cells[0]) * self.fs)
                end_idx = int(float(cells[1]) * self.fs)

                if cells[2] == 'speech\n':
                    speech_time_stamps += [(start_idx, end_idx)]
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

    def read_label(self, start_idx, end_idx, speech_time_stamps):
        for i in range(0, speech_time_stamps.shape[0]):
            if start_idx >= speech_time_stamps[i, 0] | start_idx <= speech_time_stamps[i, 1]:
                return 1
            elif end_idx >= speech_time_stamps[i, 0] | end_idx <= speech_time_stamps[i, 1]:
                return 1
        return 0

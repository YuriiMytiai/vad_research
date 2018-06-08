from DataCollectorClass import DataCollector
from FeatureExtractorClass import FeatureExtractor
import tqdm


def print_dataset_info(num_speech_events, num_nonspeech_events):
    num_events = num_speech_events + num_nonspeech_events
    print('Dataset size: {}'.format(num_events))
    print('Speech events: {} examples, {:.2f}%'.format(num_speech_events, num_speech_events / num_events * 100))
    print('Nonspeech events: {} examples, {:.2f}%'.format(num_nonspeech_events, num_nonspeech_events / num_events * 100))


path_to_wavs = "/home/nv1050ti/datasets/qut-noise/QUT-NOISE/QUT-NOISE-TIMIT"
path_to_h5 = "/media/hdd/datasets/qut-noise-timit-specgrams-high_snr/"

data = DataCollector(path_to_wavs)
data.load_data()
data.preprocess_files()

train_dataset_size = len(data.data_set_files_list["train_wavs"])
validation_dataset_size = len(data.data_set_files_list["validation_wavs"])
test_dataset_size = len(data.data_set_files_list["test_wavs"])

h5filename_train = path_to_h5 + '/train/' + 'dataset.hdf5'
h5filename_validation = path_to_h5 + '/validation/' + 'dataset.hdf5'
h5filename_test = path_to_h5 + '/test/' + 'dataset.hdf5'

feature_extractor = FeatureExtractor(h5filename_train)
for i in tqdm.tqdm(range(train_dataset_size)):
    feature_extractor.extract_features_from_wav_to_h5(data.data_set_files_list["train_wavs"][i])
feature_extractor.close_files()
print('Train dataset:')
print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)


feature_extractor = FeatureExtractor(h5filename_validation)
for i in tqdm.tqdm(range(validation_dataset_size)):
    feature_extractor.extract_features_from_wav_to_h5(data.data_set_files_list["validation_wavs"][i])
feature_extractor.close_files()
print('Validation dataset:')
print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)


feature_extractor = FeatureExtractor(h5filename_test)
for i in tqdm.tqdm(range(test_dataset_size)):
    feature_extractor.extract_features_from_wav_to_h5(data.data_set_files_list["test_wavs"][i])
feature_extractor.close_files()
print('Test dataset:')
print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)



print('finished')

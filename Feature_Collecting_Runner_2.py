from Data_Collector_2 import DataCollector2
from Feature_Extractor_2 import FeatureExtractor2
import tqdm
import threading


def print_dataset_info(num_speech_events, num_nonspeech_events):
    num_events = num_speech_events + num_nonspeech_events
    print('Dataset size: {}'.format(num_events))
    print('Speech events: {} examples, {:.2f}%'.format(num_speech_events, num_speech_events / num_events * 100))
    print('Nonspeech events: {} examples, {:.2f}%'.format(num_nonspeech_events, num_nonspeech_events / num_events * 100))


path_to_speech = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/other_speech"
path_to_noise = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/other_noises"
path_to_h5 = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/h5-new_data"

data = DataCollector2(speech_path=path_to_speech, noise_path=path_to_noise)
data.load_data()
data.preprocess_files()

train_dataset_size = len(data.data_set_files_list["train_wavs"])
validation_dataset_size = len(data.data_set_files_list["validation_wavs"])
test_dataset_size = len(data.data_set_files_list["test_wavs"])

h5filename_train = path_to_h5 + '/train/' + 'dataset.hdf5'
h5filename_validation = path_to_h5 + '/validation/' + 'dataset.hdf5'
h5filename_test = path_to_h5 + '/test/' + 'dataset.hdf5'


def train_feature_extractor(h5filename_train, data):
    feature_extractor = FeatureExtractor2(h5filename_train, full_spec=True, segment_normalization=False)
    for i in tqdm.tqdm(range(train_dataset_size)):
        feature_extractor.extract_features_from_audio_to_h5(data.data_set_files_list["train_wavs"][i])
    feature_extractor.close_files()
    print('Train dataset:')
    print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)


def validation_feature_extractor(h5filename_validation, data):
    feature_extractor = FeatureExtractor2(h5filename_validation, full_spec=True, segment_normalization=False)
    for i in tqdm.tqdm(range(validation_dataset_size)):
        feature_extractor.extract_features_from_audio_to_h5(data.data_set_files_list["validation_wavs"][i])
    feature_extractor.close_files()
    print('Validation dataset:')
    print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)


def test_feature_extractor(h5filename_test, data):
    feature_extractor = FeatureExtractor2(h5filename_test, full_spec=True, segment_normalization=False)
    for i in tqdm.tqdm(range(test_dataset_size)):
        feature_extractor.extract_features_from_audio_to_h5(data.data_set_files_list["test_wavs"][i])
    feature_extractor.close_files()
    print('Test dataset:')
    print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)


threading.Thread(target=train_feature_extractor, args=(h5filename_train, data)).start()
threading.Thread(target=validation_feature_extractor, args=(h5filename_validation, data)).start()
threading.Thread(target=test_feature_extractor, args=(h5filename_test, data)).start()

print('finished')

from DataCollectorClass import DataCollector
from FeatureExtractorClass import FeatureExtractor
import tqdm


def print_dataset_info(num_speech_events, num_nonspeech_events):
    num_events = num_speech_events + num_nonspeech_events
    print('Dataset size: {}'.format(num_events))
    print('Speech events: {} examples, {:.2f}%'.format(num_speech_events, num_speech_events / num_events))
    print('Nonspeech events: {} examples, {:.2f}%'.format(num_nonspeech_events, num_nonspeech_events / num_events))


path_to_wavs = "C:\\Users\\User\\Desktop\\vad_research\\datasets\\qut-noise-timit\\qutnoise\\QUT-NOISE\\QUT-NOISE-TIMIT"
path_to_h5 = "D:\\h5dataset"

data = DataCollector(path_to_wavs)
data.load_data()
data.preprocess_files()

train_dataset_size = len(data.data_set_files_list["train_wavs"])
train_dataset_size = 100
validation_dataset_size = len(data.data_set_files_list["validation_wavs"])
validation_dataset_size = 10

h5filename_train = path_to_h5 + '\\train\\' + 'dataset.hdf5'
h5filename_validation = path_to_h5 + '\\validation\\' + 'dataset.hdf5'
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




print('finished')

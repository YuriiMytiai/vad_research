from DataCollectorClass import DataCollector
from FeatureExtractorClass import FeatureExtractor
import tqdm


def print_dataset_info(num_speech_events, num_nonspeech_events):
    num_events = num_speech_events + num_nonspeech_events
    print('Dataset size: {}'.format(num_events))
    print('Speech events: {} examples, {.2f}%'.format(num_speech_events, num_speech_events / num_events))
    print('Nonspeech events: {} examples, {.2f}%'.format(num_nonspeech_events, num_nonspeech_events / num_events))


path_to_wavs = "C:\\Users\\User\\Desktop\\vad_research\\datasets\\qut-noise-timit\\qutnoise\\QUT-NOISE\\QUT-NOISE-TIMIT"
path_to_h5 = "D:\\h5dataset"

data = DataCollector(path_to_wavs)
data.load_data()
data.preprocess_files()


feature_extractor = FeatureExtractor()

for i in tqdm.tqdm(range(len(data.data_set_files_list["train_wavs"]))):
    h5filename = path_to_h5 + '\\train\\' + 'file_' + str(i)
    feature_extractor.extract_features_from_wav_to_h5(data.data_set_files_list["train_wavs"][i], h5filename)
print('Train dataset:')
print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)
feature_extractor.reset_events_counter()


for i in tqdm.tqdm(range(0, len(data.data_set_files_list["validation_wavs"]))):
    h5filename = path_to_h5 + '\\validation\\' + 'file_' + str(i)
    feature_extractor.extract_features_from_wav_to_h5(data.data_set_files_list["validation_wavs"][i], h5filename)
print('Validation dataset:')
print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)
feature_extractor.reset_events_counter()



#for i in tqdm.tqdm(range(0, len(data.data_set_files_list["test_wavs"]))):
#    h5filename = path_to_h5 + '\\test\\' + 'file_' + str(i) + '.hdf5'
#    feature_extractor.extract_features_from_wav(data.data_set_files_list["test_wavs"][i], h5filename, 'test_data')
#print('Test dataset:')
#print_dataset_info(feature_extractor.num_speech_events, feature_extractor.num_nonspeech_events)
#feature_extractor.reset_events_counter()

print('finished')

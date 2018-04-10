from DataCollectorClass import DataCollector
from FeatureExtractorClass import FeatureExtractor

path = "C:\\Users\\User\\Desktop\\vad_research\\datasets\\qut-noise-timit\\qutnoise\\QUT-NOISE\\QUT-NOISE-TIMIT"

data = DataCollector(path)
data.load_data()

print(data.wavs_list[0])
print(len(data.wavs_list))


data.preprocess_files()
print(data.data_set_files_list["train_wavs"][0])
print(data.data_set_files_list["train_labels"][0])

feature_extractor = FeatureExtractor()
print(feature_extractor.extract_features_from_wav(data.data_set_files_list["train_wavs"][0]).shape)
print(feature_extractor.extract_labels_from_eventlab(data.data_set_files_list["train_labels"][0]).shape)
print('dg')

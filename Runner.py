from DataCollectorClass import DataCollector
from FeatureExtractorClass import FeatureExtractor
import tqdm

path_to_wavs = "C:\\Users\\User\\Desktop\\vad_research\\datasets\\qut-noise-timit\\qutnoise\\QUT-NOISE\\QUT-NOISE-TIMIT"
path_to_h5 = "D:\\h5dataset"

data = DataCollector(path_to_wavs)
data.load_data()
data.preprocess_files()


feature_extractor = FeatureExtractor()


for i in tqdm.tqdm(range(len(data.data_set_files_list["train_wavs"]))):
    h5filename = path_to_h5 + '\\train\\' + 'file_' + str(i)
    feature_extractor.extract_features_from_wav(data.data_set_files_list["train_wavs"][i], h5filename, dataset_name='data')


for i in tqdm.tqdm(range(0, len(data.data_set_files_list["validation_wavs"]))):
    h5filename = path_to_h5 + '\\validation\\' + 'file_' + str(i)
    feature_extractor.extract_features_from_wav(data.data_set_files_list["train_wavs"][i], h5filename, 'validation_data')




#for i in tqdm.tqdm(range(0, len(data.data_set_files_list["test_wavs"]))):
#    h5filename = path_to_h5 + '\\test\\' + 'file_' + str(i) + '.hdf5'
#    feature_extractor.extract_features_from_wav(data.data_set_files_list["test_wavs"][i], h5filename, 'test_data')


print('finished')

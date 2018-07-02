import os
from Energy_VAD import EnergyVAD

path_to_audio = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/other_speech/LibriSpeech/train-clean-360"
vad = EnergyVAD(threshold_ratio=5)
#filename = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/other_speech/LibriSpeech/train-clean-360/1603/140931/1603-140931-0005.flac"
#vad.create_csv(filename)

num_bad_files = 0
for root, dirs, files in os.walk(path_to_audio):
    for file in files:
        if file.endswith(".flac"):
            try:
                vad.create_csv(os.path.join(root, file))
            except:
                num_bad_files += 1
                print("Something wrong with file: {}\nNumber of bad files: {}".format(file, num_bad_files))



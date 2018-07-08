import tensorflow as tf
import numpy as np
import h5py
import tqdm

h5_filename = "/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/h5-new_data/train/dataset.hdf5"
tfrec_filename = h5_filename.split('.')[0] + ".tfrecords"

h5_file = h5py.File(h5_filename, 'r')


def extract_example(idx):
    spectrogram = h5_file['data'][idx, :]
    label = h5_file['data_labels'][idx, 0]
    spectrogram = np.float32(spectrogram)
    label = np.int64(label)
    return spectrogram, [label]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# open the TFRecords file
writer = tf.python_io.TFRecordWriter(tfrec_filename)

for example_idx in tqdm.tqdm(range(0, h5_file['data'].shape[0])):
    spectrogram, label = extract_example(example_idx)
    # Create a feature
    feature = {'label': _int64_feature(label),
               'spectrogram': _bytes_feature(spectrogram.tostring())}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()

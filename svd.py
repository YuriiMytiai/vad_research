import tensorflow as tf
import numpy as np
import h5py
import gc


#train_file = h5py.File("/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/h5-new_data/train/dataset.hdf5", 'r')
#validation_file = h5py.File("/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/h5-new_data/validation/dataset.hdf5", 'r')

#t = train_file['data'].shape[0]
#v = validation_file['data'].shape[0]

#def close_files():
#    for obj in gc.get_objects():  # Browse through ALL objects
#        if isinstance(obj, h5py.File):  # Just HDF5 files
#            try:
#                obj.close()
#            except:
#                pass  # Was already closed


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print('The end of the training at {}'.format(str(datetime.now())))

#data_path = '/media/yurii/021f412c-0a12-4716-aaa2-e1d8c03e4188/datasets/h5-new_data/validation/dataset.tfrecords'  # address to save the hdf5 file
#with tf.Session() as sess:
#    feature = {'train/image': tf.FixedLenFeature([], tf.string),
#               'train/label': tf.FixedLenFeature([], tf.int64)}
#    # Create a list of filenames and pass it to a queue
#    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
#    # Define a reader and read the next record
#    reader = tf.TFRecordReader()
#    _, serialized_example = reader.read(filename_queue)
#    # Decode the record read by the reader
#    features = tf.parse_single_example(serialized_example, features=feature)
#    # Convert the image data from string back to the numbers
#    #image = tf.decode_raw(features['train/image'], tf.float32)
#    image = tf.cast(features["train/image"], tf.float32)
##
#
#    # Cast label data into int32
#    label = tf.cast(features['train/label'], tf.int32)
#    # Reshape image data into the original shape
#    image = tf.reshape(image, [21, 256, 2])3
#
#    # Any preprocessing here ...
#
#    # Creates batches by randomly shuffling tensors
#    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
#                                            min_after_dequeue=10)


#f = h5py.File("imagetest.hdf5")
#dset2 = f.create_dataset('timetraces4', (1000, 100, 200), maxshape=(None, 100, 200), chunks=(1, 100, 200))

#for i in range(1000):
#    arr += 1
#    add_trace_2(arr)

#dset2.resize((ntraces, 100, 200))

#print(dset2.chunks)




#close_files()

#from tensorflow.python.client import device_lib

#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devices()
 #   return [x.name for x in local_device_protos if x.device_type == 'GPU']

#get_available_gpus()


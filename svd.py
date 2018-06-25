import tensorflow as tf
import numpy as np
import h5py
import gc


def close_files():
    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, h5py.File):  # Just HDF5 files
            try:
                obj.close()
            except:
                pass  # Was already closed

#f = h5py.File("imagetest.hdf5")
#dset2 = f.create_dataset('timetraces4', (1000, 100, 200), maxshape=(None, 100, 200), chunks=(1, 100, 200))

#for i in range(1000):
#    arr += 1
#    add_trace_2(arr)

#dset2.resize((ntraces, 100, 200))

#print(dset2.chunks)

close_files()

#from tensorflow.python.client import device_lib

#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devices()
 #   return [x.name for x in local_device_protos if x.device_type == 'GPU']

#get_available_gpus()


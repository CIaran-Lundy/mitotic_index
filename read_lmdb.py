import tensorflow as tf
from tensorpack.dataflow import *
import cv2
#todo: do something like the following to avoid those weird messages about nvidia:
#import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi


lmdb_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/lmdb_dir/single_lmdb/data.mdb"

output_types= tf.dtypes.int8, tf.dtypes.string
output_shapes= tf.TensorShape([16, 16]), tf.TensorShape([1, 3])
dataset = tf.raw_ops.LMDBDataset(filenames="/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/lmdb_dir/single_lmdb/data.mdb",output_types=output_types, output_shapes=output_shapes, name=None)

print(dataset.get_shape())

print("Eager execution: {}".format(tf.executing_eagerly()))

#new = tf.data.Dataset.from_tensors(dataset)


tf.compat.v1.disable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))
data = tf.compat.v1.LMDBReader(options=lmdb_dir)


ds = LMDBSerializer.load(lmdb_dir, shuffle=False)
ds = BatchData(ds, 2, use_list=True)
TestDataSpeed(ds).start()

ds = LMDBSerializer.load(lmdb_dir, shuffle=False)
ds = LocallyShuffleData(ds, 50000)
ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
ds = BatchData(ds, 2)


#todo: so then, it seems that ive chosen to make this much harder for myself by trying to implement efficient dataflow.
# good job idiot. now then, to make this work, im going to need to do the following:
# make lmdb (done)
# read some entries from the lmdb into a buffer ( as many as possible for the given amount of ram)
# shuffle entries in buffer
# read entries and process them so that they are ready to be presented to the CNN
# put shuffled entries into a queue
# make threads that will take entries from this queue and put them into batches
# make a keras model that will ask for a batche, wait until it runs through the model, and then ask for another batch - keras fit_generator might be a shout for this
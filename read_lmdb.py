import tensorflow as tf
from tensorpack.dataflow import *

#todo: do something like the following to avoid those weird messages about nvidia:
#import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi


lmdb_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/lmdb_dir/single_lmdb/data.mdb"

output_types= tf.dtypes.int8, tf.dtypes.string
output_shapes= tf.TensorShape([16, 16]), tf.TensorShape([1, 3])
#dataset = tf.raw_ops.LMDBDataset(filenames="/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/lmdb_dir/single_lmdb/data.mdb",output_types=output_types, output_shapes=output_shapes, name=None)


print("Eager execution: {}".format(tf.executing_eagerly()))

#new = tf.data.Dataset.from_tensors(dataset)


tf.compat.v1.disable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))
data = tf.compat.v1.LMDBReader(options=lmdb_dir)


ds = LMDBSerializer.load(lmdb_dir, shuffle=False)
ds = BatchData(ds, 2, use_list=True)
TestDataSpeed(ds).start()
#todo: this should work, now i just need to change the format that the image is saved in
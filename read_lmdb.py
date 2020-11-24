import tensorflow as tf

lmdb_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/lmdb_dir"

output_types= tf.dtypes.int8, tf.dtypes.string
output_shapes= tf.TensorShape([16, 16]), tf.TensorShape([1, 3])
dataset = tf.raw_ops.LMDBDataset(filenames="/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/lmdb_dir/single_lmdb/data.mdb",output_types=output_types, output_shapes=output_shapes, name=None)


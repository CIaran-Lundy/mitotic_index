######## CNN trainer
#LOAD FILE
#READ METADATA
#SELECT AREA AROUND CO-ORDINATE THAT SAYS IT IS OR IS NOT A MITOTIC EVENT AND SAVE IMAGE
#ADD TO YARP OR NARP DATASET
#IDENTIFY CELLS FOR MITOTIC INDEX CALC
#GET ALL LABELLED BITS
#APPEND IMAGE SECTION THAT IS ASSOCIATED TO A LABEL
#APPEND LABEL TO ANOTHER LIST
#
#NOW YOU HAVE A TRAINING DATASET

# train a ResNet style
import tensorflow as tf
import os
import cv2
import csv
import lmdb
import pickle
import numpy as np
import pandas as pd
import itertools
metadata_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/"
images_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/frames/x40/"
image_format = "tiff"
metadata_format = "csv"
metadata_delimiter = ","
lmdb_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/lmdb_dir"

if not image_format.startswith('.'):
    image_format = '.' + image_format
if not metadata_format.startswith('.'):
    metadata_format = '.' + metadata_format

class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


def store_single_lmdb(image, image_id, label, lmdb_dir):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 1000
    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir + '/' + "single_lmdb"), map_size=map_size)
    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        #value = CIFAR_Image(image, label)
        value = (image, label)
        key = image_id
        txn.put(key.encode("ascii"), pickle.dumps(value))
    #print(env.stat())
    env.close()


def read_single_lmdb(image_id, lmdb_dir):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir + '/' + "single_lmdb"), readonly=True)
    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(image_id.encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        data = pickle.loads(data)
        # Retrieve the relevant bits
        image = data[0]
        label = data[1]
        #image = cifar_image.get_image()
        #label = cifar_image.label
        #cv2.imshow(winname="test_image", mat=image)
        #cv2.waitKey(0)
    env.close()
    #txn.drop(env)
    return image, label


def get_files(metadata_dir, images_dir, image_format, metadata_format):
    """find all image files of a specified file format in a specified data directory,
    and pair them into a dictionary structure with metadata files that share the same
     basename, and have the specified file format"""
    all_metadata_files = [x for x in set(os.listdir(metadata_dir)) if x.endswith(metadata_format)]
    all_image_files = [x for x in set(os.listdir(images_dir)) if x.endswith(image_format)]
    images_and_metadata = {}
    for metadata, image in itertools.product(all_metadata_files, all_image_files):
        if image.split('.')[0] in metadata:
                images_and_metadata[metadata] = image
    return images_and_metadata


def create_cropped_image_database(images_and_metadata, metadata_delimiter, metadata_dir, images_dir, lmdb_dir):
    """read image files, crop based on metadata, store in a LMDB"""
    passed_metadata_files_counter = 0
    empty_metadata_files_counter = 0
    for metadata_name, image_name in images_and_metadata.items():
        image = cv2.imread(images_dir+image_name)
        metadata = open(metadata_dir+metadata_name).read().split('\n')[:-1]
        metadata = [x for x in metadata if len(x) > 0]
        for data_point in metadata:
            data_point = data_point.split(metadata_delimiter)
            data_point.append(metadata_name.split("_", 2)[-1].split('.')[0])
            width = int(16)
            height = int(16)
            y_1 = max(0, int(data_point[0]) - height)
            y_2 = max(0, int(data_point[0]) + height)
            x_1 = max(0, int(data_point[1]) - width)
            x_2 = max(0, int(data_point[1]) + width)
            cropped_image = image[x_1:x_2, y_1:y_2]
            if not len(cropped_image) == len(image):
                store_single_lmdb(cropped_image, metadata_name.split('.')[0], data_point, lmdb_dir)
                passed_metadata_files_counter += 1
            else:
                print(len(image))
    print("passed_metadata_files_counter: {}".format(passed_metadata_files_counter))
    print("empty_metadata_files_counter: {}".format(empty_metadata_files_counter))


images_and_metadata = get_files(metadata_dir, images_dir, image_format, metadata_format)
print("got_files")
create_cropped_image_database(images_and_metadata, metadata_delimiter, metadata_dir, images_dir, lmdb_dir)

file = read_single_lmdb("A03_05Cc_not_mitosis", lmdb_dir)
print(file)


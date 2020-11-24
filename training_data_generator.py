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

import os
import cv2
import csv
import lmdb
import pickle
import numpy as np

data_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis"
image_format = "jpg"
metadata_format = "csv"
metadata_delimiter = ","

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


def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """

    map_size = image.nbytes * 1000
    lmdb_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis/lmdb_dir"
    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir + '/' + "single_lmdb"), map_size=map_size)
    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = image_id
        txn.put(key.encode("ascii"), pickle.dumps(value))
    #print(env.stat())
    env.close()


def get_files(data_dir, image_format, metadata_format):
    """find all image files of a specified file format in a specified data directory,
    and pair them into a dictionary structure with metadata files that share the same
     basename, and have the specified file format"""
    all_files = set(os.listdir(data_dir))
    images_and_metadata = {}
    for file in all_files:
        if file.endswith(image_format):
            try:
                metadata = file.split('.')[0] + metadata_format
                os.path.exists(metadata)
                images_and_metadata[file] = metadata
            except:
                print("{file} - file has no metadata")
    return images_and_metadata


def create_image_database_method_1(images_and_metadata, metadata_delimiter, data_dir):
    """read image files, crop based on metadata, store in a LMDB"""
    working_directory = os.getcwd()
    os.chdir(data_dir)
    passed_metadata_files_counter = 0
    empty_metadata_files_counter = 0
    for image_name, metadata in images_and_metadata.items():
        image = cv2.imread(image_name)
        metadata = open(metadata).read().split('\n')
        store_single_lmdb(image, image_name, metadata)
        for data_point in metadata:
            data_point = data_point.split(metadata_delimiter)
            try:
                width = int(10)
                height = int(10)
                y_1 = int(data_point[0]) - height
                y_2 = int(data_point[0]) + height

                x_1 = int(data_point[1]) - width
                x_2 = int(data_point[1]) + width

                crop_image = image[x_1:x_2, y_1:y_2]
                #cv2.imshow("Cropped", crop_image)
                #cv2.waitKey(0)
                store_single_lmdb(image, image_name, metadata)
                passed_metadata_files_counter += 1
            except:
                if len(data_point) == 1:
                    empty_metadata_files_counter += 1
                else:
                    print(data_point)
    print("passed_metadata_files_counter: {}".format(passed_metadata_files_counter))
    print("empty_metadata_files_counter: {}".format(empty_metadata_files_counter))
    os.chdir(working_directory)


images_and_metadata = get_files(data_dir, image_format, metadata_format)
create_image_database_method_1(images_and_metadata, metadata_delimiter, data_dir)


def get_files_method_2(data_dir, image_format, metadata_format):
    """find all image files of a specified file format in a specified data directory,
    and pair them into a dictionary structure with metadata files that share the same
     basename, and have the specified file format

     do all metadata in a step beofre image loading?"""
    all_files = set(os.listdir(data_dir))
    images_and_metadata = {}
    for file in all_files:
        if file.endswith(image_format):
            try:
                metadata = file.split('.')[0] + metadata_format
                os.path.exists(metadata)
                images_and_metadata[file] = metadata
            except:
                print("{file} - file has no metadata")
    return images_and_metadata

def create_image_database_method_2(file, images_and_metadata):
    """read image files, crop based on metadata, store in a LMDB"""
    for image, metadata in images_and_metadata.items():
        image = cv2.imread(image)
        metadata = csv.read
        y = 0
        x = 0
        h = 300
        w = 510
        crop_image = image[x:w, y:h]
        #cv2.imshow("Cropped", crop_image)
        #cv2.waitKey(0)

        region_of_interest = 0
        tag = file.split("_", 2)[-1]
        print(tag)

#for file in list of files at training_aperio/A03/mitosis/:
#   get file name
#   split string
#   if it contains "not":
#       add to narp dataset
#   elif it doesnt contain not:

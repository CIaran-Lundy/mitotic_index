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
data_dir = "/home/ciaran/PycharmProjects/mitotic_index/training_aperio/A03/mitosis"
image_format = "jpg"
metadata_format = "csv"

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

def create_image_database_method_1(images_and_metadata):
    """read image files, crop based on metadata, store in a LMDB"""
    for image, metadata in images_and_metadata.items():
        image = cv2.imread(image)
        metadata = csv.read
        y = 0
        x = 0
        h = 300
        w = 510
        crop_image = image[x:w, y:h]
        cv2.imshow("Cropped", crop_image)
        cv2.waitKey(0)
        region_of_interest =
        tag = file.split("_", 2)[-1]
        print(tag)



#for file in list of files at training_aperio/A03/mitosis/:
#   get file name
#   split string
#   if it contains "not":
#       add to narp dataset
#   elif it doesnt contain not:

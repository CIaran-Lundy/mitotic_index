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
    metadata_and_cropped_images = []
    for metadata_name, image_name in images_and_metadata.items():
        image = cv2.imread(images_dir+image_name)
        metadata = open(metadata_dir+metadata_name).read().split('\n')[:-1]
        metadata = [x for x in metadata if len(x) > 0]
        for data_point in metadata:

            data_point = data_point.split(metadata_delimiter)
            training_label = metadata_name.split("_", 2)[-1].split('.')[0]
            width = int(16)
            height = int(16)
            y_1 = max(0, int(data_point[0]) - height)
            y_2 = max(0, int(data_point[0]) + height)
            x_1 = max(0, int(data_point[1]) - width)
            x_2 = max(0, int(data_point[1]) + width)
            cropped_image = image[x_1:x_2, y_1:y_2]
            if not len(cropped_image) == len(image):
                metadata_image_pair = (cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY), metadata_name.split('.')[0])
                metadata_and_cropped_images.append(metadata_image_pair)
                passed_metadata_files_counter += 1
            else:
                print(len(image))
    print("passed_metadata_files_counter: {}".format(passed_metadata_files_counter))
    print("empty_metadata_files_counter: {}".format(empty_metadata_files_counter))
    return metadata_and_cropped_images


images_and_metadata = get_files(metadata_dir, images_dir, image_format, metadata_format)
print("got_files")
final_dict = create_cropped_image_database(images_and_metadata, metadata_delimiter, metadata_dir, images_dir, lmdb_dir)

total_dict_length = len(final_dict)
training_dict_length = round(int(total_dict_length * 0.9), 0)
training_set = list(itertools.islice(final_dict, training_dict_length))
testing_set = list(itertools.islice(final_dict, total_dict_length - training_dict_length))
IMAGE_SIZE = 32

testing_images = []
testing_labels = []
training_images = []
training_labels = []

for image, label in testing_set:
    if image.shape == (32, 32):
        testing_images.append(image)
        if "not" in label:
            label = 0
        else:
            label = 1
        testing_labels.append(label)

for image, label in training_set:
    if image.shape == (32, 32):
        training_images.append(image)
        if "not" in label:
            label = 0
        else:
            label = 1
        training_labels.append(label)

testing_images = np.asarray(testing_images).astype('uint8')
testing_labels = np.asarray(testing_labels).astype('float64')
training_images = np.asarray(training_images).astype('float64')
training_labels = np.asarray(training_labels).astype('uint8')

number_of_images = training_images.shape[0]
IMAGE_HEIGHT = training_images.shape[1]
IMAGE_WIDTH = training_images.shape[2]

from tensorflow import keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=30)


test_loss, test_acc = model.evaluate(testing_images,  testing_labels, verbose=2)

print('\nTest accuracy:', test_acc)
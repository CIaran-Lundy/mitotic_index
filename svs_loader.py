import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,50).__str__()
import cv2 # import after setting OPENCV_IO_MAX_IMAGE_PIXELS
import matplotlib
import py_wsi
import openslide
import py_wsi.imagepy_toolkit as tk
import pprint
#import bioformats
import opencv_wrapper
import numpy as np
image_file = '/home/ciaran/PycharmProjects/mitotic_index/'

file_dir = '/home/ciaran/PycharmProjects/mitotic_index/'
db_location = '/home/ciaran/PycharmProjects/mitotic_index/'
xml_dir = file_dir
patch_size = 64
level = 10
db_name = "patch_db"
overlap = 0
label_map = {}
storage_type = "lmdb"

turtle = py_wsi.Turtle(file_dir, db_location, db_name, xml_dir)

level_count, level_tiles, level_dims = turtle.retrieve_tile_dimensions(file_name='c4b95da36e32993289cb.svs', patch_size=128)

print("Level count:         " + str(level_count))
print("Level tiles:         " + str(level_tiles))
print("Level dimensions:    " + str(level_dims))

patch_1 = turtle.retrieve_sample_patch("c4b95da36e32993289cb.svs", 128, 1, overlap=12)
#patch_2 = turtle.retrieve_sample_patch("c4b95da36e32993289cb.svs", 128, 2, overlap=12)
#patch_4 = turtle.retrieve_sample_patch("c4b95da36e32993289cb.svs", 128, 4, overlap=12)
#patch_8 = turtle.retrieve_sample_patch("c4b95da36e32993289cb.svs", 128, 8, overlap=12)
#patch_16 = turtle.retrieve_sample_patch("c4b95da36e32993289cb.svs", 128, 16, overlap=12)
patch_17 = turtle.retrieve_sample_patch("c4b95da36e32993289cb.svs", 128, 17, overlap=12)
#tk.show_images([patch_17], 1, 1)
print(type(patch_17))
#image1 =cv2.imread(patch_17)
patch_17 = cv2.cvtColor(np.array(patch_17), cv2.COLOR_RGB2GRAY)
#patch_17 = patch_17[:, :, ::-1].copy()
print(type(patch_17))
thresh1 = cv2.adaptiveThreshold(patch_17, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
print(type(thresh1))
cv2.imshow("image1", thresh1)
cv2.imshow("image", patch_17)
cv2.waitKey(0)
#print(type(patch_1))
#tk.show_images([patch_1, patch_2, patch_4, patch_8, patch_16, patch_17], 3, 2)
#tk.show_images([patch_1, patch_2, patch_4, patch_8, patch_16], 5, 1)
exit()
image = openslide.OpenSlide(image_file)
prop =str(image.properties)
prop = prop.split(",")
for p in prop:
    print(p)
print(image.level_count)

thumb = image.get_thumbnail(size=(128, 128))

img = thumb.thumbnail(size=(128, 128))



# Python program to illustrate
# Otsu thresholding type on an image


# path to input image is specified and
# image is loaded with imread command
image1 =image
image1 =cv2.imread(image_file)
#hresh1 = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
#image1 = opencv_wrapper.threshold_adaptive(image1,255)
# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# applying Otsu thresholding
# as an extra flag in binary
# thresholding
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                             cv2.THRESH_OTSU)

cv2.threshold
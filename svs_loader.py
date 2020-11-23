import py_wsi
import openslide
import pprint
#import bioformats
image_file = '/home/ciaran/c4b95da36e32993289cb.svs'
image = openslide.OpenSlide(image_file)
prop =str(image.properties)
prop = prop.split(",")
for p in prop:
    print(p)
print(image.level_count)

thumb = image.get_thumbnail(size=(128, 128))

img = thumb.thumbnail(size=(128, 128))

import cv2

# Python program to illustrate
# Otsu thresholding type on an image

# organizing imports
import cv2
import numpy as np

# path to input image is specified and
# image is loaded with imread command

image1 =cv2.imread(image_file)

# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# applying Otsu thresholding
# as an extra flag in binary
# thresholding
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                             cv2.THRESH_OTSU)

import opencv_wrapper


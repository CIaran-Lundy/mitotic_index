#main pre-processing script

#load svs

#tile svs with py_wsi

#threshold tiles with cv2

#keep tiles with >90% tile coverage by tissue

#for all kept tiles, go UP a few levels (i.e. go from level 10 to 17) to increase tile resolution

# tile the high res kept tiles to reduce size of object being handled

# use cell profiler to estimate cell density, and select top ~30 most dense tiles on whole slide

# normalize tiles with respect to staining

# use these tiles to find mitotic events with CNN

import cv2
import py_wsi
import numpy as np

file_dir = '/home/ciaran/PycharmProjects/mitotic_index/'
db_location = '/home/ciaran/PycharmProjects/mitotic_index/'
xml_dir = file_dir
db_name = "patch_db"


# initialise turtle svs manager object
turtle = py_wsi.Turtle(file_dir, db_location, db_name, xml_dir)

# tile svs file(s) found by turtle in file_dir, where:
#       size of each tile in pixels == patch_size
#       "resolution" of image == level (higher the more detailed)
#       overlap between tiles == overlap
#       number of tiles loaded into memory at any one time == rows_per_txn
patches = turtle.sample_and_store_patches(patch_size=128, level=11, overlap=12, rows_per_txn=10)
# todo: CHANGE THE ABOVE SETTINGS TO BE APPROPRIATE
#  - should i have the largest patch size and lowest level/resolution i can for the sake of speed?
#  - should i have patch size == 10 High Power Fields ~= 2mm^2 from the off so that it is inline with B&R grading?
#  - how does overlap effect things here?
#  how many images can I get away with having loaded at once? is there even a benefit to maximising this?

#patches = turtle.get_patches_from_file("c4b95da36e32993289cb.svs")
exit()
patch_17 = turtle.retrieve_sample_patch("c4b95da36e32993289cb.svs", 128, 17, overlap=12)

print(type(patch_17))

patch_17 = cv2.cvtColor(np.array(patch_17), cv2.COLOR_RGB2GRAY)

print(type(patch_17))
thresh1 = cv2.adaptiveThreshold(patch_17, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
print(type(thresh1))
cv2.imshow("image1", thresh1)
cv2.imshow("image", patch_17)
cv2.waitKey(0)

exit()
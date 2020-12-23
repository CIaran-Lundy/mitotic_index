This repo will contain a tensorflow workflow that reads data from a set of canine cancer microscopy images to train a binary classifier.
This binary classifier will categorise cells in the image into "mitotic" and "not mitotic"/ "interphase" classes.
the absolute numbers of mitotic and non-mitotic cells will be used to calculate mitotic index with the formula below:

mitotic index = (number of mitotic cells) / (total number of cells) * 100

to achieve this, computer vision techniques will need to be implemented in at least 2 stages:

- classify image features into "cell" vs "not-cell" **OR** "nucleus" vs "not nucleus" to allow counting of total number of nucleated cells in sample
- classify cells into mitotic vs not-mitotic

data will be accessed from:
https://springernature.figshare.com/articles/dataset/c4b95da36e32993289cb_svs_-_Whole_slide_image_of_canine_mammary_carcinoma/12186612


having read: 
*A unified framework for tumor proliferation score prediction in breast histopathology* 
*https://ysbecca.github.io/programming/2018/05/22/py-wsi.html*

i will do the following:

 - identify tissue blobs on the slide

 - identify Regions of interest where there are likely to be mitotic cells

 - identify mitotic cells with a CNN
`
by using the following steps:

 - load svs

 - tile svs with py_wsi

 - threshold tiles with cv2

 - keep tiles with >90% tile coverage by tissue

 - for all kept tiles, go UP a few levels (i.e. go from level 10 to 17) to increase tile resolution

 -  tile the high res kept tiles to reduce size of object being handled

 -  use cell profiler to estimate cell density, and select top ~30 most dense tiles on whole slide

 - normalize tiles with respect to staining

 - use these tiles to find mitotic events with CNN
 
 
 PARALLELISATION:
 
  - parallel filesystem
  - parallel image segmenting
  - parallel training
    - batching
  - parallel testing

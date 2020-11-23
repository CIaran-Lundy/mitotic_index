This repo will contain a tensorflow workflow that reads data from a set of canine cancer microscopy images to train a binary classifier.
This binary classifier will categorise cells in the image into "mitotic" and "not mitotic"/ "interphase" classes.
the absolute numbers of mitotic and non-mitotic cells will be used to calculate mitotic index with the formula below:

mitotic index = (number of mitotic cells) / (total number of cells) * 100

to achieve this, computer vision techniques will need to be implemented in at least 2 stages:

- classify image features into "cell" vs "not-cell" **OR** "nucleus" vs "not nucleus" to allow counting of total number of nucleated cells in sample
- classify cells into mitotic vs not-mitotic

data will be accessed from:
https://springernature.figshare.com/articles/dataset/c4b95da36e32993289cb_svs_-_Whole_slide_image_of_canine_mammary_carcinoma/12186612


having read: *A unified framework for tumor proliferation score prediction in breast histopathology* 

i will do the following:

- identify tissue blobs using Otsu’s method, and binary dilation

- extract patches corresponding to 10 consecutive high power fields (HPFs) region, an area of approximately 2mm^2
	- microscope used affects size of area to sample: microscope is: https://med.virginia.edu/biomolecular-analysis-facility/services/shared-instrumentation/aperio-scanscope-slide-scanner/
- 

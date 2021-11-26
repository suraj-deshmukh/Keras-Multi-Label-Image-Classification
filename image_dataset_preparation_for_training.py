
# USAGE
# python train.py --dataset dataset_path

from imutils import paths
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")

IMAGE_DIMS = (150, 150, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	try:
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
        #uncomment for gray scale
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(imagePath.split(os.path.sep)[-1],image)
	except:
		os.remove(imagePath)

print("[INFO] done with image resizing...")

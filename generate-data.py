import numpy as np
import h5py
import cv2
import scipy.io as sio
import os
from skimage import io
from skimage.transform import resize

target_path = "miml-image-data/miml data.mat"   # target labels
image_path = "miml-image-data/original"     # images

y = sio.loadmat(target_path)
y = y['targets']
y = y.transpose()
y = np.array([[elem if elem == 1 else 0 for elem in row]for row in y])
x = []

for i in range(1,2001):
    print "reading image:"+str(i) + ".jpg"
    img = image_path + "/" + str(i) + ".jpg"
    img = cv2.imread(img)
    img = cv2.resize(img,(100,100))
    img = img.transpose((2,0,1))
    # img = io.imread(img)
    # img = resize(img,(100,100))
    # img = img.transpose()
    x.append(img)
    
x = np.array(x)
f = h5py.File("dataset.h5")
f['x'] = x
f['y'] = y
f.close()

# Multi label Image Classification

The objective of this study is to develop a deep learning model that will identify the natural scenes from images. This type of problem comes under multi label image classification where an instance can be classified into multiple classes among the predefined classes. https://suraj-deshmukh.github.io/Multi-Label-Image-Classification/

# Dataset

The complete description of dataset is given on http://lamda.nju.edu.cn/data_MIMLimage.ashx. The dataset contains 2000 natural scenes images.

# Keras Model Architecture

![all tag](https://github.com/suraj-deshmukh/Multi-Label-Image-Classification/blob/master/model.png)

# Preprocessing

Resized all images to 100 by 100 pixels and created two sets i.e train set and test set. Train set contains 1600 images and test set contains 200 images.

# Training
As this is multi label image classification, the loss function was binary crossentropy and activation function used was sigmoid at the output layer. Keras doesn't have provision to provide multi label output so after training there is one probabilistic threshold method which find out the best threshold value for each label seperately, the performance of threshold values are evaluated using Matthews Correlation Coefficient and then uses this thresholds to convert those probabilites into one's and zero's.

# Result

Below table shows the result on test set

Accuracy | Value
--------- | ---------
Hamming loss | 0.1395
Exact Match | 0.54

# Preprocessed Dataset and Weight file download link

Dataset: https://drive.google.com/open?id=0BxGfPTc19Ac2a1pDd1dxYlhIVlk

Weight file: https://drive.google.com/open?id=0BxGfPTc19Ac2X1RqNnEtRnNBNUE

# Ipython notebook

Jupyter/iPython Notebook has been provided to know about the model and its working. https://github.com/suraj-deshmukh/Multi-Label-Image-Classification/blob/master/miml.ipynb 

# Visualization

Command:

python visualization.py

link: localhost:5000

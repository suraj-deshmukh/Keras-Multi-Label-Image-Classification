## Download files

Dataset: https://drive.google.com/open?id=0BxGfPTc19Ac2a1pDd1dxYlhIVlk

Weight file: https://drive.google.com/open?id=0BxGfPTc19Ac2X1RqNnEtRnNBNUE

weights.hdf5
dataset.h5

### download miml-image-data.zip

#### download from kaggle
miml-image-data.zip
from
https://www.kaggle.com/twopothead/miml-image-data/version/3

#### unzip
$ unzip miml-image-data.zip
miml-image-data/processed
miml-image-data/original
miml-image-data

#### locate "miml data.mat"
locate "miml data.mat" file as
miml-image-data/miml data.mat


## scripts

### ganerate data

$ python python generate-data.py 

This will create dataset.h5.
generate-data.py assumes "miml-image-data/miml data.mat"

getdata.py is module.


### jupyter notebook

$ jupyter notebook &

Then you can run `miml.ipynb` from the browser.


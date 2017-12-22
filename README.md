# dilated_seg
Vision project on experimenting with semantic segmentation using Dilated Conv Nets

This repository is built upon : https://github.com/nicolov/segmentation_keras 


@TODO : To be modified as a Fork of `segmentation_keras` repo

## Steps

Pre-trained model:
`curl -L https://github.com/nicolov/segmentation_keras/releases/download/model/nicolov_segmentation_model.tar.gz | tar xvf -`

Install dependencies:
```
pip install -r requirements.txt
# For GPU support 
pip install tensorflow-gpu==1.3.0
```
Prediction:
`python predict.py --weights_path conversion/converted/dilation8_pascal_voc.npy`

Download Stanford Background Dataset from:
`http://dags.stanford.edu/data/iccv09Data.tar.gz`

We have to pre-process the Stanford dataset...


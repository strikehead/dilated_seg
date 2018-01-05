# dilated_seg
Computer Vision final project for [COMPSCI 670](http://people.cs.umass.edu/~smaji/teaching/670/index.html) done by [Shubham Mukherjee](https://www.linkedin.com/in/shubhammukherjee/) and [Deep Chakraborty](https://www.linkedin.com/in/deepc94/). 

In this project, we experiment with semantic segmentation using Dilated Conv Nets [[1]](https://arxiv.org/abs/1511.07122) on the Stanford Background Dataset [[2]](http://dags.stanford.edu/projects/scenedataset.html). Key contributions in this project:
* Code for end-to-end Training of Front end + Context module
* Training using Batch Norm [[3]](https://arxiv.org/abs/1502.03167) in the Context module to enable random initialization and eliminate the need for careful initialization techniques
* Fine-Tuning the PASCAL VOC pre-trained models on the Stanford Background Dataset to obtain near state-of-the-art accuracy.

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


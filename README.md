# Keras training performance check
This is the performance check for Keras training speed.
We will be using Cifar10 as dataset and pretrain model, which is Inception V3.

## Installation
### Requirements
* Python3.5 and up
* Tensorflow
* Scikit-image
* Scikit-learn
* h5py

`$ pip install -r requirements.txt`

## Training
Clone, and `cd` into the repo directory.

The keras training performance can be tested in the following method.

### Default training.
This will use the default setting to train the model.

`$ python train.py`

### Batch size & Epochs
This changes how many images per batch and the epoch size.

`$ python train.py --batch_size 16 --epochs 10`

### Training and validation images number size.
This changes how many images will be used for training and validation.

`$ python train.py --training_num 3000 --validation_num 2000`

### Dimension of the images.
This helps resize the images feeded into the model for training. The dimension should be (139x139) or higher.

`$ python train.py --image_size 224`

or 

`$ python train.py --image_size 220 224`

### Save model weight.
This helps user to save the model's training weight.
The user needs to provide the path and the file name.

`$ python train.py --save_model True --model_name Test --path /xxx/ooo/`



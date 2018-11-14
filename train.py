#!/usr/bin/env python
# coding: utf-8

from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.inception_v3 import *
from tensorflow.python.keras.optimizers import SGD
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from NCHC_CallBack_Function import NCHC_CallBack
from argparse import ArgumentParser as argu

import numpy as np
import skimage.io as io
import tensorflow.python.keras.callbacks
import time, os, sys, argparse
import tensorflow as tf
import tensorflow.python.keras
import tensorflow.python.keras.models
import json
import requests

def helpResize(x_train,x_validation,image_size):
    '''
    HelpResize(x_train = np.array, x_validation = np.array, image_size = int or tuple)

    This is a function to resize the image dimension.
    '''
    X_train = []
    X_validation = []

    if type(image_size) is int:
        image = (image_size,image_size)
    else:
        image = tuple(image_size)

    for num,img in enumerate(x_train):
        X_train.append(resize(img,image,mode='constant',anti_aliasing=True))
    for num,img in enumerate(x_validation):
        X_validation.append(resize(img,image,mode='constant',anti_aliasing=True))

    X_train = np.array(X_train)
    X_validation = np.array(X_validation)
    return X_train,X_validation


def saving_H5_Weights(model,weight_name,path):
    '''
    Saving_H5_Weights(model, weight_name = Str, path = Str)

    This is to save the trained model.
    '''

    if not os.path.isdir(path):
        os.makedirs(path)
    model_path = os.path.join(path, weight_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

def start(argv):
    '''
    This the main training function.
    '''

    parser = argu(description="This is for training image classification performance check, using Cifar 10 dataset and Inception V3 model to train.", add_help=False)

    parser.add_argument('-h','--help',
                        action ='help',
                        help='Show the arguments explanation and useage.')

    parser.add_argument('-t','--training_num',
                        help = 'The total number of images will be used for training.',
                        dest = "training_num",
                        default = 2000,
                        type = int)

    parser.add_argument('-v','--validation_num',
                        help = 'The total number of images will be used for validation.',
                        dest = "validation_num",
                        default = 1000,
                        type = int)

    parser.add_argument('-i','--image_size',
                        help = "Change the dimension of the input image",
                        dest = "image_size",
                        default = 139,
                        type = int or tuple,
                        nargs = 2)

    parser.add_argument('-b','--batch_size',
                        help = "Set how many images per batch during training.",
                        dest = "batch_size",
                        default = 32,
                        type = int)

    parser.add_argument('-e','--epochs',
                        help = 'Set how many times loop through the model before end of the training.',
                        dest = "epochs",
                        default = 1,
                        type = int)

    parser.add_argument('-roc','--save_roc_curve',
                        help = 'If the user needs to save ROC curve of current training.',
                        dest = "need_roc",
                        default = False,
                        type = bool)

    parser.add_argument('-model','--save_model',
                        help = 'If the user needs to save the weight of the model.',
                        dest = "need_weight",
                        default = False,
                        type = bool)

    parser.add_argument('-modeln','--model_name',
                        help = 'The name of the trained weight.',
                        dest = 'weight_name',
                        default = '{}'.format('_'.join(time.ctime().split(' ')).replace(':','_')),
                        type = str)

    parser.add_argument('-p','--path',
                        help = 'Enter the path for model weight.',
                        dest = "path",
                        default = os.path.dirname(os.path.realpath(__file__)),
                        type = str)
    parser.add_argument('-s','--is_serve',
                        help = 'This it load the image and predicted through Flask.',
                        dest = 'isServe',
                        default = False,
                        type = bool)

    args = parser.parse_args()

    # Assign Arguments
    training_num = args.training_num
    validation_num = args.validation_num
    image_size = args.image_size
    batch_size = args.batch_size
    epochs = args.epochs
    need_roc = args.need_roc
    need_weight = args.need_weight
    path = args.path
    weight_name = args.weight_name

    # Check if image dimension is correct.
    if type(image_size) is list:
        val1 = image_size[0]
        val2 = image_size[1]
        if val1 < 139 or val2 < 139:
            print("The size is not ok....")
            sys.exit(2)
        elif type(image_size) is int:
            if image_size <139:
                print("The size is not ok...")
                sys.exit(2)

    # Show training conditions
    print("Image size is {}".format(image_size))
    print("The batch_size is {}".format(batch_size))
    print("The epochs is {}".format(epochs))

    # Preprocessing Training and Validation Data
    (x_train, y_train), (x_validation, y_validation) = cifar10.load_data()

    # Load part of train and test data.
    x_train = x_train[:training_num]
    x_validation = x_validation[:validation_num]
    Y_train = y_train[:training_num]
    Y_validation = y_validation[:validation_num]

    print("Total Train & Validation Num as shown below")
    print("Num of training images : {}".format(x_train.shape[0]))
    print("Num of validation images : {}".format(x_validation.shape[0]))

    X_train,X_validation = helpResize(x_train,x_validation,image_size)

    # Check if both of the list has the correct length.
    Y_new_train = np.array([np.zeros(10) for x in range(len(Y_train))],dtype='float32')
    for i,x in enumerate(Y_train):
        Y_new_train[i][x] = 1
    Y_new_val = np.array([np.zeros(10) for x in range(len(Y_validation))],dtype='float32')
    for i,x in enumerate(Y_validation):
        Y_new_val[i][x] = 1

    # This could also be the output of a different Keras model or layer
    if type(image_size) is list:
        input_shape = tuple(image_size) + (3,)
    else:
        input_shape = (image_size,image_size,3)
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=input_shape)

    # Get the output of the Inception V3 pretrain model.
    x = base_model.output

    # Works same as Flatten(), but Flatten has larger dense layers, it might cause worse overfitting.
    # However, if the user has a larger dataset then the user can use Flatten() instead of GlobalAveragePooling2D or GlobalMaxPooling2D
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Use SGD as an optimizer
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=0,  # Randomly rotate images in the range (degrees, 0 to 180)
        # Randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # Randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        zoom_range=0.,  # set range for random zoom
        # Set the mode for filling points outside the input boundaries
        horizontal_flip=True,  # randomly flip images
    )
    datagen.fit(X_train)
    histories = NCHC_CallBack()
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train,
                                 Y_new_train,
                                 batch_size=batch_size),
                    callbacks = [ histories ], #added here
                    epochs=epochs,
                    validation_data=(X_validation, Y_new_val),
                    workers=4)

    pyversion = '{}.{}'.format(sys.version_info[0],sys.version_info[1])
    tfversion = '{}'.format(tf.__version__)
    avg_batch = '{:0.3f}'.format(np.average(histories.batch_time))
    img_sec = 'Images per second : {:6.3f}'.format(np.average(histories.img_per_sec))
    hist_lose = 'Losses : ' + str(histories.losses)
    if len(histories.aucs) <= 0:
        hist_aucs = "Due to validation data inbalance, Can't calculate ROC curve"
    else:
        hist_aucs = " ".join(histories.aucs)

    # Put training results to dictionary.
    t_cuda_version = " ".join(open("/usr/local/cuda/version.txt", 'r').readlines() )
    results = {"text":"Python Version : {}\nTensorflow Version : {}\nAverage batch per second : {:0.3f}\n*Images per second : {:0.3f}*\nLosses :\n{}\nAUC scores: \n{}\n".format(sys.version[:6],tf.__version__,np.average(histories.batch_time),np.average(histories.img_per_sec)," ".join(histories.losses),hist_aucs)}
    results['text'] += "\n:+1:{} | nv-tf: {} | {}-{}".format(
                                      os.environ['HOSTNAME'],
                                      os.environ['NVIDIA_TENSORFLOW_VERSION'],
                                      t_cuda_version, os.environ['NVIDIA_BUILD_ID'])
    # Transfer dictionary to string
    results_json = json.dumps(results)
    # Sending data to slack
    r = requests.post('https://hooks.slack.com/services/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',
                      data = results_json,
                      headers = {'Content-type':'application/json'})

    # Setting the path for the training result text file.
    file_path = os.path.dirname(os.path.realpath("train.py"))+'/test_results/'

    if not os.path.exists(file_path):
        os.mkdir(file_path)

    fname = file_path + 'test_results.txt'
    result_txt = "Date : {}\nPython Version : {}\nTensorflow Version : {}\nAverage batch per second : {:0.3f}\nImages per second : {:0.3f}\nLosses :\n{}\nAUC scores: \n{}\n".format(time.ctime(),sys.version[:6],tf.__version__,np.average(histories.batch_time),np.average(histories.img_per_sec)," ".join(histories.losses),hist_aucs)


    if os.path.exists(fname):
        with open(fname,'a') as f:
            f.write(result_txt)
            f.write('====================================\n')
    else:
        with open(fname,'w') as f:
            f.write(result_txt)
            f.write('====================================\n')

    if need_weight:
        saving_H5_Weights(model,weight_name,path)

if __name__ == "__main__":
    start(sys.argv[1:])

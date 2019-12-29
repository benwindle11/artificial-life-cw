# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:46:05 2019

@author: benny
"""


import numpy as np

import keras
from keras import Input, Model
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, \
    Concatenate
from keras.layers import BatchNormalization, Activation, LeakyReLU, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential



input_layer = Input(shape = (80,80))

branch1 = Conv2D(filters = 16,
                 kernel_size = (10, 23),
                 strides = (1,  1),
                 padding='same',
                 activation=LeakyReLU())(input_layer)

branch1 = MaxPooling2D()(branch1)

branch1 = Conv2D(filters = 32,
                 kernel_size = (5, 11),
                 strides = (1,  1),
                 padding='same',
                 activation=LeakyReLU())(branch1)

branch1 = MaxPooling2D()(branch1)

branch1 = Conv2D(filters = 64,
                 kernel_size = (3, 5),
                 strides = (1,  1),
                 padding='same',
                 activation=LeakyReLU())(branch1)

branch1 = MaxPooling2D()(branch1)

branch1 = Conv2D(filters = 128,
                 kernel_size = (2, 4),
                 strides = (1,  1),
                 padding='same',
                 activation=LeakyReLU())(branch1)

branch1 = MaxPooling2D(pool_size = (1,5))(branch1)


##BRANCH2
branch2 = Conv2D(filters = 16,
                 kernel_size = (21, 10),
                 strides = (1,  1),
                 padding='same',
                 activation=LeakyReLU())(input_layer)

branch2 = MaxPooling2D()(branch2)

branch2 = Conv2D(filters = 32,
                 kernel_size = (10, 5),
                 strides = (1,  1),
                 padding='same',
                 activation=LeakyReLU())(branch2)

branch2 = MaxPooling2D()(branch2)

branch2 = Conv2D(filters = 64,
                 kernel_size = (5, 3),
                 strides = (1,  1),
                 padding='same',
                 activation=LeakyReLU())(branch2)

branch2 = MaxPooling2D()(branch2)

branch2 = Conv2D(filters = 128,
                 kernel_size = (4, 2),
                 strides = (1,  1),
                 padding='same',
                 activation=LeakyReLU())(branch2)

branch2 = MaxPooling2D(pool_size = (5,1))(branch2)

#what is axis?
layer = Concatenate(axis=1)([branch1, branch2])

model = Model(input_layer, layer)

layer = Dropout(0.25)(layer)

layer = Dense(units=200, activation=LeakyReLU())(layer)

#need to add softmax

model.summary()

model.compile('adam', 'categorical_crossentropy', ['accuracy'])



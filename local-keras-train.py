import argparse, os
import numpy as np
import time
import boto3

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input, Concatenate, Reshape
from keras.utils import multi_gpu_model
from keras.datasets import fashion_mnist
from keras.activations import relu
import pickle


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default='model')
    parser.add_argument('--training', type=str, default='data')
    parser.add_argument('--testing', type=str, default='data')
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    testing_dir = args.testing
    
    # Download the data set
    os.makedirs("./data", exist_ok = True)
    with open("pickled-data/train_spect_music_data.pkl", "rb") as f:
        train_music_data = pickle.load(f)
        x_train = train_music_data["audio"]
        y_train = train_music_data["labels"]

    with open("pickled-data/test_spect_music_data.pkl", "rb") as f:
        test_music_data = pickle.load(f)
        x_val = test_music_data["audio"]
        y_val = test_music_data["labels"]

    np.savez('./data/training', image=x_train, label=y_train)
    np.savez('./data/testing', image=x_val, label=y_val)
    
    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val  = np.load(os.path.join(testing_dir, 'testing.npz'))['image']
    y_val  = np.load(os.path.join(testing_dir, 'testing.npz'))['label']

    # input image dimensions
    img_rows, img_cols = 80, 80

    # channels last
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    batch_norm_axis=-1

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')
    
    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val   = keras.utils.to_categorical(y_val, num_classes)

    input_layer = Input(shape = (80,80, 1))

    branch1 = Conv2D(filters = 16,
                    kernel_size = (10, 23),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(input_layer)

    branch1 = MaxPooling2D()(branch1)

    branch1 = Conv2D(filters = 32,
                    kernel_size = (5, 11),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(branch1)

    branch1 = MaxPooling2D()(branch1)

    branch1 = Conv2D(filters = 64,
                    kernel_size = (3, 5),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(branch1)

    branch1 = MaxPooling2D()(branch1)

    branch1 = Conv2D(filters = 128,
                    kernel_size = (2, 4),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(branch1)

    branch1 = MaxPooling2D(pool_size = (1,5))(branch1)

    branch1 = Flatten()(branch1)


    ##BRANCH2
    branch2 = Conv2D(filters = 16,
                    kernel_size = (21, 10),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(input_layer)

    branch2 = MaxPooling2D()(branch2)

    branch2 = Conv2D(filters = 32,
                    kernel_size = (10, 5),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(branch2)

    branch2 = MaxPooling2D()(branch2)

    branch2 = Conv2D(filters = 64,
                    kernel_size = (5, 3),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(branch2)

    branch2 = MaxPooling2D()(branch2)

    branch2 = Conv2D(filters = 128,
                    kernel_size = (4, 2),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(branch2)

    branch2 = MaxPooling2D(pool_size = (5,1))(branch2)

    branch2 = Flatten()(branch2)

    #what is axis?
    layer = Concatenate(axis=1)([branch1, branch2])

    layer = Dropout(0.25)(layer)

    layer = Dense(units=200, activation=relu)(layer)

    layer = Dense(num_classes, activation='softmax')(layer)

    model = Model(input_layer, layer)

    

    #need to add softmax

    model.summary()
    
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(
                      learning_rate=0.00005,
                      beta_1=0.9,
                      beta_2=0.999),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, 
              batch_size=batch_size,
              validation_data=(x_val, y_val), 
              epochs=epochs,
              verbose=1)
    
    score = model.evaluate(x_val, y_val, verbose=0)
    print('testing loss    :', score[0])
    print('testing accuracy:', score[1])
    
    # save Keras model for Tensorflow Serving
    curr_time = time.gmtime()
    curr_timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", curr_time)
    model.save("model/model"+curr_timestamp+".h5")
    

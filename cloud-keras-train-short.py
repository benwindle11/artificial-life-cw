import argparse, os
import numpy as np
import time
import boto3
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, AveragePooling2D, Input, Concatenate, Reshape
from keras.utils import multi_gpu_model
from keras.datasets import fashion_mnist
from keras.activations import relu
import pickle


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

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

    branch1 = AveragePooling2D(pool_size=(80,4))(branch1)

    branch1 = Flatten()(branch1)

    ##BRANCH2
    branch2 = Conv2D(filters = 16,
                    kernel_size = (21, 10),
                    strides = (1,  1),
                    padding='same',
                    activation=relu)(input_layer)

    branch2 = AveragePooling2D(pool_size=(4,80))(branch2)

    branch2 = Flatten()(branch2)

    #what is axis?
    layer = Concatenate(axis=1)([branch1, branch2])

    layer = Dropout(0.1)(layer)

    layer = Dense(units=200, activation=None)(layer)

    layer = Dense(num_classes, activation='softmax')(layer)

    model = Model(input_layer, layer)

    

    #need to add softmax
    model.summary()

    # model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    # model = Sequential()
    
    # # 1st convolution block
    # model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=input_shape))
    # model.add(BatchNormalization(axis=batch_norm_axis))
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2,2), strides=2))
    
    # # 2nd convolution block
    # model.add(Conv2D(128, kernel_size=(3,3), padding='valid'))
    # model.add(BatchNormalization(axis=batch_norm_axis))
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2,2), strides=2))

    # # Fully connected block
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.3))

    # # Output layer
    # model.add(Dense(num_classes, activation='softmax'))
    
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(
                      lr=0.00005,
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
    filename = "shortmodel"+curr_timestamp+".h5"
    model.save(filename)

    s3 = boto3.client('s3')
    with open(filename, 'rb') as f:
        s3.upload_fileobj(f, "windle-cw", filename)
    




    

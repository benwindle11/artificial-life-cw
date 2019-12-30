import os
import time
import numpy as np
import pickle

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser


# WORKING DIRECTORY #
#####################

# Define path where model and output files will be stored.
# The user is responsible for cleaning up this temporary directory.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'snn'))

# os.makedirs(path_wd)
# print(path_wd)
# GET DATASET #
###############

# (x_train, _), (x_test, y_test) = 
def store_data():
    with open("pickled-data/train_spect_music_data.pkl", "rb") as f:
        train_data = pickle.load(f)

    x_train = train_data["audio"]
    x_train_np = np.array(x_train)
    x_train_np = x_train_np.reshape(x_train_np.shape[0], 80, 80, 1)

    with open("pickled-data/test_spect_music_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    x_test = test_data["audio"]
    y_test = test_data["labels"]

    y_test_hot = []
    for y in y_test:
        label = [0 for i in range(0,10)]
        label[y] = 1
        y_test_hot.append(label)

    x_test_np = np.array(x_test)
    x_test_np = x_test_np.reshape(x_test_np.shape[0], 80, 80, 1)
    print(x_test_np.shape)
    # Save dataset so SNN toolbox can find it.
    np.savez_compressed(os.path.join(path_wd, 'x_test'), x_test_np)
    np.savez_compressed(os.path.join(path_wd, 'y_test'), y_test_hot)
    # SNN toolbox will not do any training, but we save a subset of the training
    # set so the toolbox can use it when normalizing the network parameters.
    np.savez_compressed(os.path.join(path_wd, 'x_norm'), x_train_np[::10])

store_data()

# CREATE ANN #
##############

# This section creates a CNN using Keras, and trains it with backpropagation.
# There are no spikes involved at this point. The model is far more complicated
# than necessary for MNIST, but serves the purpose to illustrate the kind of
# layers and topologies supported (in particular branches).

# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()
model_name = "model2019-12-30T00-01-17"

config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'normalize': False               # Normalize weights for full dynamic range.
}

config['simulation'] = {
    'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
    'duration': 50,                 # Number of time steps to run each sample.
    'num_to_test': 100,             # How many test samples to run.
    'batch_size': 50,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}

config['output'] = {
    'plot_vars': {                  # Various plots (slows down simulation).
        'spiketrains',              # Leave section empty to turn off plots.
        'spikerates',
        'activations',
        'correlation',
        'v_mem',
        'error_t'}
}

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

main(config_filepath)
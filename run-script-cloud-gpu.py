import sagemaker
import os
import keras
import pickle
import numpy as np
import argparse
from sagemaker.tensorflow import TensorFlow

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs

    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

    with open('data/train_spect_music_data.pkl', 'rb') as f:
        train_music_data = pickle.load(f)
        x_train = train_music_data["audio"]
        y_train = train_music_data["labels"]
    
    with open('data/test_spect_music_data.pkl', 'rb') as f:
        test_music_data = pickle.load(f)
        x_val = test_music_data["audio"]
        y_val = test_music_data["labels"]

    np.savez('./data/training', image=x_train, label=y_train)
    np.savez('./data/validation', image=x_val, label=y_val)

    prefix = 'gtzan-music'

    training_input_path   = sess.upload_data('data/training.npz', key_prefix=prefix+'/training')
    validation_input_path = sess.upload_data('data/validation.npz', key_prefix=prefix+'/validation')

    print(training_input_path)
    print(validation_input_path)

    tf_estimator = TensorFlow(entry_point='cloud-keras-train-long.py', 
                          role=role,
                          train_instance_count=1, 
                          train_instance_type='ml.p3.2xlarge',
                          framework_version='1.12', 
                          py_version='py3',
                          script_mode=True,
                          hyperparameters={
                              'epochs': epochs,
                              'batch-size': 256}
                         )

    print("estimator created, fitting...")
    
    tf_estimator.fit({'training': training_input_path, 'validation': validation_input_path})

import boto3
import os

os.makedirs("./data", exist_ok = True)
s3 = boto3.client('s3')
s3.download_file('windle-cw', 'train_spect_music_data.pkl', 'data/train_spect_music_data.pkl')
s3.download_file('windle-cw', 'test_spect_music_data.pkl', 'data/test_spect_music_data.pkl')

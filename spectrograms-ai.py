# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
import time
import librosa
import matplotlib.pyplot as plt
import librosa.display as disp
import numpy as np
import math


# %%
genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

class MusicData:
    def __init__(self):
        self.audio = []
        self.labels = []
track_length = 30
number_of_segments = 32
segment_length = math.floor(22050*track_length/number_of_segments)


# %%
def melspectrogram(audio):
    spec = librosa.stft(audio, n_fft=512, window='hann', hop_length=256, win_length=512, pad_mode='constant')
    mel_basis = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80)
    mel_spec = np.dot(mel_basis, np.abs(spec))
    return np.log(mel_spec + 1e-6)


# %%
def pad_with_zeros(list_to_pad):
    curr_array = np.array(list_to_pad)
    height, width = curr_array.shape
    if(height > 80 or width > 80):
        print("not paddable")
        return
    height_pad = 80 - height
    width_pad = 80 - width
    padded_array = np.pad(curr_array, ((0, height_pad), (0, width_pad)))
    return padded_array.tolist()


# %%
train_music_data = { 
    "audio": [],
    "labels": []
}
test_music_data = {
    "audio": [],
    "labels": []
}
for k, genre in enumerate(genre_list):
    for i in range(100):
        i_str = str(i)
        base = "00000"
        keep_len = len(base) - len(i_str)
        filenumber = base[:keep_len] + i_str
        filename = "data/genres/"+genre+"/"+genre+"."+filenumber+".wav"
        with open(filename, "rb") as f:
            data, rate = librosa.load(f)
            for j in range(1, number_of_segments+1):
                upper_bound = segment_length * j
                lower_bound = segment_length * (j-1)
              
                # music_data.audio.append(np.array(data[lower_bound:upper_bound]))
                data_slice = data[lower_bound:upper_bound]
                mel_data_slice = melspectrogram(data_slice)
                mel_data_slice_truncate = mel_data_slice[:80, :80]
                height, width = np.shape(mel_data_slice_truncate)
                if(height != width or height != 80):
                    mel_data_slice_truncate = pad_with_zeros(mel_data_slice_truncate)
                    print(np.shape(mel_data_slice_truncate))
                
                if(j >= 25):
                    test_music_data["audio"].append(mel_data_slice_truncate)
                    test_music_data["labels"].append(k)
                else:
                    train_music_data["audio"].append(mel_data_slice_truncate)
                    train_music_data["labels"].append(k)


# %%
print(np.shape(train_music_data["audio"][0]))
dataset_size = len(train_music_data["audio"])
print(dataset_size)

print(np.shape(train_music_data["labels"][0]))
dataset_size = len(train_music_data["labels"])
print(dataset_size)

print(np.shape(test_music_data["audio"][0]))
print(len(test_music_data["audio"]))


print(np.shape(test_music_data["labels"][0]))
print(len(test_music_data["labels"]))


# %%
import random 
for i in range(0, 5):
    random_track = random.randint(0, dataset_size)
    plt.figure()
    disp.specshow(train_music_data["audio"][random_track])
    print(random_track)
    print(train_music_data["labels"][random_track])
    print(train_music_data["audio"][random_track].shape)


# %%
with open("pickled-data/train_spect_music_data.pkl", "wb") as f:
    pickle.dump(train_music_data, f)

with open("pickled-data/test_spect_music_data.pkl", "wb") as f:
    pickle.dump(test_music_data, f)


# %%
print(len(test_music_data["audio"]))
print(len(train_music_data["audio"]))


# %%



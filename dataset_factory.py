"""
Data processing functionality.
Modified from https://github.com/mulimani/Sound-Event-Detection

"""

import os

import numpy as np
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import resample
import torch
from torch.utils.data import Dataset

###############################################################################
# upper level data class
###############################################################################
class BatchData(Dataset):
    def __init__(self, mels, labels):
        self.mels = mels
        self.labels = labels

    def __getitem__(self, index):
        mels = self.mels[index]
        label = self.labels[index]
        return mels, label

    def __len__(self):
        return len(self.mels)


###############################################################################
# function for loading metadata information
###############################################################################
def load_metadata(meta_file, class_labels):

    meta_dict = {}
    for line in open(meta_file, 'r'):

        words = line.strip().split('\t')
        # Removing 'mixture a001 (audio file name)' after the class label
        words = words[:len(words) - 2]
        name = words[0].split('/')[-1]

        if name not in meta_dict:
            meta_dict[name] = []
        meta_dict[name].append([
            float(words[2]), 
            float(words[3]), 
            class_labels[words[-1]]
        ])
    return meta_dict


###############################################################################
# extracting the data
###############################################################################
class MelData(Dataset):
    def __init__(
            self, 
            root, 
            class_labels, 
            sample_rate=32000, 
            n_mels=64, 
            n_fft=1024, 
            hop_length=320
        ):

        self.root = root
        self.class_labels = class_labels

        # Spectrogram parameters (the same as librosa.stft)
        self.sample_rate = sample_rate

        self.n_fft = n_fft
        self.hop_length = hop_length
        win_length = n_fft
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        is_mono = True

        # Mel parameters (the same as librosa.feature.melspectrogram)
        self.n_mels = n_mels
        self.fmin = 20
        self.fmax = 14000

        # Power to db parameters (the same as default settings of librosa.power_to_db
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.melspec = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin, 
            f_max=self.fmax
        )

        self.mel_tensor, self.label_tensor = None, None
        self.mel_list, self.label_list = [], []
        self.extract_data()

    def extract_data(self):

        meta = os.path.join(self.root + 'meta.txt')
        meta_dict = load_metadata(meta, self.class_labels)

        files_path = os.path.join(self.root + 'audio/' + 'street')
        for audio_file in os.listdir(files_path):
            audio_path = os.path.join(files_path + audio_file)

            y, sr = torchaudio.load(audio_path)

            # make it mono
            y = torch.mean(y, dim=0)
            if sr != self.sample_rate:
                y = resample(y, orig_freq=sr, new_freq=self.sample_rate)
            
            # extract mel spectrogram
            mels = self.melspec(y)
            mels = torch.transpose(mels, 0, 1)
            mels = torch.log(mels + torch.finfo(torch.float32).eps)

            # split data into frames
            label = torch.zeros((mels.shape[0], len(self.class_labels)))
            tmp_data = np.array(meta_dict[audio_file])
            frame_start = np.floor(tmp_data[:, 0] * self.sample_rate / self.hop_length).astype(int)
            frame_end = np.ceil(tmp_data[:, 1] * self.sample_rate / self.hop_length).astype(int)
            se_class = tmp_data[:, 2].astype(int)
            for ind, val in enumerate(se_class):
                label[frame_start[ind]:frame_end[ind], val] = 1

            # append new extracted data
            if self.mel_tensor is None:
                self.mel_tensor, self.label_tensor = mels, label
            else:
                self.mel_tensor = torch.concat((self.mel_tensor, mels), dim=0)
                self.label_tensor = torch.concat((self.label_tensor, label), dim=0)

            self.mel_list.append(mels)
            self.label_list.append(label)

        print("Total audio files =", len(self.mel_list))

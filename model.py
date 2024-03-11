"""
Model Architecture.

Modified from https://github.com/mulimani/Acoustic-Scene-Classification/
These PANN's CNN Architectures are take from https://github.com/qiuqiangkong/audioset_tagging_cnn/tree/master
"""


from torch import mean, max, sigmoid
from torch.nn import Module, Linear, BatchNorm2d, Conv2d
from torch.nn.init import xavier_uniform_
from torch.nn.functional import dropout, relu_, max_pool2d, avg_pool2d, \
                                max_pool1d, avg_pool1d
from torchlibrosa.augmentation import SpecAugmentation

import numpy as np

from utils import interpolate, pad_framewise_output
import config


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3), 
            stride=(1, 1),
            padding=(0, 1), 
            bias=False
        )
        self.conv2 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 3), 
            stride=(1, 1),
            padding=(0, 1), 
            bias=False
        )
        self.bn1 = BatchNorm2d(out_channels)
        self.bn2 = BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = relu_(self.bn1(self.conv1(x)))
        x = relu_(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = avg_pool2d(x, kernel_size=pool_size)
            x2 = max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x


class Cnn14_DecisionLevelMax(Module):
    def __init__(
            self, 
            sample_rate, 
            window_size, 
            hop_size, 
            mel_bins, 
            fmin, 
            fmax, 
            classes_num
        ):
        
        super(Cnn14_DecisionLevelMax, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.interpolate_ratio = 32     # Downsampled ratio

        """
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        """

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = BatchNorm2d(mel_bins) #64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = Linear(2048, 2048, bias=True)
        self.fc_audioset = Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        """
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        """
        #x = input  # (batch_size, 1, time_steps, freq_bins)
        x = input.unsqueeze(1)

        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)
        x = mean(x, dim=3)
        
        x1 = max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = relu_(self.fc1(x))
        x = dropout(x, p=0.5, training=self.training)
        segmentwise_output = sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = max(segmentwise_output, dim=1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {'framewise_output': framewise_output, 
            'clipwise_output': clipwise_output}

        return output_dict
    

class Cnn14(Module):
    def __init__(
            self, 
            sample_rate, 
            window_size, 
            hop_size, 
            mel_bins, 
            fmin,
            fmax, 
            classes_num
        ):

        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.interpolate_ratio = 32     # Downsampled ratio

        self.bn0 = BatchNorm2d(config.nb_mel_bands)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = Linear(2048, 2048, bias=True)
        self.fc_audioset = Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = input  # (batch_size, 1, time_steps, freq_bins)
        x = input.unsqueeze(1)
        print(x.shape) # TODO this was missing expected dim due to data
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = dropout(x, p=0.2, training=self.training)
        x = mean(x, dim=3)

        (x1, _) = max(x, dim=2)
        x2 = mean(x, dim=2)
        x = x1 + x2
        x = dropout(x, p=0.5, training=self.training)
        x = relu_(self.fc1(x))
        embedding = dropout(x, p=0.5, training=self.training)
        clipwise_output = sigmoid(self.fc_audioset(x))

        """
        x = mean(x, dim=3)
        
        x1 = max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = relu_(self.fc1(x))
        x = dropout(x, p=0.5, training=self.training)
        segmentwise_output = sigmoid(self.fc_audioset(x))
        (clipwise_output, _) = max(segmentwise_output, dim=1)
        """

        # Get framewise output
        framewise_output = interpolate(clipwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {
            'clipwise_output': clipwise_output, 
            'embedding': embedding,
            'framewise_output': framewise_output, 
        }
        return output_dict

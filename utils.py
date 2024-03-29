"""
Utility functions associated with the project

Copied from https://github.com/mulimani/Sound-Event-Detection
"""

from torch import split, cat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np


def drop(_X, _Y, _seq_len):
    if _X[-1].size(0) != _seq_len:
        _X, _Y = _X[:-1], _Y[:-1]
    return _X, _Y


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len):
    """ split into sequences

    Args:
        _X (_type_): training features
        _Y (_type_): training labels
        _X_test (_type_): test features
        _Y_test (_type_): test labels
        _seq_len (_type_): desired sequence length

    Returns:
        _type_: data split into sequences
    """
    
    _X = split(_X, _seq_len, dim=0)
    _Y = split(_Y, _seq_len, dim=0)

    _X_test = split(_X_test, _seq_len, dim=0)
    _Y_test = split(_Y_test, _seq_len, dim=0)

    _X, _Y = drop(_X, _Y, _seq_len)
    _X_test, _Y_test = drop(_X_test, _Y_test, _seq_len)

    return _X, _Y, _X_test, _Y_test

## CNN14 DecisionLevelMax

def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

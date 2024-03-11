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
def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out

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

## Plotting results
# implemented to match https://github.com/msilaev/Sound-Event-Detection-course-project/blob/main/utils.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

"""
def plot_results(target_list, preds_list, CLASS_LABELS_DICT):

    # Colors for different classes
    class_labels = {v: k for k, v in CLASS_LABELS_DICT.items()}
    class_colors_dict = {
        'brakes squeaking': '#E41A1C',  # Red
        'car': '#377EB8',  # Blue
        'children': '#4DAF4A',  # Green
        'large vehicle': '#984EA3',  # Purple
        'people speaking': '#FF7F00',  # Orange
        'people walking': '#FFFF33'  # Yellow
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for time_i in range(len(target_list)):
        
        target_class = np.argmax(target_list[time_i])
        pred_class = np.argmax(preds_list[time_i])

        target_color = class_colors_dict[class_labels[target_class]]
        pred_color = class_colors_dict[class_labels[pred_class]]

        # Plot ground truth bar
        ax.plot([time_i, time_i + 1], [1, 1], color=target_color, linewidth=10, linestyle='-', label='Annotated')

        # Plot prediction bar
        ax.plot([time_i, time_i + 1], [0.995, 0.995], color=pred_color, linewidth=10, linestyle='--', label='Model Output')

    # Create legends
    class_legend = [plt.Line2D([0], [0], color=color, linewidth=10, linestyle='-') for color in class_colors_dict.values()]
    class_legend += [plt.Line2D([0], [0], color=color, linewidth=10, linestyle='--') for color in class_colors_dict.values()]

    # Add legends to the plot
    ax.legend(handles=class_legend, title="Classes", loc='upper left')

    ax.set_yticks([])  # Hide Y axis
    ax.set_xlabel('Time')
    ax.set_title('Audio Class Activity Over Time')
    ax.set_ylim(0.98, 1.02)

    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()
"""
        
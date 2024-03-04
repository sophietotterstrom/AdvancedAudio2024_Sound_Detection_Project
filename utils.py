"""
Utility functions associated with the project

Copied from https://github.com/mulimani/Sound-Event-Detection
"""

from torch import split


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

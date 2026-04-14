import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    max = np.max(x, axis=-1, keepdims=True)
    sum = np.sum(np.exp(x-max), axis=-1, keepdims=True)
    return np.exp(x - max) / sum
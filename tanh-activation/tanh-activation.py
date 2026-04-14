import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x, dtype = float)
    top = np.exp(x) - np.exp(-x)
    bottom = np.exp(x) + np.exp(-x)
    return top / bottom
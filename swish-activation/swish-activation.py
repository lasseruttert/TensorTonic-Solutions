import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype = float)
    return x * sigmoid(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
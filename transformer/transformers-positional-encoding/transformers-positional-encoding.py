import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos_e = np.ndarray(shape=(seq_length, d_model))

    for pos in range(0, seq_length):
        for i in range(0, d_model):
            if i % 2 == 0:
                pos_e[pos, i] = np.sin(pos/(10000**(i / d_model)))
            else:
                pos_e[pos, i] = np.cos(pos/(10000**((i-1) / d_model)))

    return pos_e
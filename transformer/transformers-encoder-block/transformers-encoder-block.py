import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    sigma = np.var(x, axis=-1, keepdims=True)

    numerator = x - mean
    denominator = np.sqrt(sigma + eps)

    return gamma * (numerator/denominator) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Qq = Q @ W_q
    Kk = K @ W_k
    Vv = V @ W_v

    Qq = Qq.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    Kk = Kk.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    Vv = Vv.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    scores = (Qq @ Kk.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    attn = weights @ Vv

    attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    return attn @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.dot(x, W1) + b1
    relu_out = np.maximum(0, hidden)
    return np.dot(relu_out, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x_one = x + multi_head_attention(x,x,x, W_q, W_k, W_v, W_o, num_heads)
    x_one = layer_norm(x_one, gamma1, beta1)

    x_two = x_one + feed_forward(x_one, W1, b1, W2, b2)
    output = layer_norm(x_two, gamma2, beta2)
    return output
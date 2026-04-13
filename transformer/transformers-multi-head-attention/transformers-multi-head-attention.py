import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
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
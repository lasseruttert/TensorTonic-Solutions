def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    for i in range(len(x)):
        if x[i] <= 0:
            x[i] = alpha * (math.exp(x[i]) - 1)
    return x
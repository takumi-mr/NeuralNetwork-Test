import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def step(x):
    y = x > 0
    return y.astype(np.int)

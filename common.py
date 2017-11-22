import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def cross_entropy(y, y_label):
    if y.ndim == 1:
        y_label = y_label.reshape(1, y_label.size)
        y = y.reshape(1, y.size)
    #turn y_label into 1darray if it's a one-hot matrix
    #if y_label.size == y.size:
    #    y_label = y_label.argmax(axis=1)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), y_label])) / batch_size

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, y_label):
    return np.sum((y - y_label) ** 2) / y.size

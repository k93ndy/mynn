import numpy as np

def union_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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
    return -np.sum(np.log(y[np.arange(batch_size), y_label.astype(np.int)])) / batch_size

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0) # prevent overflow
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # prevent overflow
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, y_label):
    return np.sum((y - y_label) ** 2) / y.size

def identity_function(x):
    return x


class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dy):
        dy[self.mask] = 0
        dx = dy
        return dx

class SigmoidLayer:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dy):
        dx = dy * (1.0 - self.out) * self.out
        return dx

class SoftmaxWithLossLayer:
    def __init__(self):
        self.loss = None
        self.y = None
        self.y_label = None

    def forward(self, x, y_label):
        self.y_label = y_label
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.y_label)
        
        return self.loss

    def backward(self, dy=1):
        batch_size = self.y_label.shape[0]
        if self.y_label.size == self.y.size: # if y_label is a one-hot matrix
            dx = (self.y - self.y_label) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.y_label.astype(np.int)] -= 1
            dx = dx / batch_size
        
        return dx

class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
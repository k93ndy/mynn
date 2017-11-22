"""my nn"""
import numpy as np
from common import *
#from gradient import numerical_gradient

class myNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.param = {}
        self.param['W1'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.param['b1'] = np.zeros((1, hidden_dim))
        self.param['W2'] = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.param['b2'] = np.zeros((1, output_dim))
    
    def predict_proba(self, X):
        a1 = X.dot(self.param['W1']) + self.param['b1']
        z1 = sigmoid(a1)
        a2 = a1.dot(self.param['W2']) + self.param['b2']
        y = softmax(a2)
        return y
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def loss(self, X, y_label):
        y = self.predict_proba(X)
        loss = cross_entropy(y, y_label)
        return loss
        
    def accuracy(self, X, y_label):
        y = self.predict(X)
        total = len(y)
        accurate = np.size(np.where((y == y_label)==True))
        return float(accurate) / total
        
    def numerical_gradient(self, X, y_label, key):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(self.param[key])
        
        it = np.nditer(self.param[key], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            origin = self.param[key][idx]
            self.param[key][idx] = float(origin) + h
            fxh1 = self.loss(X, y_label) # f(x+h)
            
            self.param[key][idx] = float(origin) - h 
            fxh2 = self.loss(X, y_label) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)
            self.param[key][idx] = origin
            
            it.iternext()   
            
        return grad
    
    def union_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def get_batch(self, X, y_label, batch_size=100):
        X_shuffled, y_label_shuffled = self.union_shuffled_copies(X, y_label)
        return X_shuffled[range(batch_size)], y_label_shuffled[range(batch_size)]
    
    def train(self, X, y_label, learning_rate=0.01, epoch=100):
        for i in range(epoch):
            for key in self.param:
                grad = self.numerical_gradient(X, y_label, key)
                self.param[key] = self.param[key] - learning_rate*grad

    def train_minibatch(self, X, y_label, learning_rate=0.01, epoch=100, batch_size=100):
        for i in range(epoch):
            for j in range(np.ceil(np.size(X) / batch_size).astype(np.int)):
                X_batch, y_label_batch = self.get_batch(X, y_label, batch_size)
                for key in self.param:
                    grad = self.numerical_gradient(X_batch, y_label_batch, key)
                    self.param[key] = self.param[key] - learning_rate*grad
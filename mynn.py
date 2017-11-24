"""my nn"""
import numpy as np
from common import *

class MyNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.param = {}
        self.param['W1'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.param['b1'] = np.zeros((1, hidden_dim))
        self.param['W2'] = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.param['b2'] = np.zeros((1, output_dim))    

    def predict_proba(self, X):
        a1 = X.dot(self.param['W1']) + self.param['b1']
        z1 = relu(a1)
        a2 = z1.dot(self.param['W2']) + self.param['b2']
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
        total_num = len(y)
        accurate_num = np.size(np.where((y == y_label)==True))
        return float(accurate_num) / total_num
        
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
    
    def get_data(self, X, y_label, batch_size=100):
        X_shuffled, y_label_shuffled = union_shuffled_copies(X, y_label)
        return X_shuffled[range(batch_size)], y_label_shuffled[range(batch_size)]
    
    def train_batch(self, X, y_label, learning_rate=0.01, epoch=100):
        for i in range(epoch):
            for key in self.param:
                grad = self.numerical_gradient(X, y_label, key)
                self.param[key] = self.param[key] - learning_rate*grad

    def train_minibatch(self, X, y_label, learning_rate=0.01, epoch=100, batch_size=100):
        for i in range(epoch):
            for j in range(np.ceil(np.size(X) / batch_size).astype(np.int)):
                X_batch, y_label_batch = self.get_data(X, y_label, batch_size)
                for key in self.param:
                    grad = self.numerical_gradient(X_batch, y_label_batch, key)
                    self.param[key] = self.param[key] - learning_rate*grad


class MyNNWithBP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        W0 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        b0 = np.zeros((1, hidden_dim))
        W1 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        b1 = np.zeros((1, output_dim))

        self.tier0_affline = AffineLayer(W0, b0)
        self.tier0_relu = ReluLayer()
        self.tier1_affline = AffineLayer(W1, b1)
        self.tier1_softmaxwithloss = SoftmaxWithLossLayer()

    def get_param(self):
        param = {}
        param['W0'] = self.tier0_affline.W
        param['b0'] = self.tier0_affline.b
        param['W1'] = self.tier1_affline.W
        param['b1'] = self.tier1_affline.b
        return param
    
    def predict_proba(self, X):
        a0 = self.tier0_affline.forward(X)
        z0 = self.tier0_relu.forward(a0)
        a1 = self.tier1_affline.forward(z0)
        y = softmax(a1)
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
        total_num = len(y)
        accurate_num = np.size(np.where((y == y_label)==True))
        return float(accurate_num) / total_num

    def get_data(self, X, y_label, batch_size=100):
        X_shuffled, y_label_shuffled = union_shuffled_copies(X, y_label)
        return X_shuffled[range(batch_size)], y_label_shuffled[range(batch_size)]

    def train_minibatch(self, X, y_label, learning_rate=0.01, epoch=100, batch_size=100, X_test=None, y_test=None, report=False):
        report_epoch_loss = []
        report_accuracy = []
        # loss = None
        for i in range(epoch):
            for j in range(np.ceil(np.size(X) / batch_size).astype(np.int)):
                X_batch, y_label_batch = self.get_data(X, y_label, batch_size)
                a0 = self.tier0_affline.forward(X_batch)
                z0 = self.tier0_relu.forward(a0)
                a1 = self.tier1_affline.forward(z0)
                loss = self.tier1_softmaxwithloss.forward(a1, y_label_batch)

                d_loss = self.tier1_softmaxwithloss.backward()
                d_tier1_affline = self.tier1_affline.backward(d_loss)
                d_tier0_relu = self.tier0_relu.backward(d_tier1_affline)
                d_tier0_affline = self.tier0_affline.backward(d_tier1_affline)

                self.tier0_affline.W = self.tier0_affline.W - self.tier0_affline.dW*learning_rate
                self.tier0_affline.b = self.tier0_affline.b - self.tier0_affline.db*learning_rate
                self.tier1_affline.W = self.tier1_affline.W - self.tier1_affline.dW*learning_rate
                self.tier1_affline.b = self.tier1_affline.b - self.tier1_affline.db*learning_rate

            if report is True:
                loss = self.loss(X_test, y_test)
                accuracy = self.accuracy(X_test, y_test)
                report_epoch_loss.append([i+1, loss])
                report_accuracy.append([i+1, accuracy])

        if report is True:
            return (report_epoch_loss, report_accuracy)
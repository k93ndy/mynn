import numpy as np
from collections import OrderedDict
from common import *
from optimizer import *

class MyNNWithBP:
    def __init__(self, shape):
        self.layer_count = len(shape)
        self.layers = OrderedDict()
        for i in range(1, self.layer_count - 1):
            W = np.random.randn(shape[i - 1], shape[i]) / np.sqrt(shape[i - 1])
            b = np.zeros((1, shape[i]))
            self.layers['affline_tier' + str(i)] = AffineLayer(W, b)
            self.layers['relu_tier' + str(i)] = ReluLayer()
        W = np.random.randn(shape[self.layer_count - 2], shape[self.layer_count - 1]) / np.sqrt(shape[self.layer_count - 2])
        b = np.zeros((1, shape[self.layer_count - 1]))
        self.layers['affline_tier' + str(self.layer_count - 1)] = AffineLayer(W, b)
        self.lastlayer = SoftmaxWithLossLayer()

    def get_param(self):
        param = {}
        for name, layer in self.layers.items():
            if 'affline_tier' in name:
                param[name] = (layer.W, layer.b)
        return param
    
    def predict_proba(self, X):
        for key in self.layers:
            X = self.layers[key].forward(X)
        y = softmax(X)
        return y

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def loss(self, X, y_label):
        for key in self.layers:
            X = self.layers[key].forward(X)
        loss = self.lastlayer.forward(X, y_label)
        return loss

    def accuracy(self, X, y_label):
        y = self.predict(X)
        total_num = len(y)
        accurate_num = np.size(np.where((y == y_label)==True))
        return float(accurate_num) / total_num

    def precision(self, X, y_label):
        delta = 1e-5
        y = self.predict(X)
        y_1 = set(np.where(y == 1)[0])
        y_label_1 = set(np.where(y_label == 1)[0])
        y_label_0 = set(np.where(y_label == 0)[0])
        tp = len(y_1&y_label_1)
        fp = len(y_1&y_label_0)
        # tp = np.sum((y == 1) == (y_label == 1))
        # fp = np.sum((y == 1) == (y_label == 0))
        return float(tp) / (tp + fp + delta)

    def recall(self, X, y_label):
        y = self.predict(X)
        positive = np.size(np.where(y_label == True))
        y_1 = set(np.where(y == 1)[0])
        y_label_1 = set(np.where(y_label == 1)[0])
        # y_label_0 = set(np.where(y_label == 0)[0])
        tp = len(y_1&y_label_1)
        # tp = np.sum((y == 1) == (y_label == 1))
        return float(tp) / positive

    def get_data(self, X, y_label, batch_size=100):
        X_shuffled, y_label_shuffled = union_shuffled_copies(X, y_label)
        return X_shuffled[range(batch_size)], y_label_shuffled[range(batch_size)]

    def train_minibatch(self, X, y_label, learning_rate=0.01, optimizer=SGD_Momentum, epoch=100, batch_size=100, X_test=None, y_test=None, report=False):
        report_train_loss = []
        report_test_loss = []
        report_accuracy = []
        report_precision = []
        report_recall = []
        for i in range(epoch):
            for j in range(np.ceil(np.size(X) / batch_size).astype(np.int)):
                X_batch, y_label_batch = self.get_data(X, y_label, batch_size)
                loss = self.loss(X_batch, y_label_batch) #forward
                dout = self.lastlayer.backward(1) #backward
                layers_backward = OrderedDict(reversed(self.layers.items()))
                for name, layer in layers_backward.items():
                    dout = layer.backward(dout)
                    # if 'affline_tier' in name:
                    #     layer.W = layer.W - layer.dW*learning_rate
                    #     layer.b = layer.b - layer.db*learning_rate
                opt = optimizer(learning_rate)
                for name, layer in self.layers.items():
                    if 'affline_tier' in name:
                        opt.update(layer, name)

            if report is True:
                train_loss = self.loss(X, y_label)
                test_loss = self.loss(X_test, y_test)
                accuracy = self.accuracy(X_test, y_test)
                precision = self.precision(X_test, y_test)
                recall = self.recall(X_test, y_test)
                report_train_loss.append([i+1, train_loss])
                report_test_loss.append([i+1, test_loss])
                report_accuracy.append([i+1, accuracy])
                report_precision.append([i+1, precision])
                report_recall.append([i+1, recall])

        if report is True:
            return (report_train_loss, report_test_loss, report_accuracy, report_precision, report_recall)
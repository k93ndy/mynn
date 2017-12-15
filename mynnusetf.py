import numpy as np
import tensorflow as tf
from collections import OrderedDict
from common import *

class MyNNUseTF:
    def __init__(self, shape, activation=tf.nn.relu):
        self.layer_count = len(shape)
        self.layers = OrderedDict()
        self.activation = activation
        self.X = []
        self.X.append(tf.placeholder(tf.float32, shape=[None, shape[0]]))
        self.y_label = tf.placeholder(tf.float32, shape=None)
        self.sess = tf.Session()
        for i in range(1, self.layer_count):
            W = np.random.randn(shape[i - 1], shape[i]) / np.sqrt(shape[i - 1])
            b = np.zeros((1, shape[i]))
            self.layers['affline_tier' + str(i)] = {}
            self.layers['affline_tier' + str(i)]['W'] = tf.Variable(W, dtype=tf.float32)
            self.layers['affline_tier' + str(i)]['b'] = tf.Variable(b, dtype=tf.float32)

    def get_data(self, X, y_label, batch_size=100):
        X_shuffled, y_label_shuffled = union_shuffled_copies(X, y_label)
        return X_shuffled[range(batch_size)], y_label_shuffled[range(batch_size)]

    def train_minibatch(self, X, y_label, learning_rate=0.01, optimizer=tf.train.GradientDescentOptimizer, epoch=100, batch_size=500, X_test=None, y_test=None, report=False):
        
        for i in range(1, self.layer_count - 1):
            self.layers['affline_tier' + str(i)]['h'] = self.activation(tf.matmul(self.X[i-1], self.layers['affline_tier' + str(i)]['W']) + self.layers['affline_tier' + str(i)]['b'])
            # self.X[i] = self.layers['affline_tier' + str(i)]['h']
            self.X.append(self.layers['affline_tier' + str(i)]['h'])
        i = i + 1
        self.layers['affline_tier' + str(i)]['a'] = tf.matmul(self.X[i-1], self.layers['affline_tier' + str(i)]['W']) + self.layers['affline_tier' + str(i)]['b']
        last_layer = self.layers['affline_tier' + str(i)]['a']
        y_pred = tf.nn.softmax(last_layer)

        crossentropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=last_layer, labels=self.y_label))
        optimizer = optimizer(learning_rate)
        train = optimizer.minimize(crossentropy_loss)

        y_pred_01 = tf.argmax(y_pred, axis=1)
        y_label_01 = tf.argmax(y_label, axis=1)

        # tf.less(y_pred, y_pred)
        # confusion_matrix = tf.confusion_matrix(labels=y_label_01, predictions=y_pred_01)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for i in range(epoch):
                for step in range(int(X.shape[0]/batch_size)):
                    X_per_step, y_per_step = self.get_data(X, y_label, batch_size=batch_size)
                    sess.run(train, {self.X[0]: X_per_step, self.y_label: y_per_step})
                if X_test is not None and y_test is not None:
                    print("epoch {0}: {1}".format(i+1, sess.run(crossentropy_loss, {self.X[0]: X_test, self.y_label: y_test})))
            print(sess.run(tf.reduce_sum(y_pred_01-y_label_01), {self.X[0]: X_test, self.y_label: y_test}))
            # print("confusion matrix: {0}".format(sess.run(confusion_matrix, {self.X[0]: X_test, self.y_label: y_test})))

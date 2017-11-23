"""playground"""
import numpy as np
from mynn import myNN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata

# raw = make_classification(n_samples=1000, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, n_classes=5, n_clusters_per_class=1)
# X_train, X_test, y_train, y_test = train_test_split(raw[0], raw[1], test_size=0.2, random_state=42)

# nn = myNN(input_dim=4, hidden_dim=5, output_dim=5)

# print(nn.param)
# print(nn.accuracy(X_test, y_test))
# print(nn.loss(X_test, y_test))



# #nn.train(X_train, y_train, epoch=1000)
# nn.train_minibatch(X_train, y_train, epoch=100, batch_size=100)
# print(nn.param)
# print(nn.accuracy(X_test, y_test))
# print(nn.loss(X_test, y_test))

mnist = fetch_mldata('MNIST original')

for key in mnist:
    print(key, "\n", mnist[key])

X_train, X_test, y_train, y_test = train_test_split(mnist['data'], mnist['target'], test_size=0.2, random_state=42)

nn = myNN(input_dim=784, hidden_dim=10, output_dim=10)

print(nn.param)
print(nn.accuracy(X_test, y_test))
print(nn.loss(X_test, y_test))

#nn.train(X_train, y_train, epoch=1000)
nn.train_minibatch(X_train, y_train, epoch=1, batch_size=10000)
print(nn.param)
print(nn.accuracy(X_test, y_test))
print(nn.loss(X_test, y_test))
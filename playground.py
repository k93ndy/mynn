"""playground"""
import numpy as np
from mynnwithbp import MyNNWithBP
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

raw = make_classification(n_samples=1000, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(raw[0], raw[1], test_size=0.2, random_state=42)

nn = MyNNWithBP([4, 3, 2])

# print(nn.get_param())
print("Accucary before training:\t\t", nn.accuracy(X_test, y_test))
print("Cross entropy loss before training:\t", nn.loss(X_test, y_test))

train_loss, test_loss, accuracy, precision, recall = nn.train_minibatch(X_train, y_train, learning_rate=0.001, epoch=3000, batch_size=100, X_test=X_test, y_test=y_test, report=True)

print("Accucary after training:\t\t", nn.accuracy(X_test, y_test))
print("Cross entropy loss after training:\t", nn.loss(X_test, y_test))
# print(nn.get_param())

forest = RandomForestClassifier()
forest.fit(X_train, y_train)

y_test_pred = forest.predict(X_test)
print(classification_report(y_test_pred, y_test))

train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
accuracy = np.array(accuracy)
plt.plot(train_loss[:, 0], train_loss[:, 1], linestyle='-.', label='loss of train set')
plt.plot(test_loss[:, 0], test_loss[:, 1], linestyle='-', label='loss of test set')
plt.plot(accuracy[:, 0], accuracy[:, 1], linestyle='--', label='accuracy')
plt.legend()
plt.show()

# mnist = fetch_mldata('MNIST original')

# X_train, X_test, y_train, y_test = train_test_split(mnist['data'], mnist['target'], test_size=0.2, random_state=42)

# nn = MyNNWithBP([784, 10, 10])

# print(nn.accuracy(X_test, y_test))
# print(nn.loss(X_test, y_test))

# cross_entropy_loss, accuracy = nn.train_minibatch(X_train, y_train, learning_rate=0.01, epoch=10, batch_size=1000, X_test=X_test, y_test=y_test, report=True)
# print(nn.accuracy(X_test, y_test))
# print(nn.loss(X_test, y_test))

# cross_entropy_loss = np.array(cross_entropy_loss)
# accuracy = np.array(accuracy)
# plt.plot(cross_entropy_loss[:, 0], cross_entropy_loss[:, 1], linestyle='solid', label='cross entropy loss')
# plt.plot(accuracy[:, 0], accuracy[:, 1], linestyle='dashed', label='accuracy')
# plt.legend()
# plt.show()
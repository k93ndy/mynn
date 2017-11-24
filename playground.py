"""playground"""
import numpy as np
from mynn import MyNN, MyNNWithBP
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt

raw = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(raw[0], raw[1], test_size=0.2, random_state=42)

nn = MyNNWithBP(input_dim=5, hidden_dim=10, output_dim=3)

# print(nn.get_param())
print("Accucary before training:\t\t", nn.accuracy(X_test, y_test))
print("Cross entropy loss before training:\t", nn.loss(X_test, y_test))

cross_entropy_loss, accuracy = nn.train_minibatch(X_train, y_train, learning_rate=0.001, epoch=1000, batch_size=100, X_test=X_test, y_test=y_test, report=True)
cross_entropy_loss = np.array(cross_entropy_loss)
accuracy = np.array(accuracy)
plt.plot(cross_entropy_loss[:, 0], cross_entropy_loss[:, 1], linestyle='solid', label='cross entropy loss')
plt.plot(accuracy[:, 0], accuracy[:, 1], linestyle='dashed', label='accuracy')
plt.legend()
plt.show()
# print(cross_entropy_loss[:, 0], cross_entropy_loss[:, 1])

# print(nn.get_param())
print("Accucary after training:\t\t", nn.accuracy(X_test, y_test))
print("Cross entropy loss after training:\t", nn.loss(X_test, y_test))

# mnist = fetch_mldata('MNIST original')

# for key in mnist:
#     print(key, "\n", mnist[key])

# X_train, X_test, y_train, y_test = train_test_split(mnist['data'], mnist['target'], test_size=0.2, random_state=42)

# nn = MyNNWithBP(input_dim=784, hidden_dim=10, output_dim=10)

# print(nn.get_param())
# print(nn.accuracy(X_test, y_test))
# print(nn.loss(X_test, y_test))

# nn.train_minibatch(X_train, y_train, learning_rate=0.01, epoch=1, batch_size=10000)
# print(nn.get_param())
# print(nn.accuracy(X_test, y_test))
# print(nn.loss(X_test, y_test))
# MNIST

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")

import matplotlib.pyplot as plt

plt.figure(1)
plt.imshow(mnist.data[2].reshape(28, 28), cmap=plt.cm.gray_r)
# plt.show()

from sklearn.model_selection import train_test_split

mnist_learn, mnist_test, target_learn, target_test = \
    train_test_split(mnist.data, mnist.target, test_size=0.05)

from sklearn.neural_network import MLPClassifier

neural_network = MLPClassifier(max_iter=1000, activation='logistic',
                               hidden_layer_sizes=(50, 30, 10))
neural_network.fit(mnist_learn, target_learn)

from sklearn.metrics import accuracy_score

print("Accu:", accuracy_score(target_test, neural_network.predict(mnist_test)))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(target_test, neural_network.predict(mnist_test)))

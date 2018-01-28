from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
print(digits)
id = 501
plt.figure(1)
plt.imshow(digits.images[id], cmap=plt.cm.gray_r)
# plt.show()

from sklearn.model_selection import train_test_split

digits_learn, digits_test, target_learn, target_test = train_test_split(
    digits.data, digits.target, test_size=0.1)

from sklearn.neural_network import MLPClassifier

neural_network = MLPClassifier(
    max_iter=1000, activation='logistic', hidden_layer_sizes=(50, 30, 10))
neural_network.fit(digits_learn, target_learn)
print(digits_learn)
print(digits.target[id])
print("Neural:", neural_network.predict(digits.data[id].reshape(1, -1)))
# plt.show()

from sklearn.metrics import accuracy_score

print("Accu:", accuracy_score(target_test,
                              neural_network.predict(digits_test)))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(target_test, neural_network.predict(digits_test)))

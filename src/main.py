""" packages for processing the data and training the model """
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model
import sklearn.neural_network as sk_neural
import sklearn.metrics as sk_metrics
import pandas as pd

# custom helpers
import helpers.data_helper as dh
import helpers.confusion_matrix_helper as cfm

np.set_printoptions(precision=2)

# Get dataset of the bank, load it via pd
DATASET = pd.read_csv('datasets/bank-full.csv', sep=';')

# We should take only categorical data https://archive.ics.uci.edu/ml/datasets/bank+marketing
CATEGORICAL_COLUMN = [
  'job', 'marital', 'education',
  'default', 'housing', 'loan',
  'contact', 'month', 'poutcome'
]

# Get data categorical data
# those which are good for classification) and are not targets
DATA = DATASET[list(CATEGORICAL_COLUMN)]

# Dummied data
DUMMIED_DATA = dh.get_dummied_data(DATA)

# Dummied target
DUMMIED_TARGET = pd.get_dummies(DATASET['y'])

# # Divide the data to the learn data and test data
DATA_LEARN, DATA_TEST, TARGET_LEARN, TARGET_TEST = sk_model.train_test_split(
  DUMMIED_DATA, DUMMIED_TARGET, test_size=0.005)

# Create a new neural_network
NEURAL_NETWORK = sk_neural.MLPClassifier(
  activation='logistic', hidden_layer_sizes=(10, 20, 20))

# Train the neural network
NEURAL_NETWORK.fit(DATA_LEARN, TARGET_LEARN)

# Mean accuracy of the test data
MEAN_ACCURACY = NEURAL_NETWORK.score(DATA_TEST, TARGET_TEST)
print("MEAN_ACCURACY is equal: " + str(MEAN_ACCURACY) + "\n")

# Calculate accuracy score of the neural network
ACCURACY_SCORE = sk_metrics.accuracy_score(TARGET_TEST, NEURAL_NETWORK.predict(DATA_TEST))

# Print accuracy score
print("ACCURACY_SCORE is equal: " + str(ACCURACY_SCORE) + "\n")

# Calculate confusion matrix
CONFUSION_MATRIX = sk_metrics.confusion_matrix(
  TARGET_TEST.values.argmax(axis=1),
  NEURAL_NETWORK.predict(DATA_TEST).argmax(axis=1)
)

# Plot non-normalized confusion matrix
plt.figure()
cfm.plot_confusion_matrix(CONFUSION_MATRIX, classes=TARGET_TEST.columns.values,
                          title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
cfm.plot_confusion_matrix(CONFUSION_MATRIX, classes=TARGET_TEST.columns.values, normalize=True,
                          title='Normalized confusion matrix')

plt.show()

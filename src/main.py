import sklearn.model_selection as sk_model
import sklearn.neural_network as sk_neural
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Get dataset of the bank, load it via pd
dataset = pd.read_csv('datasets/bank-full.csv', sep=';')
target = dataset['y']

# Get the columns names
# ['age', 'job', 'marital', 'education', 'default',
# 'balance', 'housing', 'loan',
# 'contact', 'day', 'month', 'duration',
# 'campaign', 'pdays', 'previous', 'poutcome', 'y']
columns_all = set(dataset.columns.tolist())

# Columns to dismiss (they don't apply for classification) or they are target data
columns_to_dismiss = {
    'age', 'duration', 'campaign', 'pdays', 'previous', 'emp', 'var', 'rate',
    'balance', 'cons', 'conf', 'idx', 'day', 'y'
}

# Dimiss those columns
data_columns = columns_all.difference(columns_to_dismiss)

# Get data with no dismissed columns (only
# those which are good for classification) and are not targets
data = dataset[list(data_columns)]

# Divide the data to the learn data and test data
data_learn, data_test, target_learn, target_test = sk_model.train_test_split(
    data, target, test_size=0.005)

# Create a new neural_network
neural_network = sk_neural.MLPClassifier(
    activation='logistic', hidden_layer_sizes=(50, 30, 10))

# Train the neural network
neural_network.fit(data_learn, target_learn)

# Show the accuracy score of the neural network
sk_metrics.accuracy_score(target_test, neural_network.predict(data_test))

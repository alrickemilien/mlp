#!/usr/bin/env python3

import csv
import sys
from os import path
import numpy as np

from mlp import NeuralNetwork, Layer

def csv2data(path):
    ret = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row

# Extract dataset path - raise on invalid path
dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
if path.isdir(dataset_path) is True:
    raise Exception(dataset_path + ': Is a directory.')
if path.exists(dataset_path) is False:
    raise Exception(dataset_path + ': No such file or directory.')

# Load data set
data = np.array(list(csv2data(dataset_path)))

# Shuffle dataset
np.random.shuffle(data)

# Take 80% of data set for train and 20% of dataset for test
data_train = data[:int(len(data) * 0.8)]
data_test = data[int(len(data) * 0.8):]

# Define dataset of train and dataset of test
y_train = data_train[:,1]
X_train = np.delete(data_train, [0, 1], axis=1).astype(np.float)

y_test = data_test[:,1]
X_test = np.delete(data_test, [0, 1], axis=1).astype(np.float)

classification = np.unique(data[:,1])

def f(v, size):
    ret = np.zeros(size)
    ret[np.where(classification == v)[0][0]] = 1
    return ret

def vectorize(f):
    def fnv(array, value) :
        return np.vstack([f(x, value) for x in array])
    return fnv

y_train = vectorize(f)(y_train, len(classification))
y_test = vectorize(f)(y_test, len(classification))

# Build the network
nn = NeuralNetwork()
nn.add_layer(Layer(n_input=X_train.shape[1], n_neurons=3, activation='sigmoid'))
nn.add_layer(Layer(n_input=3, n_neurons=3, activation='sigmoid'))
nn.add_layer(Layer(n_input=3, n_neurons=y_train.shape[1], activation='sigmoid'))

# Train
errors = nn.train(X_train, y_train, learning_rate=0.03, max_epochs=50)

# Print weights
list(map(lambda x: print('weighs', x.weights, 'bias', x.bias), nn._layers))

# Test
print('errors', errors)
print('predict', nn.predict(X_test))


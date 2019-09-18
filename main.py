#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from os import path
import numpy as np

import preprocessing
from mlp import NeuralNetwork, Layer

def csv2data(path):
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
X_train, y_train, X_test, y_test = preprocessing(list(csv2data(dataset_path)))

# Build the network
nn = NeuralNetwork()
nn.add_layer(Layer(n_input=X_train.shape[1], n_neurons=3, activation='sigmoid'))
nn.add_layer(Layer(n_input=3, n_neurons=3, activation='sigmoid'))
nn.add_layer(Layer(n_input=3, n_neurons=2, activation='sigmoid'))

# Train
errors = nn.train(X_train, y_train, learning_rate=0.1, max_epochs=50)

# Print weights
list(map(lambda x: print('weights', x.weights, 'bias', x.bias), nn._layers))

# Test
print('errors', errors)

def f(x):
    return np.where(x == 1)[0][0]

print('Accuracy', nn.accuracy(
    y_pred=nn.predict(X_test),
    y_true=[f(x) for x in y_test]
))

nn.save()

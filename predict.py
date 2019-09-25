#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from os import path
import numpy as np

import preprocessing
from mlp import NeuralNetwork, Layer

import tools.csv2data as csv2data

# Extract dataset path - raise on invalid path
dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
if path.isdir(dataset_path) is True:
    raise Exception(dataset_path + ': Is a directory.')
if path.exists(dataset_path) is False:
    raise Exception(dataset_path + ': No such file or directory.')

# Extract nn path - raise on invalid path
nn_path = sys.argv[2] if len(sys.argv) > 2 else 'save.model.npy'
if path.isdir(nn_path) is True:
    raise Exception(nn_path + ': Is a directory.')
if path.exists(nn_path) is False:
    raise Exception(nn_path + ': No such file or directory.')

"""
The format of the save is as following
[layer:[activation,n_input,neurons,weights,biases]]
"""
nn_load = np.load(nn_path, allow_pickle=True)

nn = NeuralNetwork()

# Load data set
X_train, y_train, X_test, y_test = preprocessing(csv2data(dataset_path))

for x in  nn_load:
    activation=x[0]
    n_input=x[1]
    n_neurons=x[2]
    weights=x[3]
    bias=x[4]
    nn.add_layer(Layer(n_input=n_input, n_neurons=n_neurons, activation=activation, bias=bias, weights=weights))

print('Accuracy: %f' % (nn.accuracy(
    y_pred=nn.predict(X_test),
    y_true=[np.where(x == 1)[0][0] for x in y_test]
)))

print('E: %f' % (nn.evaluate(nn.feed_forward(X_test), y_test)))

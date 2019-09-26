#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import path
import numpy as np
import optparse

import preprocessing
from mlp import NeuralNetwork, Layer
import tools.csv2data as csv2data
import matplotlib.pyplot as plt

"""
According to dtaset structure, the idea to split the dataset into 3 parts
And learn through 3 black boxes perceptron
This way, we reduce data complexity between them 
"""


# COMMAND LINE OPTIONS
parser = optparse.OptionParser(usage='usage: %prog [options] file')

options, args = parser.parse_args()

# Extract dataset path - raise on invalid path
dataset_path = args[0] if len(args) > 0 else 'data.csv'
if path.isdir(dataset_path) is True:
    raise Exception(dataset_path + ': Is a directory.')
if path.exists(dataset_path) is False:
    raise Exception(dataset_path + ': No such file or directory.')

# Load data set
X_train, y_train, X_test, y_test = preprocessing(csv2data(dataset_path))

clusters = [
    np.arange(0, 10),
    np.arange(10, 20),
    np.arange(20, 30),
]

cluster_nn = []

# Clusterise the dataset according to its description
for (xi, x) in enumerate(clusters):
    X_train_cluster = np.array(X_train.T[x]).T
    y_train_cluster = y_train
    print('X_train_cluster',  X_train_cluster)

    # Build the network
    nn = NeuralNetwork()
    nn.add_layer(Layer(n_input=X_train_cluster.shape[1], n_neurons=2, activation='sigmoid'))
    nn.add_layer(Layer(n_input=2, n_neurons=2, activation='sigmoid'))
    nn.add_layer(Layer(n_input=2, n_neurons=2, activation='sigmoid'))
    nn.add_layer(Layer(n_input=2, n_neurons=y_train_cluster.shape[1], activation='softmax'))

    # Train
    mses, _ = nn.train(X_train_cluster, y_train_cluster, learning_rate=0.5, max_epochs=100)

    out = 'save.%d.model' % xi
    nn.save(out=out)
    cluster_nn.append(nn)

# Predict
for (xi, x) in enumerate(clusters):
    X_test_cluster = np.array(X_test.T[x]).T
    y_predict = np.vstack([cluster_nn[xi].feed_forward(y) for y in X_test_cluster])
    print('Accuracy: %f' % (nn.accuracy(
        y_pred=np.argmax(y_predict, axis=1),
        y_true=[np.where(x == 1)[0][0] for x in y_test]
    )))
    print('E: %f' % (nn.evaluate(y_predict, y_test)))

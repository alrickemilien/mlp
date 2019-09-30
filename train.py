#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import path
import numpy as np
import optparse

import preprocessing
from mlp import NeuralNetwork, Layer
import tools.csv2data as csv2data
import matplotlib.pyplot as plt
import dataconfig as cfg

# COMMAND LINE OPTIONS
parser = optparse.OptionParser(usage='usage: %prog [options] file')

# Compare
parser.add_option('-p', '--plot',
action="store_true", dest="plot",
help="Plot stats on the training", default="false")

options, args = parser.parse_args()

# Extract dataset path - raise on invalid path
dataset_path = args[0] if len(args) > 0 else 'data.csv'
if path.isdir(dataset_path) is True:
    raise Exception(dataset_path + ': Is a directory.')
if path.exists(dataset_path) is False:
    raise Exception(dataset_path + ': No such file or directory.')

# Load data set
X_train, y_train, _, _ = preprocessing(csv2data(dataset_path))

# Build the network
nn = NeuralNetwork()
seed = cfg.preprocessing['weights_seed']
nn.add_layer(Layer(n_input=X_train.shape[1], n_neurons=5, activation='sigmoid', seed=seed))
nn.add_layer(Layer(n_input=5, n_neurons=5, activation='sigmoid', seed=seed))
nn.add_layer(Layer(n_input=5, n_neurons=3, activation='sigmoid', seed=seed))
nn.add_layer(Layer(n_input=3, n_neurons=y_train.shape[1], activation='softmax', seed=seed))

# Train
mses, _ = nn.train(X_train, y_train, learning_rate=cfg.preprocessing['learning_rate'], max_epochs=cfg.preprocessing['epoch'])

if (options.plot is True):
    nn.plot(mses)

nn.save()

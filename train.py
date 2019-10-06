#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import path
import numpy as np
import optparse
import yaml

import preprocessing
from mlp import NeuralNetwork, Layer
import tools.csv2data as csv2data
import matplotlib.pyplot as plt


# COMMAND LINE OPTIONS
parser = optparse.OptionParser(usage='usage: %prog [options] file')

# Compare
parser.add_option('-p', '--plot',
action="store_true", dest="plot",
help="Plot stats on the training", default="false")

# Configuration file to use
parser.add_option('-c', '--configure',
action="store", dest="configure",
help="specific configure file path", default="dataconfig.yml")

parser.add_option('-e', '--error',
action="store", dest="error",
help="error function to use for training", default="cee")

options, args = parser.parse_args()

# Extract dataset path - raise on invalid path
dataset_path = args[0] if len(args) > 0 else 'data.csv'
if path.isdir(dataset_path) is True:
    raise Exception(dataset_path + ': Is a directory.')
if path.exists(dataset_path) is False:
    raise Exception(dataset_path + ': No such file or directory.')

# Extract configuration
if path.isdir(options.configure) is True:
    raise Exception(options.configure + ': Is a directory.')
if path.exists(options.configure) is False:
    raise Exception(options.configure + ': No such file or directory.')
with open(options.configure, 'r') as yfile:
    cfg = yaml.load(yfile, Loader=yaml.BaseLoader)

# Load data set
X_train, y_train, X_test, y_test = preprocessing(cfg, csv2data(dataset_path))

# Build the network
nn = NeuralNetwork(error=options.error)
w_seed = int(cfg['weights_seed'])
b_seed = int(cfg['bias_seed'])
nn.add_layer(Layer(n_input=X_train.shape[1], n_neurons=100, activation='tanh', weights_seed=w_seed, bias_seed=b_seed))
nn.add_layer(Layer(n_input=100, n_neurons=100, activation='tanh', weights_seed=w_seed, bias_seed=b_seed))
nn.add_layer(Layer(n_input=100, n_neurons=y_train.shape[1], activation='softmax', weights_seed=w_seed, bias_seed=b_seed))

# Train
mses, cees = nn.train(X_train, y_train, X_test, y_test, learning_rate=float(cfg['learning_rate']), max_epochs=int(cfg['epoch']))

if (options.plot is True):
    nn.plot(mses, cees)

nn.save()

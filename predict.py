#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import path
import numpy as np
import optparse
import yaml

import preprocessing
import tools.csv2data as csv2data
from mlp import NeuralNetwork, Layer

# COMMAND LINE OPTIONS
parser = optparse.OptionParser(usage='usage: %prog [options] file')

# Compare
parser.add_option('-m', '--model',
action="store", dest="model",
help="specific model to use", default="save.model.npy")

# Configuration file to use
parser.add_option('-c', '--configure',
action="store", dest="configure",
help="specific configure file path", default="dataconfig.yml")

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

# Extract nn path - raise on invalid path
nn_path = options.model
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
_, _, X_test, y_test = preprocessing(cfg, csv2data(dataset_path))

for x in nn_load:
    activation=x[0]
    n_input=x[1]
    n_neurons=x[2]
    weights=x[3]
    bias=x[4]
    nn.add_layer(Layer(n_input=n_input, n_neurons=n_neurons, activation=activation, bias=bias, weights=weights))

y_predict = nn.feed_forward(X_test)

print('Accuracy: %f' % (nn.accuracy(
    y_pred=np.argmax(y_predict, axis=1),
    y_true=[np.where(x == 1)[0][0] for x in y_test]
)))

print('CEE: %f' % (nn.evaluate(y_predict, y_test)))

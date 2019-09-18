#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from os import path
import numpy as np
import pprint as pp

def describe_numeric_vector(data, index):
    stats = {}

    data = np.sort(data)

    # print('data', data)

    stats['empty'] = len(data[np.isnan(data)])

    # Remove empty values
    data = data[~np.isnan(data)]

    stats['index'] = index
    stats['count'] = len(data)
    stats['mean'] = sum(data) / stats['count']
    stats['var'] = (1 / (stats['count'] - 1) * np.sum(np.power(data - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    stats['precision'] = np.sqrt(stats['var'])
    return stats

def describe_classification_vector(y):
    stats = {}

    stats['empty'] = len(y[np.char.chararray.isspace(y)])

    # Remove empty values
    y = y[~np.char.chararray.isspace(y)]

    classification = np.unique(y)

    def vectorize(f):
        def fnv(array) :
            return [f(x) for x in array]
        return fnv

    def f(v):
        return np.where(classification == v)[0][0]

    y = vectorize(f)(y)

    stats['index'] = 0
    stats['count'] = len(y)
    stats['mean'] = sum(y) / stats['count']
    stats['var'] = (1 / (stats['count'] - 1) * np.sum(np.power(y - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    return stats

def describe(data):
    data = np.array(data)

    y = data[:,1]
    X = np.delete(data, [0, 1], axis=1).astype(np.float)

    pp.pprint(describe_classification_vector(y))
    
    for (xi, x) in enumerate(X.T):
        # Add 2 at the index because X is the dataset with the two first columns substracted
        pp.pprint(describe_numeric_vector(x, xi + 2))

def csv2data(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row

def main():
    # Extract dataset path - raise on invalid path
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
    if path.isdir(dataset_path) is True:
        raise Exception(dataset_path + ': Is a directory.')
    if path.exists(dataset_path) is False:
       raise Exception(dataset_path + ': No such file or directory.')
    
    describe(list(csv2data(dataset_path)))

if __name__ == '__main__':
    main()

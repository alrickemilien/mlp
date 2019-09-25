#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from os import path
import numpy as np
import pprint as pp

import tools.csv2data as csv2data

import dataconfig as cfg

def describe_numeric_feature(data, index):
    stats = {}

    data = np.sort(data)

    # Handle missing datas
    if (cfg.preprocessing['missing_data'] is False):
        stats['empty'] = 0        
    else:
        unique, counts = np.unique(data, return_counts=True)
        z = dict(zip(unique, counts))
        try:
            stats['empty'] = z[cfg.preprocessing['missing_data']]
        except KeyError:
            stats['empty'] = 0        
            pass
        data = data[np.where(data != cfg.preprocessing['missing_data'])]

    stats['index'] = cfg.preprocessing['features_start_index'] + index
    stats['count'] = len(data)
    stats['mean'] = sum(data) / stats['count']
    stats['var'] = (1 / (stats['count'] - 1) * np.sum(np.power(data - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    stats['precision'] = np.sqrt(stats['var'])
    return stats

def describe_classification_feature(y, index):
    stats = {}

    if (cfg.preprocessing['missing_data'] is not False):
        unique, counts = np.unique(y, return_counts=True)
        z = dict(zip(unique, counts))
        try:
            stats['empty'] = z[cfg.preprocessing['missing_data']]
        except KeyError:
            stats['empty'] = 0
            pass
        y = y[np.where(y != str(cfg.preprocessing['missing_data']))]
    else:
        stats['empty'] = 0

    classification = np.unique(y)

    def vectorize(f):
        def fnv(array) :
            return [np.where(classification == v)[0][0] for x in array]
        return fnv

    y = vectorize(f)(y)

    stats['index'] = index
    stats['count'] = len(y)
    stats['mean'] = sum(y) / stats['count']
    stats['var'] = (1 / (stats['count'] - 1) * np.sum(np.power(y - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    return stats

def describe(data):
    y = data[:,1]
    X = np.delete(data, [0, 1], axis=1).astype(np.float)

    pp.pprint(describe_classification_feature(y, 1))
    
    for (xi, x) in enumerate(X.T):
        pp.pprint(describe_numeric_feature(x, xi + cfg.preprocessing['features_start_index']))

def main():
    # Extract dataset path - raise on invalid path
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
    if path.isdir(dataset_path) is True:
        raise Exception(dataset_path + ': Is a directory.')
    if path.exists(dataset_path) is False:
       raise Exception(dataset_path + ': No such file or directory.')
    
    describe(csv2data(dataset_path))

if __name__ == '__main__':
    main()

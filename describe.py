#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from os import path
import numpy as np
import pprint as pp
import optparse
import yaml

import tools.csv2data as csv2data

def describe_numeric_feature(data, index, features_start_index=0, missing_data=False):
    stats = {}

    data = np.sort(data)

    # Handle missing datas
    if (missing_data is False):
        stats['empty'] = 0        
    else:
        unique, counts = np.unique(data, return_counts=True)
        z = dict(zip(unique, counts))
        try:
            stats['empty'] = z[missing_data]
        except KeyError:
            stats['empty'] = 0        
            pass
        data = data[np.where(data != missing_data)]

    stats['index'] = missing_data + index
    stats['count'] = len(data)
    stats['mean'] = sum(data) / stats['count']
    stats['var'] = (1 / (stats['count'] - 1) * np.sum(np.power(data - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    stats['precision'] = np.sqrt(stats['var'])
    return stats

def describe_classification_feature(y, index, features_start_index=0, missing_data=False):
    stats = {}

    if (missing_data is not False):
        unique, counts = np.unique(y, return_counts=True)
        z = dict(zip(unique, counts))
        try:
            stats['empty'] = z[missing_data]
        except KeyError:
            stats['empty'] = 0
            pass
        y = y[np.where(y != str(missing_data))]
    else:
        stats['empty'] = 0

    classification = np.unique(y)

    y = [np.where(classification == v)[0][0] for v in y]

    stats['index'] = index
    stats['count'] = len(y)
    stats['mean'] = sum(y) / stats['count']
    stats['var'] = (1 / (stats['count'] - 1) * np.sum(np.power(y - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    return stats

def describe(data, features_start_index=0, missing_data=False):
    y = data[:,1]
    X = np.delete(data, [0, 1], axis=1).astype(np.float)

    pp.pprint(describe_classification_feature(y, 1, features_start_index, missing_data))
    
    for (xi, x) in enumerate(X.T):
        pp.pprint(describe_numeric_feature(x, xi + features_start_index, features_start_index, missing_data))

def main():
    # COMMAND LINE OPTIONS
    parser = optparse.OptionParser(usage='usage: %prog [options] file')

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

    describe(csv2data(dataset_path), int(cfg['features_start_index']), bool(cfg['missing_data']))

if __name__ == '__main__':
    main()

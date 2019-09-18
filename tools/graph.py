#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
from os import path

import matplotlib.pyplot as plt
import numpy as np

def csv2data(dataset_path):
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row

def preprocessing(data):
    data = np.array(data)
    def f(v):
        print('v', v)
        try:
            return v.astype(np.float)
        except:
            classification = np.unique(v)
            print('classification', classification )

            def indexof(vv):
                return np.where(classification == vv)
            return np.vstack(indexof(x) for x in v)
        return data
    return np.vstack([f(x) for x in data.T]).T

if __name__ == '__main__':
    # Extract dataset path - raise on invalid path
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'data.csv'
    if path.isdir(dataset_path) is True:
        raise Exception(path + ': Is a directory.')
    if path.exists(dataset_path) is False:
        raise Exception(path + ': No such file or directory.')

    data = preprocessing(list(csv2data(dataset_path)))

    print('data', data)

    plt.scatter(data, data, c = 'red')

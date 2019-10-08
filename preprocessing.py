# -*- coding: utf-8 -*-

import sys
import numpy as np

import pandas as pd

from describe import describe_numeric_feature

def scale(X):
    def _scaling(v, index):
        stats = describe_numeric_feature(v, index)
        return (v - stats['min']) / (stats['max'] - stats['min'])
        # return (v - stats['mean']) / stats['std']

    # for (i, x) in enumerate(X.T):
    #     stats = describe_numeric_feature(x, i)
    #     print('INDEX: %d, MEAN: %f, STD: %f' % (i, stats['mean'], stats['std']))

    # df = pd.DataFrame(data=X,    # values
    #             index=np.arange(X.shape[0]),    # 1st column as index
    #             columns=np.arange(X.shape[1]))  # 1st row as the column names
    # print(df.describe(include = 'all'))

    return np.vstack([_scaling(x, xi) for (xi, x) in enumerate(X.T)]).T

def shuffle_along_axis(a, axis, shuffle_seed=0):
    np.random.seed(shuffle_seed)
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

def split(data, batch_percentage=1):
    if (batch_percentage == 1):
        return data, data
    # Take N% of data set for train and (N - 100)% of dataset for test
    data_train = data[:int(len(data) * batch_percentage)]
    data_test = data[int(len(data) * batch_percentage):]
    return data_train, data_test

def classify(classification, v):
    ret = np.zeros(classification.shape[0])
    ret[np.where(classification == v)[0][0]] = 1
    return ret

def preprocessing(cfg, data):
    """
    Preprocess the dataset
    - Shuffle
    - Split into train set and test set
    - Extract classification
    - Vectorize
    """

    # Shuffle dataset
    data = shuffle_along_axis(data, 0, int(cfg['shuffle_seed']))

    data = np.concatenate((
        data[:,(0,1)],
        scale(np.delete(data, [0, 1], axis=1).astype(np.float))
    ), axis=1)

    data_train, data_test = split(data, float(cfg['batch_size']))

    # Define dataset of train and dataset of test
    y_train = data_train[:,1]
    X_train = np.delete(data_train, [0, 1], axis=1).astype(np.float)

    y_test = data_test[:,1]
    X_test = np.delete(data_test, [0, 1], axis=1).astype(np.float)

    classification = np.unique(data[:,1])

    # Replace 1d array of   ['B', 'M', 'M', 'B', 'B', ...]
    # To a 2d arrayd of     [[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], ...]
    y_train = np.vstack([classify(classification, x) for x in y_train])
    y_test = np.vstack([classify(classification, x) for x in y_test])

    return X_train, y_train, X_test, y_test

sys.modules[__name__] = preprocessing

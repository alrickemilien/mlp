# -*- coding: utf-8 -*-

import sys
import numpy as np

from describe import describe_numeric_vector

def scaling(X):
    def vectorize(f):
        def fnv(array) :
            return np.vstack([f(x, xi) for (xi, x) in enumerate(array)])
        return fnv

    def f(v, index):
        stats = describe_numeric_vector(v, index)
        return (v - stats['mean']) / stats['std']
    return vectorize(f)(X.T).T

def preprocessing(data):
    """
    Preprocess the dataset
    - Shuffle
    - Split into train set and test set
    - Extract classification
    - Vectorize
    """

    # To numpy array
    data = np.array(data)

    # Shuffle dataset
    np.random.shuffle(data)

    data = np.concatenate(
        (
            data[:,(0,1)],
            scaling(np.delete(data, [0, 1], axis=1).astype(np.float))
        ),
        axis=1)

    print('data', data)

    # Take 80% of data set for train and 20% of dataset for test
    data_train = data[:int(len(data) * 0.8)]
    data_test = data[int(len(data) * 0.8):]

    # Define dataset of train and dataset of test
    y_train = data_train[:,1]
    X_train = np.delete(data_train, [0, 1], axis=1).astype(np.float)

    y_test = data_test[:,1]
    X_test = np.delete(data_test, [0, 1], axis=1).astype(np.float)

    classification = np.unique(data[:,1])

    def f(v, size):
        ret = np.zeros(size)
        ret[np.where(classification == v)[0][0]] = 1
        return ret

    # Replace 1d array of   ['B', 'M', 'M', 'B', 'B', ...]
    # To a 2d arrayd of     [[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], ...]
    y_train = np.vstack([f(x, len(classification)) for x in y_train])
    y_test = np.vstack([f(x, len(classification)) for x in y_test])

    return X_train, y_train, X_test, y_test

sys.modules[__name__] = preprocess

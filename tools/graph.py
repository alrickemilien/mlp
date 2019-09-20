#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import itertools
import optparse
from os import path

import matplotlib.pyplot as plt
import numpy as np

plot_index = 0

def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""

    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    print('0')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[x], data[y], **kwargs)

    print('A')

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    print('B')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig

def csv2data(dataset_path):
    with open(dataset_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row

def preprocessing(data):
    data = np.array(data)
    
    def f(v):
        try:
            return v.astype(np.float)
        except:
            classification = np.unique(v)
            print('classification', classification )

            def indexof(vv):
                return np.where(classification == vv)[0][0]
            return [indexof(x) for x in v]
        return data
    
    ret = np.vstack([f(x) for x in data.T])
    
    return ret.T

def compare_string_to_indexes(s):
    splitted = s.split(',')

    return int(splitted[0]), int(splitted[1])

if __name__ == '__main__':
    #
    # COMMAND LINE OPTIONS
    #
    parser = optparse.OptionParser(usage='usage: %prog [options] [file]')

    options, args = parser.parse_args()

    # Compare
    parser.add_option('-c', '--compare',
    action="store", dest="compare",
    help="(x,y) compares x feature with y feature", default="")

    # Extract dataset path - raise on invalid path
    dataset_path = args[0] if len(args) > 0 else 'data.csv'
    if path.isdir(dataset_path) is True:
        raise Exception(dataset_path + ': Is a directory.')
    if path.exists(dataset_path) is False:
        raise Exception(dataset_path + ': No such file or directory.')

    data = preprocessing(list(csv2data(dataset_path)))

    print('data', data)

    scatterplot_matrix(np.delete(data.T, np.arange(20), axis=0), [], c = 'red')
    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
import os
import itertools
import optparse
from os import path

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np

import csv2data

def scatter_plot_matrix_view(fig, axes):
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

def scatterplot_matrix(data, names, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""

    numclass = len(data)
    numvars = len(data[0][0])
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))

    scatter_plot_matrix_view(fig, axes)

    # Plot the data.
    for c in range(numclass):
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i,j), (j,i)]:
                u = np.array(data[c]).T
                axes[x,y].scatter(u[x], u[y], cmap=c, alpha=0.3)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig

classification_index = 0
classification = []

def preprocessing(data, i=0, j=1):
    """
    Return an array like :
    [
        ClassA [
            [v1]
            [v3]
            ...
        ],
        Class B[
            [v2]
            ...
        ],
        ...
        Class N [
            [vn]
            ...
        ]
    ]
    """
    def f(vi, v):
        global classification_index
        global classification

        try:
            return v.astype(np.float)
        except:
            classification_index = vi
            classification = np.unique(v)

            def indexof(vv):
                return np.where(classification == vv)[0][0]
            return [indexof(x) for x in v]
        return data
    
    return np.vstack([f(xi, x) for (xi, x) in enumerate(data.T)]).T

def compare_string_to_indexes(s):
    splitted = s.split(',')

    return int(splitted[0]), int(splitted[1])

def cut(data, i, j):
    global classification
    global classification_index

    ret = [[] for x in range(len(classification))]

    for v in data:
        ret[int(v[classification_index])].append(np.concatenate((v[i:i+1], v[j:j+1])))
    
    return np.array(ret)

def to_pdf(data):
    if not os.path.exists('out'):
        os.makedirs('out')
    pdf = matplotlib.backends.backend_pdf.PdfPages('out/output.pdf')
    l = len(data[0])
    for i in range(2, l):
        for j in range(i, l):
            if j <= i: continue
            print('Plotting column ' + str(i) + ' with column ' + str(j))
            fig = scatterplot_matrix(cut(data, i, j), [str(i), str(j)])
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close()

if __name__ == '__main__':
    # COMMAND LINE OPTIONS
    parser = optparse.OptionParser(usage='usage: %prog [options] [file]')

    # Compare
    parser.add_option('-c', '--compare',
    action="store", dest="compare",
    help="(x,y) compares x feature with y feature", default="")

    options, args = parser.parse_args()

    # Extract dataset path - raise on invalid path
    dataset_path = args[0] if len(args) > 0 else 'data.csv'
    if path.isdir(dataset_path) is True:
        raise Exception(dataset_path + ': Is a directory.')
    if path.exists(dataset_path) is False:
        raise Exception(dataset_path + ': No such file or directory.')

    data = np.array(preprocessing(csv2data(dataset_path)))

    if (len(options.compare) == 0):
        to_pdf(data)
    else:
        i, j = compare_string_to_indexes(options.compare)
        scatterplot_matrix(data, [])
        plt.show()

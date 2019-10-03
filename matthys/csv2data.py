#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import numpy as np

def _csv2data(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row

def csv2data(path):
    return np.array(list(_csv2data(path)))

sys.modules[__name__] = csv2data

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# This script generates data according to configuration
#

import sys
import optparse

import xml.etree.cElementTree as ET

from sklearn.datasets import make_classification

#
# FIELD
#

# This class represents a field column of data
class Field:

  # Keep track on groups/classes labels
  classes = []

  def __init__(self, name, dtype, values=[] ,groupBy=False, vmin=0, vmax=100):
    self.name = name
    self.dtype = dtype
    self.values = values
    self.groupByAttribute = groupBy
    self.vmin = vmin
    self.vmax = vmax
  
  # Build field from XML
  def fromXML(x):
   groupBy = x.get('groupBy')
   if groupBy is None:
    groupBy = False

   xmlvalues = x.find('enum')
   if xmlvalues is None:
    xmlvalues = []

   values = list(map(lambda x : x.text, xmlvalues))

   if bool(x.get('groupBy')) is True:
    for y in values:
     Field.classes.append(y)

   return Field(name=x.find('name').text,
                dtype=x.find('type').text,
                groupBy=bool(x.get('groupBy')), 
                values=values)

  # Find groupBy field among fields list
  def getGroupByField(l):
   for x in l:
       if x.groupByAttribute is True:
           return x
   return False

  def __str__(self):
   return 'Field[%s] - %s' % (self.dtype, self.name)

#
# GENERATOR
#

def generator(configuration_file_path):
 tree = ET.parse(configuration_file_path)

 root = tree.getroot() 
 
 fields = []

 for x in root:
  fields.append(Field.fromXML(x))

 # Print the header
 if (bool(options.header)):
  print(*map(lambda x : x.name, fields), sep=",")

 # Build data vectors
 X, y = make_classification(n_samples=int(options.size),
                            n_features=len(fields) - 2,
                            n_informative=3, 
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=len(Field.getGroupByField(fields).values),
                            n_clusters_per_class=3,
                            class_sep=10.5,
                            flip_y=0)
 for xi in range(len(X)):
  print('%d,%s,%s' % (xi, Field.classes[y[xi]], ','.join(map(str, X[xi]))))

if __name__ == '__main__':
 USAGE='usage: %prog [options] configuration_file.xml'
 #
 # COMMAND LINE OPTIONS
 #
 parser = optparse.OptionParser(usage=USAGE)

 # Size
 parser.add_option('-s', '--size',
    action="store", dest="size",
    help="size integer", default="10")

 # Header
 parser.add_option('--header', '--header',
    action="store", dest="header",
    help="do we set header into csv output", default="false")

 options, args = parser.parse_args()

 if len(args) < 1:
  sys.exit(USAGE)

 generator(sys.argv[1])

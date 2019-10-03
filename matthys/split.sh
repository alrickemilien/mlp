#!/bin/bash

cat $1 | cut -f2 -d, | sed 's/B/0/g' | sed 's/M/1/g' > output.csv
cat $1 | cut -f3-32 -d, > input.csv

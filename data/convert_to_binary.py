import sys
import numpy

with open(sys.argv[1], 'r') as fin:
    lines = fin.readlines()
    arr = numpy.asarray([[float(token) for token in line.strip().split(' ')] for
        line in lines])
    arr.tofile(sys.argv[2])


import numpy
import sys


with open(sys.argv[1], 'r') as fpred:
    lines = fpred.readlines()
    a = numpy.asarray([int(l) for l in lines])

with open(sys.argv[2], 'r') as fY:
    lines = fY.readlines()
    b = numpy.asarray([int(l.strip().split(' ')[-2]) for l in lines])
    #c = numpy.asarray([int(round(float(l.split(' ')[-2]))) for l in lines])

print(a.size, ' vs ', b.size)
min_size = min(a.size, b.size)

print(1.0 - numpy.mean(numpy.abs(a[:min_size]-b[:min_size])))

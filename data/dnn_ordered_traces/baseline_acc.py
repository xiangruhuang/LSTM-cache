import sys
import numpy

fin = open(sys.argv[1], 'r')

lines = fin.readlines()

acc = numpy.mean([float(l.strip().split(' ')[-1]) for l in lines])

print('acc=%f' % acc)

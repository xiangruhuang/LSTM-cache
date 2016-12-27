import numpy
import sys

for i in range(30000):
    sum_i = 0.0
    g = numpy.random.normal(size=[16])

    for j in range(16):
        nj = g[j]
        sum_i += nj
        sys.stdout.write(str(nj))
        sys.stdout.write(' ')
    sys.stdout.write('1 ')
    if sum_i >= 0:
        sys.stdout.write('1\n')
    else:
        sys.stdout.write('0\n')
    
    for j in range(16):
        nj = g[j]
        sum_i += nj
        sys.stdout.write(str(nj))
        sys.stdout.write(' ')
    sys.stdout.write('1 ')
    if sum_i >= 0:
        sys.stdout.write('0\n')
    else:
        sys.stdout.write('1\n')



import sys

with open(sys.argv[1], 'r') as fin:
    a = [float(line.strip().split(',')[2]) for line in fin.readlines()[1:]]
    at_3fourth = a[int(len(a)/4.0)*3]
    allset = a[-1]
    ans = (a[-1] - at_3fourth*0.75)*4.0
    print('test acc=%f' % ans)

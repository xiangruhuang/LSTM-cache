import sys

fin = open(sys.argv[1], 'r')
lines = fin.readlines()
fin.close()

instr_addrs = set([l.split(' ')[2] for l in lines])
load_addrs = set([l.split(' ')[1] for l in lines])

print('#instr=', len(instr_addrs))
print('#data_addr=', len(load_addrs))

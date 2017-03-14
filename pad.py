import sys
import os
sys.path.append('/work/04603/xrhuang/maverick/Projects/LSTM-cache/')
from Config import *

config = Config()
i = 9999
filename = ''
while i > 0:
    s = str(i).zfill(4)
    if os.path.exists(sys.argv[1]+'.'+s):
        filename = sys.argv[1]+'.'+s
        break
    i -= 1

print('padding %s...' % filename)
fin = open(filename, 'r')
lines = fin.readlines()
fin.close()

last_line = lines[-1]

L = config.batch_size*config.num_steps

fout = open(filename, 'a')
for i in range(L-len(lines)):
    fout.writelines([last_line])

fout.close()

import sys
import json

fin = open(sys.argv[1], 'r')
lines = fin.readlines()

instr_set = set([l.strip().split(' ')[2] for l in lines])
instr_dict = {addr:i for i, addr in enumerate(instr_set)}
data_set = set([l.strip().split(' ')[1] for l in lines])
data_dict = {addr:i for i, addr in enumerate(data_set)}

print(len(data_dict))
json.dump(instr_dict, open(sys.argv[2]+'.instr_map', 'w'))
json.dump(data_dict, open(sys.argv[2]+'.data_map', 'w'))

data = [l.strip().split(' ')[1:] for l in lines] # skip some lines

with open(sys.argv[2], 'w') as fout:
    for x in data:
        #x[0] = data_dict[x[0]]
        #x[1] = instr_dict[x[1]]
        for i in range(len(x)):
            fout.write(str(x[i]))
            fout.write(' ')
        fout.write('\n')


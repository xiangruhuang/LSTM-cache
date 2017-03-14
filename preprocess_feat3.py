import sys
import json
import numpy
from Config import *

class Record(object):
    def __init__(self, t, data, instr, prob, Y):
        self.t = t
        self.data = data
        self.instr = instr
        self.prob = float(prob) if abs(float(prob) - 2.0) > 1e-3 else 0.5
        self.Y = int(Y)
        assert(Y != 0.5)

class Records(object):
    def __init__(self, data_map, instr_map, config):
        self.T = config.T
        
        self.data_map = data_map
        self.instr_map = instr_map

        num_instr = len(instr_map)
        num_data = len(data_map)
        self.data_hist = {i:[] for i in range(num_data)}
        self.instr_hist = {i:[] for i in range(num_instr)}
        self.records = []
        self.unknown_count = 0
        self.total = 0
        self.feature = {i:[] for i in range(num_instr)}
        self.pos = [] # pos[t] = position of records[t] in instr_hist[records[t].instr]
        self.unknown = {i:0 for i in range(num_instr)}
        self.dist_list = {i:[] for i in range(num_instr)}

    # data_hist[d] contains every record with this data address until now.
    # If record r is not the latest record in this list, OPTGEN must know its optimal output
    # otherwise if record r is not the only record, then OPTGEN must know the optimal output for the previous one
    # otherwise we have no choice better than a guess (output 0.5)
    def get_recent_true_label(self, r):
        dh = self.data_hist[r.data]
        assert(len(dh) > 0)
        if dh[-1] != r:
            return r.Y
        elif len(dh) > 1:
            return dh[-2].Y
        else:
            return 0.5

    def add_sample(self, t, sample):
        r = Record(t, data=self.data_map[sample[0]], instr=self.instr_map[sample[1]], prob=sample[2], Y=sample[3])
        self.records.append(r)
        self.data_hist[r.data].append(r)
        self.pos.append(len(self.instr_hist[r.instr]))
        self.instr_hist[r.instr].append(r)
        self.feature[r.instr].append(self.get_recent_true_label(r))
        if len(self.instr_hist[r.instr]) > 1:
            self.dist_list[r.instr].append(1)
            self.dist_list[r.instr].append(r.t-self.instr_hist[r.instr][-2].t)
        #self.unknown[r.instr] += 1
        if len(self.data_hist[r.data]) > 1:
            last_record = self.data_hist[r.data][-2]
            offset = self.pos[last_record.t]
            self.feature[last_record.instr][offset] = last_record.Y
            #self.unknown[last_record.instr] -= 1
    
    """ pad x to arr from head, until length L is reached, then take tail L elements from it"""
    def pad_and_trunc(self, arr, x, L):
        if len(arr) < L:
            return [x]*(L-len(arr)) + arr
        else:
            return arr[-L:]

    def get_feature(self, t):
        r_t = self.records[t]
        #h = self.feature[r_t.instr][:-1]
        L = len(self.feature[r_t.instr])-1
        pad_len = int(max(self.T - L, 0))
        feature_len = int(min(self.T, L))
        h = self.feature[r_t.instr][-(feature_len+1):-1]

        dl = self.pad_and_trunc(self.dist_list[r_t.instr], 0, self.T*2)

        feature = [r_t.prob] + [0.5]*(pad_len) + h + dl
        ul = len([f for f in h if (abs(float(f) - 0.5) < 1e-3)])
        assert(ul <= feature_len)
        self.unknown_count += ul
        self.total += feature_len

        #h = []
        #count = 0
        #"""Check every record history for this instruction"""
        #for r in self.instr_hist[r_t.instr][:-1]:
        #    Y_r = self.get_recent_true_label(r)
        #    assert(Y_r == self.feature[r_t.instr][count])
        #    if abs(float(Y_r) - 0.5) < 1e-3:
        #        self.unknown_count += 1
        #    h.append(Y_r)
        #    count += 1
        #    self.total += 1
        #    #if count >= self.T:
        #    #    break
        #if count > self.T:
        #    count = self.T
        #h = h[-count:]
        #feature = [r_t.prob] + [0.5] * (self.T - count) + h 

        return feature, r_t.Y

fin = open(sys.argv[1], 'r')
lines = fin.readlines()
config = Config()

instr_set = set([l.strip().split(' ')[2] for l in lines])
instr_dict = {addr:i for i, addr in enumerate(instr_set)}
data_set = set([l.strip().split(' ')[1] for l in lines])
data_dict = {addr:i for i, addr in enumerate(data_set)}

json.dump(instr_dict, open(sys.argv[2]+'.instr_map', 'w'))
json.dump(data_dict, open(sys.argv[2]+'.data_map', 'w'))

"""data_addr, instr_addr, prob, truelabel"""
samples = [l.strip().split(' ')[1:5] for l in lines] # skip some lines

records = Records(data_map = data_dict, instr_map = instr_dict, config = config)

unknown_count = 0
total = 0
fout = open(sys.argv[2], 'w')
for (t, sample) in enumerate(samples):
    """Updates"""
    records.add_sample(t, sample)
    
    feature_t, Y_t = records.get_feature(t)
    assert(len(feature_t) == records.T*3+1)
    if t % 10000 == 0:
        print('t=%d, unknown=%d, total=%d, fail_prob=%f' % (t, records.unknown_count, records.total, float(records.unknown_count)/float(records.total+1)))
    for f in feature_t:
        fout.write(str(f))
        fout.write(' ')
    fout.write(str(Y_t))
    fout.write('\n')

fout.close()

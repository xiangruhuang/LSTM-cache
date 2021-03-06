import sys, os
sys.path.append(os.path.join(sys.path[0],'../'))
import json
import numpy
from Config import Config

class Sample(object):
    def __init__(self, instr_id, time_stamp, feature, y, hacc, wakeup_id=-1):
        self.instr_id = instr_id
        self.t = time_stamp
        self.feature = feature
        self.y = y
        self.hacc = hacc
        self.wakeup_id = wakeup_id

    @classmethod
    def sample_length(cls):
        return 16

    def get_baseline_prediction(self):
        prob = float(self.feature[1])
        baseline_prediction = 1
        if prob < 0.5:
            baseline_prediction = 0
        return baseline_prediction
    
    @classmethod
    def from_line(cls, line, time_stamp):
        tokens = line.strip().split(' ')
        instr_id = int(tokens[0])
        wakeup_id = int(tokens[1])
        feature = [float(t) for t in tokens[2:-2]]
        y = int(tokens[-2])
        hacc = float(tokens[-1])
        return cls(instr_id, time_stamp, feature, y, hacc, wakeup_id)

    @classmethod
    def from_array(cls, arr, time_stamp):
        instr_id = int(arr[0])
        wakeup_id = int(arr[1])
        feature = [float(t) for t in arr[2:-2]]
        y = int(arr[-2])
        hacc = float(arr[-1])
        return cls(instr_id, time_stamp, feature, y, hacc, wakeup_id)

class Record(object):
    last_instr = 0
    def __init__(self, t, data, instr, prob, Y, hacc):
        self.t = t
        self.data = int(data, 16)

        self.instr = int(instr, 16)
        dist = self.instr - Record.last_instr
        if (dist <= 100 and dist > 0):
            self.forward = 1
        else:
            self.forward = 0
        Record.last_instr = self.instr

        self.prob = float(prob) if abs(float(prob) - 2.0) > 1e-3 else 0.5
        self.Y = int(Y)
        self.hacc = float(hacc)

class Records(object):

    def __init__(self, filename, feature_type):
        with open(filename, 'r') as fin:
            lines = fin.readlines()

        self.feature_type = feature_type
        config = Config()
        instr_set = set([int(l.strip().split(' ')[2], 16) for l in lines])
        data_set = set([int(l.strip().split(' ')[1], 16) for l in lines])
        """data_addr, instr_addr, prob, truelabel"""
        samples = [l.strip().split(' ')[1:6] for l in lines] # skip some lines

        self.T = config.history_len
        #self.code_history_len = config.code_history_len
        self.wakeup_id = []
        num_instr = len(instr_set)
        num_data = len(data_set)
        self.instr_dict = {addr:i for i, addr in enumerate(instr_set)}
        self.data_hist = {i:[] for i in data_set}
        self.instr_hist = {i:[] for i in instr_set}
        self.records = []
        self.unknown_count = 0
        self.total = 0
        self.feature = {i:[] for i in instr_set}
        self.dist_list = {i:[] for i in instr_set}
        """pos[t] = position of records[t] in instr_hist[records[t].instr]"""
        self.pos = [] 
        self.unknown = {i:0 for i in instr_set}
        self.samples = []
        #self.context = [0]*self.code_history_len
        
        samples_for_instr = {instr:[] for instr in instr_set}
        for (t, sample) in enumerate(samples):
            """Updates"""
            self.add_sample(t, sample)

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
    
    def get_feature(self, instr):
        """feature of this instruction"""
        #if self.feature_type=='feat5':
        #    r_t = self.instr_hist[instr][-1]
        #    L = len(self.feature[r_t.instr])-1
        #    pad_len = int(max(self.T - L, 0))
        #    feature_len = int(min(self.T, L))
        #    h = self.feature[r_t.instr][-(feature_len+1):-1]
        #    feature = self.context[-self.code_history_len:] + [r_t.prob] + [0.5]*(pad_len) + h
        if self.feature_type=='feat5':
            r_t = self.instr_hist[instr][-1]
            L = len(self.feature[r_t.instr])-1
            pad_len = int(max(self.T - L, 0))
            feature_len = int(min(self.T, L))
            h = self.feature[r_t.instr][-(feature_len+1):-1]
            feature = [r_t.forward] + [r_t.prob] + [0.5]*(pad_len) + h
        else:
            raise(ValueError('Not A Valid Feature Type'))
       
        return feature, r_t.Y

    def add_sample(self, t, sample):
        r = Record(t, data=sample[0], instr=sample[1], prob=sample[2],
            Y=sample[3], hacc=sample[4])
        self.records.append(r)
        
        self.data_hist[r.data].append(r)
        self.pos.append(len(self.instr_hist[r.instr]))
        self.instr_hist[r.instr].append(r)
        self.feature[r.instr].append(self.get_recent_true_label(r))
        #if r.forward == 0:
        #    self.context[-1] += 1
        #else:
        #    self.context.append(0)
        #self.unknown[r.instr] += 1
        wakeup_id = -1
        if len(self.data_hist[r.data]) > 1:
            last_record = self.data_hist[r.data][-2]
            wakeup_id = last_record.t
            offset = self.pos[last_record.t]
            self.feature[last_record.instr][offset] = last_record.Y
            #self.unknown[last_record.instr] -= 1
        feature_t, Y_t = self.get_feature(r.instr)
        #assert(len(feature_t) == self.T+2)
        self.samples.append(Sample(self.instr_dict[r.instr], r.t, feature_t,
            Y_t, r.hacc))
        self.samples[-1].wakeup_id = wakeup_id

    def dump(self, filename):
        with open(filename+'.'+'all', 'w') as fout_all:
            for sample in self.samples:
                fout_all.write(str(sample.instr_id)+' '+str(sample.wakeup_id)+' ')
                for f in sample.feature:
                    fout_all.write(str(f)+' ')
                fout_all.write(str(sample.y)+ ' ')
                fout_all.write(str(sample.hacc) + '\n')



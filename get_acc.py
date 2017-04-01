from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import sys
import numpy
import os
sys.path.append('data/')
from Record import *

arr = numpy.fromfile(sys.argv[1]+'/instr.all.bin').reshape([-1,
    Sample.sample_length()])

length = arr.shape[0]//10000*10000
arr = arr[:length, :]

num_instr=1498
num_learners=100

instr_ids = arr[:, 0].astype(int)

"""get distribution of data"""
dist = [0 for i in range(num_instr)]
for instr_id in instr_ids:
    dist[instr_id] += 1
pairs = sorted([(i, dist[i]) for i in 
    range(num_instr)], key=lambda x:x[1], reverse=True)

id_to_learner = [0 for i in range(num_instr)]

ids = [[] for i in range(num_learners)]
for i in range(num_instr):
    (instr_id, freq) = pairs[i]
    l = min(i, num_learners-1)
    ids[l].append(instr_id)
    id_to_learner[instr_id] = l

learner_id = numpy.asarray([id_to_learner[instr_ids[i]] for i in
    range(length)])
baseline_pred = [1.0 if arr[i, 3] >=0.5 else 0.0 for i in range(length)]
labels = arr[:, -2]

pred = []
for i in range(100):
    suffix = str(i)
    if (i + 1) == num_learners:
        suffix = str(i)+':'
    pred_i = []
    #with open('omnetpp/online_t10_b100_n10/pred'+suffix) as fin:
    if i <= 50:
        with open('omnetpp/'+sys.argv[2]+'/pred'+suffix) as fin:
            pred_i = [float(line.strip())for line in fin.readlines()]
    pred.append(pred_i)

st = [0 for i in range(num_learners)]

up = 0.0
down = 0.0
up_ = [0.0 for i in range(num_learners)]
down_ = [0.0 for i in range(num_learners)]
#print(len([t for t in range(length) if learner_id[t] == 87]))
for t in range(length):
    i = learner_id[t]
    #print('t=%d, i=%d, st_i=%d, len_i=%d' % (t, i, st[i], len(pred[i])))
    if st[i] < len(pred[i]):
        pred_t = pred[i][st[i]]
    else:
        pred_t = baseline_pred[t]
    st[i] += 1
    if abs(pred_t - labels[t])<=1e-3:
        up += 1.0
        up_[i] += 1.0
    down += 1.0
    down_[i] += 1.0
    #print('t=%d, i=%d, st[i]=%d' % (t, i, st[i]))

print('up=%f, down=%f' % (up, down))
for i in range(num_learners):
    print('i=%d, acc_i=%f' % (i, up_[i]/down_[i]))
print('total_acc=%f' % (up/down))

#pred = [float(line.strip().split(' ')[0]) for line in lines]
#baseline_pred = [float(line.strip().split(' ')[1]) for line in lines]
#true_label = [float(line.strip().split(' ')[2]) for line in lines]
#T = 30000
#pred[:T] = baseline_pred[:T]
##for t in range(T, len(lines)):
##    if abs(pred[t] - baseline_pred[t]) > 1e-3:
##        print(t)
##bsubs = numpy.asarray([float(line.strip().split(' ')[3]) for line in lines])
##lsubs = numpy.asarray([float(line.strip().split(' ')[4]) for line in lines])
##mask = numpy.asarray([float(line.strip().split(' ')[5]) for line in lines])
#
##for t in range(len(lines)):
##    print(pred[t], baseline_pred[t], true_label[t], lsubs[t], bsubs[t])
##    assert abs(lsubs[t] - (1.0-abs(pred[t]-true_label[t])))<=1e-3
##    assert abs(bsubs[t] - (1.0-abs(baseline_pred[t]-true_label[t])))<=1e-3
##    assert abs(mask[t] - 1.0) <= 1e-3
#pred = numpy.asarray(pred)
#baseline_pred = numpy.asarray(baseline_pred)
#true_label = numpy.asarray(true_label)
#test_acc = 1.0-numpy.mean(numpy.abs(pred-true_label))
#baseline_acc = 1.0-numpy.mean(numpy.abs(baseline_pred-true_label))
#print('test_acc=%f, baseline_acc=%f' % (test_acc, baseline_acc)) 

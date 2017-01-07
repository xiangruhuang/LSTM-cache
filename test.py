from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

#import utils
import collections
import os
import sys
import json

import tensorflow as tf

from flags import *
from Reader import *
from LSTMModel import *
from Config import *

def main(_):
    """Initialization"""
    if not FLAGS.data_path:
        raise ValueError("Set data path with '--data_path=<data_path>' ")
    config = Config()
    config.num_instr=int(FLAGS.num_instr)

    if not (os.path.exists(FLAGS.data_path+'.train') and os.path.exists(FLAGS.data_path+'.test') and os.path.exists(FLAGS.data_path+'.valid')):
        split_file(config, FLAGS.data_path, 10000, [3, 1, 0])
    basename = (FLAGS.data_path).split('/')[-1].split('.')[0]

    print('dataset=', basename, ', #instr=', config.num_instr)
    instr_map = json.load(open(FLAGS.data_path+'.instr_map', 'r'))
    assert(config.num_instr == len(instr_map))
    data_map = json.load(open(FLAGS.data_path+'.data_map', 'r'))
    #train_reader = Reader(FLAGS.data_path+'.train', config, instr_map, shuffle=True)
    #test_reader = Reader(FLAGS.data_path+'', config, instr_map, shuffle=False)
    model = LSTMModel(config)

    """Define Computation Graph"""
    init = tf.global_variables_initializer()
   
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        #restore LSTM model
        saver.restore(sess, tf.train.latest_checkpoint(basename+'/'))
        test_acc = 0.0
        fin = open('./data/dnn_ordered_traces/'+basename, 'r')
        samples = [l.strip().split()[1:5] for l in fin.readlines()]

        pred = []
        Y = []
        addrs = []
        T = config.input_dim-1
        data = []
        instr = [] # instr[t] is the t-th sample's instruction address's id
        prob = [] 
        offset = [] # for t-th sample, t will be appended at the history list of instr_t, so we need to track
        hist = {i:[] for i in range(config.num_instr)}
        last_visit = {i:-1 for i in range(len(data_map))}
        visible = [] # whether the OPTGEN's output for t-th sample is visible to scheduler (is it computed yet?)
        
        err = 0
        #for output
        fout = open(basename+".pred_real", 'w')
        for (t, sample) in enumerate(samples):
            #s[0] is data addr string, s[1] is instr addr string, s[2] is prob, s[3] is true label
            data_t = data_map[sample[0]] # data address's id
            instr_t = instr_map[sample[1]] # instr address's id
            prob_t = float(sample[2]) # cache hit probability
            data.append(data_t)
            instr.append(instr_t)
            prob.append(prob_t)
            if int(prob_t) == 2:
                prob_t = 0.5
            """Since we're accessing <data address>=data_t, OPTGEN can output true label for the last visit of data_t"""
            last_t = last_visit[data_t]
            if last_t >= 0:
                last_instr = instr[last_t]
                visible[last_t] = True
                hist[last_instr][offset[last_t]] = Y[last_t] # Y[last_t] is OPTGEN's result (true label) for last_t-th sample, is visible now.
            """generate feature vector for <time>=t, <instruction>=instr_t"""
            h = hist[instr_t]

            l1 = min(T, len(h))
            l2 = max(0, T - len(h))
            feature = [prob_t] + [0.5] * l2 + h[-l1:]
            assert(len(feature) == T+1)
    
            #initialize model, you can adjust this
            if t % config.num_steps == 0:
                model.inite(sess)

            #get prediction
            output = model.predict(sess, feature)
            output = int(numpy.round(numpy.squeeze(output)))

            err += abs(output-float(sample[3]))
            if (t+1) % 10000 == 0:
                print(str(t)+':\tacc='+str(1.0 - float(err)/float(t)))
            """Add t to instr_t's history, but for now OPTGEN's result is invisible, it becomes visible when the same data_address is accessed again"""
            visible.append(False)
            offset.append(len(hist[instr_t]))
            hist[instr_t].append(0.5) # <time step>=t, <true_label>=unknown (for now)
            last_visit[data_t] = t
            Y_t = int(sample[3])
            Y.append(Y_t)
            pred.append(output)
            fout.write(sample[0])
            fout.write(' ')
            fout.write(sample[1])
            fout.write(' ')
            fout.write(str(output))
            fout.write('\n')

        fout.close()
        pred = numpy.asarray(pred)
        Y = numpy.asarray(Y)
        
        #compute accuracy
        test_acc = numpy.mean(1.0 - abs(pred-Y))

        print("\tTest acc=%f" % test_acc)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()

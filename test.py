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
    train_reader = Reader(FLAGS.data_path+'.train', config, instr_map, shuffle=True)
    test_reader = Reader(FLAGS.data_path+'.test', config, instr_map, shuffle=False)
    model = LSTMModel(config, train_reader, test_reader)

    """Define Computation Graph"""
    init = tf.initialize_all_variables()
   
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        #restore LSTM model
        saver.restore(sess, tf.train.latest_checkpoint(basename+'/'))
        #for output
        fout = open(basename+".pred", 'w')
        test_acc = 0.0
        fin = open(FLAGS.data_path+'.test', 'r')
        data = [l.strip().split()[1:4] for l in fin.readlines()]
        pred = []
        Y = []
        for (i, d) in enumerate(data):
            #d[0] is instr addr, d[1] is prob, d[2] is true label
            d[0] = instr_map[d[0]]

            #initialize model, you can adjust this
            if i % config.num_steps == 0:
                model.inite(sess)
                
            #get prediction
            output = model.predict(sess, d[0], d[1])
            output = int(numpy.round(numpy.squeeze(output)))

            #store prediction and truelabel
            pred.append(output)
            Y.append(int(d[2]))
        pred = numpy.asarray(pred)
        Y = numpy.asarray(Y)
        
        #compute accuracy
        test_acc = numpy.mean(1.0 - abs(pred-Y))

        print("\tTest acc=%f" % test_acc)
        
        #output prediction to file
        for p in pred:
            fout.write(str(int(p)))
            fout.write('\n')
        fout.close()

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()

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
import time
from tensorflow.python.platform import gfile
from subprocess import call

def main(_):
    """Initialization"""
    if not FLAGS.data_path:
        raise ValueError("Set data path with '--data_path=<data_path>' ")
    if not (os.path.exists(FLAGS.data_path)):
        """ should extract the feature """
        raise ValueError("data_path: %s does not exist" % FLAGS.data_path)
    if not (os.path.exists(FLAGS.model_path)):
        raise ValueError("model_path: %s does not exist" % FLAGS.model_path)
    """e.g. foldername=gcc.feat2"""
    """     basename=gcc        """
    """     feattype=feat2      """
    foldername = (FLAGS.data_path).split('/')[-1]
    basename = foldername.split('.')[0]
    feattype = foldername.split('.')[1]
    if not (os.path.exists(FLAGS.data_path+'.train') and os.path.exists(FLAGS.data_path+'.test') and os.path.exists(FLAGS.data_path+'.valid')):
        #raise ValueError(".train/.test/.valid file(s) missing")
        config = Config(feattype=feattype)
        split_file(config, FLAGS.data_path, 10000, [3, 1, 0])
    """Split train&test into separate files"""
    #call(['rm', '-f', FLAGS.data_path+'.test.*'])
    #call(['./split.sh', FLAGS.data_path+'.test'])

    """     check if feature extracted  """

    C = tf.Graph()
    graph_name = '/graph.meta'
    with C.as_default():
        #if os.path.exists(foldername+graph_name):
        #    with gfile.FastGFile(foldername+graph_name, 'rb') as f:
        #        graph_def = tf.GraphDef()
        #        graph_def.ParseFromString(f.read())
        #        print("loading graph from " + foldername + graph_name + ' ....', end="")
        #        tf.train.import_meta_graph(foldername + graph_name)
        #        print("done")
        #else:
        with tf.name_scope('config'):
            config = Config(feattype=feattype)
            instr_map = None #json.load(open(FLAGS.data_path+'.instr_map', 'r'))

        print('dataset=', basename)
        print(config.to_string())
        
        """Define Computation Graph"""
        print("Building Computation Graph")
        #READ = tf.Graph()
        #with READ.as_default():
        with tf.name_scope('readers'):
            #train_reader = Reader(FLAGS.data_path+'.train', config, instr_map, shuffle=True)
            test_reader = Reader(FLAGS.data_path+'.test', config, instr_map, shuffle=False)
        #with tf.Session(graph=READ) as sess:
        #    sess.run(tf.global_variables_initializer())
        
        #MODEL = tf.Graph()
        #with MODEL.as_default():
        with tf.name_scope('LSTM_model'):
            model = LSTMModel(config)
            packed = tf.placeholder("float", [config.batch_size, config.sample_dim])
            vals = model.run_batch(packed)
            saver = tf.train.Saver(tf.trainable_variables())
            init = tf.global_variables_initializer()
        print([(v.name, v.get_shape()) for v in tf.trainable_variables()])
        
        #with tf.Session(graph=MODEL) as sess:
        #    sess.run(init)
        
        #print("creating new computation graph, saved as " + foldername + graph_name + " ....", end="")
        #tf.train.export_meta_graph(foldername + graph_name)
        #print("done")
        #merged = tf.summary.merge_all()
        C.finalize() 

    best_acc = 0.0
    countdown = 0

    #with tf.Session(graph=C) as sess:
    #    sess.run(init)
    #    print("Initialization Done")
    ##with tf.Session() as sess:
    #
    #    #if FLAGS.is_training:
    #    current_epoch = 0
    #    #if not (tf.train.latest_checkpoint('./'+foldername+'/') is None):
    #        #lc = tf.train.latest_checkpoint('./'+foldername+'/')
    #        #print('restoring from ' + lc + '...')
    #        #saver.restore(sess, lc)
    #        #current_epoch = int(lc.split('-')[-1]) + 1

    #conf_proto = tf.ConfigProto()
    #conf_proto.gpu_options.allow_growth = True

    with tf.Session(graph=C) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #train_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train', sess.graph)
        #test_writer = tf.summary.FileWriter(FLAGS.log_dir+'/test', sess.graph)
        current_epoch = 0
        if not (tf.train.latest_checkpoint(FLAGS.model_path) is None):
            lc = tf.train.latest_checkpoint(FLAGS.model_path)
            print('restoring from ' + lc + ' ....', end="")
            saver.restore(sess, lc)
            print("done")
            current_epoch = int(lc.split('-')[-1]) + 1
        else:
            raise ValueError("No Available Checkpoint.")
        fout = open(foldername+".pred", 'w')
        foutY = open(foldername+".truelabel", 'w')
        test_acc = 0.0
        for i in range(test_reader.num_batches):
            test_packed = numpy.asarray(sess.run(test_reader.packed))
            test_acc_i, pred_i, Y_i = sess.run([vals['acc'], vals['pred'], vals['Y']], feed_dict={packed:test_packed})
            Y_i = numpy.asarray(Y_i)
            pred_i = numpy.asarray(pred_i)
            alt_acc = 1.0 - numpy.mean(abs(Y_i - pred_i))
            test_acc += test_acc_i
            pred_i = numpy.transpose(pred_i).reshape([config.num_steps*config.batch_size])
            Y_i = numpy.transpose(Y_i).reshape([config.num_steps*config.batch_size])
            print('\ttest_acc=%f' % test_acc_i)
            for q in range(pred_i.size):
                fout.write( str(int( numpy.round(pred_i[q]) )))
                fout.write('\n')
            for q in range(Y_i.size):
                foutY.write( str(int( numpy.round(Y_i[q]) )))
                foutY.write('\n')
        test_acc /= test_reader.num_batches
        print("\tTest acc=%f" % test_acc)
        foutY.close()
        fout.close()
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()

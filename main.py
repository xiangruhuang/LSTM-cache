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
    call(['rm', '-f', FLAGS.data_path+'.train.*'])
    call(['./split.sh', FLAGS.data_path+'.train'])

    call(['rm', '-f', FLAGS.data_path+'.test.*'])
    call(['./split.sh', FLAGS.data_path+'.test'])

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
            config.num_instr=int(FLAGS.num_instr)
            instr_map = None #json.load(open(FLAGS.data_path+'.instr_map', 'r'))

        print('dataset=', basename, ', #instr=', config.num_instr)
        print(config.to_string())
        
        """Define Computation Graph"""
        print("Building Computation Graph")
        #READ = tf.Graph()
        #with READ.as_default():
        with tf.name_scope('readers'):
            train_reader = Reader(FLAGS.data_path+'.train', config, instr_map, shuffle=True)
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
        print("models will be saved under ./"+foldername)
        if not (tf.train.latest_checkpoint('./'+foldername+'/') is None):
            lc = tf.train.latest_checkpoint('./'+foldername+'/')
            print('restoring from ' + lc + ' ....', end="")
            saver.restore(sess, lc)
            print("done")
            current_epoch = int(lc.split('-')[-1]) + 1
        for e in range(current_epoch, config.max_epoch):
            print('epoch %d...' % e)
            train_acc = 0.0
            states = []
            pred = []
            cost = 0.0
            #config.learning_rate *= config.lr_decay
            grad_norm = 0
            #step_size = 1000
            for i in range(train_reader.num_batches):
                #reopen
                #init_time -= time.time()
                #if i % step_size == 0:
                #    sess = tf.Session(graph=C, config=conf_proto)
                #    sess.run(init)
                #    if not (tf.train.latest_checkpoint('./'+foldername+'/') is None):
                #        lc = tf.train.latest_checkpoint('./'+foldername+'/')
                #        print('restoring from ' + lc + '...')
                #        saver.restore(sess, lc)
                #        print('Done restoring')
                #    coord = tf.train.Coordinator()
                #    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                #init_time += time.time()

                #with tf.Session(graph=C) as sess:
                #    sess.run(init)
                #    if not (tf.train.latest_checkpoint('./'+foldername+'/') is None):
                #        lc = tf.train.latest_checkpoint('./'+foldername+'/')
                #        print('restoring from ' + lc + '...')
                #        saver.restore(sess, lc)
                #        print('Done restoring')
                #    coord = tf.train.Coordinator()
                #    threads = tf.train.start_queue_runners(coord=coord)
                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()

                train_packed = numpy.asarray(sess.run(train_reader.packed))#, options=run_options, run_metadata=run_metadata))

                train_vals, weights = sess.run([vals, model.weights['out']], feed_dict={packed:train_packed})
                grad = train_vals['grad']
                grad_norm = 0.0
                for gv in grad:
                    #print(gv[1])
                    #print("Variable=", gv[1], ", shape=", numpy.asarray(gv[0]).shape)
                    gg = numpy.asarray(gv)
                    gg = numpy.reshape(gv, [-1])
                    grad_norm += numpy.linalg.norm(gg, 2)
                #index = e*train_reader.num_batches+i
                #train_writer.add_run_metadata(run_metadata, 'batch%d' % index)
                #train_writer.add_summary(summary, index)

                pred=train_vals['pred']
                Y = train_vals['Y']
                alt_acc = 1.0 - numpy.mean(abs(numpy.asarray(pred)-numpy.asarray(Y)))
                assert(abs(alt_acc - train_vals['acc']) < 1e-3)
                train_acc += train_vals['acc']
                cost += train_vals['cost']
                print('train_acc=%f, grad_norm=%f' % (train_vals['acc'], grad_norm))

                #close
                #if ((i % step_size == step_size-1) or i == train_reader.num_batches-1):
                #    #write model to file, close session and reopen, re-initialize
                #    print("saving model...")
                #    saver.save(sess, './'+foldername+'/ckpt', global_step=e)
                #    coord.request_stop()
                #    coord.join(threads)
                #    sess.close()
                #init_time += time.time()

            train_acc /= train_reader.num_batches
            print("\tTrain acc=%f, cost=%f, grad_norm=%f" % (train_acc, cost, grad_norm))

            #testing_time -= time.time()
            fout = open(foldername+"/pred@"+str(e), 'w')
            foutY = open(foldername+"/truelabel@"+str(e), 'w')
            test_acc = 0.0
            #sess = tf.Session(graph=C, config=conf_proto)
            #sess.run(init)
            #coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            #lc = tf.train.latest_checkpoint('./'+foldername+'/')
            #assert( not (lc is None) )
            #saver.restore(sess, lc)
            for i in range(test_reader.num_batches):
                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()
                test_packed = numpy.asarray(sess.run(test_reader.packed))
                test_acc_i, pred_i, Y_i = sess.run([vals['acc'], vals['pred'], vals['Y']], feed_dict={packed:test_packed})
                #index = e*test_reader.num_batches+i
                #test_writer.add_run_metadata(run_metadata, 'batch%d' % index)
                #test_writer.add_summary(summary, index)
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
            #coord.request_stop()
            #coord.join(threads)
            #sess.close()
            test_acc /= test_reader.num_batches
            foutY.close()
            fout.close()
            #testing_time += time.time()
            #print("Init=%f, Reading=%f, Training=%f, Testing=%f" % (init_time, reading_time, training_time, testing_time))
                #print("saving model...")
            saver.save(sess, './'+foldername+'/ckpt', global_step=e)
            if test_acc > best_acc:
                best_acc = test_acc
                countdown = 0
            else:
                countdown += 1
            print("\tTest acc=%f, Best acc=%f, countdown=%d" % (test_acc, best_acc, countdown))
            if countdown >= 30:
                break
        coord.request_stop()
        coord.join(threads)
    #train_writer.close()
    #test_writer.close()
    #else:
    #    #restore LSTM model
    #    saver.restore(sess, tf.train.latest_checkpoint('./'+foldername+'/'))
    #    #for output
    #    fout = open(foldername+".pred2", 'w')
    #    foutY = open(foldername+".truelabel2", 'w')
    #    test_acc = 0.0
    #    for i in range(test_reader.num_batches):
    #        test_packed = numpy.asarray(sess.run(test_reader.packed))
    #        test_acc_i, pred_i, Y_i = sess.run([vals['acc'], vals['pred'], vals['Y']], feed_dict={packed:test_packed})
    #        Y_i = numpy.asarray(Y_i)
    #        pred_i = numpy.asarray(pred_i)
    #        alt_acc = 1.0 - numpy.mean(abs(Y_i - pred_i))
    #        test_acc += test_acc_i
    #        pred_i = numpy.transpose(pred_i).reshape([config.num_steps*config.batch_size])
    #        Y_i = numpy.transpose(Y_i).reshape([config.num_steps*config.batch_size])
    #        for x in pred_i:
    #            fout.write( str(int( numpy.round(x) )))
    #            fout.write('\n')
    #        for q in Y_i:
    #            foutY.write( str(int( numpy.round(x) )))
    #            foutY.write('\n')

    #    test_acc /= test_reader.num_batches
    #    print("\tTest acc=%f" % test_acc)
    #    foutY.close()
    #    fout.close()
                #coord.request_stop()
                #coord.join(threads)


if __name__ == "__main__":
    tf.app.run()

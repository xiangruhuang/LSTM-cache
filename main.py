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
    print(config.to_string())
    instr_map = json.load(open(FLAGS.data_path+'.instr_map', 'r'))
    train_reader = Reader(FLAGS.data_path+'.train', config, instr_map, shuffle=True)
    test_reader = Reader(FLAGS.data_path+'.test', config, instr_map, shuffle=False)
    model = LSTMModel(config, train_reader, test_reader)

    """Define Computation Graph"""
    packed = tf.placeholder("float", [config.batch_size, config.sample_dim])
    is_training = tf.placeholder(tf.bool)
    vals = model.run_batch(packed)
    #test = model.run_batch(packed, test_reader, False, model.cell)
    init = tf.initialize_all_variables()
   
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        best_acc = 0.0
        countdown = 0
        if FLAGS.is_training:
            for e in range(config.max_epoch):
                print('epoch %d...' % e)
                train_acc = 0.0
                states = []
                pred = []
                cost = 0.0
                config.learning_rate *= config.lr_decay
                grad = []
                grad_norm = 0
                for i in range(train_reader.num_batches):
                    train_packed = numpy.asarray(sess.run(train_reader.packed))
                    train_vals = sess.run(vals, feed_dict={packed:train_packed})
                    pred=train_vals['pred']
                    Y = train_vals['Y']
                    #print("======================================================")
                    #print(numpy.asarray(pred).shape)
                    #print("======================================================")
                    #print(numpy.asarray(train_vals['Y']).shape)
                    alt_acc = 1.0 - numpy.mean(abs(numpy.asarray(pred)-numpy.asarray(Y)))
                    #print("======================================================")
                    #print(alt_acc, ' ', train_vals['acc'])
                    #assert(abs(alt_acc - train_vals['acc']) < 1e-3)
                    train_acc += train_vals['acc']
                    cost += train_vals['cost']
                    grad += train_vals['grad']
                    #if i % 100 == 0:
                        #print(i, '/', train_reader.num_batches)
                    #    print('cost=%f, acc=%f, grad_norm=%f' % (train_vals['cost'], train_acc/float(i), nm))
                for gv in grad:
                    gg = numpy.asarray(gv)
                    gg = numpy.reshape(gv, [-1])
                    grad_norm += numpy.linalg.norm(gg, 2)
                train_acc /= train_reader.num_batches
                print("\tTrain acc=%f, cost=%f, grad_norm=%f" % (train_acc, cost, grad_norm))

                fout = open(basename+"/pred@"+str(e), 'w')
                foutY = open(basename+"/truelabel@"+str(e), 'w')
                test_acc = 0.0
                for i in range(test_reader.num_batches):
                    test_packed = numpy.asarray(sess.run(test_reader.packed))
                    #print(test_packed)
                    test_acc_i, pred_i, Y_i = sess.run([vals['acc'], vals['pred'], vals['Y']], feed_dict={packed:test_packed})
                    Y_i = numpy.asarray(Y_i)
                    pred_i = numpy.asarray(pred_i)
                    alt_acc = 1.0 - numpy.mean(abs(Y_i - pred_i))
                    #print(alt_acc, ' ', test_acc_i)
                    #assert(abs(alt_acc - test_acc_i) < 1e-5)
                    test_acc += test_acc_i
                    pred_i = numpy.transpose(pred_i).reshape([config.num_steps*config.batch_size])
                    Y_i = numpy.transpose(Y_i).reshape([config.num_steps*config.batch_size])
                    #print('Shapes:', Y_i.shape, ' ', pred_i.shape)
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
                saver.save(sess, './'+basename+'/ckpt', global_step=e)
                if test_acc > best_acc:
                    best_acc = test_acc
                else:
                    countdown += 1
                    if countdown >= 3:
                        break
        else:
            #restore LSTM model
            saver.restore(sess, tf.train.latest_checkpoint(basename+'/'))
            #for output
            fout = open(basename+".pred", 'w')
            test_acc = 0.0
            fin = open(FLAGS.data_path+'.test', 'r') 
            data = [l[1:4] for l in fin.readlines()]
            pred = []
            Y = []
            for (i, d) in enumerate(data):
                d[0] = instr_map[d[0]]
                #d[0] is instr addr, d[1] is prob, d[2] is true label

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

            #for i in range(test_reader.num_batches):
            #    test_packed = numpy.asarray(sess.run(test_reader.packed))
            #    
            #    test_packed = numpy.split(test_packed, config.num_steps, axis=1)
            #    counter = 0
            #    pred_i = []
            #    Y_i = []
            #    for b in range(config.batch_size):
            #        model.inite(sess)
            #        for t in range(config.num_steps):
            #            s_t = test_packed[t][b, :]
            #            #print(s_t.shape)
            #            output = model.predict(sess, s_t[0], s_t[1])
            #            output = numpy.squeeze(output)
            #            #fout.write(str(int(numpy.round(output)))+'\n')
            #            pred_i.append(int(numpy.round(output)))
            #            #foutY.write(str(int(numpy.round(s_t[2])))+'\n')
            #            Y_i.append(int(numpy.round(s_t[2])))
            #            #sys.stdout.write('.')
            #    pred_i = numpy.asarray(pred_i)
            #    Y_i = numpy.asarray(Y_i)
            #    test_acc += numpy.mean(1.0 - abs(pred_i - Y_i)) #(1.0-abs(numpy.round(output)-s_t[2]))
            #    sys.stdout.write('\n')
            #    for q in range(pred_i.size):
            #        fout.write( str(int( numpy.round(pred_i[q]) )))
            #        fout.write('\n')
            #    #for q in range(Y_i.size):
            #    #    foutY.write( str(int( numpy.round(Y_i[q]) )))
            #    #    foutY.write('\n')
            #
            #test_acc /= test_reader.num_batches
            
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

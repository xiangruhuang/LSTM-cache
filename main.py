from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import os

import sys

from utils import *

import tensorflow as tf
import numpy
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple

from rnn_utils import *

from flags import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device

from LSTMModel import *
from Config import *
from Reader import Reader
from Model import Model


#"""Generate random batch from data.
#Inputs:
#    samples: 
#    features: has shape [batch_size*num_steps, input_dim]
#    labels: has shape [batch_size*num_steps, output_dim]
#    batch_size:
#    num_steps:
#    ids
#Outputs:
#    feature_batch: has shape [num_steps, batch_size, input_dim]
#    label_batch: has shape [num_steps, batch_size, output_dim]
#    condition_batch: has shape [num_steps, batch_size, num_learners]
#"""
#"""output has shape [num_steps, batch_size, input_dim]"""
#
#def get_random_batch(samples, batch_size, num_steps, instr_ids, part, split):
#    batch_len = batch_size*num_steps
#    l = len(samples)
#    upper = l//batch_len
#    mid = get_split(upper, split)
#    if part == 'train':
#        start = numpy.random.randint(0, mid)*batch_len
#    elif part == 'test':
#        start = numpy.random.randint(mid, upper)*batch_len
#    else:
#        raise(ValueError("part must be either 'train' or 'test'"))
#    end = start + batch_len
#    subsamples = samples[start:end]
#    features = [sample.feature for sample in subsamples]
#    labels = [[sample.y] for sample in subsamples]
#    #baseline_predictions = []
#    #for sample in subsamples:
#    #    if sample.feature[1] < 0.5:
#    #        baseline_predictions.append([0])
#    #    else:
#    #        baseline_predictions.append([1])
#    
#    baseline_ups = [0.0 for i in range(len(instr_ids))]
#    baseline_downs = [0.0 for i in range(len(instr_ids))]
#    for i, id_subset in enumerate(instr_ids):
#        for s, sample in enumerate(subsamples):
#            if sample.instr_id in id_subset:
#                baseline_downs[i] += 1.0
#                pred = 0
#                if sample.feature[1] >= 0.5:
#                    pred = 1
#                if pred == sample.y:
#                    baseline_ups[i] += 1.0
#    
#    hawkeye_ups = [0.0 for i in range(len(instr_ids))]
#    hawkeye_downs = [0.0 for i in range(len(instr_ids))]
#    for i, id_subset in enumerate(instr_ids):
#        for s, sample in enumerate(subsamples):
#            if sample.instr_id in id_subset:
#                hawkeye_downs[i] += 1.0
#                if int(sample.hacc) == 1:
#                    hawkeye_ups[i] += 1.0
#
#    conditions = [[bool(sample.instr_id in id_subset) for id_subset in
#        instr_ids] for sample in subsamples]
#    #print('loads=', [numpy.mean(
#    #    [1.0 if sample.instr_id in id_subset else 0.0 for sample in subsamples]) 
#    #    for id_subset in instr_ids])
#    """feature has shape [batch_size*num_steps, input_dim]"""
#    """feature_batch has shape [num_steps, batch_size, input_dim]"""
#    feature_batch = numpy.transpose(numpy.array_split(features, batch_size),
#        [1,0,2])
#    """label_batch has shape [num_steps, batch_size, output_dim]"""
#    label_batch = numpy.transpose(numpy.array_split(labels, batch_size),
#        [1,0,2])
#    """condition_batch has shape [num_steps, batch_size, len(instr_ids)]"""
#    condition_batch = numpy.transpose( numpy.array_split(conditions,
#        batch_size), [1,0,2])
#    #print('feature batch has shape %s', feature_batch.shape)
#    #print('label   batch has shape %s', label_batch.shape)
#    #print('condition batch has shape %s', condition_batch.shape)
#    return feature_batch, label_batch, condition_batch, baseline_ups, \
#        baseline_downs, hawkeye_ups, hawkeye_downs

#def get_ups_and_downs(instr_ids, label_batch, condition_batch, pred_batch):
#    ups = [0.0 for ids in range(len(instr_ids))]
#    downs = [0.0 for ids in range(len(instr_ids))]
#    num_steps = len(label_batch)
#    batch_size = len(label_batch[0])
#    for t in range(num_steps):
#        for b in range(batch_size):
#            subacc = 1.0 - abs(label_batch[t, b, 0] - pred_batch[t, b, 0])
#            for i in range(len(instr_ids)):
#                if condition_batch[t, b, i] == True:
#                    ups[i] += subacc
#                    downs[i] += 1.0
#    return ups, downs

#def run_batch(config, num_batches, name, global_features, labels, conditions,\
#        samples, ids, sess, evals, split, reader, \
#        summary_writer=None, num_prints=None):
#
#    if (num_prints is None) or (num_prints == 0):
#        print_period = num_batches+1
#    else:
#        print_period = num_batches//num_prints
#
#    st = time.time()
#    baseline_ups = numpy.asarray([0.0 for i in
#        range(config.num_learners)])
#    baseline_downs = numpy.asarray([0.0 for i in
#        range(config.num_learners)])
#    hawkeye_ups = numpy.asarray([0.0 for i in
#        range(config.num_learners)])
#    hawkeye_downs = numpy.asarray([0.0 for i in
#        range(config.num_learners)])
#    lstm_ups = numpy.asarray([0.0 for i in range(config.num_learners)])
#    lstm_downs = numpy.asarray([0.0 for i in
#        range(config.num_learners)])
#
#    for Iter in range(num_batches):
#        """output has shapes [num_steps, batch_size, XXX]"""
#        if reader=='random':
#            global_feature_data, label_data, condition_data, \
#            baseline_up, baseline_down, hawkeye_up, hawkeye_down = \
#            get_random_batch(samples, config.batch_size,
#                config.num_steps, ids, name, split)
#        else:
#            global_feature_data, label_data, condition_data, \
#            baseline_up, baseline_down, hawkeye_up, hawkeye_down = \
#            get_artificial_batch(samples, config.batch_size,
#                config.num_steps, ids, name, split)
#        
#        #if summary_writer is not None:
#        #    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#        #    #run_metadata = tf.RunMetadata()
#        
#        values = sess.run(evals, feed_dict={ global_features:global_feature_data
#            , labels:label_data, conditions:condition_data})
#        
#        num_samples = values['num_samples']
#        if summary_writer is not None:
#            #summary_writer.add_run_metadata(run_metadata, )
#            summary_writer.add_summary(values['summary'], num_samples)
#        
#        baseline_ups += numpy.asarray(baseline_up)
#        baseline_downs += numpy.asarray(baseline_down)
#        baseline_acc = float(baseline_ups.sum())/float(baseline_downs.sum())
#
#        hawkeye_ups += numpy.asarray(hawkeye_up)
#        hawkeye_downs += numpy.asarray(hawkeye_down)
#        hawkeye_acc = float(hawkeye_ups.sum())/float(hawkeye_downs.sum())
#
#        lstm_up, lstm_down = get_ups_and_downs(ids, label_data,
#                condition_data, values['pred'])
#        lstm_ups += numpy.asarray(lstm_up)
#        lstm_downs += numpy.asarray(lstm_down)
#        lstm_acc = float(lstm_ups.sum())/float(lstm_downs.sum())
#
#        if (Iter+1) % print_period == 0:
#            ed = time.time()
#            print('%sing:\titer=%d, num_samples=%d, lstm acc=%f, ' %  \
#                (name, Iter, num_samples, lstm_acc), end="")
#            print('baseline acc=%f, hawkeye acc=%f, elapsed time=%f' % \
#                (baseline_acc, hawkeye_acc, (ed-st)))
#            print('baseline\t\thawkeye\t\tlstm')
#            for i in range(config.num_learners):
#                print('%.5f\t%.5f\t%.5f' % (baseline_ups[i]/baseline_downs[i],
#                    hawkeye_ups[i]/hawkeye_downs[i],
#                    lstm_ups[i]/lstm_downs[i]))
#            st = ed
#
#    lstm_acc = float(lstm_ups.sum())/float(lstm_downs.sum())
#    hawkeye_acc = float(hawkeye_ups.sum())/float(hawkeye_downs.sum())
#    baseline_acc = float(baseline_ups.sum())/float(baseline_downs.sum())
#    num_samples = num_batches * config.num_steps * config.batch_size
#    print('%sing:\tnum_samples=%d, lstm acc=%f' 
#            % (name, num_samples, lstm_acc), end="")
#    print(', hawkeye acc=%f, baseline acc=%f' % (hawkeye_acc, baseline_acc))
#    print('baseline\t\thawkeye\t\tlstm')
#    for i in range(config.num_learners):
#        print('%.5f\t%.5f\t%.5f' % (baseline_ups[i]/baseline_downs[i],
#            hawkeye_ups[i]/hawkeye_downs[i],
#            lstm_ups[i]/lstm_downs[i]))


def main(_):
    """Initialization"""
    if not FLAGS.data_path:
        raise ValueError("Set data path with '--data_path=<data_path>' ")
    """e.g. FLAGS.data_path=/home/xiangru/Projects/LSTM-cache/data/gcc.feat2"""
    """e.g. foldername=gcc.feat2"""
    """     basename=gcc        """
    """     feattype=feat2      """
    foldername = (FLAGS.data_path).split('/')[-1]
    basename = foldername.split('.')[0]
    feattype = foldername.split('.')[1]
    model_dir = FLAGS.model_dir
    
    #configProto = tf.ConfigProto(allow_soft_placement=True)
    #configProto.gpu_options.allow_growth=True

    config = Config(feattype=feattype, FLAGS=FLAGS)

    print('dataset=', basename, ', #instr=', config.num_instr)

    model = Model(config)
             
    with tf.Session(graph=model.graph) as sess:
        if config.mode == 'offline':
            model.offline(sess)
        else:
            model.online(sess)
    
#    train_summary_writer = tf.summary.FileWriter(
#            FLAGS.model_dir+'/tensorboard/train')
#    test_summary_writer = tf.summary.FileWriter(
#            FLAGS.model_dir+'/tensorboard/test')
#    for learner, id_set in zip(model.learners, reader.ids):
#        learner.instr_id_set = id_set
#
#        acc_sum = 0.0
#        current_epoch = -1
#        if FLAGS.is_training:
#            lc = tf.train.latest_checkpoint(model_dir+'/')
#            if lc is not None:
#                print("restoring model from "+model_dir+'/')
#                saver.restore(sess, lc)
#                current_epoch = int(str(lc).split('ckpt-')[-1])
#        else:
#            lc = tf.train.latest_checkpoint(model_dir+'/')
#            assert not (lc is None) 
#            saver.restore(sess, lc)
#
#        num_batches = len(samples)//(config.batch_size*config.num_steps)
#        num_test_batches = (num_batches - get_split(num_batches,
#            FLAGS.split))//config.max_epoch*4
#        num_train_batches = get_split(num_batches,
#                FLAGS.split)//config.max_epoch
#        print('starting with epoch %d ....' % current_epoch)
#        
#        """copy weights to the other lstm"""
#        with tf.variable_scope('') as scope:
#            scope.reuse_variables()
#            w0 = tf.get_variable('local0'+'/inputs_'+
#                    str(config.local_params.input_dim)
#                    +'/multi_rnn_cell/cell_0/basic_lstm_cell/weights')
#            b0 = tf.get_variable('local0'+'/inputs_'+
#                    str(config.local_params.input_dim)
#                    +'/multi_rnn_cell/cell_0/basic_lstm_cell/biases')
#            ow0 = tf.get_variable('local0'+'/outputs_'+
#                    str(config.local_params.output_dim)
#                    +'/out_weights')
#            ob0 = tf.get_variable('local0'+'/outputs_'+
#                    str(config.local_params.output_dim)
#                    +'/out_biases')
#            assign_ops = []
#            for i in range(1, config.num_learners):
#                w = tf.get_variable('local'+str(i)+'/inputs_'+
#                        str(config.local_params.input_dim)
#                        +'/multi_rnn_cell/cell_0/basic_lstm_cell/weights')
#                b = tf.get_variable('local'+str(i)+'/inputs_'+
#                        str(config.local_params.input_dim)
#                        +'/multi_rnn_cell/cell_0/basic_lstm_cell/biases')
#                ow = tf.get_variable('local'+str(i)+'/outputs_'+
#                        str(config.local_params.output_dim)
#                        +'/out_weights')
#                ob = tf.get_variable('local'+str(i)+'/outputs_'+
#                        str(config.local_params.output_dim)
#                        +'/out_biases')
#                assign_ops.append(w.assign(w0))
#                assign_ops.append(b.assign(b0))
#                assign_ops.append(ow.assign(ow0))
#                assign_ops.append(ob.assign(ob0))
#    
#        """Pretrain"""
#        #if current_epoch == -1:
#        #    for pretrain in range(3):
#        #        print('pretraining %d' % pretrain)
#        #        """pretraining"""
#        #        run_batch(config, num_train_batches, 'train', global_features,
#        #                labels, conditions, samples, ids, sess, train_evals,
#        #                FLAGS.split, 'artificial',
#        #                summary_writer=train_summary_writer, num_prints=0)
#
#        #        sess.run(assign_ops)
#        #        
#        #        """testing"""
#        #        run_batch(config, num_test_batches, 'test', global_features, labels,
#        #                conditions, samples, ids, sess, test_evals, FLAGS.split,
#        #                'random', summary_writer=test_summary_writer, num_prints=0)
#       
#        """Simulation"""
#        transfered_samples = []
#        prep_time = 0.0
#        train_time = 0.0
#        predict_time = 0.0
#        
#        online_evals = {}
#        local_pos = {}
#
#        #prepare feature_tensor
#
#        for t, sample in enumerate(samples):
#            prep_time -= time.time()
#            label = sample.y
#            baseline_pred = 0.0
#            if sample.feature[1] >= 0.5:
#                baseline_pred = 1.0
#            instr_id = sample.instr_id
#            
#            #global_feature_t = [[sample.feature]]
#            #local_feature_t = sess.run(local_input,
#            #        feed_dict={global_features:global_feature_t})
#
#            if sample.wakeup_id != -1:
#                tf.scatter_update(label_ts, tf.constant([sample.wakeup_id],
#                    tf.int64), [label])
#                """Adding Samples to One Learner"""
#                #s = local_pos[sample.wakeup_id]
#                #last_sample = samples[sample.wakeup_id]
#                #last_instr_id = last_sample.instr_id
#                #for i, learner in enumerate(learners):
#                #    if last_instr_id in learner.instr_id_set:
#                #        if len(learner.train_samples) == learner.capacity-1:
#                #            """Now active for training"""
#                #            online_evals['acc'+str(i)] = \
#                #            learner.train_evals['acc']
#                #            online_evals['opt'+str(i)] = \
#                #            learner.train_evals['opt']
#                #        """True Label adding in"""
#                #        learner.train_samples[s].y = last_sample.y
#                #        break
#            
#            tf.scatter_update(label_ts, tf.constant([t],
#                tf.int64), [baseline_pred])
#
#            #new_sample = Sample(instr_id, t, local_feature_t, baseline_pred, sample.hacc)
#            #for i, learner in enumerate(learners):
#            #    if new_sample.instr_id in learner.instr_id_set:
#            #        local_pos[t] = len(learner.train_samples)
#            #        learner.add_sample(new_sample)
#            learner.add_sample(t)
#
#            prep_time += time.time()
#
#            if (t % (config.capacity) == 0) and (len(online_evals) > 0):
#                prep_time -= time.time()
#                feed_dict = {}
#                for learner in learners:
#                    inputs, labels = learner.train()
#                    if (inputs is None) or (labels is None):
#                        continue
#                    feed_dict[learner.inputs] = inputs
#                    feed_dict[learner.labels] = labels
#                prep_time += time.time()
#                
#                train_time -= time.time()
#                vals = sess.run(online_evals, feed_dict=feed_dict)
#                for i, learner in enumerate(learners):
#                    acc_i = vals.get('acc'+str(i), None)
#                    if acc_i is not None:
#                        learner.train_ups += acc_i
#                        learner.train_downs += 1.0
#
#                train_time += time.time()
#
#            predict_time -= time.time()
#            for learner in learners:
#                if new_sample.instr_id in learner.instr_id_set:
#                    learner.predict(sess, local_feature_t, label, baseline_pred)
#                    break
#            predict_time += time.time() 
#
#            if (t % (10*config.capacity) == 0):
#                print('time=%d, prep_time=%f, train_time=%f, predict_time=%f' %
#                        (t, prep_time, train_time, predict_time))
#                print('#sample \t', end="")
#                for learner in learners:
#                    print('%7d' % len(learner.train_samples), end=" ")
#                print('')
#                print('tr_acc  \t', end="")
#                for learner in learners:
#                    print('%.5f' % (learner.train_acc()), end=" ")
#                print('')
#
#                print('lstm:    \t', end="")
#                count_ups = 0.0
#                count_downs = 0.0
#                for learner in learners:
#                    count_ups += learner.lstm_ups
#                    count_downs += learner.downs
#                    if learner.downs == 0:
#                        print('%.5f ' % (float(0.0)), end="")
#                    else:
#                        print('%.5f ' % (float(learner.lstm_ups/learner.downs)), end="")
#                print(', acc=%.5f' % float(count_ups/count_downs))
#                
#                count_ups = 0.0
#                count_downs = 0.0
#                print('baseline:\t', end="")
#                for learner in learners:
#                    count_ups += learner.baseline_ups
#                    count_downs += learner.downs
#                    if learner.downs == 0:
#                        print('%.5f ' % (float(0.0)), end="")
#                    else:
#                        print('%.5f ' % (float(learner.baseline_ups/learner.downs)), end="")
#                print(', acc=%.5f' % float(count_ups/count_downs))
#                print('')
#
#        """Offline"""
#        #for e in range(current_epoch+1, current_epoch+config.max_epoch):
#        #    print('epoch=%d, #train=%d, #test=%d...' % (e, num_train_batches,
#        #        num_test_batches))
#        #    """training"""
#        #    if (e+1) % 1 == 0:
#        #        run_batch(config, num_train_batches, 'train', global_features,
#        #                labels, conditions, samples, ids, sess, train_evals,
#        #                FLAGS.split, 'random', summary_writer=train_summary_writer,
#        #                num_prints=100)
#        #    else:
#        #        run_batch(config, num_train_batches, 'train', global_features,
#        #                labels, conditions, samples, ids, sess,
#        #                train_evals_without_context, FLAGS.split, 'random',
#        #                summary_writer=train_summary_writer, num_prints=100)
#
#        #    saver.save(sess, model_dir+'/ckpt', global_step=e)
#        #    
#        #    """testing"""
#        #    run_batch(config, num_test_batches, 'test', global_features, labels,
#        #            conditions, samples, ids, sess, test_evals, FLAGS.split,
#        #            'random', summary_writer=test_summary_writer, num_prints=0)
            

if __name__ == "__main__":
    tf.app.run()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import collections
import os


import sys

import tensorflow as tf
import numpy
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple

from rnn_utils import *

from flags import *
#from Reader import *
from LSTMModel import *
from Config import *
import time
sys.path.append('./data/')
from Record import * 


class Learner(object):

    @classmethod
    def _get_unique_name(cls, name):
        if not hasattr(cls, 'name_count_map'):
            cls.name_count_map = {}
        if not (name in cls.name_count_map):
            cls.name_count_map[name] = 0
        name_count = cls.name_count_map[name]
        cls.name_count_map[name] += 1
        fullname = name + str(name_count)
        return fullname 

    def __init__(self, params):
        self.fullname = Learner._get_unique_name('learner')
        with tf.variable_scope(self.fullname):
            self.instr_id = -1
            self.lstm = LSTMModel(params)
            self.log = None
            #cost = tf.nn.l2_loss(self.lstm.packed_outputs - expected_outputs)
            #"""minimizer for this lstm"""
            #optimizer = tf.train.AdamOptimizer(
            #        learning_rate=Config().learning_rate)
            ##get a list of tuple (gradient, variable)
            #grads_and_vars = optimizer.compute_gradients(cost)
            #self.optimizer = optimizer.apply_gradients(grads_and_vars)

    def attach(self, log):
        self.log = log
    
    #def initialize(self, sess):
    #    sess.run(self.lstm.initializers)

class Log(object):
    def __init__(self, instr_id):
        self.instr_id = instr_id
        self.train_samples = []
        self.baseline_predictions = []
        self.baseline_acc = 0.0
        self.lstm_predictions = []
        self.lstm_acc = 0.0
        self.capacity = 20
        self.train_up = 0.0
        self.train_down = 0.0
        self.sample_count = 0

    def add_sample(self, sample):
        self.train_samples.append(sample)
        if len(self.train_samples) > self.capacity*2:
            self.train_samples = self.train_samples[-self.capacity:]
        self.sample_count += 1
        #l = len(self.train_samples)
        #if sample.y == self.baseline_predictions[l]:
        #    self.baseline_acc += 1.0
    
    def bacc(self):
        l = len(self.baseline_predictions)
        if l == 0:
            return -1
        else:
            return self.baseline_acc/float(l)

    def lstmacc(self):
        l = len(self.lstm_predictions)
        if l == 0:
            return -1
        else:
            return self.lstm_acc/float(l)
    
    def train_acc(self):
        if self.train_down == 0:
            return 0
        else:
            return self.train_up / self.train_down

def get_split(num_batches):
    mid = num_batches//4*2
    return mid

"""Generate random batch from data.
Inputs:
    samples: 
    features: has shape [batch_size*num_steps, input_dim]
    labels: has shape [batch_size*num_steps, output_dim]
    batch_size:
    num_steps:
    ids
Outputs:
    feature_batch: has shape [num_steps, batch_size, input_dim]
    label_batch: has shape [num_steps, batch_size, output_dim]
    condition_batch: has shape [num_steps, batch_size, num_learners]
"""
"""output has shape [num_steps, batch_size, input_dim]"""

def get_random_batch(samples, batch_size, num_steps, instr_ids, part):
    batch_len = batch_size*num_steps
    l = len(samples)
    upper = l//batch_len
    mid = get_split(upper)
    if part == 'train':
        start = numpy.random.randint(0, mid)*batch_len
    elif part == 'test':
        start = numpy.random.randint(mid, upper)*batch_len
    else:
        raise(ValueError("part must be either 'train' or 'test'"))
    end = start + batch_len
    subsamples = samples[start:end]
    features = [sample.feature for sample in subsamples]
    labels = [[sample.y] for sample in subsamples]
    baseline_predictions = []
    for sample in subsamples:
        if sample.feature[1] < 0.5:
            baseline_predictions.append([0])
        else:
            baseline_predictions.append([1])
    baseline_acc = 1.0 - numpy.mean(abs(numpy.asarray(labels)-numpy.asarray(baseline_predictions)))
    ups = [0.0 for i in range(len(instr_ids))]
    downs = [0.0 for i in range(len(instr_ids))]
    for i, id_subset in enumerate(instr_ids):
        for s, sample in enumerate(subsamples):
            if sample.instr_id in id_subset:
                downs[i] += 1.0
                if baseline_predictions[s][0] == sample.y:
                    ups[i] += 1.0
    conditions = [[bool(sample.instr_id in id_subset) for id_subset in instr_ids] for sample in subsamples]
    #print('loads=', [numpy.mean(
    #    [1.0 if sample.instr_id in id_subset else 0.0 for sample in subsamples]) 
    #    for id_subset in instr_ids])
    """feature has shape [batch_size*num_steps, input_dim]"""
    """feature_batch has shape [num_steps, batch_size, input_dim]"""
    feature_batch = numpy.transpose(numpy.array_split(features, batch_size), [1, 0, 2])
    """label_batch has shape [num_steps, batch_size, output_dim]"""
    label_batch = numpy.transpose(numpy.array_split(labels, batch_size), [1, 0, 2])
    """condition_batch has shape [num_steps, batch_size, len(instr_ids)]"""
    condition_batch = numpy.transpose(numpy.array_split(conditions, batch_size), [1, 0, 2])
    #print('feature batch has shape %s', feature_batch.shape)
    #print('label   batch has shape %s', label_batch.shape)
    #print('condition batch has shape %s', condition_batch.shape)
    return feature_batch, label_batch, condition_batch, baseline_acc, ups, downs

def get_ups_and_downs(instr_ids, label_batch, condition_batch, pred_batch):
    ups = [0.0 for ids in range(len(instr_ids))]
    downs = [0.0 for ids in range(len(instr_ids))]
    num_steps = len(label_batch)
    batch_size = len(label_batch[0])
    for t in range(num_steps):
        for b in range(batch_size):
            subacc = 1.0 - abs(label_batch[t, b, 0] - pred_batch[t, b, 0])
            for i in range(len(instr_ids)):
                if condition_batch[t, b, i] == True:
                    ups[i] += subacc
                    downs[i] += 1.0
    return ups, downs

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

    C = tf.Graph()
    with C.as_default():
        with tf.name_scope('config'):
            config = Config(feattype=feattype, FLAGS=FLAGS)
            config.num_instr=int(FLAGS.num_instr)

        print('dataset=', basename, ', #instr=', config.num_instr)
        print(config.to_string())
       
        print("Reading Data...")
        with tf.name_scope('readers'):
            with open(FLAGS.data_path+'/instr.all') as fin:
                lines = fin.readlines()
                samples = [Sample.from_line(line, t) for 
                        (t, line) in enumerate(lines)]
                """get distribution of data"""
                dist = {i:0 for i in range(config.num_instr)}
                for sample in samples:
                    dist[sample.instr_id] += 1
                pairs = sorted([(i, dist[i]) for i in 
                    range(config.num_instr)], key=lambda x:x[1], reverse=True)
        

        print("Building Computation Graph...")
        with tf.name_scope('Model'):
            """Buidling Network"""
            #learners = [Learner(config.local_params) 
            #        for i in range(config.num_learners)]
            
            """global feature : <forward, 1> 
                <prob, 1> <hist, self.history_len>
                shape = [global_input_dim]
            """
            global_features = tf.placeholder(tf.float32, shape = 
                    [config.num_steps, config.batch_size
                    , config.global_input_dim])

            """label: <True Label, 1>
                shape = [num_steps, batch_size, 1]
            """
            labels = tf.placeholder(tf.float32, shape = 
                    [config.num_steps, config.batch_size
                    , config.local_params.output_dim])
            
            """conditions on outputs/states:
                shape = [num_steps, batch_size, num_learners]
            """
            conditions = tf.placeholder(dtype=tf.bool
                            , shape=[config.num_steps, config.batch_size
                            , config.num_learners]) 
            
            #conditions = tf.Print(conditions, [conditions], 'feed in', summarize=10000)

            #switchs = tf.placeholder(dtype=tf.int32
            #                , shape=[config.num_steps, config.batch_size])
            
            """Context LSTM.
                Input: <forward, [batch_size, 1]>
                Hidden Layers: [30]
                Output: <context_feature, [batch_size, 20]>
            """
            context = LSTMModel(config.context_params)
            learners = [Learner(config.local_params)
                    for i in range(config.num_learners)]
            cells = [learner.lstm for learner in learners]
            states = [[learner.lstm.initial_state(1)
                    for learner in learners]
                    for b in range(config.batch_size)]
            context_state = context.initial_state(config.batch_size)
            outputs = []
            
            #context_input = tf.unstack(global_features[:, :, 0:1])
            #context_output =tf.contrib.rnn.static_rnn(context
            #        , inputs = context_input
            #        , state = context_state
            #        , output_dim = context.params.output_dim
            #        , activation=tf.sigmoid)['outputs']
            #assert(context_output.get_shape().as_list()==
            #        [config.batch_size, context.params.output_dim])

            for t in range(config.num_steps):
                var_sizes = [numpy.product(list(map(int, v.get_shape())))*v.dtype.size
                    for v in tf.global_variables()]
                print("time step %d:" % t)
                global_feature_t = global_features[t, :, 0:1]
                assert(global_feature_t.get_shape().as_list()==
                        [config.batch_size, config.context_params.input_dim])
                context_feature, context_state = context.feed_forward(
                        global_feature_t, state = context_state
                        , output_dim=context.params.output_dim
                        , activation=tf.sigmoid)
                local_input = tf.concat([context_feature
                        , global_features[t, :, 1:]], axis=1)
                assert(local_input.get_shape().as_list()==
                        [config.batch_size, config.local_params.input_dim])
                output_t, states = switched_batched_feed_forward(cells, local_input
                        , conditions[t, :, :], states)
                outputs.append(output_t)
            
            outputs = tf.stack(outputs, axis=0)
            assert(outputs.get_shape().as_list()==[config.num_steps
                , config.batch_size, config.local_params.output_dim])
            cost = tf.nn.l2_loss(outputs - labels)
            int_pred = tf.round(outputs)
            acc = 1.0 - tf.reduce_mean(tf.abs(int_pred-labels))
            """minimizer for this lstm"""
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=config.learning_rate)
            #get a list of tuple (gradient, variable)
            grads_and_vars = optimizer.compute_gradients(cost)
            for (grad, var) in grads_and_vars:
                if grad is not None:
                    grad = tf.Print(grad, [grad], str(var.name))
            optimizer = optimizer.apply_gradients(grads_and_vars)
            if FLAGS.is_training:
                evals = {'optimizer':optimizer, 'acc':acc, 'pred':int_pred}
            else:
                evals = {'acc':acc, 'pred':int_pred}
            
            init = tf.global_variables_initializer()
            #print(global_vars)
            #for tt in init.inputs:
            #    print(tt)
            #print('=========================')
            #for tt in init.control_inputs:
            #    print(tt)
            #print('=========================')
            #for tt in init.outputs:
            #    print(tt)
            #print('=========================')
            #print(init)
            saver = tf.train.Saver(tf.trainable_variables())
            
        print([(v.name, v.get_shape()) for v in tf.trainable_variables()])

    configProto = tf.ConfigProto(allow_soft_placement=True)
    configProto.gpu_options.allow_growth=True
    with tf.Session(graph=C, config=configProto) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ids = [[] for i in range(config.num_learners)]
        for i in range(config.num_instr):
            (instr_id, freq) = pairs[i]
            l = min(i, config.num_learners-1)
            l = (l + 1) % config.num_learners
            ids[l].append(instr_id)
        print(ids)
        
        acc_sum = 0.0
        if FLAGS.is_training:
            lc = tf.train.latest_checkpoint(model_dir+'/')
            if lc is not None:
                saver.restore(sess, lc)
        else:
            lc = tf.train.latest_checkpoint(model_dir+'/')
            assert( not (lc is None) )
            saver.restore(sess, lc)

        num_batches = len(samples)//(config.batch_size*config.num_steps)
        num_train_batches = num_batches//(4*config.max_epoch)*3 #get_split(num_batches)
        num_test_batches = num_batches//(4*config.max_epoch) #num_batches - num_train_batches
        for e in range(config.max_epoch):
            print('epoch=%d, #train=%d, #test=%d...' % (e, num_train_batches, num_test_batches))
            print('training...', end="")
            """training"""
            baseline_train_acc = 0.0
            train_acc = 0.0
            st = time.time()
            baseline_ups = numpy.asarray([0.0 for i in range(config.num_learners)])
            baseline_downs = numpy.asarray([0.0 for i in range(config.num_learners)])
            lstm_ups = numpy.asarray([0.0 for i in range(config.num_learners)])
            lstm_downs = numpy.asarray([0.0 for i in range(config.num_learners)])
            for train_iter in range(num_train_batches):
                """output has shapes [num_steps, batch_size, XXX]"""
                global_feature_data, label_data, condition_data, baseline_acc, \
                    ups, downs = get_random_batch(samples
                        , config.batch_size, config.num_steps, ids, 'train')
                
                values = sess.run(evals, feed_dict={
                        global_features:global_feature_data
                        , labels:label_data, conditions:condition_data})
                train_acc += values['acc']
                baseline_ups += numpy.asarray(ups)
                baseline_downs += numpy.asarray(downs)
                lstm_up, lstm_down = get_ups_and_downs(ids, label_data, condition_data, values['pred'])
                lstm_ups += numpy.asarray(lstm_up)
                lstm_downs += numpy.asarray(lstm_down)
                baseline_train_acc += baseline_acc
                if (train_iter+1) % 10 == 0:
                    ed = time.time()
                    print('\titer=%d, train acc=%f, baseline train acc=%f, \n\t\tbaseline_details=%s'
                        ', \n\t\tlstm_details=%s, \n\t\telapsed time=%f' 
                        % (train_iter, train_acc/float(train_iter+1)
                            , baseline_train_acc/float(train_iter+1)
                            , str(baseline_ups/baseline_downs)
                            , str(lstm_ups/lstm_downs)
                            , (ed-st)))
                    st = ed
            print('train acc=%f' % (train_acc/float(num_train_batches)))
            saver.save(sess, model_dir+'/ckpt', global_step=e)
            """testing"""
            print('testing...', end="")
            test_acc = 0.0
            baseline_test_acc = 0.0
            st = time.time()
            up_total = numpy.asarray([0.0 for i in range(config.num_learners)])
            down_total = numpy.asarray([0.0 for i in range(config.num_learners)])
            lstm_ups = numpy.asarray([0.0 for i in range(config.num_learners)])
            lstm_downs = numpy.asarray([0.0 for i in range(config.num_learners)])
            for test_iter in range(num_test_batches):
                global_feature_data, label_data \
                    , condition_data, baseline_acc, ups, downs= \
                    get_random_batch(samples, config.batch_size
                            , config.num_steps, ids, 'test')
                test_acc_t, pred_t = sess.run([evals['acc'], evals['pred']], feed_dict={
                    global_features:global_feature_data
                    , labels:label_data, conditions:condition_data})
                test_acc += test_acc_t
                baseline_test_acc += baseline_acc
                up_total += numpy.asarray(ups)
                down_total += numpy.asarray(downs)
                lstm_up, lstm_down = get_ups_and_downs(ids, label_data, condition_data, pred_t)
                lstm_ups += numpy.asarray(lstm_up)
                lstm_downs += numpy.asarray(lstm_down)
                if (test_iter+1) % 10 == 0:
                    ed = time.time()
                    print('\titer=%d, test acc=%f, baseline test acc=%f, \n\t\tbaseline_details=%s, \n\t\tlstm_details=%s, \n\t\telapsed time=%f' 
                        % (test_iter, test_acc/float(test_iter+1)
                            , baseline_test_acc/float(test_iter+1)
                            , str(baseline_ups/baseline_downs)
                            , str(lstm_ups/lstm_downs)
                            , (ed-st)))
                    st = ed
            print('test acc=%f' % (test_acc/float(num_test_batches)))
            
        coord.request_stop()
        coord.join(threads)


#    with tf.Session(graph=C) as sess:
#        sess.run(init)
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#        if config.mode=='online':
#            assigned_learner = {i:None for i in range(config.num_instr)}
#            num_assigned = 0
#            log = [Log(i) for i in range(config.num_instr)]
#            for sample in samples:
#                instr_id = int(sample.instr_id)
#                wakeup_id = int(sample.wakeup_id)
#                if wakeup_id != -1:
#                    sample_t = samples[wakeup_id]
#                    last_instr_id = sample_t.instr_id
#                    log[last_instr_id].add_sample(sample_t)
#                    if (len(log[last_instr_id].train_samples) >= 20) and (num_assigned < 5) and (assigned_learner[last_instr_id]==None) and (log[last_instr_id].bacc() < 0.85):
#                        assigned_learner[last_instr_id] = learners[num_assigned]
#                        learners[num_assigned].attach(log[last_instr_id])
#                        num_assigned += 1
#                        print('attaching lstm with instr id=%d, current_bacc=%f\n' % (last_instr_id, log[last_instr_id].bacc()), end="")
#                        log[last_instr_id].baseline_predictions = []
#                        log[last_instr_id].baseline_acc = 0.0
#
#                baseline_prediction = sample.get_baseline_prediction()
#                
#                log[instr_id].baseline_predictions.append(baseline_prediction)
#                """feature 5: <instruction ID, 1> <wakeup_id, 1> <forward, 1> <prob, 1> <hist, self.history_len> <True Label, 1>"""
#                true_label = int(sample.y)
#                if baseline_prediction == true_label:
#                    log[instr_id].baseline_acc += 1.0
#                
#                if not (assigned_learner[instr_id] == None):
#                    learner = assigned_learner[instr_id]
#                    lstm_prediction = numpy.squeeze(sess.run(learner.lstm.packed_outputs, feed_dict={global_input:[[sample.feature]]}))
#                    lstm_prediction = int(numpy.round(lstm_prediction))
#                    log[instr_id].lstm_predictions.append(lstm_prediction)
#                    print(lstm_prediction, ' ', true_label)
#                    if lstm_prediction == true_label:
#                        log[instr_id].lstm_acc += 1.0
#                    print('monitoring id \t%d: #train samples=\t%d, baseline acc=\t%f, lstm acc=\t%f, train acc=\t%f\n' % (instr_id, learner.log.sample_count, learner.log.bacc(), learner.log.lstmacc(), learner.log.train_acc()), end="")
#
#                """learners training"""
#                for learner in learners:
#                    if learner.log == None:
#                        continue
#                    l = len(learner.log.train_samples)
#                    for T in range(10):
#                        random_index = numpy.random.random_integers(0, l-1)
#                        sample = learner.log.train_samples[random_index]
#                        output, _ = sess.run([learner.lstm.packed_outputs, learner.optimizer], feed_dict={global_input:[[sample.feature]], expected_outputs:[[[sample.y]]]})
#                        output = numpy.squeeze(output)
#                        output = int(numpy.round(output))
#                        if output == int(sample.y):
#                            learner.log.train_up += 1.0
#                        learner.log.train_down += 1.0
#                    log_instr_id = learner.log.instr_id
#        else:
#            """offline mode for training context extractor"""
#            """build a input producer"""
#            assigned_learner = {i:None for i in range(config.num_instr)}
#            num_assigned = 0
#            log = [Log(i) for i in range(config.num_instr)]
#            for e in range(config.max_epoch):
#                print('epoch %d...' % e)
#                feature_batch, label_batch = get_random_batch(features, labels, config.batch_size, config.num_steps)
#
#
#        coord.request_stop()
#        coord.join(threads)

#    best_acc = 0.0
#    countdown = 0
#
#    with tf.Session(graph=C) as sess:
#        sess.run(init)
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#        current_epoch = 0
#        if config.mode == 'online':
#            config.max_epoch = 1
#        for e in range(current_epoch, config.max_epoch):
#            print('epoch %d...' % e)
#            train_acc = 0.0
#            states = []
#            pred = []
#            cost = 0.0
#            #config.learning_rate *= config.lr_decay
#            grad_norm = 0
#            #step_size = 1000
#            stacked_data = []
#            model.inite(sess)
#            for i in range(train_reader.num_batches):
#                #####reopen
#                #####init_time -= time.time()
#                #####if i % step_size == 0:
#                #####    sess = tf.Session(graph=C, config=conf_proto)
#                #####    sess.run(init)
#                #####    if not (tf.train.latest_checkpoint('./'+foldername+'/') is None):
#                #####        lc = tf.train.latest_checkpoint('./'+foldername+'/')
#                #####        print('restoring from ' + lc + '...')
#                #####        saver.restore(sess, lc)
#                #####        print('Done restoring')
#                #####    coord = tf.train.Coordinator()
#                #####    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#                #####init_time += time.time()
#
#                #####with tf.Session(graph=C) as sess:
#                #####    sess.run(init)
#                #####    if not (tf.train.latest_checkpoint('./'+foldername+'/') is None):
#                #####        lc = tf.train.latest_checkpoint('./'+foldername+'/')
#                #####        print('restoring from ' + lc + '...')
#                #####        saver.restore(sess, lc)
#                #####        print('Done restoring')
#                #####    coord = tf.train.Coordinator()
#                #####    threads = tf.train.start_queue_runners(coord=coord)
#                #####run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#                #####run_metadata = tf.RunMetadata()
#                
#                train_packed = numpy.asarray(sess.run(train_reader.packed))#, options=run_options, run_metadata=run_metadata))
#                #train_packed = numpy.squeeze(numpy.asarray(train_packed))
#                output = model.predict(sess, train_packed[:-1])
#                output = int(numpy.round(numpy.squeeze(output)))
#                
#                #train_vals = sess.run(vals, feed_dict={packed:train_packed})
#
#                #pred=train_vals['pred']
#                #Y = train_vals['Y']
#                #alt_acc = 1.0 - numpy.mean(abs(numpy.asarray(pred)-numpy.asarray(Y)))
#                #assert(abs(alt_acc - train_vals['acc']) < 1e-3)
#                #train_acc += train_vals['acc']
#                #cost += train_vals['cost']
#                if output == train_packed[-1]:
#                    train_acc += 1.0
#                print('train_acc_avg=%f' % (train_acc/float(i+1)))
#                stacked_data.append(train_packed)
#                temp_acc = 0.0
#                if len(stacked_data) < config.num_steps:
#                    continue
#                stacked_data = stacked_data[-20:]
#
#                print('stacked_data:', numpy.asarray(stacked_data).shape)
#                packed_data = []
#                for d in stacked_data:
#                    packed_data += d
#                print('packed_data:', numpy.asarray(packed_data).shape)
#                for t in range(10):
#                    temp_acc = 0.0
#                    train_vals = sess.run(vals, feed_dict={packed:packed_data})
#                    temp_acc = train_vals['acc']
#                    print('\ttemp_acc = %f' % (temp_acc))
#            train_acc /= train_reader.num_batches
#            print("\tOnline acc=%f, cost=%f" % (train_acc, cost))
#
#            ##testing_time -= time.time()
#            #fout = open(foldername+"/pred@"+str(e), 'w')
#            #foutY = open(foldername+"/truelabel@"+str(e), 'w')
#            #test_acc = 0.0
#            ##sess = tf.Session(graph=C, config=conf_proto)
#            ##sess.run(init)
#            ##coord = tf.train.Coordinator()
#            ##threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#            ##lc = tf.train.latest_checkpoint('./'+foldername+'/')
#            ##assert( not (lc is None) )
#            ##saver.restore(sess, lc)
#            #for i in range(test_reader.num_batches):
#            #    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#            #    #run_metadata = tf.RunMetadata()
#            #    test_packed = numpy.asarray(sess.run(test_reader.packed))
#            #    test_acc_i, pred_i, Y_i = sess.run([vals['acc'], vals['pred'], vals['Y']], feed_dict={packed:test_packed})
#            #    #index = e*test_reader.num_batches+i
#            #    #test_writer.add_run_metadata(run_metadata, 'batch%d' % index)
#            #    #test_writer.add_summary(summary, index)
#            #    Y_i = numpy.asarray(Y_i)
#            #    pred_i = numpy.asarray(pred_i)
#            #    alt_acc = 1.0 - numpy.mean(abs(Y_i - pred_i))
#            #    test_acc += test_acc_i
#            #    pred_i = numpy.transpose(pred_i).reshape([config.num_steps*config.batch_size])
#            #    Y_i = numpy.transpose(Y_i).reshape([config.num_steps*config.batch_size])
#            #    print('\ttest_acc=%f' % test_acc_i)
#            #    for q in range(pred_i.size):
#            #        fout.write( str(int( numpy.round(pred_i[q]) )))
#            #        fout.write('\n')
#            #    for q in range(Y_i.size):
#            #        foutY.write( str(int( numpy.round(Y_i[q]) )))
#            #        foutY.write('\n')
#            ##coord.request_stop()
#            ##coord.join(threads)
#            ##sess.close()
#            #test_acc /= test_reader.num_batches
#            #foutY.close()
#            #fout.close()
#            ##testing_time += time.time()
#            ##print("Init=%f, Reading=%f, Training=%f, Testing=%f" % (init_time, reading_time, training_time, testing_time))
#            #    #print("saving model...")
#            #saver.save(sess, './'+foldername+'/ckpt', global_step=e)
#            #if test_acc > best_acc:
#            #    best_acc = test_acc
#            #    countdown = 0
#            #else:
#            #    countdown += 1
#            #print("\tTest acc=%f, Best acc=%f, countdown=%d" % (test_acc, best_acc, countdown))
#            #if countdown >= 30:
#            #    break
#        coord.request_stop()
#        coord.join(threads)
#    #train_writer.close()
#    #test_writer.close()
#    #else:
#    #    #restore LSTM model
#    #    saver.restore(sess, tf.train.latest_checkpoint('./'+foldername+'/'))
#    #    #for output
#    #    fout = open(foldername+".pred2", 'w')
#    #    foutY = open(foldername+".truelabel2", 'w')
#    #    test_acc = 0.0
#    #    for i in range(test_reader.num_batches):
#    #        test_packed = numpy.asarray(sess.run(test_reader.packed))
#    #        test_acc_i, pred_i, Y_i = sess.run([vals['acc'], vals['pred'], vals['Y']], feed_dict={packed:test_packed})
#    #        Y_i = numpy.asarray(Y_i)
#    #        pred_i = numpy.asarray(pred_i)
#    #        alt_acc = 1.0 - numpy.mean(abs(Y_i - pred_i))
#    #        test_acc += test_acc_i
#    #        pred_i = numpy.transpose(pred_i).reshape([config.num_steps*config.batch_size])
#    #        Y_i = numpy.transpose(Y_i).reshape([config.num_steps*config.batch_size])
#    #        for x in pred_i:
#    #            fout.write( str(int( numpy.round(x) )))
#    #            fout.write('\n')
#    #        for q in Y_i:
#    #            foutY.write( str(int( numpy.round(x) )))
#    #            foutY.write('\n')
#
#    #    test_acc /= test_reader.num_batches
#    #    print("\tTest acc=%f" % test_acc)
#    #    foutY.close()
#    #    fout.close()
#                #coord.request_stop()
#                #coord.join(threads)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.device)

if __name__ == "__main__":
    tf.app.run()

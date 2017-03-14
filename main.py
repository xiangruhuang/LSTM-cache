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
    mid = num_batches//4*3
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
    baseline_acc = 1.0 - numpy.mean( abs(numpy.asarray( labels )-numpy.asarray(
        baseline_predictions)))
    ups = [0.0 for i in range(len(instr_ids))]
    downs = [0.0 for i in range(len(instr_ids))]
    for i, id_subset in enumerate(instr_ids):
        for s, sample in enumerate(subsamples):
            if sample.instr_id in id_subset:
                downs[i] += 1.0
                if baseline_predictions[s][0] == sample.y:
                    ups[i] += 1.0
    conditions = [[bool(sample.instr_id in id_subset) for id_subset in
        instr_ids] for sample in subsamples]
    #print('loads=', [numpy.mean(
    #    [1.0 if sample.instr_id in id_subset else 0.0 for sample in subsamples]) 
    #    for id_subset in instr_ids])
    """feature has shape [batch_size*num_steps, input_dim]"""
    """feature_batch has shape [num_steps, batch_size, input_dim]"""
    feature_batch = numpy.transpose(numpy.array_split(features, batch_size),
        [1,0,2])
    """label_batch has shape [num_steps, batch_size, output_dim]"""
    label_batch = numpy.transpose(numpy.array_split(labels, batch_size),
        [1,0,2])
    """condition_batch has shape [num_steps, batch_size, len(instr_ids)]"""
    condition_batch = numpy.transpose( numpy.array_split(conditions,
        batch_size), [1,0,2])
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
                var_sizes = [numpy.product(list(map(int,
                    v.get_shape())))*v.dtype.size for v in
                    tf.global_variables()]
                print("time step %d:" % t)
                global_feature_t = global_features[t, :, 0:1]
                assert global_feature_t.get_shape().as_list() == \
                [config.batch_size, config.context_params.input_dim]
                context_feature, context_state = context.feed_forward(
                        global_feature_t, state = context_state
                        , output_dim=context.params.output_dim
                        , activation=tf.sigmoid)
                local_input = tf.concat([context_feature
                        , global_features[t, :, 1:]], axis=1)
                assert local_input.get_shape().as_list()== [config.batch_size,
                    config.local_params.input_dim]
                output_t, states = switched_batched_feed_forward(cells,
                    local_input , conditions[t, :, :], states)
                outputs.append(output_t)
            
            outputs = tf.stack(outputs, axis=0)
            assert outputs.get_shape().as_list()==[config.num_steps ,
                config.batch_size, config.local_params.output_dim]
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
        heuristic_init = []
        with tf.variable_scope('') as scope:
            scope.reuse_variables()
            for i in range(config.num_learners):
                w = tf.get_variable('local'+str(i)+'/inputs_'+
                    str(config.local_params.input_dim)
                    +'/multi_rnn_cell/cell_0/basic_lstm_cell/weights')
                x_size = config.local_params.input_dim
                h_size = config.local_params.hidden_sizes[0]
                w_init = []
                for x_i in range(x_size):
                    w_i = []
                    w_i.append(tf.constant(0.0, shape=[1, h_size]))
                    w_i.append(tf.constant(0.0, shape=[1, h_size]))
                    w_i.append(tf.constant(0.0, shape=[1, h_size]))
                    if x_i == config.context_dims[-1]:
                        w_i.append(tf.constant(1.0, shape=[1, h_size]))
                    else:
                        w_i.append(tf.constant(0.0, shape=[1, h_size]))
                    w_i = tf.concat(w_i, 1)
                    w_init.append(w_i)

                w_init.append(tf.constant(0.0, shape=[h_size, h_size*4]))

                w_init = tf.concat(w_init, 0)
                heuristic_init.append(w.assign(w_init))
        
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

    configProto = tf.ConfigProto(allow_soft_placement=True)
    configProto.gpu_options.allow_growth=True
    with tf.Session(graph=C, config=configProto) as sess:
        sess.run(init)
        sess.run(heuristic_init)
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
                print("restoring model from "+model_dir+'/')
                saver.restore(sess, lc)
        else:
            lc = tf.train.latest_checkpoint(model_dir+'/')
            assert not (lc is None) 
            saver.restore(sess, lc)

        num_batches = len(samples)//(config.batch_size*config.num_steps)
        num_train_batches = get_split(num_batches//config.max_epoch)
        num_test_batches = num_batches//config.max_epoch - num_train_batches
        for e in range(config.max_epoch):
            print('epoch=%d, #train=%d, #test=%d...' % (e, num_train_batches,
                num_test_batches))
            print('training...', end="")
            """training"""
            baseline_train_acc = 0.0
            train_acc = 0.0
            st = time.time()
            baseline_ups = numpy.asarray([0.0 for i in
                range(config.num_learners)])
            baseline_downs = numpy.asarray([0.0 for i in
                range(config.num_learners)])
            lstm_ups = numpy.asarray([0.0 for i in range(config.num_learners)])
            lstm_downs = numpy.asarray([0.0 for i in
                range(config.num_learners)])
            for train_iter in range(num_train_batches):
                """output has shapes [num_steps, batch_size, XXX]"""
                global_feature_data, label_data, condition_data, baseline_acc, \
                        ups, downs = get_random_batch(samples,
                            config.batch_size, config.num_steps, ids, 'train')
                
                values = sess.run(evals, feed_dict={
                    global_features:global_feature_data , labels:label_data,
                    conditions:condition_data})
                train_acc += values['acc']
                baseline_ups += numpy.asarray(ups)
                baseline_downs += numpy.asarray(downs)
                lstm_up, lstm_down = get_ups_and_downs(ids, label_data,
                        condition_data, values['pred'])
                lstm_ups += numpy.asarray(lstm_up)
                lstm_downs += numpy.asarray(lstm_down)
                baseline_train_acc += baseline_acc
                if (train_iter+1) % 10 == 0:
                    ed = time.time()
                    print('\titer=%d, train acc=%f, baseline train acc=%f,'
                        'elapsed time=%f' % (train_iter,
                            train_acc/float(train_iter+1),
                            baseline_train_acc/float(train_iter+1), (ed-st)))
                    print('train baseline\tlstm')
                    for i in range(config.num_learners):
                        print('%f\t%f' % (baseline_ups[i]/baseline_downs[i],
                            lstm_ups[i]/lstm_downs[i])) 
                    st = ed

            print('train acc=%f' % (train_acc/float(num_train_batches)))
            saver.save(sess, model_dir+'/ckpt', global_step=e)
            """testing"""
            print('testing...', end="")
            test_acc = 0.0
            baseline_test_acc = 0.0
            st = time.time()
            up_total = numpy.asarray([0.0 for i in range(config.num_learners)])
            down_total = numpy.asarray([0.0 for i in
                range(config.num_learners)])
            lstm_ups = numpy.asarray([0.0 for i in range(config.num_learners)])
            lstm_downs = numpy.asarray([0.0 for i in
                range(config.num_learners)])
            for test_iter in range(num_test_batches):
                global_feature_data, label_data , condition_data, baseline_acc,
                ups, downs= get_random_batch(samples, config.batch_size ,
                        config.num_steps, ids, 'test')
                test_acc_t, pred_t = sess.run([evals['acc'], evals['pred']],
                        feed_dict={ global_features:global_feature_data ,
                            labels:label_data, conditions:condition_data})
                test_acc += test_acc_t
                baseline_test_acc += baseline_acc
                up_total += numpy.asarray(ups)
                down_total += numpy.asarray(downs)
                lstm_up, lstm_down = get_ups_and_downs(ids, label_data,
                    condition_data, pred_t)
                lstm_ups += numpy.asarray(lstm_up)
                lstm_downs += numpy.asarray(lstm_down)
                if (test_iter+1) % 10 == 0:
                    ed = time.time()
                    print('\titer=%d, test acc=%f, baseline test acc=%f,'
                        'elapsed time=%f' % (test_iter,
                            test_acc/float(test_iter+1),
                            baseline_test_acc/float(test_iter+1), (ed-st)))

                    print('test baseline\tlstm')
                    for i in range(config.num_learners):
                        print('%f\t%f' % (baseline_ups[i]/baseline_downs[i],
                            lstm_ups[i]/lstm_downs[i])) 
                    st = ed
            print('test acc=%f' % (test_acc/float(num_test_batches)))
            
        coord.request_stop()
        coord.join(threads)



if __name__ == "__main__":
    tf.app.run()

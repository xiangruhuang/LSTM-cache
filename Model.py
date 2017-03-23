from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

from utils import *

from LSTMModel import LSTMModel
from Learner import Learner

from rnn_utils import *
from Reader import Reader
import time

class Log(object):
    def __init__(self):
        self.indices = []
        self.mask = []
        self.sample_count = 0
        self.loc_dict = {}

    def add_index(self, index):
        self.loc_dict[index] = self.sample_count
        self.indices.append(index)
        self.mask.append(0)
        self.sample_count += 1

    def set_mask(self, index):
        loc = self.loc_dict[index]
        self.mask[loc] = 1

    def random_indices(self, length, window_size=None):
        if window_size is None:
            window_size = self.sample_count
        st = numpy.random.randint(0, 
                window_size-length+1) + self.sample_count - window_size
        ed = st + length
        selected_indices = self.indices[st:ed]
        return selected_indices

    def random_indices_and_mask(self, window_size, length): 
        st = numpy.random.randint(0, 
                window_size-length+1) + self.sample_count - window_size
        ed = st + length
        selected_indices = self.indices[st:ed]
        selected_mask = self.mask[st:ed]
        return selected_indices, selected_mask

    def num_batches(self, length):
        return self.sample_count//length

class Model(object):

    def construct_online(self):
        config = self.config
        if config.context_output_dim > 0:
            self.reader.local_feature(self.context)
        self.logs = [Log() for i in range(config.num_learners)]

        context_variables = []
        for v in tf.trainable_variables():
            if v.name.startswith('context0'):
                context_variables.append(v)
        #print([(v.name, v.get_shape()) for v in context_variables])
        self.saver = tf.train.Saver(context_variables)

    def construct_offline(self):
        
        config = self.config
        reader = self.reader

        self.train_log = Log()
        self.test_log = Log()

        local_features = reader.feature_batch
        labels = reader.label_batch
        conditions = reader.condition_batch
        baseline_preds = reader.baseline_batch
        
        start_time = time.time()
        print("Building Offline Graph..., elapsed time=%f" % (time.time()-start_time))

        #"""label: <True Label, 1>
        #    shape = [num_steps, batch_size, 1]
        #"""
        #self.labels = tf.placeholder(tf.float32, shape = [config.num_steps,
        #    config.batch_size , config.local_params.output_dim])
        #
        #"""conditions on outputs/states:
        #    shape = [num_steps, batch_size, num_learners]
        #"""
        #self.conditions = tf.placeholder(dtype=tf.bool, shape=[config.num_steps,
        #    config.batch_size, config.num_learners])
        
        cells = self.cells
        states = [[cell.initial_state(1) for cell in cells]
                for b in range(config.batch_size)]
        outputs = [] 

        context_state = self.context.initial_state( config.batch_size )
        for t in range(config.num_steps):
            #if config.context_output_dim==0:
            #    local_input = tf.concat([self.global_features[t, :, 1:]],
            #            axis=1)
            #else:
            #    global_feature_t = self.global_features[t, :, 0:1]
            #    assert global_feature_t.get_shape().as_list() == \
            #    [config.batch_size, config.context_params.input_dim]
            #    context_feature, context_state = self.context.feed_forward(
            #            global_feature_t, context_state,
            #            self.context.params.output_dim, tf.sigmoid)
            #    local_input = tf.concat([context_feature,
            #        local_features[t,:,1:]], axis=1)
            #assert local_input.get_shape().as_list()== [config.batch_size,
            #    config.local_params.input_dim]
            output_t, states = switched_batched_feed_forward(cells,
                    local_features[t, :, :], conditions[t, :, :], states)
            outputs.append(output_t)
        
        print("Computing stats, elapsed time=%f" % (time.time()-start_time))
        
        inc_num_train_samples = self.num_train_samples.assign_add( tf.constant(
            config.num_steps * config.batch_size, dtype = tf.int64 ) )
        inc_num_test_samples = self.num_test_samples.assign_add( tf.constant(
            config.num_steps * config.batch_size, dtype = tf.int64 ) )
        
        outputs = tf.stack(outputs, axis=0)
        assert outputs.get_shape().as_list()==[config.num_steps,
                config.batch_size, config.local_params.output_dim]
        cost = tf.nn.l2_loss(outputs - labels)
        int_pred = tf.round(outputs)
        lstm_subs = tf.ones_like(int_pred,
                dtype=tf.float32)-tf.abs(int_pred-labels)
        baseline_subs = tf.ones_like(baseline_preds,
                dtype=tf.float32)-tf.abs(baseline_preds-labels)
       
        lstm_ups = self.lstm_ups.assign_add( tf.reduce_sum(
            tf.multiply(lstm_subs, reader.indicator_batch), axis=[0, 1]))
        baseline_ups = self.baseline_ups.assign_add( tf.reduce_sum(
            tf.multiply(baseline_subs, reader.indicator_batch), axis=[0, 1]))
        downs = self.downs.assign_add( tf.reduce_sum(reader.indicator_batch,
            axis=[0, 1]))
        lstm_accs = tf.div(lstm_ups, downs)
        baseline_accs = tf.div(baseline_ups, downs)
        lstm_acc = tf.div( tf.reduce_sum(lstm_ups), tf.reduce_sum(downs) )
        baseline_acc = tf.div( tf.reduce_sum(baseline_ups),tf.reduce_sum(downs))

        """minimizer for this lstm"""
        optimizer = tf.train.AdamOptimizer(
                learning_rate=config.learning_rate)
        #get a list of tuple (gradient, variable)
        
        print("Computing gradients, elapsed time=%f" % (time.time()-start_time))
        optimizer = optimizer.minimize(cost)
        #grads_and_vars_without_context = []
        #for (grad, var) in grads_and_vars:
        #    if grad is not None:
        #        if var.name.startswith('context'):
        #            train_summary.append( tf.summary.scalar(
        #                var.name, tf.global_norm([grad]) ) )
        #        else:
        #            grads_and_vars_without_context.append((grad, var))
        #optimizer_without_context = optimizer.apply_gradients(
        #        grads_and_vars_without_context )
        #optimizer = optimizer.apply_gradients(grads_and_vars)


        print("Computing summary, elapsed time=%f" % (time.time()-start_time))
        
        lstm_train_summary = []
        lstm_test_summary = []
        for i in range(config.num_learners):
            lstm_train_summary.append( tf.summary.scalar('lstm_train_acc'+str(i),
                lstm_accs[i]) )
            lstm_test_summary.append( tf.summary.scalar('lstm_test_acc'+str(i),
                lstm_accs[i]) )
        lstm_train_summary.append( tf.summary.scalar('lstm_train_acc', lstm_acc) )
        lstm_test_summary.append( tf.summary.scalar('lstm_test_acc', lstm_acc) )
        
        baseline_train_summary = []
        baseline_test_summary = []
        for i in range(config.num_learners):
            baseline_train_summary.append( tf.summary.scalar('baseline_train_acc'+str(i),
                baseline_accs[i]) )
            baseline_test_summary.append( tf.summary.scalar('baseline_test_acc'+str(i),
                baseline_accs[i]) )
        baseline_train_summary.append( tf.summary.scalar('baseline_train_acc', baseline_acc) )
        baseline_test_summary.append( tf.summary.scalar('baseline_test_acc', baseline_acc) )

        self.train_evals = {'optimizer':optimizer, 'summary':tf.summary.merge(
            lstm_train_summary + baseline_train_summary),
            'num_samples':inc_num_train_samples}
        #self.train_evals_without_context={'optimizer':optimizer_without_context,
        #        'acc':acc, 'summary':tf.summary.merge(train_summary),
        #        'num_samples':inc_num_train_samples }

        self.test_evals = {'summary':tf.summary.merge( lstm_test_summary +
            baseline_test_summary), 'num_samples':inc_num_test_samples}

        self.saver = tf.train.Saver( tf.trainable_variables() )

        print("Done, elapsed time=%f" % (time.time()-start_time))


    """
        load model and return current epoch
    """
    def load(self, sess, model_dir):
        lc = tf.train.latest_checkpoint(model_dir+'/')
        if lc is not None:
            print("restoring model from "+model_dir+'/')
            self.saver.restore(sess, lc)
            return int(str(lc).split('ckpt-')[-1])
        else:
            return -1

    def online(self, sess):
        sess.run(self.init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('online')
        coord.request_stop()
        coord.join(threads)

    def offline(self, sess):
        config = self.config
        sess.run(self.init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader = self.reader
        for t in range(reader.num_samples()):
            if t < reader.get_split():
                self.train_log.add_index(t)
            else:
                self.test_log.add_index(t)
        
        current_epoch = self.load(sess, config.model_dir)
        num_train_batches = self.train_log.num_batches(
                config.num_steps*config.batch_size)//config.max_epoch
        num_test_batches = self.test_log.num_batches(
                config.num_steps*config.batch_size)//config.max_epoch*3

        clear_stats = []
        clear_stats.append( self.lstm_ups.assign(tf.zeros_like(self.lstm_ups)) )
        clear_stats.append( self.downs.assign(tf.zeros_like(self.downs)) )
        clear_stats.append(
                self.baseline_ups.assign(tf.zeros_like(self.baseline_ups)) )

        for e in range(current_epoch+1, current_epoch+config.max_epoch):
            print('epoch=%d, #train=%d, #test=%d...' % (e, num_train_batches,
                num_test_batches))
            
            sess.run(clear_stats)
            """training"""
            for Iter in range(num_train_batches):
                random_indices = self.train_log.random_indices(
                        config.batch_size*config.num_steps)
                vals = sess.run(self.train_evals,
                        feed_dict={reader.indices:random_indices})
                self.train_summary_writer.add_summary(vals['summary'],
                        vals['num_samples'])

            for Iter in range(num_test_batches):
                random_indices = self.test_log.random_indices(
                        config.batch_size*config.num_steps)
                vals = sess.run(self.test_evals,
                        feed_dict={reader.indices:random_indices})
                self.test_summary_writer.add_summary(vals['summary'],
                        vals['num_samples'])

            self.saver.save(sess, config.model_dir+'/ckpt', global_step=e)

        coord.request_stop()
        coord.join(threads)
    
    def __init__(self, config):
        print(config.to_string())
        self.config = config
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            self.train_summary_writer = tf.summary.FileWriter(
                    config.model_dir+'/tensorboard/train')
            self.test_summary_writer = tf.summary.FileWriter(
                    config.model_dir+'/tensorboard/test')

            self.num_train_samples = tf.Variable(tf.constant(0, dtype=tf.int64))
            self.num_test_samples = tf.Variable(tf.constant(0, dtype=tf.int64))
            self.lstm_ups = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))
            self.baseline_ups = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))
            self.downs = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))

            """Context LSTM.
                Input: <forward, [batch_size, 1]>
                Hidden Layers: [30]
                Output: <context_feature, [batch_size, context_output_dim]>
            """
            self.reader = Reader(config)
            if config.context_output_dim > 0:
                self.context = LSTMModel(config.context_params)
            self.cells = [LSTMModel(config.local_params) for i in
                    range(config.num_learners)]

            if config.mode == 'online':
                self.construct_online()
            elif config.mode == 'offline':
                self.construct_offline()
            else:
                raise(ValueError('mode not acceptable'+config.mode))

            
            self.init = tf.global_variables_initializer()

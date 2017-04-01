from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

from utils import *

from LSTMModel import LSTMModel

from rnn_utils import *
from Reader import *
import time

class OfflineModel(object):

    def construct(self): 
        config = self.config

        start_time = time.time()
        print("Building Offline Graph..., elapsed time=%f" % (time.time()-start_time))
        
        """Context LSTM.
            Input: <forward, [batch_size, 1]>
            Hidden Layers: [30]
            Output: <context_feature, [batch_size, context_output_dim]>
        """
        self.reader = Reader(config)
        feed_dict, evals = self.reader.get_batch( config.batch_size,
                config.num_steps)
        features = evals['feature_batch']
        labels = evals['label_batch']
        conditions = evals['condition_batch']
        baseline_preds = evals['baseline_batch']
        indicator_batch = evals['indicator_batch']
        
        if config.context_dims[-1] > 0:
            self.context = LSTMModel(config.context_params)
            local_features = []
            context_state = self.context.initial_state(config.batch_size,
                    dtype=tf.float32)
            input_dim = self.context.params.input_dim
            output_dim = self.context.params.output_dim
            print(features.shape)
            for t in range(config.num_steps):
                local_t, context_state = self.context.feed_forward(
                        features[t,:,:input_dim], context_state,
                        output_dim, tf.sigmoid)
                local_features.append(tf.concat([local_t, features[t,:,:]],
                    axis=1))
        else:
            local_features = tf.unstack(features)
        
        local_features[0] = tf.Print(local_features[0], [local_features[0]],
                summarize=1000, message='local_feature:3645000')

        """Local LSTM"""
        self.cells = [LSTMModel(config.local_params) for i in
                range(config.num_learners)]

        cells = self.cells
        states = [[cell.initial_state(1) for cell in cells]
                for b in range(config.batch_size)]
        outputs = [] 

        context_state = self.context.initial_state( config.batch_size )
        for t, input_t in zip(range(config.num_steps), local_features):
            print('time step=%d' % t)
            output_t, states = switched_batched_feed_forward(cells,
                    input_t, conditions[t, :, :], states)
            outputs.append(output_t)
        
        print("Computing stats, elapsed time=%f" % (time.time()-start_time))
        
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
            tf.multiply(lstm_subs, indicator_batch), axis=[0, 1]))
        baseline_ups = self.baseline_ups.assign_add( tf.reduce_sum(
            tf.multiply(baseline_subs, indicator_batch), axis=[0, 1]))
        downs = self.downs.assign_add( tf.reduce_sum(indicator_batch,
            axis=[0, 1]))
        lstm_accs = tf.div(lstm_ups, downs)
        baseline_accs = tf.div(baseline_ups, downs)
        lstm_acc = tf.div( tf.reduce_sum(lstm_ups), tf.reduce_sum(downs) )
        baseline_acc = tf.div(tf.reduce_sum(baseline_ups), tf.reduce_sum(downs))

        print("Computing gradients, elapsed time=%f" % (time.time()-start_time))
        """minimizer for this lstm"""
        if config.is_training:
            optimizer = tf.train.AdamOptimizer( learning_rate=config.learning_rate)
            optimizer = optimizer.minimize(cost)

        """get a list of tuple (gradient, variable)"""
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
            lstm_train_summary.append(
                    tf.summary.scalar('lstm_train_acc'+str(i), lstm_accs[i]) )
            lstm_test_summary.append( tf.summary.scalar('lstm_test_acc'+str(i),
                lstm_accs[i]) )
        lstm_train_summary.append(tf.summary.scalar('lstm_train_acc', lstm_acc))
        lstm_test_summary.append( tf.summary.scalar('lstm_test_acc', lstm_acc) )
        
        baseline_train_summary = []
        baseline_test_summary = []
        for i in range(config.num_learners):
            baseline_train_summary.append( tf.summary.scalar(
                'baseline_train_acc'+str(i), baseline_accs[i]) )
            baseline_test_summary.append( tf.summary.scalar(
                'baseline_test_acc'+str(i), baseline_accs[i]) )
        baseline_train_summary.append( tf.summary.scalar('baseline_train_acc',
            baseline_acc) )
        baseline_test_summary.append( tf.summary.scalar('baseline_test_acc',
            baseline_acc) )

        if config.is_training:
            train_evals = {'optimizer':optimizer, 'summary':tf.summary.merge(
                lstm_train_summary + baseline_train_summary),
                'num_samples':self.inc_num_train_samples}

        test_evals = {'summary':tf.summary.merge( lstm_test_summary +
            baseline_test_summary), 'num_samples':self.inc_num_test_samples}
        
        """Build Learners for train and test"""
        batch_len = config.batch_size*config.num_steps
        window_size = None
        train_log = Log(batch_len, window_size)
        test_log = Log(batch_len, window_size)
        #for t in range(self.reader.num_samples()):
        print(self.reader.get_split())
        for t in range(3645000, 3645001):
            if t < self.reader.get_split():
                train_log.add_index(t)
            else:
                test_log.add_index(t)
        if config.is_training:
            self.trainer = OfflineLearner(config, train_evals, feed_dict['indices'],
                    self.train_summary_writer, train_log)
        self.tester = OfflineLearner(config, test_evals, feed_dict['indices'],
                self.test_summary_writer, test_log)


        print("Done, elapsed time=%f" % (time.time()-start_time))


    """
        load model and return current epoch
    """
    def load(self, sess, save_dir):
        lc = tf.train.latest_checkpoint(save_dir+'/')
        if lc is not None:
            print("restoring model from "+save_dir+'/')
            self.loader.restore(sess, lc)
            return int(str(lc).split('ckpt-')[-1])
        else:
            return -1

    def run(self):
        config = self.config
        reader = self.reader
        sess = self.sess
        sess.run(self.init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        current_epoch = load_vars(sess, var_list, load_dir)
        if config.is_training:
            num_train_batches = self.trainer.log.num_batches()//config.max_epoch
        else:
            num_train_batches = 0
        num_test_batches = self.tester.log.num_batches()//config.max_epoch*3

        #sess.run(self.num_train_samples.assign( tf.constant((current_epoch+1)*
        #    num_train_batches*config.batch_size*config.num_steps,
        #    dtype=tf.int64)))

        #sess.run(self.num_test_samples.assign(tf.constant((current_epoch+1)*
        #        num_test_batches*config.batch_size*config.num_steps,
        #        dtype=tf.int64)))

        for e in range(current_epoch+1, current_epoch+config.max_epoch+1):
            print('epoch=%d, #train_batch=%d, #test_batch=%d...' % (e,
                num_train_batches, num_test_batches))
           
            """Clear Ups and Downs for this epoch"""
            #sess.run(self.clear_ups_and_downs)

            """Training"""
            if config.is_training:
                self.trainer.run(sess, num_train_batches)

            """Testing"""
            self.tester.run(sess, num_test_batches)

            """Saving"""
            if config.is_training:
                self.saver.save(sess, config.save_dir+'/ckpt', global_step=e)

        coord.request_stop()
        coord.join(threads)
    
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.graph = tf.get_default_graph()
        self.load_dir = config.load_dir
        with self.graph.as_default():
   
            """Summary Writers"""
            self.train_summary_writer = tf.summary.FileWriter(
                    config.save_dir+'/tensorboard/train')
            self.test_summary_writer = tf.summary.FileWriter(
                    config.save_dir+'/tensorboard/test')
            
            """Statistics to plot"""
            self.num_train_samples = tf.Variable(tf.constant(0, dtype=tf.int64))
            self.num_test_samples = tf.Variable(tf.constant(0, dtype=tf.int64))
            self.inc_num_train_samples = self.num_train_samples.assign_add(
                    tf.constant( config.num_steps * config.batch_size, dtype =
                        tf.int64 ) )
            self.inc_num_test_samples = self.num_test_samples.assign_add(
                    tf.constant( config.num_steps * config.batch_size, dtype =
                        tf.int64 ) )

            self.lstm_ups = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))
            self.baseline_ups = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))
            self.hawkeye_ups = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))

            self.downs = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))
            
            self.clear_ups_and_downs = []
            self.clear_ups_and_downs.append(
                    self.lstm_ups.assign(tf.zeros_like(self.lstm_ups)) )
            self.clear_ups_and_downs.append(
                    self.downs.assign(tf.zeros_like(self.downs)) )
            self.clear_ups_and_downs.append(
                    self.baseline_ups.assign(tf.zeros_like(self.baseline_ups)) )

            """Construct Model"""
            self.construct()
            
            self.init = tf.global_variables_initializer()
            self.loader = tf.train.Saver( tf.trainable_variables() )
            self.saver = tf.train.Saver( tf.trainable_variables() )
    
class OfflineLearner(object):
    def __init__(self, config, evals, feed_indices, summary_writer, log):
        self.config = config
        self.log = log
        self.evals = evals
        self.feed_indices = feed_indices
        self.summary_writer = summary_writer

    def run(self, sess, num_batches=None):
        config = self.config
        log = self.log
        evals = self.evals
        summary_writer = self.summary_writer
        if num_batches is None:
            num_batches = log.num_batches()
        for Iter in range(num_batches):
            indices = log.random_indices()
            vals = sess.run(evals, feed_dict={self.feed_indices:indices})
            summary_writer.add_summary(vals['summary'], vals['num_samples'])

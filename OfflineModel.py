from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

from utils import *

from MultiLSTMCell import MultiLSTMCell

from rnn_utils import *
from Reader import *
import time

class OfflineModel(object):

    def construct(self): 
        config = self.config

        batch_size = config.batch_size
        num_steps = config.num_steps
        num_learners = config.num_learners

        start_time = time.time()
        print("Building Offline Graph...", end="")
        
        """Context LSTM.
            Input: <forward, [batch_size, 1]>
            Hidden Layers: [30]
            Output: <context_feature, [batch_size, context_output_dim]>
        """
        self.reader = Reader(config)
        feed_dict, evals = self.reader.get_batch( batch_size, num_steps)
        features = evals['feature_batch']
        labels = evals['label_batch']
        #conditions = evals['condition_batch']
        baseline_preds = evals['baseline_batch']
        #indicator_batch = evals['indicator_batch']
        switch = evals['switch']
        
        if config.baseline_only:
            outputs = baseline_preds
        else:
            if config.context_dims[-1] > 0:
                self.context = MultiLSTMCell(config.context_params)
                local_features = []
                context_state = self.context.zero_state(batch_size, tf.float32)
                input_dim = self.context.params.input_dim
                output_dim = self.context.params.output_dim

                with tf.variable_scope('context_b%d_i%d' % (batch_size,
                    input_dim)) as scope:
                    for t in range(num_steps):
                        if t > 0:
                            scope.reuse_variables()
                        output_t, context_state = self.context(
                                features[t,:,:input_dim], context_state, None)
                        output_t = linear(output_t, output_dim)
                        local_features.append(tf.concat([output_t,
                            features[t,:,:]], axis=1))
            else:
                local_features = tf.unstack(features)

            """Local LSTM"""
            self.local = MultiLSTMCell(config.local_params)

            state = self.local.zero_state(batch_size, tf.float32)
            outputs = []

            with tf.variable_scope('local_b%d_i%d' % (batch_size,
                config.local_params.input_dim)) as scope:
                for t, input_t in zip(range(num_steps), local_features):
                    if t > 0:
                        scope.reuse_variables()
                    output_t, state = self.local(input_t, state, switch[t, :])
                    output_t = switched_linear(output_t, num_learners, 1,
                            switch[t,:])
                    outputs.append(output_t)
            outputs = tf.stack(outputs, axis=0)

        assert outputs.shape.as_list() == [num_steps, batch_size, 1]
        cost = tf.nn.l2_loss(outputs - labels)
        int_pred = tf.round(outputs)
        sub_acc = tf.ones_like(int_pred,
                dtype=tf.float32)-tf.abs(int_pred-labels)
        #sub_acc = tf.Print(sub_acc, [sub_acc], summarize=100, message='sub_acc')
        sub_acc_flat = tf.reshape(sub_acc, shape=[-1])
        #baseline_subs = tf.ones_like(baseline_preds,
        #        dtype=tf.float32)-tf.abs(baseline_preds-labels)
       
        sparse_indices = expand_indices(switch, axis=2)
        ups_delta = tf.SparseTensor(sparse_indices, sub_acc_flat, [num_steps,
            batch_size, num_learners])
        ups = self.ups.assign_add(tf.sparse_reduce_sum(ups_delta, [0, 1]))
        #ups = tf.Print(ups, [ups], summarize=100, message='ups')

        downs_delta = tf.SparseTensor(sparse_indices, tf.ones_like(
            sub_acc_flat), [num_steps, batch_size, num_learners])
        downs = self.downs.assign_add(tf.sparse_reduce_sum(downs_delta,
            axis=[0,1]))
        #downs = tf.Print(downs, [downs], summarize=100, message='downs')

        accs = tf.div(ups, downs)
        #baseline_accs = tf.div(baseline_ups, downs)
        acc = tf.div(tf.reduce_sum(ups), tf.reduce_sum(downs))
        #baseline_acc = tf.div(tf.reduce_sum(baseline_ups), tf.reduce_sum(downs))

        """minimizer for this lstm"""
        if config.is_training:
            global_step = tf.Variable(0, trainable=False)
            #learning_rate = tf.train.exponential_decay(config.learning_rate,
            #        global_step, 1000, 0.96)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            opt = optimizer.minimize(cost, var_list=tf.trainable_variables(),
                    global_step=global_step)

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

        train_summary = []
        test_summary = []
        for i in range(config.num_learners):
            train_summary.append(tf.summary.scalar('train_acc'+str(i), accs[i]))
            test_summary.append(tf.summary.scalar('test_acc'+str(i), accs[i]))
        train_summary.append(tf.summary.scalar('train_acc', acc))
        test_summary.append(tf.summary.scalar('test_acc', acc))

        if config.is_training:
            train_evals = {'opt':opt, 'summary':tf.summary.merge(
                train_summary), 'num_samples':self.inc_num_train_samples}

        test_evals = {'summary':tf.summary.merge(test_summary),
                'num_samples':self.inc_num_test_samples}
        
        """Build Learners for train and test"""
        window_size = None
        train_log = Log(batch_size, num_steps, window_size)
        test_log = Log(batch_size, num_steps, window_size)
        for t in range(self.reader.num_samples()):
            if t < self.reader.get_split():
                train_log.add_index(t)
            else:
                test_log.add_index(t)
        if config.is_training:
            self.trainer = OfflineLearner(config, train_evals,
                    feed_dict['indices'], self.train_summary_writer, train_log)
        self.tester = OfflineLearner(config, test_evals, feed_dict['indices'],
                self.test_summary_writer, test_log)

        print("Done, elapsed time=%f" % (time.time()-start_time))

    def run(self):
        config = self.config
        reader = self.reader
        sess = self.sess
        sess.run(self.init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        current_epoch = load_vars(sess, tf.trainable_variables(), config.load_dir)
        if config.is_training:
            num_train_batches = self.trainer.log.num_batches()
        else:
            num_train_batches = 0
        num_test_batches = self.tester.log.num_batches()

        sess.run(self.num_train_samples.assign( tf.constant((current_epoch+1)*
            num_train_batches*config.batch_size*config.num_steps,
            dtype=tf.int64)))

        sess.run(self.num_test_samples.assign(tf.constant((current_epoch+1)*
                num_test_batches*config.batch_size*config.num_steps,
                dtype=tf.int64)))

        for e in range(current_epoch+1, current_epoch+config.max_epoch+1):
            print('epoch=%d, #train_batch=%d, #test_batch=%d...' % (e,
                num_train_batches, num_test_batches))

            """Clear Ups and Downs for this epoch"""
            sess.run(self.clear_ups_and_downs)

            """Training"""
            train_time = -time.time()
            if config.is_training:
                self.trainer.run(sess, num_train_batches, False)
            train_time += time.time()
            print('\ttrain_time=%f' % train_time)

            """Testing"""
            test_time = -time.time()   
            self.tester.run(sess, num_test_batches, True)
            test_time += time.time()
            print('\ttest_time=%f' % test_time)

            """Saving"""
            save_time = -time.time()   
            if config.is_training:
                self.saver.save(sess, config.save_dir+'/ckpt', global_step=e)
            save_time += time.time()
            print('\tsave_time=%f' % save_time)

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

            self.ups = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))

            self.downs = tf.Variable(tf.constant(0,
                shape=[config.num_learners], dtype=tf.float32))
            
            self.clear_ups_and_downs = []
            self.clear_ups_and_downs.append(
                    self.ups.assign(tf.zeros_like(self.ups)) )
            self.clear_ups_and_downs.append(
                    self.downs.assign(tf.zeros_like(self.downs)) )

            """Construct Model"""
            self.construct()
            
            self.init = tf.global_variables_initializer()
            self.loader = tf.train.Saver( tf.trainable_variables() )
            self.saver = tf.train.Saver( tf.trainable_variables() )
        print(tf.trainable_variables())
    
class OfflineLearner(object):
    def __init__(self, config, evals, feed_indices, summary_writer, log):
        self.config = config
        self.log = log
        self.evals = evals
        self.feed_indices = feed_indices
        self.summary_writer = summary_writer

    def run(self, sess, num_batches=None, sequential=False):
        config = self.config
        log = self.log
        evals = self.evals
        summary_writer = self.summary_writer
        if num_batches is None:
            num_batches = log.num_batches()
        for Iter in range(num_batches):
            if sequential:
                st = Iter*log.batch_len
                ed = (Iter+1)*log.batch_len
                indices = log.indices[st:ed]
            else:
                indices = log.random_indices()
            vals = sess.run(evals, feed_dict={self.feed_indices:indices})
            summary_writer.add_summary(vals['summary'], vals['num_samples'])

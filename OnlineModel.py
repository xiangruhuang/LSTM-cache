from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

from utils import *

from LSTMModel import LSTMModel

from rnn_utils import *
from Reader import *
import time

class OnlineLearner(object):
    def __init__(self, config, evals, feed_indices, feed_masks, summary_writer,
            log):
        self.config = config
        self.log = log
        self.evals = evals
        self.feed_indices = feed_indices
        self.feed_masks = feed_masks
        self.summary_writer = summary_writer

    def run(self, sess, num_batches, fout=None):
        config = self.config
        log = self.log
        evals = self.evals
        summary_writer = self.summary_writer
        if num_batches is None:
            num_batches = log.num_batches()
        for Iter in range(num_batches):
            if self.feed_masks is not None:
                indices, masks = log.random_indices_and_masks()
                if (masks is None) or (indices is None):
                    continue
                vals = sess.run(evals, feed_dict={self.feed_indices:indices,
                    self.feed_masks:masks})
            else:
                indices = log.random_indices()
                vals = sess.run(evals, feed_dict={self.feed_indices:indices})
                if fout is not None:
                    int_pred = vals['pred']
                    for i in range(self.log.batch_len):
                        batch = i // config.num_steps
                        step = i % config.num_steps
                        fout.write('%f\n' % (int_pred[step, batch, 0]))
            summary_writer.add_summary(vals['summary'], vals['num_samples'])

class OnlineModel(object):

    class Stats(object):
        def __init__(self):
            self.up= tf.Variable(tf.constant(0.0, tf.float32),
                    trainable=False)
            self.down = tf.Variable(tf.constant(0.0, tf.float32),
                    trainable=False)

    def build_dataflow(self, batch_size, num_steps, use_mask=True,
            optimizer=True, summary_name=None, output_pred=False,
            baseline_only=False):
        """Compute Outputs"""
        feed_dict, evals = self.reader.get_batch(batch_size, num_steps)
        local_features = tf.unstack(evals['feature_batch'])
        labels = evals['label_batch']
        #conditions = evals['condition_batch']
        baseline_pred = evals['baseline_batch']
        #indicator_batch = evals['indicator_batch']

        config = self.config 
        #if config.context_dims[-1] > 0:
        #    local_features = []
        #    context_state = self.context.initial_state(batch_size,
        #            dtype=tf.float32)
        #    input_dim = self.context.params.input_dim
        #    output_dim = self.context.params.output_dim
        #    for t in range(config.num_steps):
        #        local_t, context_state = self.context.feed_forward(
        #                features[t,:,:input_dim], context_state, output_dim,
        #                tf.sigmoid)
        #        local_features.append(
        #                tf.concat([local_t,features[t,:,:]], axis=1))
        #        #local_features.append(local_t)
        #else:
        #    local_features = tf.unstack(features)
         
        if baseline_only:
            outputs = baseline_pred
        else:
            cell_state = self.cell.initial_state(batch_size, dtype=tf.float32)
            input_dim = self.cell.params.input_dim
            output_dim = self.cell.params.output_dim
            outputs = []
            for local_feature in local_features:
                output_t, cell_state = self.cell.feed_forward(local_feature,
                        cell_state, output_dim, tf.sigmoid)
                outputs.append(output_t)
            
            outputs = tf.stack(outputs)
        assert outputs.shape.as_list()==[num_steps, batch_size,
                config.local_params.output_dim]
        
        if use_mask:
            mask = tf.placeholder(shape=[batch_size*num_steps],
                    dtype=tf.float32)
            batch_mask = tf.reshape(mask, shape=[config.batch_size,
                config.num_steps])
            batch_mask = tf.transpose(batch_mask)
            batch_mask = tf.expand_dims(batch_mask, -1)
            labels = tf.multiply(labels, batch_mask)+tf.multiply(baseline_pred,
                    (tf.ones_like(batch_mask, dtype=tf.float32)-batch_mask))
            weights = tf.nn.bias_add(batch_mask, tf.constant([0.01], tf.float32))
            mask_percentage = tf.reduce_mean(batch_mask)
            #outputs = tf.multiply(outputs, batch_mask)
            #labels = tf.multiply(labels, batch_mask)
            #baseline_pred = tf.multiply(baseline_pred, batch_mask)
        else:
            weights = tf.ones_like(labels, dtype=tf.float32)

        #baseline_subs = tf.ones_like(baseline_pred,
        #        dtype=tf.float32)-tf.abs(baseline_pred-labels)
        #if use_mask:
        #    baseline_subs = tf.multiply(baseline_subs, weights)

        cost = tf.nn.l2_loss(outputs - labels)
        int_pred = tf.round(outputs)
        subs = tf.ones_like(int_pred, dtype=tf.float32)-tf.abs(int_pred-labels) 
        subs = tf.multiply(subs, weights)

        stats = OnlineModel.Stats()
        up = stats.up
        down = stats.down
       
        inc_up = up.assign_add(tf.reduce_sum(subs))
        inc_down = down.assign_add(tf.reduce_sum(weights))
        
        acc = tf.div(up, down)

        evals = {'cost':cost, 'acc':acc, 'inc_up':inc_up,
                'num_samples':inc_down}
        if (output_pred is not None):
            evals['pred'] = int_pred
        if summary_name is not None:
            summary_writer = tf.summary.FileWriter(
                    config.save_dir+'/tensorboard/'+ summary_name+
                    config.expr_suffix)
            summaries = []
            summaries.append(tf.summary.scalar(
                summary_name+'_acc'+config.expr_suffix, acc))
            if use_mask:
                summaries.append( tf.summary.scalar(
                    summary_name+'_mask_percentage'+config.expr_suffix,
                    mask_percentage))
            evals['summary'] = tf.summary.merge(summaries)

        if optimizer:
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=config.learning_rate)
            noncontext_variables = [v for v in tf.trainable_variables() if
                    (not v.name.startswith(self.context.fullname))]
            optimizer = optimizer.minimize(cost, var_list=noncontext_variables)
            evals['optimizer'] = optimizer

        feed_dict = {'indices':feed_dict['indices']}
        if use_mask:
            feed_dict['mask'] = mask
       
        return_dict = {'feed_dict':feed_dict, 'evals':evals}
        if summary_name is not None:
            return_dict['summary_writer'] = summary_writer
        return return_dict

    """One Learner Version

    """
    def construct(self): 
        config = self.config

        """Context LSTM.
            Input: <forward, [batch_size, 1]>
            Hidden Layers: [30]
            Output: <context_feature, [batch_size, context_output_dim]>
        """
        self.reader = Reader(config)
        if config.context_dims[-1] > 0:
            self.context = LSTMModel(config.context_params)

            """replace features"""
            self.reader.replace_features(self.sess, self.context, self.load_dir)
        else:
            raise(ValueError('needs a positive context output dim'))
        
        """Local LSTM"""
        self.cell = LSTMModel(config.local_params)

        """Build Learners for train and test"""
        if config.is_training:
            train_dict = self.build_dataflow( config.batch_size,
                    config.num_steps, use_mask=True, optimizer=True,
                    summary_name='train')
            train_feed_dict = train_dict['feed_dict']
            train_evals = train_dict['evals']
            train_summary_writer = train_dict['summary_writer']
            window_size = config.window_size
            train_log = Log(config.batch_size, config.num_steps, window_size)
            self.trainer = OnlineLearner(config, train_evals,
                    train_feed_dict['indices'], train_feed_dict['mask'],
                    train_summary_writer, train_log)


        test_dict = self.build_dataflow(1, config.num_steps, use_mask=False,
                optimizer=False, summary_name='test',
                baseline_only=config.baseline_only)
        test_feed_dict = test_dict['feed_dict']
        test_evals = test_dict['evals']
        test_summary_writer = test_dict['summary_writer']
        test_log = Log(1, config.num_steps, config.num_steps)
        self.tester = OnlineLearner(config, test_evals,
                test_feed_dict['indices'], None, test_summary_writer, test_log)


    def run(self):
        config = self.config
        reader = self.reader
        sess = self.sess
        sess.run(self.init)
        
        if config.is_training:
            context_variables = [v for v in tf.trainable_variables() if
                    v.name.startswith(self.context.fullname)]
            load_vars(sess, context_variables, self.load_dir)
        else:
            load_vars(sess, tf.trainable_variables(), self.load_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        batch_len = config.num_steps
        num_batch = reader.num_samples()//batch_len

        instr_ids = reader.instr_ids
        wakeup_set = reader.wakeup_set
        if config.is_training:
            trainer = self.trainer
        tester = self.tester
        fout = open(config.save_dir+'/pred'+config.expr_suffix, 'w')
        print('num_batch = %d' % num_batch)
        mask_up = 0.0
        mask_down = 0.0
        for b in range(num_batch):
            offset = b*batch_len
            for t in range(batch_len):
                idx = t + offset
                tester.log.add_index(idx)
                tester.log.set_mask(idx)
            tester.run(sess, 1, fout)
            if config.baseline_only:
                continue
            if config.is_training:
                for t in range(batch_len):
                    idx = t + offset
                    trainer.log.add_index(idx)
                    mask_down += 1.0
                    for wakeup_id in wakeup_set[idx]:
                        trainer.log.set_mask(wakeup_id)
                        mask_up += 1.0
                if b + 1 < config.batch_size:
                    continue
                trainer.run(sess, 10)
        fout.close()

        coord.request_stop()
        coord.join(threads)
    
    def __init__(self, config, sess):
        self.config = config
        self.graph = tf.get_default_graph()
        self.sess = sess
        self.load_dir = config.load_dir
        self.save_dir = config.save_dir
        with self.graph.as_default():
            """Construct Model"""
            self.construct()
            meta = tf.train.export_meta_graph(
                    config.save_dir+'/meta'+config.expr_suffix)
            self.init = tf.global_variables_initializer()
    

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import sys
from utils import *
sys.path.append(os.path.join(sys.path[0], 'data'))
from Record import *
import numpy
numpy.set_printoptions(threshold=numpy.nan)
import time
from rnn_utils import *
import heapq

class Reader(object):
    
    def __init__(self, config):
        self.config = config

        construct_time = -time.time()
        arr = numpy.fromfile(config.data_path+'/instr.all.bin').reshape([-1,
            Sample.sample_length()])
        self.length = arr.shape[0]//10000*10000
        arr = arr[:self.length, :]

        self.instr_ids = arr[:, 0].astype(int)
        """get distribution of data"""
        dist = [0 for i in range(config.num_instr)]
        for instr_id in self.instr_ids:
            dist[instr_id] += 1
        pairs = sorted([(i, dist[i]) for i in 
            range(config.num_instr)], key=lambda x:x[1], reverse=True)

        if config.instr_set is not None:
            print('Considering PC set', [pairs[rank] for rank in config.instr_set])
            config.instr_set = [pairs[instr_rank][0] for instr_rank in config.instr_set]
            selected = []
            if config.num_learners > len(config.instr_set):
                config.num_learners = len(config.instr_set)
            for (n, instr_id) in enumerate(self.instr_ids):
                if instr_id in config.instr_set:
                    selected.append(n)
            self.selected = selected
            self.wakeup_ids = arr[:, 1].astype(int)
            self.wakeup_set = []
            current_wakeup_set = []
            self.time_to_time = {}
            count = 0
            for t in range(self.length):
                if (self.wakeup_ids[t] != -1) and (self.time_to_time.get(
                        self.wakeup_ids[t], None) is not None):
                    current_wakeup_set.append(self.time_to_time[self.wakeup_ids[t]])
                if self.instr_ids[t] in config.instr_set:
                    self.time_to_time[t] = count
                    count += 1
                    self.wakeup_set.append(current_wakeup_set)
                    current_wakeup_set = []
            self.length = len(selected) 

            self.wakeup_ids = None
            self.instr_ids = arr[selected, 0].astype(int)
            self.features = arr[selected, 2:-2]
            self.all_features = arr[:, 2:-2]
            self.labels = arr[selected, -2]
            self.hacc = arr[selected, -1]
            self.baseline_preds = [1.0 if feature[1] >= 0.5 else 0.0 for feature in
                    self.features]
        else:
            if config.num_learners > config.num_instr:
                config.num_learners = config.num_instr
            self.wakeup_ids = numpy.asarray(range(self.length))
            self.features = arr[:, 2:-2]
            self.labels = arr[:, -2]
            self.hacc = arr[:, -1]
            self.baseline_preds = [1.0 if feature[1] >= 0.5 else 0.0 for feature in
                    self.features]        

        self.id_to_learner = [0 for i in range(config.num_instr)]

        self.ids = [[] for i in range(config.num_learners)]
        for i in range(config.num_instr):
            (instr_id, freq) = pairs[i]
            l = min(i, config.num_learners-1)
            l = (l + 1) % config.num_learners
            self.ids[l].append(instr_id)
            self.id_to_learner[instr_id] = l

        self.label_ts = tf.convert_to_tensor(self.labels, tf.float32)
        self.feature_ts = tf.convert_to_tensor(self.features, dtype=tf.float32)
        self.baseline_pred_ts = tf.convert_to_tensor(self.baseline_preds, tf.float32)
        self.learner_ids = numpy.asarray([self.id_to_learner[self.instr_ids[i]]
            for i in range(self.length)])

        self.learner_id_ts = tf.convert_to_tensor(self.learner_ids, dtype=tf.int64)

    def replace_features(self, sess, context, load_dir):
        config = self.config
        input_dim = context.params.input_dim
        output_dim = context.params.output_dim
        all_features = self.all_features
        
        batch_size = 100
        num_steps = 100
        batch_len = batch_size*num_steps
        num_batches = len(all_features)//batch_len

        batch_features_ph = tf.placeholder(shape=[batch_len, input_dim],
                dtype=tf.float32)
        batch_features = tf.stack(tf.split(batch_features_ph, batch_size))
        batch_features = tf.transpose(batch_features, [1, 0, 2])

        context_state = context.initial_state(batch_size, dtype=tf.float32) 
        local_features = []
        for t in range(num_steps):
            local_t, context_state = context.feed_forward(
                batch_features[t,:,:input_dim], context_state, output_dim,
                    tf.sigmoid)
            local_features.append( tf.concat([local_t,
                batch_features[t,:,:]], axis=1))
        local_features = tf.convert_to_tensor(local_features)
        features = []
        sess.run(tf.global_variables_initializer())
        load_vars(sess, [v for v in tf.trainable_variables() if
            v.name.startswith(context.fullname)], load_dir)
        print('', end="")
        for b in range(num_batches):
            print('\r%d / %d' % (b, num_batches), end="")
            st = b*batch_len
            ed = st + batch_len
            local_feature_b = sess.run(local_features,
                    feed_dict={batch_features_ph:all_features[st:ed]})
            local_feature_b = numpy.transpose(local_feature_b, [1,0,2])
            local_feature_b = numpy.reshape(local_feature_b, (batch_len,
                output_dim+config.global_input_dim))
            features.append(local_feature_b)
        print('replaced features.')
        features = numpy.concatenate(features, axis=0)
        features = features[self.selected, :]
        self.feature_ts = tf.convert_to_tensor(features)

    def get_batch(self, batch_size, num_steps):
        config = self.config
        feed_dict = {}
        evals = {}

        indices = tf.placeholder(shape=[num_steps*batch_size], dtype=tf.int32)
        feed_dict['indices'] = indices

        batch_indices = tf.stack(tf.split(indices, batch_size))
        batch_indices = tf.transpose(batch_indices)
        #feed_dict['batch_indices'] = batch_indices
        
        if config.mode == 'offline':
            ranges = tf.constant([ [b,t] for t in range(num_steps) for b
                in range(batch_size) ], dtype=tf.int64)
            selected_learner_ids = tf.gather(self.learner_id_ts, indices)
            #selected_learner_ids = tf.expand_dims(selected_learner_ids, -1)
            #condition_indices = tf.concat([ranges,selected_learner_ids], axis=1)
            #condition_indices_sparse = tf.SparseTensor( condition_indices,
            #        tf.constant( True, shape=[ num_steps*batch_size ],
            #            dtype=tf.bool ), [ batch_size, num_steps,
            #                config.num_learners ])
            
            #condition_batch = tf.sparse_tensor_to_dense(
            #        condition_indices_sparse, default_value=False )
            #condition_batch = tf.transpose(condition_batch, [1,0,2])
            #evals['condition_batch']=condition_batch
            
            #indicator_batch = tf.to_float(condition_batch)
            #evals['indicator_batch']=indicator_batch

            switch_batch = tf.stack(tf.split(selected_learner_ids, batch_size))
            switch_batch = tf.transpose(switch_batch)
            evals['switch'] = switch_batch

        feature_batch = tf.gather(self.feature_ts, batch_indices)
        evals['feature_batch'] = feature_batch

        label_batch = tf.gather(self.label_ts, batch_indices)
        label_batch = tf.expand_dims(label_batch, axis=-1)
        #label_batch = tf.ones_like(label_batch)
        evals['label_batch'] = label_batch

        baseline_batch = tf.gather(self.baseline_pred_ts, batch_indices)
        baseline_batch = tf.expand_dims(baseline_batch, axis=-1)
        #baseline_batch = tf.ones_like(baseline_batch)
        evals['baseline_batch'] = baseline_batch

        return feed_dict, evals

#    """
#    Convert global feature to local features, given fixed context extractor
#    """
#    def local_feature(self, context, feature_batch, context_state=None):
#
#        if context_state is None:
#            batch_size = feature_batch.shape.as_list()[1]
#            context_state = context.initial_state(batch_size)
#
#        """shape = [num_steps, batch_size, dim]"""
#        unstacked_features = tf.unstack(feature_batch)
#        outputs = []
#        for input_t in unstacked_features:
#            output_t, context_state = context.feed_forward(input_t,
#                    context_state, context.params.output_dim, tf.sigmoid)
#            outputs.append(output_t)
#        return tf.convert_to_tensor(outputs), context_state

    def num_samples(self):
        return self.length

    def get_split(self):
        config = self.config
        train = int(config.split.split(':')[0])
        test = int(config.split.split(':')[1])
        return self.length//(train+test)*train

class Log(object):
    def __init__(self, batch_size, num_steps, window_size):
        self.indices = []
        self.mask = []
        self.sample_count = 0
        self.loc_dict = {}
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.batch_len = batch_size*num_steps
        self.window_size = window_size
        self.heap = []

    def add_index(self, index):
        self.loc_dict[index] = self.sample_count
        self.indices.append(index)
        self.mask.append(0.0)
        self.sample_count += 1
        if (self.sample_count % self.num_steps) == 0:
            idx = (self.sample_count // self.num_steps) - 1
            """sorted tuples (priority, #train, idx)"""
            heapq.heappush(self.heap, (0.0, 0.0, idx))

    def set_mask(self, index):
        loc = self.loc_dict[index]
        self.mask[loc] = 1.0

    def random_indices(self):
        window_size = self.window_size
        batch_len = self.batch_len
        if window_size is None:
            window_size = self.sample_count
        if window_size > self.sample_count:
            window_size = self.sample_count
        st = numpy.random.randint(0,
                window_size-batch_len+1)+self.sample_count-window_size
        ed = st + batch_len
        selected_indices = self.indices[st:ed]
        return selected_indices

    def random_indices_and_masks(self):
        window_size = self.window_size
        batch_len = self.batch_len
        num_steps= self.num_steps
        batch_size = self.batch_size

        if window_size is None:
            window_size = self.sample_count
        if window_size > self.sample_count:
            window_size = self.sample_count
       
        num_batch = self.sample_count // num_steps
        if num_batch < batch_size:
            raise(ValueError('num_batch=%d < batch_size=%d'% (num_batch,
                batch_size) ))
        window_size_in_batch = window_size // num_steps
        threshold = num_batch - window_size_in_batch

        #selected_indices = []
        #selected_mask = []
        #tuple_list = []
        #average_mask_rate = 0.0
        #for b in range(batch_size):
        #    val, num_train, idx = heapq.heappop(self.heap)
        #    while idx < threshold:
        #        val, num_train, idx = heapq.heappop(self.heap)
        #    st = idx*num_steps
        #    ed = (idx+1)*num_steps
        #    selected_indices += self.indices[st:ed]
        #    selected_mask += self.mask[st:ed]
        #    #print('[%d, %d], mask_percentage = %f' % (st, ed,
        #    #    numpy.mean(self.mask[st:ed])))
        #    num_train += 1.0
        #    mask_rate = numpy.mean(self.mask[st:ed])
        #    average_mask_rate += mask_rate
        #    val = num_train
        #    tuple_list.append((val, num_train, idx))

        #average_mask_rate /= batch_size

        st = numpy.random.randint(0,
                window_size-batch_len+1)+self.sample_count-window_size
        ed = st + batch_len
        selected_indices = self.indices[st:ed]
        selected_mask = self.mask[st:ed]

        assert len(selected_indices) == batch_len
        assert len(selected_mask) == batch_len
        
        #print('mask_percentage = %f' % numpy.mean(self.mask))
        #for tup in tuple_list:
        #    heapq.heappush(self.heap, tup)

        return selected_indices, selected_mask

    def num_batches(self):
        return self.sample_count//self.batch_len




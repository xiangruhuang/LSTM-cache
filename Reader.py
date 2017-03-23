from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import sys
from utils import *
sys.path.append(os.path.join(sys.path[0], 'data'))
from Record import *
import numpy
import time

class Reader(object):
    
    def __init__(self, config):
        self.config = config

        construct_time = -time.time()
        arr = numpy.fromfile(config.data_path+'/instr.all.bin').reshape([-1,
            Sample.sample_length()])
        self.length = arr.shape[0]

        self.instr_ids = arr[:, 0].astype(int)
        self.time_stamps = arr[:, 0].astype(int)
        self.features = arr[:, 2:-2]
        self.labels = arr[:, -2]
        self.hacc = arr[:, -1]
        self.baseline_preds = [1.0 if feature[1] >= 0.5 else 0.0 for feature in
                self.features]

        #self.samples = [Sample.from_array(arr[t, :], t) for 
        #        t in range(len(arr))]
        
        """get distribution of data"""
        dist = [0 for i in range(config.num_instr)]
        for instr_id in self.instr_ids:
            dist[instr_id] += 1
        pairs = sorted([(i, dist[i]) for i in 
            range(config.num_instr)], key=lambda x:x[1], reverse=True)

        self.id_to_learner = [0 for i in range(config.num_instr)]

        self.ids = [[] for i in range(config.num_learners)]
        for i in range(config.num_instr):
            (instr_id, freq) = pairs[i]
            l = min(i, config.num_learners-1)
            l = (l + 1) % config.num_learners
            self.ids[l].append(instr_id)
            self.id_to_learner[instr_id] = l

        self.indices = tf.placeholder(shape=[config.num_steps*
            config.batch_size], dtype=tf.int32)
        batch_indices = tf.parallel_stack(tf.split(self.indices,
            config.batch_size), 'batch_indices')
        batch_indices = tf.transpose(batch_indices)

        label_ts = tf.convert_to_tensor(self.labels, tf.float32)
        feature_ts = tf.convert_to_tensor(self.features, dtype=tf.float32)
        baseline_pred_ts = tf.convert_to_tensor(self.baseline_preds, tf.float32)

        self.learner_ids = numpy.asarray([self.id_to_learner[self.instr_ids[i]]
            for i in range(self.length)])

        learner_id_ts = tf.convert_to_tensor(self.learner_ids, dtype=tf.int64)
        
        if config.mode == 'offline':
            ranges = tf.constant([ [b,t] for t in range(config.num_steps) for b
                in range(config.batch_size) ], dtype=tf.int64)
            selected_learner_ids = tf.gather(learner_id_ts, self.indices)
            selected_learner_ids = tf.expand_dims(selected_learner_ids, -1)
            condition_indices = tf.concat([ranges, selected_learner_ids], axis=1)
            condition_indices_sparse = tf.SparseTensor(condition_indices,
                    tf.constant(True, shape=[config.num_steps*config.batch_size],
                        dtype=tf.bool), [config.batch_size, config.num_steps,
                            config.num_learners])

            self.condition_batch =tf.sparse_tensor_to_dense(
                    condition_indices_sparse, default_value=False )
            self.condition_batch = tf.transpose(self.condition_batch, [1,0,2])
            print(self.condition_batch.shape)
            self.indicator_batch = tf.to_float(self.condition_batch)
        

        self.feature_batch = tf.gather(feature_ts, batch_indices)
        self.label_batch = tf.gather(label_ts, batch_indices)
        self.label_batch = tf.expand_dims(self.label_batch, axis=-1)
        self.baseline_batch = tf.gather(baseline_pred_ts, batch_indices)
        self.baseline_batch = tf.expand_dims(self.baseline_batch, axis=-1)

        construct_time += time.time()
        print('Reader: construct time=%f' % (construct_time)) 
        
    """
    Convert global feature to local features, given fixed context extractor
    """
    def local_feature(self, context):
        config = self.config
        context_state = context.initial_state(config.batch_size)

        """shape = [num_steps, batch_size, dim]"""
        self.features = tf.unstack(self.feature_batch)
        outputs = []
        for input_t in self.features:
            output_t, context_state = context.feed_forward(input_t,
                    context_state, context.params.output_dim, tf.sigmoid)
            outputs.append(output_t)
        self.feature_batch = tf.convert_to_tensor(outputs)

    def num_samples(self):
        return self.length

    def get_split(self):
        config = self.config
        train = int(config.split.split(':')[0])
        test = int(config.split.split(':')[1])
        return self.length//(train+test)*train

from utils import *
import numpy

class Reader(object):
    def __init__(self, _data_path, config, instr_map, shuffle=False):
        self.config = config
        self.data_path = _data_path
        fin = open(_data_path, 'r')
        lines = fin.readlines()
        self.data = [l.strip().split(' ')[1:4] for l in lines] # skip some lines
        for d in self.data:
            d[0] = instr_map[d[0]]
        self.data = numpy.asarray(self.data)

        self.data = numpy.reshape(self.data, [-1])
        print(_data_path)
        batch_len = self.batch_len = config.batch_size * config.sample_dim
        num_batches = self.num_batches = self.data.size / batch_len
        print(self.data.size, batch_len, num_batches)

        self.data = tf.convert_to_tensor(self.data, name="data", dtype=tf.float32)

        #self.data = self.data[0 : num_batches * batch_len]
        self.data = tf.reshape(self.data[0 : num_batches * batch_len], [num_batches, batch_len])

        #self.data = tf.train.shuffle_batch(self.data, batch_size=batch_size, num_threads=4, capacity=50000, min_after_dequeue=10000)

        i = tf.train.range_input_producer(num_batches, shuffle=shuffle).dequeue()
        
        self.packed = tf.slice(self.data, [i, 0], [1, batch_len])
        self.packed = tf.reshape(self.packed, [config.batch_size, config.sample_dim])

    def unpack(self, packed):
        config = self.config
        #next_batch = numpy.hsplit(packed, config.num_steps)
        splitted = tf.split(1, config.num_steps, packed)
        X = [sample[:, :config.input_dim] for sample in splitted]
        Y = [sample[:, config.input_dim:(config.input_dim+config.output_dim)] for sample in splitted]
        #Z = [numpy.rint(sample[:, config.input_dim+1]).astype(int) for sample in next_batch]
        #Z = [tf.cast(tf.round(sample[:, config.input_dim+1]), tf.int32) for sample in splitted]
        return {'features':X, 'output':Y}

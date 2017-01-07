from utils import *
import numpy
import random
import json
import os
import re

class Reader(object):
    def __init__(self, _data_path, config, instr_map=None, shuffle=False):
        self.config = config
        self.data_path = _data_path
        name = self.data_path.split('/')[-1]
        basename = self.data_path.split('/')[-1].split('.')[0]
        path = os.path.dirname(self.data_path)

        flist = [os.path.join(path, f) for f in os.listdir(path) if (f.split('.')[:-1] == name.split('.'))]
        #if len(flist) == 0:
        #    fin = open(_data_path, 'r')
        #    lines = fin.readlines()
        #    self.data = [l.strip().split(' ') for l in lines] # skip some lines
        #    #for d in self.data:
        #    #    d[0] = instr_map[d[0]]
        #    self.data = numpy.asarray(self.data)

        #    self.data = numpy.reshape(self.data, [-1])
        #    batch_len = self.batch_len = config.batch_size * config.sample_dim
        #    num_batches = self.num_batches = self.data.size / batch_len

        #    #self.data = tf.convert_to_tensor(self.data, name="data", dtype=tf.float32)
        #    self.data = numpy.reshape(self.data[0 : num_batches * batch_len], [num_batches, batch_len])
        #    self.data = numpy.split(self.data, num_batches)
        #    
        #    for i in range(self.num_batches):
        #        flist.append(self.data_path+'-'+str(i).zfill(3))
        #        if os.path.exists(self.data_path+'-'+str(i).zfill(3)):
        #            continue
        #        fout_i = open(self.data_path+'-'+str(i).zfill(3), 'w')
        #        d = numpy.squeeze(numpy.asfarray(self.data[i]))
        #        d = numpy.reshape(d, [config.batch_size, config.sample_dim])
        #        for x in d:
        #            for x_i in x:
        #                fout_i.write(str(x_i))
        #                fout_i.write(' ')
        #            fout_i.write('\n')
        #        fout_i.close()
        #    self.data = []
        #else:
        batch_len = self.batch_len = config.batch_size * config.sample_dim
        num_batches = self.num_batches = len(flist)
        assert(self.num_batches == len(flist))
        print(name.split('.')[-1]+': '+str(self.num_batches)+' batches')
        #self.data = self.data[0 : num_batches * batch_len]
        #self.data = tf.reshape(self.data[0 : num_batches * batch_len], [num_batches, batch_len])

        #self.data = tf.train.shuffle_batch(self.data, batch_size=batch_size, num_threads=4, capacity=50000, min_after_dequeue=10000)

        #i = tf.train.range_input_producer(num_batches, shuffle=shuffle).dequeue()
        flist.sort()
        self.shuffle = shuffle
        #self.packed = tf.slice(self.data, [i, 0], [1, batch_len])
        #self.packed = self.data[i]
        #self.packed = tf.reshape(self.packed, [config.batch_size, config.sample_dim])
        filename_queue = tf.train.string_input_producer(flist, shuffle=self.shuffle)
        reader = tf.TextLineReader()
        keys, values = reader.read_up_to(filename_queue, config.num_steps*config.batch_size)
        record_defaults = [[0.0] for i in range(config.input_dim+config.output_dim)]
        cols = tf.decode_csv(values, record_defaults=record_defaults, field_delim=' ')
        self.packed = tf.transpose(tf.pack(cols), perm=[1, 0])
        #self.packed = tf.Print(self.packed, [self.packed])
        self.packed = tf.reshape(self.packed, [-1])
        self.packed = tf.reshape(self.packed, [config.batch_size, config.sample_dim])
        #print(self.packed.get_shape())
        #with tf.Session() as sess:
        #    coord = tf.train.Coordinator()
        #    threads = tf.train.start_queue_runners(coord=coord)
        #    f1 = sess.run(self.packed)
        #    print(f1)
        #    f2 = sess.run(self.packed)
        #    print(f2)
        #    coord.request_stop()
        #    coord.join(threads)

    #def next_batch(self):
    #    if (self.current == 0) and self.shuffle:
    #        random.shuffle(self.indices)
    #    i = self.indices[self.current]
    #    packed = json.load(open(self.data_path+'.'+str(i), 'r'))#self.data[i]
    #    packed = tf.convert_to_tensor(packed, dtype=tf.float32)
    #    #packed = tf.reshape(packed, [self.config.batch_size, self.config.sample_dim])
    #    self.current = (self.current + 1) % self.num_batches
    #    return packed

    #def unpack(self, packed):
    #    config = self.config
    #    #next_batch = numpy.hsplit(packed, config.num_steps)
    #    splitted = tf.split(1, config.num_steps, packed)
    #    X = [sample[:, :config.input_dim] for sample in splitted]
    #    Y = [sample[:, config.input_dim:(config.input_dim+config.output_dim)] for sample in splitted]
    #    #Z = [numpy.rint(sample[:, config.input_dim+1]).astype(int) for sample in next_batch]
    #    #Z = [tf.cast(tf.round(sample[:, config.input_dim+1]), tf.int32) for sample in splitted]
    #    return {'features':X, 'output':Y}

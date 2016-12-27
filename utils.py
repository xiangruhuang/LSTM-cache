from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 
import collections
import os

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy

#def read_numbers(filename):
#    filename_queue = tf.train.string_input_producer([filename])
#    with tf.TextLineReader() as reader:
#        key, value = reader.read(filename_queue)
#    return key, value
#
#read_numbers('../simple-examples/data/ptb.train.txt')

def split_file(config, filename, offset, ratio):
    """Split the file into train, test, validate sets w.r.t. (ratio), each sample contains (config.num_steps) lines """
    with open(filename, 'r') as fin:
        """skip the first (offset) lines"""
        lines = fin.readlines()[offset:]
        num_samples = int(len(lines)/config.num_steps)
        lines = lines[: (num_samples * config.num_steps)]
        indices = numpy.arange(num_samples)
        numpy.random.shuffle(indices)
        print(indices)

        print("#samples=%d, #lines=%d" % (num_samples, len(lines)))

        lens = numpy.asarray(ratio) / numpy.sum(ratio) * num_samples

        train_len = int(lens[0])
        test_len = int(lens[1])
        valid_len = num_samples - train_len - test_len
        
        print("spliting samples w.r.t. (train:test:valid) = (%d:%d:%d)" % (train_len, test_len, valid_len))

        train_indices = numpy.asarray([numpy.arange(index*config.num_steps, (index+1)*config.num_steps, 1) for index in indices[0:train_len]]).flatten()
        print(train_indices.shape)
        #train_indices = [index for index in indices[0:train_len]]
        test_indices = numpy.asarray([numpy.arange(index*config.num_steps, (index+1)*config.num_steps) for index in indices[train_len:train_len + test_len]]).flatten()
        valid_indices = numpy.asarray([numpy.arange(index*config.num_steps, (index+1)*config.num_steps) for index in indices[train_len+test_len:train_len + test_len+valid_len]]).flatten()

        with open(filename + '.train', 'w') as train:
            train.writelines([lines[index] for index in train_indices])
        with open(filename + '.test', 'w') as test:
            test.writelines([lines[index] for index in test_indices])
        with open(filename + '.valid', 'w') as valid:
            valid.writelines([lines[index] for index in valid_indices])

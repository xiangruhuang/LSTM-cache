import tensorflow as tf

from rnn_utils import *
from LSTMModel import LSTMModel
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
import time

def _test1():

    bs=10
    a = LSTMModel.from_raw_params([31, 30, 1], 1, bs, name='lstm')
    b = LSTMModel.from_raw_params([31, 30, 1], 1, bs, name='lstm')
    c = LSTMModel.from_raw_params([31, 30, 1], 1, bs, name='lstm')
    cells = [a, b, c]
    
    inputs = tf.constant(0.0, shape=[bs, 31])
    switch = tf.constant([[True if (i % len(cells) == j) 
        else False for j in range(len(cells))] for i in range(bs)], shape=[bs, len(cells)])
    states = [[cell.initial_state(1) 
                        for cell in cells] 
                        for b in range(bs)]
    
    
    #vals = a.feed_forward(inputs, output_dim=1, activation=tf.sigmoid)
    
    outputs, next_states = switched_batched_feed_forward(cells, inputs, switch, states)

    outputs, next_states = switched_batched_feed_forward(cells, inputs, switch, next_states)
    
    #print([var.initializer for var in tf.global_variables()])
    #$init = tf.global_variables_initializer()
    #$
    #$aaa = a.cell.zero_state(batch_size=5, dtype=tf.float32)
    #$print(a.cell.state_size)
    #$print(nest.flatten(aaa))
    #$l = nest.pack_sequence_as(a.cell.state_size, nest.flatten(aaa))
    #$print(aaa)
    #$print(l)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess, tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        sess.run([outputs, next_states])

def _on_device(fn, device):
    if device:
        with ops.device(device):
            return fn()
    else:
        return fn()

def _test2():
    a = tf.constant(1.0, shape=[10000, 10000])
    b = tf.constant(0.0, shape=[10000, 10000])
    s = tf.placeholder(shape=[1000], dtype=tf.bool)
    
    c = tf.constant(0.0, shape=[10000, 10000])
    
    for i in range(1000):
        #c = _on_device(lambda: array_ops.where(i % 2 == 0, a, b), device=a.op.device)
        c = tf.cond(s[i], lambda: c, lambda: tf.add(c, a))
    
    with tf.Session() as sess:
        st = time.time()
        sess.run(tf.global_variables_initializer())
        cc = sess.run(c, feed_dict={s:[True if i % 2 == 0 else True for i in range(1000)]})
        print(cc)
        ed = time.time()
        print('time=%f' % (ed-st))

_test1()

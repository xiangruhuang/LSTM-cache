import tensorflow as tf

from rnn_utils import *
from LSTMModel import LSTMModel

bs=10
a = LSTMModel.from_raw_params([31, 30, 1], 1, bs, name='lstm')
b = LSTMModel.from_raw_params([31, 30, 1], 1, bs, name='lstm')
c = LSTMModel.from_raw_params([31, 30, 1], 1, bs, name='lstm')
cells = [a, b, c]

inputs = tf.constant(0.0, shape=[bs, 31])
switch = tf.constant([[True if (i % len(cells) == j) else False for j in range(len(cells))] for i in range(bs)], shape=[bs, len(cells)])
states = [[cell.initial_state(1) 
                    for cell in cells] 
                    for b in range(bs)]


#vals = a.feed_forward(inputs, output_dim=1, activation=tf.sigmoid)

outputs, next_states = switched_batched_feed_forward(cells, inputs, switch, states)

#print([var.initializer for var in tf.global_variables()])
init = tf.global_variables_initializer()

aaa = a.cell.zero_state(batch_size=5, dtype=tf.float32)
l = to_tuple_states(a.cell.output_size, to_flat_states(aaa))
print(aaa)
print(l)

#with tf.Session() as sess:
#    sess.run(init)
#    sess.run([outputs, next_states])

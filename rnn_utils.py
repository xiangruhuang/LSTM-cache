from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.python.util import nest
from tensorflow.python.ops import nn_ops
import tensorflow as tf

"""Build a One Step switched feed-forward data flow through a group of RNNCells

Fields:
    cells: list of LSTMModel of size N
    inputs: Tensor or placeholder with shape [batch_size, input_dim]
    switch: Tensor of shape [batch_size, N] and dtype bool, 
        each indicate whether to go through this cell.
    states: list of tuple of size N, total size is 
        [batch_size, [N, 
            (num_layers_n, LSTMStateTuple(2, [1, hidden_size_n_i]))]]

Return:
    outputs: Tensor of shape [batch_size, output_dim]
        , if not flew through any cell
        , fill with zero
    next_state: same shape as state, if not flew through any cell
        , return unchanged state
"""

def switched_batched_feed_forward(cells, inputs, switch, states):
    #print('switched_feed_forward: ', end="")
    #print([cell.fullname for cell in cells])
    [batch_size, input_dim] = inputs.get_shape().as_list()
    N = len(cells)
    outputs = []
    output_dim = 1
    next_states = []
    gate = tf.constant(0)
    zero_output = tf.constant(0.0, shape=[1, output_dim])
    flat_zero_output = nest.flatten(zero_output)
    zero_state = cells[0].initial_state(1)
    flat_zero_state = nest.flatten(zero_state)
    for b in range(batch_size):
        pred_fn_pairs_outputs = []
        next_states_b = []
        
        vals = tf.case([(switch[b, i] , lambda i=i, b=b:
            cells[i].feed_forward(inputs[b:b+1, :], state=states[b][i] ,
                output_dim=output_dim, activation=tf.sigmoid , flat=True)) for i
            in range(N)] , default=lambda:nest.flatten(zero_output)+
            nest.flatten(zero_state) , exclusive = False)
        output_b = vals[:len(flat_zero_output)]
        for structural, flat in zip(output_b, flat_zero_output):
            structural.set_shape(flat.get_shape())
        output_b = nest.pack_sequence_as(structure=zero_output,
            flat_sequence=output_b)
        state_b = vals[len(flat_zero_output):]
        for structural, flat in zip(state_b, flat_zero_state):
            structural.set_shape(flat.get_shape())
        state_b = nest.pack_sequence_as(structure=states[b][i],
            flat_sequence=state_b)
        #print(output_b)
        #print(state_b)
        for i in range(N):
            states[b][i] = cond_tuple(switch[b, i], state_b, states[b][i])
        #next_states.append(next_states_b)
        #output_b = tf.case([(switch[b, i], lambda i=i:
        #        cells[i].feed_forward(inputs[i:i+1, :], state=states[b][i]
        #        , output_dim=output_dim, activation=tf.sigmoid)['outputs'])
        #        for i in range(N)]
        #    , default=lambda:tf.constant(0.0, shape=[1, output_dim])
        #    , exclusive=False)
        
        #output_b = tf.add_n([
        #        cells[i].feed_forward(inputs[i:i+1, :], state=states[b][i]
        #        , output_dim=output_dim, activation=tf.sigmoid)['outputs']
        #        for i in range(N)])
        #output_b.set_shape([1, output_dim])
        assert(output_b.get_shape().as_list() == [1, output_dim])
        outputs.append(output_b)
    outputs = tf.concat(outputs, axis=0)
    assert(outputs.get_shape().as_list() == [batch_size, output_dim])
    
    with tf.get_default_graph().control_dependencies([gate]):
        return outputs, states

"""get a structured tuple as the same shape of cell.state_size

Fields:
    cell: must be an instance of MultiRNNCell
    batch_size: batch size
    dtype: data type
    builder: can be tf.placeholder(default) or any initializer

Returns:
    states: a tuple (LSTMStateTuple_0, ..., LSTMStateTuple_{num_layers-1})
    LSTMStateTuple_i = (c_i, h_i)
    c_i = placeholder(shape=[batch_size_i, hidden_size_i])
    h_i = placeholder(shape=[batch_size_i, hidden_size_i])
"""

def get_cell_states(cell, batch_size, dtype=tf.float32, builder=tf.placeholder):
    if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
        states = []
        for lstm_state in cell.state_size:
            assert isinstance(lstm_state, LSTMStateTuple)
            (size_c, size_h) = lstm_state
            c = builder(dtype=dtype, shape=[batch_size, size_c])
            h = builder(dtype=dtype, shape=[batch_size, size_h])
            states.append(LSTMStateTuple(c, h))
        return tuple(states)
    else:
        raise(ValueError('Invalid cell type'))


"""Generalized tf.cond

Inputs:
    iftrue and iffalse are structured tuple of lambda functions

Returns:
    ans is a structured tuple, same shape as ifture and iffalse

"""

def cond_tuple(switch, iftrue, iffalse):
    flat_iftrue=nest.flatten(iftrue)
    flat_iffalse=nest.flatten(iffalse)
    nest.assert_same_structure(iftrue, iffalse)
    flat_ans = [tf.cond(switch, lambda:flat_iftrue[i], lambda:flat_iffalse[i])
        for i in range(len(flat_iftrue))]
    ans = nest.pack_sequence_as(structure=iftrue, flat_sequence=flat_ans)
    return ans

"""Converted tupled states into list of tensors, meta data will not be included.

Inputs:
    states: cell states of any RNNCell

Outputs:
    tensors: list of tensor: the flatten states
    
"""

def to_flat_states(states):
    return nest.flatten(states)

"""Convert a flatten list of tensors into tuple states.

Inputs:
    structure: meta data of original tuple structure
        , should be returned by <RNNCell>.output_size 
    flat_states: list of tensor
    
Outputs:
    tuple states, looks like RNNCell.zero_state(..)

"""

def to_tuple_states(structure, flat_states):
    return nest.pack_sequence_as(structure=structure, flat_sequence=flat_states)



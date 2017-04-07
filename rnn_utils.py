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

"""Load variables and Return current epoch
Args:
    var_list: list of variables to restore
    load_dir: directory containing checkpoints, latest checkpoint is loaded

Returns:
    current_epoch: i.e. the suffix of the latest checkpoint file
"""
def load_vars(sess, var_list, load_dir):
    if len(var_list) == 0:
        return
    loader = tf.train.Saver(var_list)

    lc = tf.train.latest_checkpoint(load_dir+'/')
    if lc is not None:
        var_names = [v.name for v in var_list]
        print("restoring %s from %s" % (str(var_names), load_dir+'/'))
        loader.restore(sess, lc)
        return int(str(lc).split('ckpt-')[-1])
    else:
        print('nothing exists in %s' % load_dir)
        return -1

"""One Step batched switched matrix multiplication, switch decides which
alternative of weights and biases should the data flow into.

Args:
    inputs: Tensor of Shape [batch_size, input_dim], dtype should be
        tf.float32
    num_alts: number of alternatives
    output_size: int
    switch: Tensor of Shape [batch_size], each element should be in range of
        [0, num_alts), dtype should be tf.int32

Returns:
    linear: Tensor of shape [batch_size, output_size]

weights will have shape [num_alts, input_dim, output_size]
biases will have shape [num_alts, output_size]
weights and biases will be created or retrived from current variable_scope

"""

def switched_linear(inputs, num_alts, output_size, switch):
    [batch_size, input_dim] = inputs.shape.as_list()
    assert switch.shape.as_list() == [batch_size], 'switch.shape'
    name_suffix = '_a%d_i%d_o%d' % (num_alts, input_dim, output_size)
    weight_shape = [num_alts, input_dim, output_size]
    weights = create_or_reuse_variable('weights'+name_suffix, weight_shape,
            tf.float32, tf.random_normal_initializer(dtype=tf.float32))

    bias_shape = [num_alts, output_size]
    biases = create_or_reuse_variable('biases'+name_suffix, bias_shape,
            tf.float32, tf.constant_initializer(0.0, dtype=tf.float32))
 
    selected_weights = tf.gather(weights, switch)
    assert selected_weights.shape.as_list() == [batch_size
            ]+weight_shape[1:]

    selected_biases = tf.gather(biases, switch)
    assert selected_biases.shape.as_list() == [batch_size
            ]+bias_shape[1:]

    matmul = tf.matmul(tf.expand_dims(inputs, 1), selected_weights)
    matmul = tf.squeeze(matmul, axis=1)
    return tf.sigmoid(matmul+selected_biases)

"""One Step batched matrix multiplication

Args:
    inputs: Tensor of Shape [batch_size, input_dim], dtype should be
        tf.float32
    output_size: int

Returns:
    linear: Tensor of shape [batch_size, output_size]

weights will have shape [num_alts, input_dim, output_size]
biases will have shape [num_alts, output_size]
weights and biases will be created or retrived from current variable_scope

"""
def linear(inputs, output_size):
    [batch_size, input_dim] = inputs.shape.as_list()
    name_suffix = '_i%d_o%d' % (input_dim, output_size)
    weight_shape = [input_dim, output_size]
    weights = create_or_reuse_variable('weights'+name_suffix, weight_shape,
            tf.float32, tf.random_normal_initializer(dtype=tf.float32))

    bias_shape = [output_size]
    biases = create_or_reuse_variable('biases'+name_suffix, bias_shape,
            tf.float32, tf.constant_initializer(0.0, dtype=tf.float32))

    matmul = tf.matmul(inputs, weights)
    return tf.sigmoid(tf.nn.bias_add(matmul, biases))

"""
compute a + b
    
Args:
    new_index: Tensor of same shape as value
    axis: D_{axis} will be added

Returns:
    A Sparse Tensor of dense_shape [D_0*...*D_{N-1}, N+1]
        same value and indices
"""

def expand_indices(new_index, axis=0, dtype=tf.int64):
    assert isinstance(new_index, tf.Tensor), "new_index should be a Tensor"
    dims = new_index.shape.as_list()
    args = (range(dim) for dim in dims)
    indices = tf.meshgrid(*args, indexing = 'ij')
    indices.insert(axis, new_index)
    for i in range(len(indices)):
        if indices[i].dtype != dtype:
            indices[i] = tf.cast(indices[i], dtype)
    indices = tf.stack(indices, axis=-1)
    indices = tf.reshape(indices, shape=[-1, (len(dims)+1)])
    return indices 

def create_or_reuse_variable(name, shape, dtype, initializer, scope=None):
    if scope is None:
        scope = tf.get_variable_scope()
    global_name = scope.name+'/'+name 
    existing_vars = [v for v in tf.global_variables() if
            v.name.startswith(global_name)]
    if len(existing_vars) > 0:
        scope.reuse_variables()
    return tf.get_variable(name, shape, dtype, initializer=initializer)


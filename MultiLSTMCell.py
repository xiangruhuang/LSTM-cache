#from __future__ import print_function
#from __future__ import division

from utils import *

from rnn_utils import *

#import tensorflow as tf
#import numpy
#from tensorflow.python.util import nest

class OneHotLSTM(tf.contrib.rnn.BasicLSTMCell):
    """
    Args:
        num_units_list: a list of length num_alt, each element is an integer
        indicating how many neurons that alternative has.
    """


    def __init__(self, num_units_each, num_alts, dtype=tf.float32):
        self._num_alts = num_alts
        self._num_units_each = num_units_each
        self._num_units = num_alts*num_units_each
        self._dtype = dtype
        super(OneHotLSTM, self).__init__( num_units=self._num_units)

    """One Step batched Feed Forward, with an extra switch indicating which
    alternative of weights and biases should the data flow into.
    
    Args:
        inputs: Tensor of Shape [batch_size, input_dim], dtype should be
            tf.float32
        state: LSTMTuple (c, h), c and h have shape 
            [batch_size, self._num_alts, self._num_units_each]
        switch: Tensor of Shape [batch_size], each element should be in range of
            [0, self._num_alts), dtype should be tf.int32

    Returns:
        state: same nested shape as input arg 'state'

    """
    def __call__(self, inputs, state, switch):
        [batch_size, input_dim] = inputs.shape.as_list()
        assert switch.shape[0] == batch_size
        switch2 = tf.stack([tf.range(0, batch_size, dtype=tf.int64), switch],
                axis=1)
        """Now Switch has shape [batch_size, 2]"""
        assert switch2.shape.as_list() == [batch_size, 2]
        
        assert isinstance(state, tf.contrib.rnn.LSTMStateTuple)
        assert state[0].shape.as_list() == state[1].shape.as_list()
        assert state[0].shape.as_list() == [batch_size, self._num_alts,
                self._num_units_each], "shape=%s, b=%d, a=%d, ne=%d" % \
        (str(state[0].shape), batch_size, self._num_alts, self._num_units_each)

        """Looks for variable_scope that contains weights and biases has input
        dimension=input_dim"""
        c, h = state

        """Select weights and biases by switch""" 
        
        selected_c = tf.gather_nd(c, switch2)
        assert selected_c.shape.as_list() == [batch_size,
            self._num_units_each]
        selected_h = tf.gather_nd(h, switch2)
        assert selected_h.shape.as_list() == [batch_size,
            self._num_units_each]
        
        #concat = _linear([inputs, selected_h], 4 *
        #        self._num_units_each, True)
        
        concat = tf.concat([inputs, selected_h], axis=1)
        #concat = tf.expand_dims(concat, 1)
        #assert concat.shape.as_list() == [batch_size, 1,
        #        input_dim+self._num_units_each]
        linear = switched_linear(concat, self._num_alts,
            self._num_units_each*4, switch)
        #matmul = tf.matmul(concat, selected_weights)
        #matmul = tf.squeeze(matmul, axis=1)
        #assert matmul.shape.as_list() == selected_biases.shape.as_list()
        
        #linear = tf.add(matmul, selected_biases)

        i, j, f, o = tf.split(value=linear, num_or_size_splits=4,
            axis=1)

        new_c = (selected_c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
             self._activation(j))
        new_h = self._activation(new_c) * tf.sigmoid(o)

        new_index, _ = tf.meshgrid(switch, tf.range(self._num_units_each,
            dtype=tf.int64), indexing='ij')
        indices = expand_indices(new_index, axis=1)
        #indices0, _ = tf.meshgrid(tf.range(0, batch_size), tf.range(0,
        #    self._num_units_each))
        #indices1, indices2 = tf.meshgrid(switch, tf.range(0, self._num_units_each))

        #indices0 = tf.reshape(tf.transpose(indices0), shape=[-1])
        #indices1 = tf.reshape(tf.transpose(indices1), shape=[-1])
        #indices2 = tf.reshape(tf.transpose(indices2), shape=[-1])
        #indices = tf.stack([indices0, indices1, indices2], axis=-1)
        indices = tf.to_int64(indices)

        delta_c = tf.reshape(new_c-selected_c,
            shape=[batch_size*self._num_units_each])
        sparse_delta_c = tf.SparseTensor(indices=indices, values = delta_c,
            dense_shape=c.shape.as_list())
        
        delta_h = tf.reshape(new_h-selected_h,
            shape=[batch_size*self._num_units_each])
        sparse_delta_h = tf.SparseTensor(indices=indices, values = delta_h,
            dense_shape=h.shape.as_list())

        c = tf.sparse_add(c, sparse_delta_c)
        h = tf.sparse_add(h, sparse_delta_h)

        new_state = tf.contrib.rnn.LSTMStateTuple(c, h)

        return new_h, new_state

    @property
    def num_alts(self):
        return self._num_alts

    @property
    def num_units(self):
        return self._num_units_each*self._num_alts

    @property
    def state_size(self):
        assert self._state_is_tuple
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    def zero_state(self, batch_size, dtype=tf.float32):
        state_shape = [batch_size, self._num_alts, self._num_units_each]
        state_c = tf.zeros(shape=state_shape, dtype=dtype)
        state_h = tf.zeros(shape=state_shape, dtype=dtype)
        return tf.contrib.rnn.LSTMStateTuple(state_c, state_h)

class MultiLSTMCell(tf.contrib.rnn.MultiRNNCell):

    """Internal Class for compactly storing Parameters
    Fields:
        dims: 
            dims[0] is input dimension (can be None),
            dims[1:-1] is hidden layers' size,
            dims[-1] is output dimension,
            e.g. [10, 20, 30, 1] means a input layer of dimension 10, 
                then two hidden layers of dimension 20 and 30, 
                then a output layer of dimension 1
    """
    class Params(object):
        def __init__(self, dims=None, num_steps=None, batch_size=None, 
                reuse=False, name='default', name_offset=None,
                num_alts=1):
            """Params.name is a template, while LSTMModel.name is 
                a instance (usually suffixed with unique numbers)"""
            self.name = name
            self.name_offset = name_offset
            assert len(dims) >= 3, 'Constructing LSTMModel.Params(%s) \
            at least one input, hidden, output layer' % name
            self.input_dim = dims[0]
            self.hidden_sizes = dims[1:-1]
            self.output_dim = dims[-1]
            self.num_layers = len(self.hidden_sizes)
            self.num_steps = num_steps
            
            self.num_steps = num_steps
            self.batch_size = batch_size
            self.reuse = reuse
            self.num_alts = num_alts

    @classmethod
    def _get_unique_name(cls, name):
        if not hasattr(cls, 'name_count_map'):
            cls.name_count_map = {}
        if not (name in cls.name_count_map):
            cls.name_count_map[name] = 0
        name_count = cls.name_count_map[name]
        cls.name_count_map[name] += 1
        fullname = name + str(name_count)
        return fullname

    def __init__(self, params):
        assert isinstance(params, self.Params), 'params should be an instance\
        of self.Params'
        self.params = params
        self.scope_map = {}

        """different variables/tensors/operations should have different names
            best way to manage names is to use scopes
        """
        fullname = self._get_unique_name(params.name)
        while (params.name_offset is not None) and (not
                fullname.endswith(params.name_offset)):
            fullname = self._get_unique_name(params.name)
        self._fullname = fullname
        with tf.variable_scope(self._fullname) as scope:
            if self.params.num_alts > 1:
                lstm_cells = [OneHotLSTM(num_units_each=hidden_size_i,
                    num_alts=self.params.num_alts) for hidden_size_i in
                    params.hidden_sizes]

            else:
                lstm_cells = [tf.contrib.rnn.BasicLSTMCell(
                    num_units=hidden_size_i, state_is_tuple=True) for
                    hidden_size_i in params.hidden_sizes]
        super(MultiLSTMCell, self).__init__(lstm_cells)

    """One Step batched Feed Forward, with an extra switch indicating which
    alternative of weights and biases should the data flow into.
    
    Args:
        inputs: Tensor of Shape [batch_size, input_dim], dtype should be
            tf.float32
        state: LSTMTuple (c, h), c and h have shape 
            [batch_size, self._num_alts, self._num_units_each]
        switch: Tensor of Shape [batch_size], each element should be in range of
            [0, self._num_alts), dtype should be tf.int32

    Returns:
        state: same nested shape as input arg 'state'

    """
    def __call__(self, inputs, state, switch):
        cur_inp = inputs
        new_states = []
        for i, cell in enumerate(self._cells):
            with tf.variable_scope('cell_%d' % i):
                if not nest.is_sequence(state):
                    raise ValueError("state should be a tuple")
                cur_state = state[i]
                if isinstance(cell, OneHotLSTM):
                    cur_inp, new_state = cell(cur_inp, cur_state, switch)
                else:
                    cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states
        #[batch_size, input_dim]=inputs.get_shape().as_list()
        #if self.params.num_alts > 1:
        #    assert switch is not None, "need a switch to guide feed forward"
        #    assert switch.shape.as_list() == [batch_size], "switch.shape"
        #if state is None:
        #    state = self.cell.zero_state(batch_size, tf.float32)
        #with tf.variable_scope( self.fullname+'/inputs_'+str(input_dim) ) as \
        #        scope:
        #    count = self.count_map.get(scope.name, 0)
        #    if count > 0:
        #        scope.reuse_variables()
        #    if self.params.num_alts > 1:
        #        outputs, next_state = self.cell(inputs, state, switch)
        #    else:
        #        outputs, next_state = self.cell(inputs, state)

        #    if count == 0:
        #        var_sizes = [numpy.product(list(map(int,
        #            v.get_shape())))*v.dtype.size for v in
        #            tf.global_variables()]
        #        print("creating variables, current variables size=%f MB" %
        #            (sum(var_sizes)/(1024**2)))
        #    self.count_map[scope.name] = count + 1
        #with tf.variable_scope(self.fullname+'/outputs_'+str(output_dim)) as \
        #        scope:
        #    count = self.count_map.get(scope.name, 0)
        #    if count > 0:
        #        scope.reuse_variables()
        #    if output_dim is not None:
        #        output_weights = tf.get_variable('out_weights' ,
        #            [self.params.hidden_sizes[-1], output_dim] , initializer =
        #            tf.random_normal_initializer())
        #        output_biases = tf.get_variable('out_biases' , [output_dim],
        #            initializer=tf.constant_initializer(0.0))
        #        outputs = tf.nn.bias_add(tf.matmul(outputs,
        #            output_weights), output_biases)
        #        if activation is not None:
        #            outputs = activation(outputs)
        #    if count == 0:
        #        var_sizes = [numpy.product(list(map(int,
        #            v.get_shape())))*v.dtype.size for v in
        #            tf.global_variables()]
        #        print("creating variables, current variables size=%f MB" %
        #            (sum(var_sizes)/(1024**2)))
        #    self.count_map[scope.name] = count + 1
        #if flat:
        #    return nest.flatten(outputs) + nest.flatten(next_state)
        #else:
        #    return outputs, next_state
    def zero_state(self, batch_size, dtype):
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)


#from flags import *
#from Config import *

#from tensorflow.python.ops import control_flow_ops
#from tensorflow.python.util import nest
#from tensorflow.python.ops import array_ops
#from tensorflow.python.framework import tensor_shape

import tensorflow as tf

def initialize_rnn_state(state):
    """Return the initialized RNN state.
    The input is LSTMStateTuple or State of RNNCells.
    Parameters
    -----------
    state : a RNN state.
    """
    if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
        c = state.c.eval()
        h = state.h.eval()
        return (c, h)
    else:
        new_state = state.eval()
        return new_state

class LSTMConfig(object):
    def __init__(self, dims=None, num_steps=None, batch_size=None, reuse=False, name='default'):
        self.name = name
        assert(len(dims) >= 3, 'Constructing LSTMConfig(%s) at least one input layer, one hidden layer, one output layer' % name)
        self.input_dim = dims[0]
        self.hidden_sizes = dims[1:-1]
        self.output_dim = dims[-1]
        self.num_layers = len(self.hidden_sizes)

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.reuse = reuse

    @property
    def input_shape(self):
        return tf.placeholder(tf.float32, shape=(self.num_steps, self.batch_size, self.input_dim))

    @property
    def output_shape(self):
        return tf.placeholder(tf.float32, shape=(self.num_steps, self.batch_size, self.output_dim))

    def to_string(self):
        s = 'LSTMConfig(' + self.name + '):'
        s += '\n\tinput dim=' + str(self.input_dim)
        s += '\n\thidden layer sizes=' + str(self.hidden_sizes)
        s += '\n\toutput dim=' + str(self.output_dim)
        s += '\n\tnumber of time steps=' + str(self.num_steps)
        s += '\n\tbatch size=' + str(self.batch_size)
        return s


name_count_map = {}

def get_unique_name(name):
    assert(name != None, 'get_unique_name: name can not be None')
    global name_count_map
    if not (name in name_count_map.keys()):
        name_count_map[name] = 0
    name_count = name_count_map[name]
    fullname = name + str(name_count)
    name_count_map[name] += 1
    return fullname

class LSTMModel(object):
    def __init__(self, config, packed_inputs=None):
        assert(isinstance(config, LSTMConfig), 'LSTMModel constructor: config must be instance of LSTMConfig')
        fullname = get_unique_name(config.name)
        with tf.variable_scope(fullname) as scope:
            self.config = config
        
            self.weights = {'out':tf.Variable(tf.random_normal([config.hidden_sizes[-1], config.output_dim]), trainable=True)}
            self.biases = {'out':tf.Variable(tf.random_normal([config.output_dim]), trainable=True)}
            lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size_i, forget_bias=1.0, state_is_tuple=True) for hidden_size_i in config.hidden_sizes]
            
            self.cell = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)
            self.init = self.cell.zero_state(config.batch_size, tf.float32)
            #if packed_inputs == None:
            #    self.packed_inputs = tf.placeholder(tf.float32, [config.num_steps, config.batch_size, config.input_dim])
            #else:
            #    self.packed_inputs = packed_inputs

            """inputs has shape [config.num_steps, config.batch_size, config.input_dim]"""
            """unpacked has shape [batch_size, config.input_dim] * config.num_steps """

            """get a placeholder that has same shape as self.cell.initial_state"""
            #state_size = self.cell.state_size
            #state_size_flat = nest.flatten(state_size)
            #self.state_flat = [tf.placeholder(shape=[config.batch_size, s], dtype=tf.float32) for s in state_size_flat]
            #self.state = nest.pack_sequence_as(structure=state_size, flat_sequence=self.state_flat)
            #self.input = tf.placeholder(tf.float32, shape=[config.batch_size, config.input_dim])
            #self.output, self.next_state = self.cell(self.input, self.state)
            #self.output = tf.sigmoid(tf.matmul(self.output, self.out_weights)+self.out_biases)
            #
            #scope.reuse_variables()
            #"""get packed outputs in shape [num_steps, batch_size, output_dim]"""
            #self.packed_outputs = self.feed_forward(self.packed_inputs, self.cell.zero_state(config.batch_size, tf.float32))

            #print('generating LSTM (%s) with input size: %s' % (name, self.packed_inputs.get_shape()))
            #print('                         output size: %s' % (self.packed_outputs.get_shape()))
            #print('             One Step     input size: %s' % (self.input.get_shape()))
            #print('                         output size: %s' % (self.output.get_shape()))
            #
            #lstm_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]
            #print([v.name for v in lstm_variables])
            #self.initializers = [v.initializer for v in lstm_variables]

    def feed_forward(self, packed_inputs, start_state):
        """packed inputs has shape [config.num_steps, config.batch_size, config.input_dim]"""
        state = start_state
        outputs = []
        for input_t in tf.unpack(packed_inputs, axis=0):
            (output_t, state) = self.cell(input_t, state)
            output_t = tf.sigmoid(tf.matmul(output_t, self.out_weights)+self.out_biases)
            outputs.append(output_t)
        return tf.pack(outputs, axis=0)

    def initialize(self, sess):
        sess.run(self.initializers)
        self.clear_state(sess)

    def clear_state(self, sess):
        self.current_state = sess.run(self.init)

    def predict(self, sess, feature):
        feed_dict={}
        for i, d in zip(self.state, self.current_state):
            feed_dict[i] = d
        feed_dict[self.input] = [feature]
        (outputs, self.current_state) = sess.run([self.output, self.next_state], feed_dict=feed_dict)
        return outputs


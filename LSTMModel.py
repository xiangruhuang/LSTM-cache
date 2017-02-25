from __future__ import print_function

import tensorflow as tf

class LSTMModel(object):

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
                reuse=False, name='default'):
            """Params.name is a template, while LSTMModel.name is 
                a instance (usually suffixed with unique numbers)"""
            self.name = name
            assert(len(dims) >= 3, 'Constructing LSTMModel.Params(%s)' 
                'at least one input, hidden, output layer' % name)
            self.input_dim = dims[0]
            self.hidden_sizes = dims[1:-1]
            self.output_dim = dims[-1]
            self.num_layers = len(self.hidden_sizes)
            self.num_steps = num_steps
            
            self.num_steps = num_steps
            self.batch_size = batch_size
            self.reuse = reuse
    
        @property
        def input_shape(self, num_steps, batch_size):
            return tf.placeholder(tf.float32, 
                    shape=(num_steps, batch_size, self._input_dim))
    
        @property
        def output_shape(self, num_steps, batch_size):
            return tf.placeholder(tf.float32, 
                    shape=(num_steps, batch_size, self._output_dim))
    
        def to_string(self):
            s  = '\tname=' + self.name+'\n'
            s += '\tinput dim=' + str(self.input_dim)+'\n'
            s += '\thidden layer sizes=' + str(self.hidden_sizes)+'\n'
            s += '\toutput dim=' + str(self.output_dim)+'\n'
            s += '\tnum steps=' + str(self.num_steps)+'\n'
            s += '\tbatch size=' + str(self.batch_size)
            return s

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

    def get_scope(self, name, reuse=False):
        if not hasattr(self, 'scope_map'):
            self.scope_map={}
        if not (name in self.scope_map):
            self.scope_map[name] = tf.VariableScope(None, name)
        elif reuse:
            self.scope_map[name].reuse_variables()
        return self.scope_map[name]

    def __init__(self, params):
        assert(isinstance(params, self.Params), "params should be an instance of"
            "LSTMModel.Params")
        self.params = params
        
        """different variables/tensors/operations should have different names
            best way to manage names is to use scopes
        """
        fullname = self._get_unique_name(params.name)
        self.fullname = fullname
        with tf.variable_scope(self.get_scope(fullname), reuse=False):
            lstm_cells = [tf.contrib.rnn.BasicLSTMCell(
                num_units=hidden_size_i, forget_bias=1.0, state_is_tuple=True
                ) for hidden_size_i in params.hidden_sizes]
            self.cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
    
    @classmethod
    def from_raw_params(cls, dims, num_steps, batch_size, 
            reuse=False, name='default'):
        params = cls.Params(dims, num_steps, batch_size, reuse, name)
        return cls(params)
        
    """Build a One Step feed-forward data flow.
    
    Fields:
        inputs: Tensor or placeholder with shape [batch_size, input_dim]
        state: Cell state to start with, if it is None, use cell.zero_state.
    
    Return:
        outputs: Tensor of shape [batch_size, output_dim]
        next_state: same shape as state
    """
    def feed_forward(self, inputs, state=None, output_dim=None,
            activation=None):
        [batch_size, input_dim]=inputs.get_shape().as_list()
        inputs = tf.Print(inputs, [0], self.fullname+'/feed_forward'+str(output_dim))
        with tf.get_default_graph().control_dependencies([inputs]):
            if state is None:
                state = self.cell.zero_state(batch_size, tf.float32)
            with tf.variable_scope(self.get_scope(self.fullname+'/inputs_'+str(input_dim)
                , reuse=True)):
                outputs, next_state = self.cell(inputs, state=state)
            with tf.variable_scope(self.get_scope(self.fullname+'/outputs_'+str(output_dim)
                , reuse=True)): 
                #outputs = tf.Print(outputs, [output_dim], 'feed forward on '+self.fullname)
                if output_dim is not None:
                    output_weights = tf.get_variable('out_weights'
                            , initializer=tf.random_normal(
                                [self.params.hidden_sizes[-1], output_dim]))
                    output_biases = tf.get_variable('out_biases'
                            , initializer=tf.random_normal([output_dim]))
                    #output_weights = tf.Variable(tf.random_normal(
                    #[self.params.hidden_sizes[-1], output_dim]), name='out_weights'+str(output_dim))
                    #output_biases = tf.Variable(tf.random_normal([output_dim])
                    #, name='out_biases'+str(output_dim))
                    outputs = tf.matmul(outputs, output_weights) + output_biases
                    if activation is not None:
                        outputs = activation(outputs)
            #outputs = tf.Print(outputs, [outputs], self.fullname+'/feed_forward'+str(output_dim))
            #with tf.get_default_graph().control_dependencies([outputs]):
            #outputs = tf.Print(outputs, [outputs], self.fullname+'/feed_forward'+str(output_dim)+'_outputs:'+message)
            ans = {}
            #with tf.get_default_graph().control_dependencies([outputs]):
            ans['outputs'] = outputs
            ans['states'] = next_state
            return ans


    def initial_state(self, batch_size):
        return self.cell.zero_state(batch_size, dtype=tf.float32)


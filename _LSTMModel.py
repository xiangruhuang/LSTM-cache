from flags import *
from itertools import chain
#from Config import *
#from Reader import *

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape

class LSTMModel(object):
    def __init__(self, config):
        print("initializing LSTM Model:")
        print("\t #hidden units=%d, #layers=%d" % (config.hidden_size, config.num_layers) )
        
        self.config = config
        self.weights = {'out':tf.Variable(tf.random_normal([config.hidden_size, config.output_dim]), trainable=True)}
        self.biases = {'out':tf.Variable(tf.random_normal([config.output_dim]), trainable=True)}

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_size, forget_bias=0)
        """Dropout"""
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
    
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        self.init = self.cell.zero_state(config.batch_size, tf.float32)
        
        if config.mode=='online':
            self.input = [tf.placeholder(shape=(1, config.input_dim), dtype=tf.float32)]

            state_size = self.cell.state_size
            state_size_flat = nest.flatten(state_size)
            self.state_flat = [tf.placeholder(shape=[1, s], dtype=tf.float32) for s in state_size_flat]
            with tf.variable_scope("RNN"):
                self.state = nest.pack_sequence_as(structure=state_size, flat_sequence=self.state_flat)
                (self.output, self.next_state) = rnn.rnn(self.cell, self.input, initial_state=self.init, dtype=tf.float32)
                self.output = tf.pack([tf.sigmoid(tf.matmul(output_t, self.weights['out']) + self.biases['out']) for output_t in self.output])

    def run_batch(self, packed):
        """packed has shape [batch_size, (config.num_steps*(config.input_dim+config.output_dim))]"""
        config = self.config
        
        """splitted has shape [batch_size, (config.input_dim + config.output_dim)]*config.num_steps"""
        splitted = tf.split(1, config.num_steps, packed)

        """each sample has shape [batch_size, (config.input_dim + config.output_dim)]"""
        """X has shape [batch_size, config.input_dim] * config.num_steps """
        X = [sample[:, :config.input_dim] for sample in splitted]

        """time_steps * batch_size * input_dim"""
        print(len(X), X[0].get_shape())

        """Y has shape [batch_size, config.output_dim] * config.num_steps """
        Y = tf.pack([sample[:, config.input_dim:(config.input_dim+config.output_dim)] for sample in splitted])

        with tf.variable_scope("RNN") as scope:
            if self.config.mode == 'online':
                scope.reuse_variables()
            outputs, states = rnn.rnn(self.cell, X, initial_state=self.init, dtype=tf.float32)
            #return outputs, states
            pred = tf.pack([tf.sigmoid(tf.matmul(output_t, self.weights['out']) + self.biases['out']) for output_t in outputs])
            int_pred = tf.round(pred)
        #Y = tf.Print(Y, [tf.reshape(Y, [-1])])
        acc = 1.0-tf.reduce_mean(tf.abs(tf.sub(int_pred, Y)))
        cost = tf.nn.l2_loss(tf.sub(pred, Y))
        #variable_summaries(acc)
        #variable_summaries(cost)
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
        #cost = tf.Print(cost, [cost], message='cost=')
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        #get a list of tuple (gradient, variable)
        grads_and_vars = optimizer.compute_gradients(cost)
        optimizer = optimizer.apply_gradients(grads_and_vars)
        return {'acc':acc, 'cost':cost, 'opt':optimizer, 'pred':int_pred, 'grad':grads_and_vars, 'Y':Y}
        #return {'acc':acc, 'cost':cost, 'optimizer':optimizer}

    def predict(self, sess, f):#instr_addr, prob):
        this_input = [f]#numpy.zeros([1, self.config.num_instr+1])
        #this_input[0, int(instr_addr)] = 1.0
        #this_input[0, -1] = prob
        this_input = [this_input]
        feed_dict = {}
        for i, d in zip(self.state, self.current_state):
            feed_dict[i] = d
        #{i: d for i, d in zip(self.state, self.current_state)}
        for i, d in zip(self.input, this_input):
            feed_dict[i] = d
        (output, self.current_state) = sess.run([self.output, self.next_state], feed_dict=feed_dict)
        return output

    def inite(self, sess):
        self.current_state=sess.run(self.cell.zero_state(1, dtype=tf.float32))
        #print(current_state)

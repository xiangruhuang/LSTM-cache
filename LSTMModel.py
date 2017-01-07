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
        #variable_summaries(self.weights['out'])
        self.biases = {'out':tf.Variable(tf.random_normal([config.output_dim]), trainable=True)}
        #variable_summaries(self.biases['out'])
        #self.train_reader = train_reader
        #self.test_reader = test_reader

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_size, forget_bias=0)
        """Dropout"""
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
    
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        self.init = self.cell.zero_state(config.batch_size, tf.float32)
        
        self.input = [tf.placeholder(shape=(1, config.input_dim), dtype=tf.float32)]

        state_size = self.cell.state_size
        state_size_flat = nest.flatten(state_size)
        self.state_flat = [tf.placeholder(shape=[1, s], dtype=tf.float32) for s in state_size_flat]
        with tf.variable_scope("RNN"):
            self.state = nest.pack_sequence_as(structure=state_size, flat_sequence=self.state_flat)
            (self.output, self.next_state) = rnn.rnn(self.cell, self.input, initial_state=self.state, dtype=tf.float32)
            self.output = tf.pack([tf.sigmoid(tf.matmul(output_t, self.weights['out']) + self.biases['out']) for output_t in self.output])

    #def run_epoch(self, session, fetches, num_batches):
    #    #self.pred = [tf.sigmoid(tf.matmul(output_t, self.weights['out']) + self.biases['out']) for output_t in outputs]
    #    #self.accuracy = tf.reduce_mean(tf.abs(tf.sub(self.pred, reader.Y)))
    #   
    #    accuracy = 0
    #    for i in range(num_batches):
    #        vals = session.run(fetches)
    #        print(vals["outputs"])
    #        #accuracy += vals["accuracy"]
    #   
    #    return accuracy/num_batches

    def run_batch(self, packed):
        config = self.config
        splitted = tf.split(1, config.num_steps, packed)
        #indices = [tf.to_int64(tf.pack(list(chain.from_iterable(([t, s_t[0]], [t, (config.num_instr+1)]) for t, s_t in enumerate(tf.unpack(sample)))))) for sample in splitted]
        #print(indices[0].get_shape())
        #values = [tf.pack(list(chain.from_iterable((1, s_t[1]) for s_t in tf.unpack(sample)))) for sample in splitted]
        #print(values[0].get_shape())
        #shape = tf.constant([config.num_steps, (config.num_instr+1)], dtype=tf.int64)
        X = [sample[:, :config.input_dim] for sample in splitted]
        print(sample[0].get_shape())
        #X = []
        #for sample in splitted:
        #    a = []
        #    for t, s_t in enumerate(tf.unpack(sample)):
        #        indices = tf.to_int64(tf.pack([[tf.to_int64(s_t[0])], [tf.constant(config.num_instr, dtype=tf.int64)]]))
        #        values = tf.pack([tf.constant(1.0), s_t[1]])
        #        b = tf.SparseTensor(indices=indices, values=values, shape=tf.constant([config.num_instr+1], dtype=tf.int64))
        #        b = tf.sparse_tensor_to_dense(b)
        #        a.append(b)
        #    a = tf.pack(a)
        #    X.append(a)
        """time_steps * batch_size * input_dim"""
        print(len(X), X[0].get_shape())

        Y = tf.pack([sample[:, config.input_dim:(config.input_dim+config.output_dim)] for sample in splitted])
        #Z = [tf.cast(tf.round(sample[:, config.input_dim+1]), tf.int32) for sample in splitted]

        #X[0] = tf.Print(X[0], [X[0].get_shape()])

        with tf.variable_scope("RNN") as scope:
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

from flags import *
from itertools import chain
#from Config import *
#from Reader import *

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape

class LSTMModel(object):
    def __init__(self, config, train_reader, test_reader):
        print("initializing LSTM Model:")
        print("\t #hidden units=%d, #layers=%d" % (config.hidden_size, config.num_layers) )
        
        self.config = config
        self.weights = {'out':tf.Variable(tf.random_normal([config.hidden_size, config.output_dim]))}
        self.biases = {'out':tf.Variable(tf.random_normal([config.output_dim]))}
        self.train_reader = train_reader
        self.test_reader = test_reader

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=config.hidden_size, forget_bias=0)
        """Dropout"""
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
    
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        #self.init = tf.initialize_all_variables()
        self.init = self.cell.zero_state(config.batch_size, tf.float32)
       
        self.input = [tf.placeholder(shape=(1, config.num_instr+1), dtype=tf.float32)]
        #outputs, states = rnn.rnn(self.cell, X, initial_state=self.init, dtype=tf.float32)
        #s = [config.batch_size, s] for s in self.cell.state_size
        
        #state_size_flat = nest.flatten(self.cell.state_size)
        #print(len(self.init), self.init[0].c)

        state_size = self.cell.state_size
        state_size_flat = nest.flatten(state_size)
        self.state_flat = [tf.placeholder(shape=[1, s], dtype=tf.float32) for s in state_size_flat]
        #for s, z in zip(state_size_flat, state_flat):
        #    z.set_shape([s])
        with tf.variable_scope("RNN"):
            self.state = nest.pack_sequence_as(structure=state_size, flat_sequence=self.state_flat)
            #print(self.cell.state)
            (self.output, self.next_state) = rnn.rnn(self.cell, self.input, initial_state=self.state, dtype=tf.float32)
            self.output = tf.pack([tf.sigmoid(tf.matmul(output_t, self.weights['out']) + self.biases['out']) for output_t in self.output])
        #state_variables = []
        #for state_c, state_h in self.cell.zero_state(config.batch_size, tf.float32):
        #    state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(tf.Variable(state_c, trainable=False), tf.Variable(state_h, trainable=False))
        #update_ops = []
        #for state_var, new_state in zip(state_variables, new_states):
        #    update_ops.extend([state_var[0].assign(new_state[0]), state_var[1].assign(new_state[1])])
        #update_ops = tf.tuple(update_ops)

        #self.cell = lstm_cell

        #self.is_training = tf.placeholder(tf.bool)

        #self.X = [tf.select(self.is_training, train_reader.X[t], test_reader.X[t]) for t in range(config.num_steps)]
        #self.Y = [tf.select(self.is_training, train_reader.Y[t], test_reader.Y[t]) for t in range(config.num_steps)]

        #self.X = [tf.placeholder(tf.float32, shape=(config.batch_size, config.input_dim))] * config.num_steps #config.input_shape
        #self.Y = config.output_shape

        #self.X = tf.Print(self.X, [self.X.get_shape()])

        #self.X = [tf.squeeze(X_t) for X_t in tf.split(0, config.num_steps, self.X)]
        #self.Y = [tf.squeeze(Y_t) for Y_t in tf.split(0, config.num_steps, self.Y)]

        #a = train_reader.X
        #b = test_reader.X

        #with tf.name_scope("train"):
        
        #with tf.variable_scope("network") as scope:
        #    train_outputs, states = rnn.rnn(self.cell, train_reader.X, dtype=tf.float32)
        #    self.train_pred = [tf.squeeze(tf.sigmoid(tf.matmul(output_t, self.weights['out']) + self.biases['out'])) for output_t in train_outputs]
        #    self.train_acc = tf.reduce_mean(tf.abs(tf.sub(self.train_pred, train_reader.Y)))
        #    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.train_pred, train_reader.Y))
        #    self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.cost)
        #    
        #    scope.reuse_variables()
        #    test_outputs, states = rnn.rnn(self.cell, test_reader.X, dtype=tf.float32)
        #    self.test_pred = [tf.squeeze(tf.sigmoid(tf.matmul(output_t, self.weights['out']) + self.biases['out'])) for output_t in test_outputs]
        #    self.test_acc = tf.reduce_mean(tf.abs(tf.sub(self.test_pred, test_reader.Y)))

        #self.cell = tf.Print(self.cell, [self.cell._num_units])
        
        #state = self.cell.zero_state(config.batch_size, tf.float32)
        #output, state = self.cell(test_reader.X[0], state)
        

        #self.pred = tf.Print(self.pred, [self.pred])
        #reader.Y = tf.Print(reader.Y, [reader.Y])
        
        #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        #sess = tf.InteractiveSession()

        #a = tf.Print(output, [output], message="Output: ")

        ## Add more elements of the graph using a
        #b = tf.add(a, a).eval()

    def run_epoch(self, session, fetches, num_batches):
        #self.pred = [tf.sigmoid(tf.matmul(output_t, self.weights['out']) + self.biases['out']) for output_t in outputs]
        #self.accuracy = tf.reduce_mean(tf.abs(tf.sub(self.pred, reader.Y)))
       
        accuracy = 0
        for i in range(num_batches):
            vals = session.run(fetches)
            print(vals["outputs"])
            #accuracy += vals["accuracy"]
       
        return accuracy/num_batches

    def run_batch(self, packed):
        config = self.config
        splitted = tf.split(1, config.num_steps, packed)
        #indices = [tf.to_int64(tf.pack(list(chain.from_iterable(([t, s_t[0]], [t, (config.num_instr+1)]) for t, s_t in enumerate(tf.unpack(sample)))))) for sample in splitted]
        #print(indices[0].get_shape())
        #values = [tf.pack(list(chain.from_iterable((1, s_t[1]) for s_t in tf.unpack(sample)))) for sample in splitted]
        #print(values[0].get_shape())
        #shape = tf.constant([config.num_steps, (config.num_instr+1)], dtype=tf.int64)
        X = []
        for sample in splitted:
            a = []
            for t, s_t in enumerate(tf.unpack(sample)):
                indices = tf.to_int64(tf.pack([[tf.to_int64(s_t[0])], [tf.constant(config.num_instr, dtype=tf.int64)]]))
                values = tf.pack([tf.constant(1.0), s_t[1]])
                b = tf.SparseTensor(indices=indices, values=values, shape=tf.constant([config.num_instr+1], dtype=tf.int64))
                b = tf.sparse_tensor_to_dense(b)
                a.append(b)
            a = tf.pack(a)
            X.append(a)
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
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
        #cost = tf.Print(cost, [cost], message='cost=')
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate);
        #get a list of tuple (gradient, variable)
        grads_and_vars = optimizer.compute_gradients(cost)
        optimizer = optimizer.apply_gradients(grads_and_vars)
        return {'acc':acc, 'cost':cost, 'opt':optimizer, 'pred':int_pred, 'grad':grads_and_vars, 'Y':Y}
        #return {'acc':acc, 'cost':cost, 'optimizer':optimizer}

    def predict(self, sess, instr_addr, prob):
        this_input = numpy.zeros([1, self.config.num_instr+1])
        this_input[0, int(instr_addr)] = 1.0
        this_input[0, -1] = prob
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

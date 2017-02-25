import tensorflow as tf
from Config import *

class OneLSTM(object):
    def __init__(self, config=Config()):
        config = config
        batch_size = config.batch_size
        
        with tf.variable_scope('lstm1'):
            input1 = tf.placeholder(shape=[batch_size, 1], dtype=tf.float32)
            lstm1 = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            output1, state1 = lstm1(inputs=input1, state=lstm1.zero_state(batch_size, dtype=tf.float32))
        
        with tf.variable_scope('lstm2'):
            lstm2 = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            input2 = tf.placeholder(shape=[batch_size, 1], dtype=tf.float32)
            input2 = tf.concat_v2([output1, input2], axis=1)
            output2, state2 = lstm2(inputs=input2, state=lstm2.zero_state(batch_size, dtype=tf.float32))

        with tf.variable_scope("Wb"):
            weights = tf.Variable(tf.random_normal([config.input_dim, config.hidden_size]), name="weights", dtype=tf.float32)
            biases = tf.Variable(tf.random_normal([1, config.hidden_size]), name="biases", dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(embedding, [5])



config = Config()
onelstm = OneLSTM(config=config)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    inputs = sess.run([onelstm.inputs])
    print(inputs)
    inputs = sess.run([onelstm.inputs])
    print(inputs)

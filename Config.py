from utils import *
class Config(object):
    def __init__(self, feattype='feat'):
        """Small Size"""
        #self.init_scale = 0.1
        self.learning_rate = 1e-4
        self.lr_decay = 0.99
        self.max_epoch = 100
        
        self.num_layers = 2
        self.hidden_size = 200
        
        self.keep_prob = 1.0
        
        self.T = 100
        self.num_steps = 300
        self.batch_size = 128
        if feattype=='feat3':
            self.input_dim = self.T*3+1
        else:
            self.input_dim = self.T+1
        self.output_dim = 1
        self.num_instr = 0

        self.sample_dim = (self.input_dim + self.output_dim) * self.num_steps
    
    @property
    def input_shape(self):
        return tf.placeholder(tf.float32, shape=(self.num_steps, self.batch_size, self.input_dim))

    @property
    def output_shape(self):
        return tf.placeholder(tf.float32, shape=(self.num_steps, self.batch_size, self.output_dim))

    def to_string(self):
        s = 'Config:'
        s += '\n\tlearning rate=' + str(self.learning_rate)
        s += '\n\tlr decay=' + str(self.lr_decay)
        s += '\n\tmax epoch=' + str(self.max_epoch)
        s += '\n\tnum layers=' + str(self.num_layers)
        s += '\n\thidden size=' + str(self.hidden_size)
        s += '\n\tkeep prob=' + str(self.keep_prob)
        s += '\n\tnum steps=' + str(self.num_steps)
        s += '\n\tinput dim=' + str(self.input_dim)
        s += '\n\toutput dim=' + str(self.output_dim)
        s += '\n\tbatch size=' + str(self.batch_size)

        return s
    

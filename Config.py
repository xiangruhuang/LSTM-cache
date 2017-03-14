from LSTMModel import LSTMModel
class Config(object):
    def __init__(self, feattype='feat5', FLAGS=None):
        """feature 5: <instruction ID, 1> <wakeup_id, 1> <forward, 1> 
            <prob, 1> <hist, self.history_len> <True Label, 1>"""

        """Data Parameters"""
        if (FLAGS is not None) and (FLAGS.history_len is not None):
            self.history_len = FLAGS.history_len
        else:
            self.history_len = 10
        if (FLAGS is not None) and (FLAGS.num_steps is not None):
            self.num_steps = FLAGS.num_steps
        else:
            self.num_steps = 100
        self.sample_dim = (3 + 1 + self.history_len + 1) * self.num_steps
        self.global_input_dim = 1 + 1 + self.history_len

        """Learning Parameters"""
        #self.init_scale = 0.1
        self.learning_rate = 1e-3
        #self.lr_decay = 0.99
        self.max_epoch = 25
        self.max_batches = self.max_epoch*1000
        self.keep_prob = 1.0 
        self.mode = 'online'
        self.feattype=feattype
        if (FLAGS is not None) and (FLAGS.num_learners is not None):
            self.num_learners = FLAGS.num_learners
        else:
            self.num_learners = 50
        
        """Network Architecture Parameters"""
        self.batch_size = 1

        """Context LSTM.
            Input: <forward, 1>
            Hidden Layers: [30]
            Output: <context_feature, 20>
        """
        self.context_dims = [1, 30, 20]
        self.context_params = LSTMModel.Params(dims=self.context_dims
            , num_steps=self.num_steps, batch_size=self.batch_size
            , name='context')

        """Local LSTM.
            Input: <context_feature, context_dims[-1]>, <prob, 1>
                , <OPTGEN's history, history_len>
            Hidden Layers: [50]
            Output: <True Label, 1>
        """
        
        if (FLAGS is not None) and (FLAGS.local_hidden_size is not None):
            self.local_hidden_size = FLAGS.local_hidden_size
        else:
            self.local_hidden_size = 50
        self.local_dims = [self.context_dims[-1] + 1 + self.history_len,
            self.local_hidden_size, 1]
        self.local_params = LSTMModel.Params(dims=self.local_dims
            , num_steps=self.num_steps, batch_size=self.batch_size
            , name='local')

    def to_string(self):
        s = 'Config:'
        s += '\n\tfeature type=' + str(self.feattype)
        s += '\n\tlearning rate=' + str(self.learning_rate)
        #s += '\n\tlr decay=' + str(self.lr_decay)
        s += '\n\tmax epoch=' + str(self.max_epoch)
        s += '\n\tnum learners=' + str(self.num_learners)
        s += '\n\tkeep prob=' + str(self.keep_prob)
        s += '\n\tlocal hidden size=' + str(self.local_hidden_size)
        s += '\n'+self.context_params.to_string()
        s += '\n'+self.local_params.to_string()
        return s
    

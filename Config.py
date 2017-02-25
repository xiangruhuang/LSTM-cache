from LSTMModel import LSTMModel
class Config(object):
    def __init__(self, feattype='feat5'):
        """feature 5: <instruction ID, 1> <wakeup_id, 1> <forward, 1> 
            <prob, 1> <hist, self.history_len> <True Label, 1>"""

        """Data Parameters"""
        self.history_len = 10
        self.num_steps = 10
        self.sample_dim = (3 + 1 + self.history_len + 1) * self.num_steps
        self.global_input_dim = 1 + 1 + self.history_len

        """Learning Parameters"""
        #self.init_scale = 0.1
        self.learning_rate = 1e-3
        #self.lr_decay = 0.99
        self.max_epoch = 100
        self.max_batches = self.max_epoch*10000
        self.keep_prob = 1.0 
        self.mode = 'online'
        self.feattype=feattype
        self.num_learners = 10
        
        """Network Architecture Parameters"""
        self.batch_size = 100

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

        self.local_dims = [self.context_dims[-1] + 1 + self.history_len, 50, 1]
        self.local_params = LSTMModel.Params(dims=self.local_dims
            , num_steps=self.num_steps, batch_size=self.batch_size
            , name='local')

    def to_string(self):
        s = 'Config:'
        s += '\n\tfeature type=' + str(self.feattype)
        s += '\n\tlearning rate=' + str(self.learning_rate)
        #s += '\n\tlr decay=' + str(self.lr_decay)
        s += '\n\tmax epoch=' + str(self.max_epoch)
        s += '\n\tkeep prob=' + str(self.keep_prob)
        s += '\n'+self.context_params.to_string()
        s += '\n'+self.local_params.to_string()
        return s
    

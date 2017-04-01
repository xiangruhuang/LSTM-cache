from LSTMModel import LSTMModel
import os
import inspect

class Config(object):
    def __init__(self, FLAGS):
        """feature 5: <instruction ID, 1> <wakeup_id, 1> <forward, 1> 
            <prob, 1> <hist, self.history_len> <True Label, 1>"""

        """     System and File I/O         """
        self.data_path = FLAGS.data_path
        self.save_dir = FLAGS.save_dir
        self.load_dir = FLAGS.load_dir
        if FLAGS.device is not None: 
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device
        

        """     Data Specs                  """
        self.global_input_dim = FLAGS.global_input_dim
        self.num_instr = FLAGS.num_instr
        #self.sample_dim = (3 + self.global_input_dim) * self.num_steps
        
        
        """     Learning Parameters         """
        self.mode = FLAGS.mode
        self.split = FLAGS.split
        self.feattype = FLAGS.feattype
        self.max_epoch = FLAGS.max_epoch
        self.num_steps = FLAGS.num_steps
        self.batch_size = FLAGS.batch_size
        self.window_size = FLAGS.window_size
        self.is_training = FLAGS.is_training
        self.num_learners = FLAGS.num_learners
        self.learning_rate = FLAGS.learning_rate
        self.instr_set = FLAGS.instr_set.split(':')
        self.baseline_only = FLAGS.baseline_only
        if len(self.instr_set[0]) == 0:
            self.instr_set = None
        elif len(self.instr_set) == 1:
            self.instr_set = [int(self.instr_set[0])]
        else:
            st = int(self.instr_set[0])
            if self.instr_set[1] == '':
                ed = self.num_instr
            self.instr_set = range(st, ed)
        self.expr_suffix = FLAGS.instr_set
        if self.instr_set is not None:
            self.name_offset = str((self.instr_set[0]+1))
            if len(self.instr_set) > 1:
                self.name_offset = '0'
        else:
            self.name_offset = None

        """     Network Architecture        """
        """     Context LSTM.
            Input: <forward, 1>
            Hidden Layers: [30]
            Output: <context_feature, context_output_dim>
        """
        self.context_dims = [int(token) for token in FLAGS.context_dims.split(',')]
        self.context_params = LSTMModel.Params(dims=self.context_dims ,
                num_steps=self.num_steps, batch_size=self.batch_size ,
                name='context')

        """     Local LSTM.
            Input: <Context&Global Feature,
                dim=context.params.output_dim+global_input_dim>
            Hidden Layers: <Hidden_0, dim=local_hidden_sizes[0]>, ...
            Output: <True Label, dim=1>
        """
        self.local_hidden_sizes = [int(token) for token in
                FLAGS.local_hidden_size.split(',')]
        self.local_dims = [self.context_dims[-1]+ self.global_input_dim]+self.local_hidden_sizes + [1]
        self.local_params = LSTMModel.Params(dims=self.local_dims,
            num_steps=self.num_steps, batch_size=self.batch_size, name='local',
            name_offset=self.name_offset)
        


    def __str__(self):
        attributes = inspect.getmembers(self, lambda a :
                not(inspect.isroutine(a)))
        toStr = 'Config:'
        for a in attributes:
            if not(a[0].startswith('__') and a[0].endswith('__')):
                toStr += '\n\t' + str(a)
        toStr += '\n'
        return toStr
        #s = 'Config:'
        #s += '\n\tLearning Params:'
        #s += '\n\t\tmode=' + str(self.mode)
        #s += '\n\t\tsplit=' + str(self.split)
        #
        #s += '\n\tfeature type=' + str(self.feattype)
        #s += '\n\tmax epoch=' + str(self.max_epoch)
        #s += '\n\t#steps=' + str(self.num_steps)
        #s += '\n\tbatch_size=' + str(self.batch_size)
        #s += '\n\twindow_size=' + str(self.window_size)
        #s += '\n\tis_training=' + str()
        #self.is_training = FLAGS.is_training
        #self.num_learners = FLAGS.num_learners 
        #s += '\n\tlearning rate=' + str(self.learning_rate)
        #s += '\n\tcapacity=' + str(self.capacity)
        ##s += '\n\tlr decay=' + str(self.lr_decay)
        #s += '\n\tmax epoch=' + str(self.max_epoch)
        #s += '\n\tnum learners=' + str(self.num_learners)
        #s += '\n\tkeep prob=' + str(self.keep_prob)
        #s += '\n\tlocal hidden size=' + str(self.local_hidden_size)
        #s += '\n'+self.context_params.to_string()
        #s += '\n'+self.local_params.to_string()
        return s
    

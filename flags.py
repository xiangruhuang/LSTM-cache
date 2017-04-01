from utils import *

flags = tf.flags


"""     System and File I/O         """
flags.DEFINE_string("data_path", None, "path to train/test/valid data.")
flags.DEFINE_string("save_dir", None, "where to output model and tensorboard")
flags.DEFINE_string("load_dir", None, "where to load variables")
flags.DEFINE_string("device", None, "visible device")


"""     Data Specs                  """
flags.DEFINE_integer("num_instr", -1, "number of unique instructions")
flags.DEFINE_integer("global_input_dim", 12, "global input dimension {12}")

"""     network architecture        """
"""         context LSTM    """
flags.DEFINE_string("context_dims", '1,30,20', "dimensions of context LSTM\
        in format <input>,<hidden>,...,<hidden>,<output> {'1,30,20'}")

"""         local LSTM      """
flags.DEFINE_string("local_hidden_size", '50', "#neurons for local LSTM {'50'}")



"""             Learning Parameters """
flags.DEFINE_float("learning_rate", 1e-3, "learing rate {1e-3}")
flags.DEFINE_integer("max_epoch", 100, "max number of epochs to run {100}")
flags.DEFINE_string("split", '3:1', "train test split ratio {'3:1'}")
flags.DEFINE_string("mode", 'offline', "online or offline mode {'offline'}")
flags.DEFINE_string("feattype", "feat5", "feature type {feat5}")
flags.DEFINE_integer("num_learners", 1, "num of learners {1}")
flags.DEFINE_integer("num_steps", 1, "num of time steps {1}")
flags.DEFINE_integer("batch_size", 1, "#minibatch each batch {1}")
flags.DEFINE_integer("window_size", 100, "capacity of each training set {100}")
flags.DEFINE_boolean("is_training", True, "trainable variables or not {True}")
flags.DEFINE_string("instr_set", '', "set of instructions to learn {''}")
flags.DEFINE_boolean("baseline_only", False, "used to get stats {False}")

FLAGS = flags.FLAGS

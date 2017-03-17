from utils import *

flags = tf.flags

flags.DEFINE_string("data_path", None, "path to train/test/valid data.")
flags.DEFINE_string("model_path", None, "path to model checkpoint folder")
flags.DEFINE_string("num_instr", None, "number of unique instructions")
flags.DEFINE_string("device", None, "device to run on")
flags.DEFINE_boolean("is_training", False, "whether to train or just load the\
    model")
flags.DEFINE_string("log_dir", './tensorboard/log/', "where to output logs for\
    tensorboard")
flags.DEFINE_string("model_dir", None, "where to output model and log")
flags.DEFINE_string("split", None, "how to split train and test set")
flags.DEFINE_integer("num_learners", None, "num of learners")
flags.DEFINE_integer("num_steps", None, "num of time steps")
flags.DEFINE_integer("history_len", None, "history length from OPTgen")
flags.DEFINE_integer("local_hidden_size", None, "#neurons for local LSTM")
flags.DEFINE_integer("context_output_dim", None, "output dimension for output\
    LSTM, set 0 to disable context LSTM")

FLAGS = flags.FLAGS

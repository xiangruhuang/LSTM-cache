from utils import *

flags = tf.flags

flags.DEFINE_string("data_path", None, "path to train/test/valid data.")
flags.DEFINE_string("model_path", None, "path to model checkpoint folder")
flags.DEFINE_string("num_instr", None, "number of unique instructions")
flags.DEFINE_boolean("is_training", False, "whether to train or just load the model")
flags.DEFINE_string("log_dir", './tensorboard/log/', "where to output logs for tensorboard")

FLAGS = flags.FLAGS

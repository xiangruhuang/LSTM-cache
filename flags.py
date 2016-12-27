from utils import *

flags = tf.flags

flags.DEFINE_string("data_path", None, "path to train/test/valid data.")
flags.DEFINE_string("num_instr", None, "number of unique instructions")
flags.DEFINE_boolean("is_training", False, "whether to train or just load the model")

FLAGS = flags.FLAGS

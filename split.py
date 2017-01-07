import sys
sys.path.append("/home/xiangru/Projects/LSTM-cache/")
from Config import *

config = Config()
split_file(config, sys.argv[1], 10000, [3, 1, 0])

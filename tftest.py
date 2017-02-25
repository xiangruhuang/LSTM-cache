
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

#import utils
import collections
import os
import sys
import json

import tensorflow as tf

from flags import *
from Reader import *
from LSTMModel import *
from SimpleLSTM import *
from Config import *

def main(_):
    """Initialization"""
    
    config = Config()
    local_config = config.local_config
    packed = local_config.input_shape
   
    simple0 = SimpleLSTM(local_config, packed)
    simple1 = SimpleLSTM(local_config, packed)
    #with tf.Session() as sess:
    #    simple0.initialize(sess)
    

if __name__ == "__main__":
    tf.app.run()

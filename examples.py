import tensorflow as tf

from rnn_utils import *
from LSTMModel import LSTMModel

""" static_rnn_example 
b batches, each batch has a different length, represented by 
`Tensor` sequence_length
"""

def static_rnn_example():
    bs=10
    a = LSTMModel.from_raw_params([31, 30, 1], 1, bs, name='lstm')
    


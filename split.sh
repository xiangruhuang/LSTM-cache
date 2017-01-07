#!/bin/bash
data=$(pwd)/$1
num_lines=`((python -c 'import sys; sys.path.append("/home/xiangru/Projects/LSTM-cache/"); from Config import *; config = Config(); print(config.num_steps*config.batch_size); ')2>& 1) | tail -1 `
echo splitting $data, ${num_lines} lines per file... 
split -d -a4 -l${num_lines} "$data" "${data}."
cd /home/xiangru/Projects/LSTM-cache/data/dnn_ordered_traces/ && python pad.py $data

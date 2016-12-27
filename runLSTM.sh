#!/bin/bash
#make data=data/dnn_ordered_traces/astar.num num_instr=54
#make data=data/dnn_ordered_traces/mcf.num num_instr=650
#make data=data/dnn_ordered_traces/tonto.num num_instr=1292
make omnetpp.train 
#make data=data/dnn_ordered_traces/sphinx3.num num_instr=1698
#make data=data/dnn_ordered_traces/xalancbmk.num num_instr=2249
make data=data/dnn_ordered_traces/soplex.num num_instr=2348
make data=data/dnn_ordered_traces/gcc.num num_instr=3280

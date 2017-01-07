#!/bin/bash
#make data=data/dnn_ordered_traces/astar.num num_instr=54
#make data=data/dnn_ordered_traces/mcf.num num_instr=650
#make data=data/dnn_ordered_traces/tonto.num num_instr=1292
#make data=data/dnn_ordered_traces/sphinx3.num num_instr=1698
#make data=data/dnn_ordered_traces/xalancbmk.num num_instr=2249
#make gcc.train feat=feat3
#make soplex.train feat=feat2
make omnetpp.train feat=feat3
make soplex.train feat=feat3

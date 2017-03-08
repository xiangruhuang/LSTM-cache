#!/bin/bash
num_learners=50
local_hidden_size=$2
echo "running omnetpp feat5 with num_steps = ${i}, device=/gpu:${1}, num_learners=${num_learners}, local_hidden_size=${local_hidden_size}" >> global_log
make device=${1} num_learners=${num_learners} num_steps=100 local_hidden_size=$2

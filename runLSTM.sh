#!/bin/bash
num_learners=50
local_hidden_size=50
num_steps=$1
train_split=3
test_split=1
split=${train_split}:${test_split}
data=omnetpp
model_no=`bash next_model.sh ${data}`
model_dir=${data}.model${model_no}
context_output_dim=20
device=0

mkdir -p ${model_dir}
echo "running omnetpp feat5 split=${split} LSTMCell with num_steps = ${num_steps}, num_learners=${num_learners}, local_hidden_size=${local_hidden_size}, split=${split}, context_output_dim=${context_output_dim} under ${model_dir}" >> global_log

cat script_template | sed "s/LOGNAME/${model_dir}\/out/" | sed "s/JOBNAME/${data}_${train_split}vs${test_split}_${model_no}_lhs${local_hidden_size}_nl${num_learners}_ns${num_steps}_cod${context_output_dim}/" > ${model_dir}/script
echo "make num_learners=${num_learners} num_steps=${num_steps} local_hidden_size=${local_hidden_size} model_dir=${model_dir} device=${device} split=${split} context_output_dim=${context_output_dim}" >> ${model_dir}/script
sbatch ${model_dir}/script

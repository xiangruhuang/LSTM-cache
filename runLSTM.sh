#!/bin/bash
num_learners=50
local_hidden_size=50
num_steps=$1
data=omnetpp
model_no=`bash next_model.sh ${data}`
model_dir=${data}.model${model_no}
device=0

mkdir -p ${model_dir}
echo "running omnetpp feat5 split=3 vs 1 LSTMCell with num_steps = ${num_steps}, num_learners=${num_learners}, local_hidden_size=${local_hidden_size} under ${model_dir}" >> global_log

cat script_template | sed "s/LOGNAME/${model_dir}\/log/" | sed "s/JOBNAME/${data}_3vs1_${model_no}_lhs${local_hidden_size}_nl${num_learners}_ns${num_steps}/" > ${model_dir}/script
echo "make num_learners=${num_learners} num_steps=${num_steps} local_hidden_size=${local_hidden_size} model_dir=${model_dir} device=${device}" >> ${model_dir}/script
sbatch ${model_dir}/script

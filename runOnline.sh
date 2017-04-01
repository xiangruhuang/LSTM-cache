data=omnetpp
device=${1}
params="device=${device} window_size=1000000 load_dir=omnetpp/offline \
num_steps=10 batch_size=100 mode='online' num_learners=1 context_dims=12,30,20 \
learning_rate=1e-3 is_training=True"

for l in `seq 0 99`; do
    if (( ${l} % 2 != ${device} % 2 )); then
        continue
    fi
    if (( ${l} == 99 )); then
        i=${l}:;
    else
        i=${l};
    fi
    if [ ! -d ${data}/baseline/tensorboard/test${i} ]; then
        #echo baseline ${i}
        make ${data}.train instr_set=${i} save_dir=${data}/baseline ${params} baseline_only=True
    fi
    save_dir=${data}/online
    if [ ! -d ${save_dir}/tensorboard/test${i} ]; then
        #echo online ${i}
        make ${data}.train instr_set=${i} save_dir=${save_dir} ${params} baseline_only=False
    fi
done

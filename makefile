datadir=$(PWD)/data/
gcc.num_instr=3280
soplex.num_instr=2348
omnetpp.num_instr=1498
mix.num_instr=0
feat=feat5
device=NA
num_learners=50
num_steps=N/A
history_len=10
local_hidden_size=50
model_dir=NA
split=NA
context_output_dim=20

all: omnetpp.train

%.train:
	$(eval b = $(basename $@))
	$(eval data = $(datadir)/$(b).$(feat))
	echo dataset=$b feature type=$(feat)
	$(eval num_instr = $($(b).num_instr))
	((stdbuf -oL python main.py --data_path=$(data) --num_instr=$(num_instr) --is_training=True --model_dir=$(model_dir) --device=$(device) --history_len=$(history_len) --num_steps=$(num_steps) --num_learners=$(num_learners) --local_hidden_size=$(local_hidden_size) --split=$(split) --context_output_dim=$(context_output_dim) ) 2>&1) >> $(model_dir)/log
	#python main.py --data_path=$(data) --num_instr=$(num_instr) --is_training=True --model_dir=$(model_dir) --device=$(device) --history_len=$(history_len) --num_steps=$(num_steps) --num_learners=$(num_learners) --local_hidden_size=$(local_hidden_size)


datadir=$(PWD)/data/
save_dir=NNNNNNNAAAAAAA
load_dir=NNNNNNNAAAAAAA
device=NNNAAA # 1 or 2

gcc.num_instr=3280
soplex.num_instr=2348
omnetpp.num_instr=1498
mcf.num_instr=650
astar.num_instr=54
tonto.num_instr=1292
global_input_dim=12

context_dims=12,30,20

local_hidden_size=50

learning_rate=1e-4
max_epoch=100
split=3:1
mode='online'
feattype=feat5
num_learners=100
num_steps=100
batch_size=100
window_size=1000000
is_training=True
instr_set=''
baseline_only=False

all: omnetpp.train

%.train: 
	$(eval b = $(basename $@))
	$(eval data_path = $(datadir)/$(b).$(feattype))
	echo dataset=$b feature type=$(feattype)
	$(eval num_instr = $($(b).num_instr))
	mkdir -p $(save_dir)
	#((stdbuf -oL python main.py \
	#	--data_path=$(data_path) --save_dir=$(save_dir) --device=$(device) \
	#	--load_dir=$(load_dir) \
	#	--num_instr=$(num_instr) --global_input_dim=$(global_input_dim) \
	#	--context_dims=$(context_dims) --local_hidden_size=$(local_hidden_size) \
	#	--learning_rate=$(learning_rate) --max_epoch=$(max_epoch) --split=$(split) \
	#	--mode=$(mode) --feattype=$(feattype) --num_learners=$(num_learners) \
	#	--num_steps=$(num_steps) --batch_size=$(batch_size) --window_size=$(window_size) \
	#	--is_training=$(is_training) --instr_set=$(instr_set) --baseline_only=$(baseline_only)\
	#	) 2>&1) >> $(save_dir)/log$(instr_set)
	
	python main.py \
		--data_path=$(data_path) --save_dir=$(save_dir) --device=$(device) \
		--load_dir=$(load_dir) \
		--num_instr=$(num_instr) --global_input_dim=$(global_input_dim) \
		--context_dims=$(context_dims) --local_hidden_size=$(local_hidden_size) \
		--learning_rate=$(learning_rate) --max_epoch=$(max_epoch) --split=$(split) \
		--mode=$(mode) --feattype=$(feattype) --num_learners=$(num_learners) \
		--num_steps=$(num_steps) --batch_size=$(batch_size) --window_size=$(window_size) \
		--is_training=$(is_training) --instr_set=$(instr_set) --baseline_only=$(baseline_only)\

offline_tensorboard:
	tensorboard --logdir=omnetpp:./omnetpp/offline_test/tensorboard/

null:=
space:= $(null) #
comma:= ,
data=omnetpp

%.tensorboard:
	$(eval names := $(basename $@))
	$(eval names := $(subst +, ,$(names)))
	$(eval LOGDIR := $(foreach name,$(names),$(name):$(data)/$(name)/tensorboard))
	$(eval LOGDIR := $(subst $(space),$(comma),$(LOGDIR)))
	tensorboard --logdir=$(LOGDIR)

all_tensorboard:
	tensorboard --logdir=selected:./tensorboard/

#data=data/hmmer/hmmer
#data=data/gcc/gcc
#data=data/gaussian/gaussian

data=data/dnn_ordered_traces/
#name=$(firstword $(subst ., , $(notdir $(data))))
#num_instr=
gcc.num_instr=3280
soplex.num_instr=2348
omnetpp.num_instr=1498
mix.num_instr=0
feat=feat2

%.train:
	$(eval b = $(basename $@))
	mkdir -p $(b).$(feat)
	$(eval data = data/dnn_ordered_traces/$(b)/$(b).$(feat))
	echo dataset=$b feature type=$(feat)
	$(eval num_instr = $($(b).num_instr))
	((stdbuf -oL python main.py --data_path=./$(data) --num_instr=$(num_instr) --is_training=True) 2>&1) >> $(b).$(feat)/log

%.test: 
	$(eval b = $(basename $@))
	mkdir -p $(b).$(feat)
	$(eval data = data/dnn_ordered_traces/$(b)/$(b).$(feat))
	echo dataset=$b feature type=$(feat)
	$(eval num_instr = 0)
	python test.py --data_path=./$(data) --num_instr=$(num_instr) --is_training=False


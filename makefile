#data=data/hmmer/hmmer
#data=data/gcc/gcc
#data=data/gaussian/gaussian

#data=data/dnn_ordered_traces/gcc.num
#name=$(firstword $(subst ., , $(notdir $(data))))
#num_instr=
gcc.num_instr=3280
soplex.num_instr=2348
omnetpp.num_instr=1498

%.train:
	$(eval b = $(basename $@))
	mkdir -p $(b)
	$(eval data = data/dnn_ordered_traces/$(b).num)
	echo dataset=$b
	$(eval num_instr = $($(b).num_instr))
	python main.py --data_path=./$(data) --num_instr=$(num_instr) --is_training=True

%.test:
	$(eval b = $(basename $@))
	mkdir -p $(b)
	$(eval data = data/dnn_ordered_traces/$(b).num)
	echo dataset=$b
	$(eval num_instr = $($(b).num_instr))
	python test.py --data_path=./$(data) --num_instr=$(num_instr) --is_training=False


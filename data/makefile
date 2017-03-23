datasets=soplex omnetpp tonto astar mcf
feature=feat5

data=

all:
	for data in $(datasets); do \
		echo extracting $(feature) from $${data}; \
		make $${data}.$(feature)/instr.all.bin data=$${data}; \
	done

$(data).$(feature)/instr.all:
	@echo constructing $(data).$(feature)/instr.all
	python preprocess.py $(data) $(feature)

$(data).$(feature)/instr.all.bin: $(data).$(feature)/instr.all
	@echo constructing $(data).$(feature)/instr.all.bin
	python convert_to_binary.py $(data).$(feature)/instr.all $(data).$(feature)/instr.all.bin

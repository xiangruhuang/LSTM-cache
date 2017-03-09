datasets=omnetpp
feature=feat5

all:
	for data in $(datasets); do \
		echo extracting $(feature) from $${data}; \
		make $${data}/$${data}.$(feature)/$${data}.$(feature); \
	done

%.$(feature):
	$(eval data := $(notdir $(basename $@)) )
	python preprocess.py $(data) $(feature)

datasets=astar omnetpp soplex tonto mcf sphinx3 xalancbmk gcc #gems gromacs hmmer lbm libquantum milc bzip2 calculix gobmk h264ref leslie3d cactusADM zeusmp 

all: $(addsuffix .num,$(datasets))

%.num:
	python preprocess.py $(basename $@) $@

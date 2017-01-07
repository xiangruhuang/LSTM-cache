data=soplex
feat=feat2
git=/home/xiangru/Projects/LSTM-cache/
all: $(data).$(feat) $(data).$(feat).train

preprocess_$(feat).py:
	cp $(git)/$@ ./

split.py:
	cp $(git)/$@ ./

split.sh:
	cp $(git)/$@ ./

pad.py:
	cp $(git)/$@ ./

%.$(feat).train: %.$(feat) split.py split.sh pad.py
	python split.py $<
	./split.sh $<.train
	./split.sh $<.test

%.$(feat): preprocess_$(feat).py
	$(eval b := $(basename $@))
	python preprocess_$(feat).py $(b) $@

clean:
	rm *.py
	rm split.sh

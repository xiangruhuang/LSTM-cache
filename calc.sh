#!/bin/bash
feat=feat2
base=data/dnn_ordered_traces/${1}/${1}.${feat}.test
for i in `ls ${1}.${feat}/pred*`; do
    echo pred=${i}, true label=${base}
    python calc_test_acc.py ${i} ${base};
    python calc_test_acc.py ${i/pred/truelabel} ${base};
    echo " ";
done

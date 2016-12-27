#!/bin/bash
base=data/dnn_ordered_traces/${1}.num.test
for i in `ls ${1}/pred*`; do
    echo pred=${i}, true label=${base}
    python calc_test_acc.py ${i} ${base};
    python calc_test_acc.py ${i//pred/truelabel} data/dnn_ordered_traces/${1}.num.test;
    echo " ";
done

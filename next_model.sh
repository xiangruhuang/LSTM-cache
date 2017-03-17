for i in `seq 1 1000`; do 
    if [ -d ${1}/${i} ]; then 
        continue;
    else 
        echo ${i};
        break;
    fi
done

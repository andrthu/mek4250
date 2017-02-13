#!/bin/bash
COUNTER=2;
COUNTER2=0;
if [ $# -gt 1 ]; then
    N=$1;
    K="$2";
else
    N=$1;
    K=1;
fi

while [ $(($K-$COUNTER2)) -gt 0 ]; do
    mpiexec -n 1 -mca btl ^openib python func_grad_speedup_test.py $N 1 1 0;
    let COUNTER2=COUNTER2+1;
done
let COUNTER2=0;
mpiexec -n 1 -mca btl ^openib python func_grad_speedup_test.py $N 1 0 0;
rm temp_time.txt;
echo "First";
while [  $COUNTER -lt 13 ]; do
    while [ $(($K-$COUNTER2)) -gt 0 ]; do
	mpiexec -n $COUNTER -mca btl ^openib python func_grad_speedup_test.py $N 1 1 0;
	let COUNTER2=COUNTER2+1;
    done
    mpiexec -n $COUNTER -mca btl ^openib python func_grad_speedup_test.py $N 1 0 0;
    let COUNTER=COUNTER+1;
    echo "new $COUNTER"
    let COUNTER2=0;
    rm temp_time.txt;
done


#!/bin/bash
COUNTER=2;
N=$1;
mpiexec -n 1 python speedup_mpi.py $N 0;
while [  $COUNTER -lt 13 ]; do
    mpiexec -n $COUNTER python speedup_mpi.py $N 1;
    let COUNTER=COUNTER+1; 
done


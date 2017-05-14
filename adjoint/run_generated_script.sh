#!/bin/bash
python generate_sbatch.py $1 $2;
python $2;
rm $2;

#!/bin/bash

## preprocess the data 
python m0_preprocess.py

## declare an array variable
declare datasets=('collins_12')
declare models=('RLbaseline')

for model in "${models[@]}"; do 
    echo Data set=collins 14 Model=$model
        python m1_fit_model.py -f=6 -c=6 -n=$model -d='collins_14'
done
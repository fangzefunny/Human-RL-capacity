#!/bin/bash

## declare an array variable
declare datasets=('collins_12')

for data in "${datasets[@]}"; do 
    echo Data set=$data Model='Pi model 2'
        python m1_fit_model.py -f=10 -c=6 -n='Pi_model_2' -d=$data
done
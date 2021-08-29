#!/bin/bash

## preprocess the data 
python m0_preprocess.py

## declare an array variable
declare datasets=('collins_12' 'collins_12')
declare models=('RLbaseline'  'Pi_model_1'  'Psi_Pi_model_3'  'Pi_Rep_Grad')

for dataset in "${datasets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$dataset 14 Model=$model
            python m1_fit_model.py -f=6 -c=6 -n=$model -d=$dataset
    done
done 
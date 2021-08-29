# Inforamtion theory and Reinforcement Learning

# Introduction

Humans' working memory is known to be capacity-limited, incapable of processing unlimited information. A natural question is raised: How the capacity affects our learning?

This repo aims at building an information-theoretic model that accounts for the human learning data (from Collins and Frank 2012 and Collins, et al. 2014). 

# Where to start?

## Use the bash script

Run:

    bash run.sh 

and go to bed. This script will preprocess, fit the models, simulate the models, and generate the figures automatically for you. It takes about 8 hrs on a 12-core computer. More cores will not shorten your waiting time, but fewer cores will increase. 

## Use the "mxxx" script

These scripts may return more details about the model. Note that the number follows 'm' indicating the step. In step 0, you preprocess the data and you can fit the model in the step 1 using,

    python m1_fit_model.py -data_set=<ds> -f=<fit time> -n=<the model you like>

Fill what you want in “<>”。

## The corresponding paper

Fang, Z., & Sims, C. (2021, July). Computationally rational reinforcement learning: modeling the influence of policy and representation complexity. Paper presented at Virtual MathPsych/ICCM 2021. Via mathpsych.org/presentation/608.
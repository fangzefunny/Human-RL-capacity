import os 
import pickle

import pandas as pd
import matplotlib.pyplot as plt 

from collections import OrderedDict
from utils.model import RLModel
from utils.analyze import empirical_Pi_Rate, Set_Size_effect
from utils.agents import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

# Hand defined RGB
Blue    = np.array([   9, 132, 227]) / 255
Green   = np.array([   0, 184, 148]) / 255
Red     = np.array([ 255, 118, 117]) / 255
Yellow  = np.array([ 253, 203, 110]) / 255
Purple  = np.array([ 108,  92, 231]) / 255

# Model Name to model
def name_to_model( name):
    if name == 'RL_baseline':
        process_model = RLbaseline
    elif name == 'Pi_model_2':
        process_model = Pi_model_2

    return process_model

# simulate data for analyze 
def simulate_data( data_set, process_name, params=[]):

    # load data
    with open( f'{path}/data/{data_set}.pkl', 'rb')as handle:
        data = pickle.load( handle)

    # select the process model and define the RL model
    process_agent = name_to_model( process_name)
    model = RLModel( process_agent)

    # if there is no input parameter,
    # choose the best model,
    # the last column is the loss, so we ignore that 
    if len(params) == 0:
        fname = f'{path}/results/params-{data_set}-{process_name}.csv'
        params = pd.read_csv( fname, index_col=0).iloc[0, 0:-1].values
    
    # synthesize the data and save
    pre_data = model.predict( data, params)
    fname = f'{path}/data/sim_{data_set}-{process_name}.csv'
    pre_data.to_csv(fname)
        
def Fig1_Set_Size( data_set):
    '''Fig1 Set Size effect

    This function is defined to replica the set size
    effect in Collins&Frank 2012.

    To make comparision, we inlcude five different agent:

        - Real human agent
        - Optimal
        - RL baseline 
        - Pi model
        - Pi+Rep model 
    '''
    models   = [ 'human',   
                 'Pi_model_2', ] # 'optimal',  'RL baseline', 'Pi+Rep model' ]
    colors   = [ Blue, Red, Green, Yellow, Purple]
    set_sizes = [ 2, 3, 4, 5, 6]

    # get data
    data_lst = []
    for agent in models:

        # Load Data 
        if agent == 'human':
            fname = f'{path}/data/{data_set}.csv'
            mode = 'human'
        else:
            fname = f'{path}/data/sim_{data_set}-{agent}.csv'
            mode = 'model'
        data = pd.read_csv( fname)
        
        # Get teh set size effect and  
        data_lst.append(Set_Size_effect( data, mode))
    
    # create figure
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots( 3, 2, figsize=( 8,12))
    plt.subplots_adjust( right=.8, bottom=.18)

    for n, model in enumerate( models):
        axis = ax[ n // 3, n % 2]
        iters = np.arange( 1, 10)
        curves = data_lst[n]
        for i, _ in enumerate( set_sizes):
            axis.plot( iters, curves[ i, :], 'o-',
                        linewidth=1, markersize=5.5, color=colors[i])
        axis.set_ylim( [ 0, 1.1])
        axis.set_xticks( range(1,10))
        if n>0:
            axis.set_yticks([])
        if model == 'human':
            axis.set_title( 'human data')
        else:
            axis.set_title( f'{model}')   
    try: 
        plt.savefig(f'{path}/figures/Fig1_set_size.png')
    except:
        os.mkdir( f'{path}/figures')
        plt.savefig(f'{path}/figures/Fig1_set_size,png')

if __name__ == '__main__':

    params = [ 0., 0.02107682, 0.01277333, 6.57572485, 
               6.20778639, 5.7483761, 5.33288481, 6.02997583]
    simulate_data( 'collins_12', 'Pi_model_2', params)

    Fig1_Set_Size('collins_12')

              




        






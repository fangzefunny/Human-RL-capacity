import os 
import pickle

import pandas as pd
import matplotlib.pyplot as plt 

from collections import OrderedDict
from utils.model import RLModel
from utils.analyze import Empirical_Pi_Rate, Model_Pi_Rate, Model_Psi_Rate, Set_Size_effect, Empirical_Rate_Reward
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
    if name == 'RLbaseline':
        process_model = RLbaseline
    elif name == 'optimal':
        process_model = optimal
    elif name == 'Pi_model_1':
        process_model = Pi_model_1
    elif name == 'Pi_model_2':
        process_model = Pi_model_2
    elif name == 'Psi_Pi_model_3':
        process_model = Psi_Pi_model_3
    elif name == 'Pi_Rep_Grad':
        process_model = Pi_Rep_Grad

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
    models   = [ 'human', 'RLbaseline', 'Pi_model_1',
                 'Pi_model_2', 'Pi_Rep_Grad', 'Psi_Pi_model_3', 'optimal']
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

    # calculate SSE
    human_data = data_lst[0]
    nM = len( models)
    for n in range( 1, nM):
        sse = np.sum((data_lst[n] - human_data)**2)
        print( f'{models[n]}: {sse}')

    # create figure
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots( 4, 2, figsize=( 8, 16))
    plt.subplots_adjust( right=.9, bottom=.18)

    for n, model in enumerate( models):
        axis = ax[ n // 2, n % 2]
        iters = np.arange( 1, 10)
        curves = data_lst[n]
        for i, _ in enumerate( set_sizes):
            axis.plot( iters, curves[ i, :], 'o-',
                        linewidth=1.5, markersize=5.5, color=colors[i])
        axis.set_ylim( [ 0, 1.1])
        axis.set_xticks( range(1,10))
        if n % 2 ==1:
            axis.set_yticks([])
        if n<5:
            axis.set_xticks([])
        if model == 'human':
            axis.set_title( 'human data')
        else:
            axis.set_title( f'{model}')   
    
    axis = ax[ 3, 1]
    for i, _ in enumerate( set_sizes):
        axis.plot( iters, 9*[np.nan], 'o-',
                    linewidth=1.5, markersize=5.5, color=colors[i])    
        axis.axis('off')
        axis.legend( ['Set Size=2', 'Set Size=3', 'Set Size=4',
                            'Set Size=5', 'Set Size=5']) 
    try: 
        plt.savefig(f'{path}/figures/Fig1_set_size-{data_set}.png', dpi=1000)
    except:
        os.mkdir( f'{path}/figures')
        plt.savefig(f'{path}/figures/Fig1_set_size-{data_set}.png', dpi=1000)

def Fig2_Emp_Pi_Rate( data_set):

    models   = [ 'human', 'RLbaseline', 
                 'Pi_model_1', 'Pi_model_2', 
                 'Psi_Pi_model_3']
    set_sizes = [ 2, 3, 4, 5, 6]

    # get the optimal curve for comparison
    fname = f'{path}/data/sim_{data_set}-optimal.csv'
    optimal_curve = Empirical_Pi_Rate( pd.read_csv( fname), prior=.1)

    # get the predictive curve
    data_lst = []
    for agent in models:

        # Load Data
        if agent == 'human':
            fname = f'{path}/data/{data_set}.csv'
        else:
            fname = f'{path}/data/sim_{data_set}-{agent}.csv'
        data = pd.read_csv( fname)
        
        # Get teh set size effect and  
        data_lst.append(Empirical_Pi_Rate( data, prior=.1))

    # create figure
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots( 3, 2, figsize=( 8, 8))
    plt.subplots_adjust( right=.8, bottom=.18)

    for n, model in enumerate( models):
        axis = ax[ n // 2, n % 2]
        curves = data_lst[n]
        axis.plot( set_sizes, optimal_curve, 'o-',
                    linewidth=1.5, markersize=5.5, color=Red)
        axis.plot( set_sizes, curves, 'o-',
                    linewidth=1.5, markersize=5.5, color=Blue)
        axis.set_ylim( [ 0, .9])
        axis.set_xticks( set_sizes)
        if n % 2 ==1:
            axis.set_yticks([])
        if n<4:
            axis.set_xticks([])
        if model == 'human':
            axis.set_title( 'human data')
        else:
            axis.set_title( f'{model}')   
    
    plt.savefig(f'{path}/figures/Fig2_Emp_Pi_Rate-{data_set}.png')

def Fig3_Model_Cogcost_Rate( data_set):

    set_sizes = [ 2, 3, 4, 5, 6]

    # get the optimal curve for comparison
    fname = f'{path}/data/sim_{data_set}-optimal.csv'
    optimal_curve = Model_Pi_Rate( pd.read_csv( fname))    

    # create figure
    plt.rcParams.update({'font.size': 15})
    nr = 3
    nc = 2
    alpha = .425
    fig, ax = plt.subplots( nr, nc, figsize=( 2.8*nc, 2.5*nr))
    plt.subplots_adjust( right=.9, bottom=.1)

    # plot Pi model 2
    axis = ax[ 0, 0]
    fname = f'{path}/data/sim_{data_set}-Pi_model_2.csv'
    pi_curves = Model_Pi_Rate( pd.read_csv( fname))

    axis.plot( set_sizes, optimal_curve, '--',
                linewidth=1.5, color=Purple)
    axis.plot( set_sizes, pi_curves, 'o-',
                linewidth=1.5, color=Blue)
    axis.fill_between( set_sizes, 0, pi_curves, 
                        color=Blue, 
                        alpha=alpha)
    axis.set_ylim( [ 0, 3.])
    axis.set_xticks([])

    # plot legend
    axis = ax[ 0, 1]
    nannan = 5*[np.nan]
    axis.plot( set_sizes, nannan, '--',
                linewidth=1.5, color='k')
    axis.plot( set_sizes, nannan, '--',
                linewidth=1.5, color=Purple)
    axis.plot( set_sizes, nannan, '--',
                linewidth=1.5, color='b')
    axis.plot( set_sizes, nannan, 'o-',
                linewidth=1.5, color=Red)
    axis.plot( set_sizes, nannan, 'o-',
                linewidth=1.5, color=Blue)
    axis.fill_between( set_sizes, 0, nannan, 
                        color=Red, 
                        alpha=alpha)
    axis.fill_between( set_sizes, 0, nannan, 
                        color=Blue, 
                        alpha=alpha)
    axis.legend( ['Effective capacity', 'Optimal policy comp.', 'Optimal cog. cost', 
                 'Model cog. cost', 'Model pol. comp.', 'Model rep. comp.', 'Model pol. comp.'],prop={'size':10} )
    axis.axis('off')

    # plot Psi Pi model 1
    axis = ax[ 1, 0]
    fname = f'{path}/data/sim_{data_set}-Pi_Rep_Grad.csv'
    pi_curves = Model_Pi_Rate( pd.read_csv( fname))
    psi_curves = Model_Psi_Rate( pd.read_csv( fname))

    axis.plot( set_sizes, optimal_curve, '--',
                linewidth=1.5, color=Purple)
    axis.plot( set_sizes, pi_curves, 'o-',
                linewidth=1.5, color=Blue)
    axis.fill_between( set_sizes, 0, pi_curves, 
                        color=Blue, 
                        alpha=alpha)
    axis.set_ylim( [ 0, 3.])
    axis.set_xticks([])
    
    axis = ax[ 1, 1]
    axis.plot( set_sizes, psi_curves+optimal_curve, '--',
                linewidth=1.5, color='b')
    axis.plot( set_sizes, pi_curves, 'o-',
                linewidth=1.5, markersize=5.5, color=Blue)
    axis.fill_between( set_sizes, 0, pi_curves, 
                        color=Blue, 
                        alpha=alpha)
    # Representation complexity
    axis.plot( set_sizes, psi_curves+pi_curves, 'o-',
                linewidth=1.5, markersize=5.5, color=Red)
    axis.fill_between( set_sizes, pi_curves, pi_curves+psi_curves, 
                        color=Red, 
                        alpha=alpha)
    # capacity
    axis.plot( set_sizes, [np.max(pi_curves+psi_curves)]*5, '--',
                linewidth=1.5, color='k')
    axis.set_ylim( [ 0, 3.])
    axis.set_xticks([])
    axis.set_yticks([])
    print( f'Psi Pi 1: {np.max(pi_curves+psi_curves)}')
    
    # plot Psi Pi model 2
    axis = ax[ 2, 0]
    fname = f'{path}/data/sim_{data_set}-Psi_Pi_model_3.csv'
    pi_curves = Model_Pi_Rate( pd.read_csv( fname))
    psi_curves = Model_Psi_Rate( pd.read_csv( fname))
    axis.plot( set_sizes, optimal_curve, '--',
                linewidth=1.5, color=Purple)
    axis.plot( set_sizes, pi_curves, 'o-',
                linewidth=1.5, color=Blue)
    axis.fill_between( set_sizes, 0, pi_curves, 
                        color=Blue, 
                        alpha=alpha)
    axis.set_ylim( [ 0, 3.])
    axis.set_xticks( set_sizes)
    #axis.set_title( 'Pi Psi Model') 

    axis = ax[ 2, 1]
    # Policy complexity
    axis.plot( set_sizes, psi_curves+optimal_curve, '--',
                linewidth=1.5, color='b')
    axis.plot( set_sizes, pi_curves, 'o-',
                linewidth=1.5, markersize=5.5, color=Blue)
    axis.fill_between( set_sizes, 0, pi_curves, 
                        color=Blue, 
                        alpha=alpha)
    # Representation complexity
    axis.plot( set_sizes, psi_curves+pi_curves, 'o-',
                linewidth=1.5, markersize=5.5, color=Red)
    axis.fill_between( set_sizes, pi_curves, pi_curves+psi_curves, 
                        color=Red, 
                        alpha=alpha)
    # capacity
    axis.plot( set_sizes, [np.max(pi_curves+psi_curves)]*5, '--',
                linewidth=1.5, color='k')
    
    axis.set_ylim( [ 0, 3.])
    axis.set_yticks( [])
    axis.set_xticks( set_sizes)

    print( f'Psi Pi 2: {np.max(pi_curves+psi_curves)}')
    
    plt.savefig(f'{path}/figures/Fig3_Model_Pi_Rate-{data_set}.png', dpi=1000)

def Fig4_Rate_Reward_curve( data_set):

    # get the optimal curve for comparison
    with open(f'{path}/data/{data_set}_subject.pkl', 'rb')as handle:
        data = pickle.load( handle)

    # analyze the data 
    Rate_Rew = Empirical_Rate_Reward( data, .1)

    # create figure
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots( 1, 2, figsize=( 8, 4))
    plt.subplots_adjust( right=.9, bottom=.1)

    # plot set size = 2
    axis = ax[ 0]
    axis.plot( Rate_Rew[ 'Rate_theo'][ :, 1], Rate_Rew[ 'Val_theo'][ :, 1],
                    color = 'k', linewidth=3)
    axis.scatter( Rate_Rew[ 'Rate_data'][ :, 1], Rate_Rew[ 'Val_data'][ :, 1],
                color=Blue, s=35)
    axis.scatter( np.mean(Rate_Rew[ 'Rate_data'][ :, 1]), np.mean(Rate_Rew[ 'Val_data'][ :, 1]),
                color='r', s=60)
    axis.legend( [ 'theory', 'human', 'mean human'], loc=4, prop={'size': 12})
    axis.set_title( 'Set Size=3')
    axis.set_xlim([  0, 1.2])
    axis.set_ylim([ .2, 1.1])

    # plot set size = 2
    axis = ax[ 1]
    axis.plot( Rate_Rew[ 'Rate_theo'][ :, 4], Rate_Rew[ 'Val_theo'][ :, 4],
                    color = 'k', linewidth=3)
    axis.scatter( Rate_Rew[ 'Rate_data'][ :, 4], Rate_Rew[ 'Val_data'][ :, 4],
                color=Blue, s=35)
    axis.scatter( np.mean(Rate_Rew[ 'Rate_data'][ :, 4]), np.mean(Rate_Rew[ 'Val_data'][ :, 4]),
                color='r', s=60)
    axis.set_title( 'Set Size=6')
    axis.set_yticks([])
    axis.set_xlim([  0, 1.2])
    axis.set_ylim([ .2, 1.1])

    plt.savefig(f'{path}/figures/Fig4_Rate_Reward-{data_set}.png', dpi=1000)

    print( np.mean(Rate_Rew[ 'Rate_data'][ :, 1]), np.mean(Rate_Rew[ 'Val_data'][ :, 1]))
    
    print( np.mean(Rate_Rew[ 'Rate_data'][ :, 4]), np.mean(Rate_Rew[ 'Val_data'][ :, 4]))
        
def Fig5_Effective_MI( data_set):

    # Set size
    set_sizes = [ 2, 3, 4, 5, 6]

    # Load human empirical
    fname = f'{path}/data/{data_set}.csv'
    emp_MI = Empirical_Pi_Rate( pd.read_csv( fname), prior=.1)

    # Get the effective mutual information 
    fname = f'{path}/data/sim_{data_set}-Psi_Pi_model_3.csv'
    pi_curves = Model_Pi_Rate( pd.read_csv( fname))
    psi_curves = Model_Psi_Rate( pd.read_csv( fname))
    effect_MI = np.zeros([5,])
    for i in range(5):
        effect_MI[i] = np.min( [ pi_curves[i], psi_curves[i]])
    
    # create figure
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=( 4, 4))
    plt.plot( set_sizes, emp_MI, 'o-', color=Blue, 
                linewidth=1.5, markersize=5.5)
    plt.plot( set_sizes, effect_MI, 'o-', color=Red, 
                linewidth=1.5, markersize=5.5)
    plt.xticks(set_sizes)

    plt.savefig(f'{path}/figures/Fig5_Effective_MI-{data_set}.png')
    


if __name__ == '__main__':

    # params = [ 0., 0.02107682, 0.01277333, 6.57572485, 
    #            6.20778639, 5.7483761, 5.33288481, 6.02997583]

#     [1.45856979e-01 1.00000000e+00 0.00000000e+00 3.91992351e-02
#  4.60110370e+01 3.40192460e+01 1.89895727e+01 1.14184622e+01
#  9.02675319e+00]

    ## simulate data 
    data_set = 'collins_12'
    #simulate_data( data_set, 'Psi_Pi_model_3')
    
    # make figures
    #Fig1_Set_Size(data_set)
    Fig3_Model_Cogcost_Rate( data_set)

    #Fig5_Effective_MI( data_set)

    





        






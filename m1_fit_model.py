import argparse 
import os 
import pickle
import datetime 

import multiprocessing as mp
import pandas as pd

from collections import OrderedDict
from utils.model import RLModel
from utils.params import set_hyperparams
from utils.agents import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set', '-d', help='which data set', default='collins_12')
parser.add_argument('--fit_num', '-f', help='fit times', type = int, default=4)
parser.add_argument('--agent_name', '-n', help='choose agent', default='Pi_Rep_model')
parser.add_argument('--n_cores', '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=0)
parser.add_argument('--seed', '-s', help='random seed', type=int, default=120)
args = parser.parse_args()

def fit_human_data( data, args):

    # define the RL2 model 
    model = RLModel( args.agent)    


     # fit the data 
     # if no init state, use parallel computing 
    if len(args.init)==0:

        # decide how many cores we need
        if args.n_cores:
            n_cores = args.n_cores
        else:
            n_cores = int( mp.cpu_count())
        pool = mp.Pool( n_cores)

        print( f'Using {n_cores} parallel CPU cores')

        start_time = datetime.datetime.now()
       
        # parameter matrix and loss matrix 
        fit_mat = np.zeros( [args.fit_num, len(args.bnds) + 1])

        # para
        seed = args.seed
        results = [ pool.apply_async(model.fit, args=(data, args.bnds, seed+2*i)
                        ) for i in range(args.fit_num)]

        for i, p in enumerate(results):
            param, loss  = p.get()
            fit_mat[ i, :-1] = param
            fit_mat[ i, -1]  = loss

        end_time = datetime.datetime.now()
        print( '\nparallel computing spend {:.2f} seconds'.format((end_time - start_time).total_seconds()))

        # choose the best params and loss 
        loss_vec = fit_mat[ :, -1]
        opt_idx, loss_opt = np.argmin( loss_vec), np.min( loss_vec)
        param_opt = fit_mat[ opt_idx, :-1]

        # save the fit results
        col = args.params_name + ['loss']
        fit_results = pd.DataFrame( fit_mat, columns=col)
        
        fname = f'{path}/results/fit_results-{args.data_set}-{args.agent_name}.csv'
        try:
            fit_results.to_csv( fname)
        except:
            os.mkdir( f'{path}/results')
            fit_results.to_csv( fname)

    # if there is init param0       
    else:
        param_opt, loss_opt = model.fit( data, 
                    bnds = args.bnds,
                    seed = 2021,
                    init = args.init) 
    
    # save opt params 
    params_mat = np.zeros( [1, len(args.bnds) + 1])
    params_mat[ 0, :-1] = param_opt
    params_mat[ 0, -1]  = loss_opt
    col = args.params_name + ['loss']
    params = pd.DataFrame( params_mat, columns=col)
    
    # save the optimal parameter
    fname = f'{path}/results/params-{args.data_set}-{args.agent_name}.csv'
    try:
        params.to_csv( fname)
    except:
        os.mkdir( f'{path}/results')
        params.to_csv( fname)

if __name__ == '__main__':

    ## STEP 0: LOAD DATA
    with open(f'{path}/data/{args.data_set}.pkl', 'rb') as handle:
        human_data = pickle.load( handle)

    ## STEP 1: HYPERPARAMETER TUNING
    args = set_hyperparams(args)   
            
    ## STEP 2: FIT
    fit_human_data( human_data, args)
        
    

    




    


   
    

    
        
        
    
    
    
    
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from utils.analyze import extra_dynamic_setsize_effect, extract_pi_complexity_setsize, mle_loss

def plot_results( model_pred, params, mode='simple'):
    trials = np.arange( 1, 10)
    setsizes = [ 2, 3, 4, 5, 6]
    ns = len( setsizes)
    task_performance = np.zeros( [ ns, 9])

    # iteration the data set to get the model prediction
    for i, sz in enumerate( setsizes):
        task_performance[ i, :] = extra_dynamic_setsize_effect( model_pred, 'accuracy')[sz]
    
    if mode == 'all':
        focus_on_val = np.zeros( [ ns, 9])
        rep_complexities = np.zeros( [ ns, 9])
        pi_complexities = np.zeros( [ ns, 9])
        free_memory = np.zeros( [ ns, 9]) 
        for i, sz in enumerate( setsizes):   
            focus_on_val[ i, :] = extra_dynamic_setsize_effect( model_pred, 'tradeoff')[sz]
            pi_comp = extra_dynamic_setsize_effect( model_pred, 'pi_complexity')[sz]
            rep_comp = extra_dynamic_setsize_effect( model_pred, 'rep_complexity')[sz]
            rep_complexities[ i, :] = rep_comp
            pi_complexities[ i, :] = pi_comp 
            C = np.max((pi_comp + rep_comp))
            free_memory[ i, :] = params[-1] - (pi_comp + rep_comp)
        focus_on_val = 1/focus_on_val
        memory_affordance = pi_complexities + rep_complexities
        loss = mle_loss( model_pred)

    # set the color for each set size
    colors   = [     'r',     'b',     'g','orange', 'violet']

    if mode == 'simple':
        plt.rcParams.update({'font.size': 15})
        fig = plt.figure()
        for j, _ in enumerate( setsizes):
            plt.plot( trials, task_performance[ j, :], 'o-', 
                            linewidth=1, markersize=5, color=colors[j])
            plt.xticks( trials)
            plt.ylim( [ 0, 1.1])
            plt.xlabel( 'Trials per stimulus')
            plt.ylabel( 'Accuracy')
            plt.legend(['setsize=2','setsize=3','setsize=4','setsize=5','setsize=6'])

    elif mode == 'all':
        fig_loc  = [ ( 0, 0), ( 0, 1), ( 1, 0), ( 1, 1), ( 2, 0), ( 2, 1) ]
        curves   = [ task_performance, focus_on_val, pi_complexities, 
                        rep_complexities, free_memory, memory_affordance]
        legends  = [ 'Accuracy', 'Focus on value', 'Policy comp.',
                        'Rep. comp.', 'Free memory', 'Memory affordance',]

        # prepare the figures 
        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots( 3, 2, figsize=(8, 10))
        plt.subplots_adjust( right=.8, bottom=.18)

        for i, line in enumerate(curves):
            x, y = fig_loc[i]
            axis = ax[ x, y]
            for j, _ in enumerate( setsizes):
                axis.plot( trials, line[ j, :], 'o-', 
                            linewidth=1, markersize=5, color=colors[j])
                axis.set_xticks( range(1,10)) 
                if i == 0:
                    axis.set_ylim([0, 1.1])
                elif i >1:
                    axis.set_ylim([ C-2.5, C+1])
                if i ==4:
                    axis.set_ylim([ params[-1]-3, params[-1]+1])
                if x<2:
                    axis.set_xticks([])
                axis.set_title( legends[i])
        fig.suptitle('neg_mle={:.4f}'.format(loss))
        fig.text(0.5, 0, 'Trials per stimuli', ha='center')
        fig.legend( ['setsize=2','setsize=3','setsize=4','setsize=5','setsize=6'])
    
    return fig 

def plot_lrcurve_setsize( ax, model_pred):

    trials = np.arange( 1, 10)
    setsizes = [ 2, 3, 4, 5, 6]
    # set the color for each set size
    colors   = [     'r',     'b',     'g','orange', 'violet']

    # iteration the data set to get the model prediction
    for i, sz in enumerate( setsizes):
        lr_curve = extra_dynamic_setsize_effect( model_pred, 'accuracy')[sz]
        ax.plot( trials, lr_curve, 'o-', 
                        linewidth=1, markersize=5, color=colors[i])
    ax.set_ylim( [ 0, 1.1])
    ax.set_xticks(range(1,10))
        
def plot_cogload_setsize( ax, model_pred, optimal_pred):
    
    setsizes = [ 2, 3, 4, 5, 6]

    # load prediction of the optimal model and get the curve 
    opt_pi_comp_curve = extract_pi_complexity_setsize( optimal_pred, mode='model')
    
    # load prediction of the selected model and get the curve
    model_pi_comp_curve = extract_pi_complexity_setsize( model_pred, mode='model')

    # plot the results
    ax.plot( setsizes, opt_pi_comp_curve, 'o-',
                linewidth=1, markersize=5, color='b')
    ax.plot( setsizes, model_pi_comp_curve, 'o-',
                linewidth=1, markersize=5, color='r')
    ax.set_ylim( [ 0, 2])
    ax.set_yticks([0,1,2])
    ax.set_xticks(range(2,7))

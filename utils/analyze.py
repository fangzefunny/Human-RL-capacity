import numpy as np 
import pandas as pd 
from scipy.special import psi, logsumexp

##########################################
##    PART1: Rate distortion package    ##
##########################################

def entropy(px):
    '''Shannon Entropy of a vector 
    H(x) = px log(px)
    '''
    return -np.sum( px * np.log( px + 1e-20))

def normalize(x):
    '''Normalize
    '''
    return x / np.sum(x)

def nats_to_bits( nats):
    '''Map nats to bits
    '''
    return nats / np.log(2)

def Blahut_Arimoto( distort, p_x, 
                    beta,
                    tol=1e-3, max_iter=50):
    
    # init for iteration
    nX, nY = distort.shape[0], distort.shape[1]
    p_y1x  = np.ones([nX, nY]) / nY
    p_y    = ( p_x.T @ p_y1x).T
    done   = False 
    i      = 0

    while not done:

        # cache  the old channel for convergence check
        old_p_y1x = p_y1x

        # update p(y|x) ∝ p(y)exp( -βD(x,y))
        log_p_y1x = - beta * distort + np.log(p_y.T + np.finfo(float).eps) 
        p_y1x     = np.exp( log_p_y1x - logsumexp( log_p_y1x, axis=-1, keepdims=True)) 

        # update p_y = ∑_x p(x)p(y|x)
        p_y = (p_x.T @ p_y1x).T 
    
        # counter 
        i += 1

        # check convergence
        if np.sum(abs( p_y1x - old_p_y1x)) < tol:
            done = True 
        
        if i >= max_iter:
            # print( f'''
            #         The BA algorithm reaches maximum iteration {max_iter},
            #         the outcome might be inaccurate
            #         ''')
            done = True

    return p_y1x, p_y 

def MI_from_data( xs, ys, prior=None):
    '''Hutter estimator of MI
    '''

    nX = len(np.unique(xs))
    nY = len(np.unique(ys))
    counts = np.zeros( [nX, nY])

    if prior==None:
        # if there is no prior
        # use empirical prior
        prior = 1 / (nX * nY)

    # estimate the mutual information 
    for i, x in enumerate(np.unique(xs)):
        for j, y in enumerate(np.unique(ys)):
            counts[ i, j] = prior + np.sum( (xs==x) * (ys==y))

    # https://papers.nips.cc/paper/2001/file/fb2e203234df6dee15934e448ee88971-Paper.pdf equation 4
    n = np.sum( counts)
    nX = np.sum( counts, axis=0, keepdims=True)
    nY = np.sum( counts, axis=1, keepdims=True)
    P = psi( counts+1) - psi( nX+1) - psi(nY+1) + psi(n+1)
    MI = np.sum( counts * P ) / n
    return MI

def mutual_info( px, qyx, py=None):
    '''Compute the mutual information
    '''
    Hy  = -np.sum( py * np.log( py + 1e-20))
    Hyx = -np.sum( px * np.sum( qyx * 
            np.log( qyx + 1e-20), axis=1, keepdims=True))
    return np.sum(Hy - Hyx)

def KLD( p_target, q_current):
    return np.sum(p_target * np.log( p_target +1e-20) - p_target * np.log( q_current + 1e-20))

def JSD( p_target, q_current):
    return .5 * (KLD( p_target, q_current) + KLD( q_current, p_target))

def opt_channel_given_R( x, y, px, cost_fn, capacity, 
                        tol=1e-3, alpha=.01, max_iter = 1e5):
    '''Find opt qyx subject to capacity

    Iteratively to find the optimal with given capacity.
    The objective function can be written as

    min E[ L(x,y)]  s.t. I(x,y) <= C

    The written lagrangian 

    min E[ L(x,y)] + tau( I(x,y) - C )

    the update scheme

    1. q_y|x = 1/Z * exp( - 1/tau * Loss + log pa)
    2. py = px @ q_y|x
    3. tau = max(0, tau + alpha*(I(x,y) - C) )  

    Inputs:
        x  ([nx, 1] vector): input list to show the input dimensions
        y  ([ny, 1] vector): output dimensions  
        px ([nx, 1] vector): input distribution
        cost_fn ([nx, ny] vector): distortion matrix
        Capacity (scalar): the given capacity

    Args:
        tol: tolerance of convergence
        alpha: learning rate of temperature
        max_iter: max iterations 

    Return:
        qyx: optimal channel 
    '''
    # extract some variables 
    nx = len(x)
    ny = len(y)
    px = np.reshape( px, [nx, 1])

    # iterate to find the optimal channel
    # initialize the iteration 
    tau   = 1
    niter = 0 
    done  = False 
    py    = np.ones( [ 1, ny]) / ny
    qyx   = np.ones( [nx, ny]) / ny
    
    while not done:
        # store the current channel 
        qyx0 = qyx 
        tau0 = tau
        # update channel 
        beta = 1 / tau
        log_qyx = np.log( py + 1e-20) - beta * cost_fn
        log_Z   = logsumexp( log_qyx, axis=1, keepdims=True)
        qyx     = np.exp( log_qyx - log_Z)
        # update marginal output distribution
        py = px.T @ qyx
        # update the tradeoff 
        rate = mutual_info(px, qyx)
        tau = np.max( [ 1e-20, tau + alpha * ( 
                 rate - capacity)])
        # check convergence 
        delta = np.sum( (qyx0 - qyx)**2 + (tau0 - tau)**2)
        if ( delta < tol) or (niter > max_iter):
            done = True
            if niter >= max_iter:
                print( 'reach maximum iteration, check the alpha')
            break  
        # record the iteration 
        niter += 1

    return qyx 

def opt_channel_given_D( x, y, px, cost_fn, D, 
                        tol=1e-3, alpha=80, max_iter=1e5):
    '''Find opt q_yx subject to distortion

    Iteratively to find the optimal wty.
    The objective function can be written as

    min I(x,y) s.t.  E[ L(x,y)] <= D

    The written lagrangian 

    min I(x,y) + beta( E[ L(x,y)] - D)

    the update scheme

    1. q_y|x = 1/Z * exp( - beta * Loss + log pa)
    2. py = px @ q_y|x
    3. beta = beta + alpha*(E[ L(x,y)] - D) 
    '''
    # extract some variables 
    nx = len(x)
    ny = len(y)
    px = np.reshape( px, [nx, 1])

    # iterate to find the optimal channel
    # initialize the iteration 
    beta  = 1
    done  = False
    py    = np.ones( [ 1, ny]) / ny
    qyx   = np.ones( [nx, ny]) / ny
    niter = 0
    
    while not done:
        # store the current channel 
        qyx0 = qyx 
        beta0 = beta 
        # update channel 
        log_qyx = - beta * cost_fn + np.log( py + 1e-20)
        log_Z   = logsumexp( log_qyx, axis=1, keepdims=True)
        qyx     = np.exp( log_qyx - log_Z)
        # update marginal output distribution
        py = px.T @ qyx
        # update the tradeoff 
        primal_con = np.sum( px.T @ (qyx * cost_fn)) - D
        beta =  beta + alpha * primal_con
        # check convergence 
        delta = np.sum( (qyx0 - qyx)**2 + (beta0 - beta)**2)
        if ( delta < tol) or (niter > max_iter):
            done = True
            if niter >= max_iter:
                print( 'reach maximum iteration, check the alpha')
            break  
        niter += 1

    return qyx 

def mle_loss( data):
    return np.sum(data.negLogLike)

def Set_Size_effect( data, mode = 'human'):
    '''Extra setsize effect 

    input: 
        data: need to input a dataframe

    args:
        mode:
            -human: extract the set size effect of human data
            -model: extract the set size effect of model prediction
            -beta:  extract the set size effect of different beta 

    output:
        out_dict: a dict with setsize as key
                  and the prediction over iterations.
                  The prediction is averaged over
                    - blocks x subjects
    '''
    # concatenate the data into a long list
    set_sizes = [ 2, 3, 4, 5, 6]  # get the setsize list
    trial_lst = range( 1, 10)                   # manual the iteration list 
    outcome = np.zeros([ len(set_sizes), len(trial_lst)])                           # output placeholder
    for i, sz in enumerate(set_sizes):  
        for j, trial in enumerate(trial_lst):
            sub2_data = data[ (data.iter == trial) &
                                  (data.setSize == sz)]
            N = len( sub2_data)
            if mode == 'human':
                # learn the probability as categorical distribution
                sub3_data = sub2_data[ sub2_data.reward == 1] 
                n = len( sub3_data)  
                outcome[ i, j] =  n / N
            elif mode == 'model':
                outcome[ i, j] = np.mean( sub2_data.prob)
            elif mode == 'tradeoff':
                outcome[ i, j] = np.mean( sub2_data.tradeoff)
            elif mode == 'constraint':
                outcome[ i, j] = np.mean( sub2_data.con)
            elif mode == 'pi_complexity':
                outcome[ i, j] = np.mean( sub2_data.pi_complexity)
            elif mode == 'rep_complexity':
                outcome[ i, j] = np.mean( sub2_data.rep_complexity)
            
    return outcome

def Model_Pi_Rate( data, min_it=1):

    set_sizes = [ 2, 3, 4, 5, 6]  # get the setsize list
    trial_lst = range( min_it, 10)                   # manual the iteration list 
    outcome = np.zeros([ len(set_sizes), len(trial_lst)])  
    for i, sz in enumerate(set_sizes):  
        for j, trial in enumerate(trial_lst):
            sub2_data = data[ (data.iter == trial) &
                              (data.setSize == sz)]  
            outcome[ i, j] = np.mean( sub2_data.pi_complexity)
    
    # take the mean over iteration
    Pi_Rate = np.mean( outcome, axis=1)
    return Pi_Rate

def Model_Psi_Rate( data, min_it=1):

    set_sizes = [ 2, 3, 4, 5, 6]  # get the setsize list
    trial_lst = range( min_it, 10)                   # manual the iteration list 
    outcome = np.zeros([ len(set_sizes), len(trial_lst)])  
    for i, sz in enumerate(set_sizes):  
        for j, trial in enumerate(trial_lst):
            sub2_data = data[ (data.iter == trial) &
                              (data.setSize == sz)]  
            outcome[ i, j] = np.mean( sub2_data.rep_complexity)
    
    # take the mean over iteration
    Psi_Rate = np.mean( outcome, axis=1)
    return Psi_Rate


def extract_pi_complexity_setsize( data, mode='model', iter_bar=1, prior=.1):

    out_lst = []                                # output placeholder
    set_lst = np.sort(np.unique(data.setSize))  # get the setsize list

    for sz in set_lst:

        if mode == 'model':
            # choose specific set size data 
            sub_data = data[ (data.iter >= iter_bar) & 
                             (data.setSize == sz)]
            lst = []
            for it in range(iter_bar, 10):
                sub2_data = sub_data[ sub_data.iter == it]
                # note that taking mean here and then take mean 
                # in the outer loop is different from directly taking
                # mean in the outer loop. This is because the number
                # of data in different iteration is different. 
                lst.append( np.mean(sub2_data.pi_complexity))
            out_lst.append(np.mean(lst))

        elif mode == 'human':
            
            # infer the data with representation
            # choose the nearly converged policy, >= 5
            sub_data = data[ (data.iter >= iter_bar) & 
                                (data.setSize == sz)]
            state  = sub_data.stimuli.values
            action = sub_data.choice.values
            # learn human policy as catgorical distribution
            state_space = sub_data.stimuli.unique()
            action_space = np.arange(0,3) + 1
            ns = len(state_space)
            # representation
            reprsentations = []
            weights = []
            actions = [] 
            for s,a in zip(state,action):
                reprsentations += list(np.arange(1,ns+1))
                w = list(np.ones([ns,]) * err / ( ns - 1))
                w[int(s)-1] = 1 - err 
                weights += w 
                actions += [int(a)] * ns
            # infer new data
            pi = prior + np.zeros( [ len(state_space), len(action_space)]) 
            # estimate human policy
            for x, a, w in zip( reprsentations, actions, weights):
                pi[x-1, a-1] += w 
            pi = pi / np.sum( pi, axis=1, keepdims=True)
            ps = np.ones( [len(state_space), 1]) * 1 / len(state_space)
            pa = np.ones( [len(action_space), 1]) * 1 / len(action_space)
            out_lst.append( mutual_info( ps, pi, pa))
    
    return out_lst 

def Empirical_Pi_Rate( data, prior):
    '''Analyze the data

    Analyze the data to get the rate distortion curve,

    Input:
        data
        prior, concentration parameter

    Output:
        the empirical estimation of MI from data
        in each block
    '''
     
    # get all different block
    sub_card   = np.unique( data.subject)
    block_card = np.unique(data.block)
    num_sub    = len( sub_card)
    num_block  = len( block_card)
    setSize    = np.array([ 2, 3, 4, 5, 6])

    # create a placeholder
    summary_Rate_data = np.empty( [ num_sub, num_block, len(setSize)]) + np.nan

    # run Blahut-Arimoto
    for subi, sub in enumerate( sub_card):
        for bi, block in enumerate(block_card):

            # select the data in the block 
            idx      = ((data.subject == sub) 
                        & (data.iter < 10) 
                        & (data.block==block))
            states   = data.state[idx].values
            actions  = data.action[idx].values
            setsize  = int(data.setSize[idx].values[0]) - 2
                
            # estimate the mutual information from the data
            Rate_data = MI_from_data( states, actions, prior)
            
            summary_Rate_data[ subi, bi, setsize] = Rate_data

    Emp_Pi_Rate = np.nanmean( np.nanmean( summary_Rate_data, axis=0),axis=0)
           
    return Emp_Pi_Rate

def Empirical_Rate_Reward( data, prior):
    '''Analyze the data

    Analyze the data to get the rate distortion curve,

    Input:
        data

    Output:
        Theoretical rate and distortion
        Empirical rate and distortion 
    '''
 
    # prepare an array of tradeoff
    betas = np.logspace( np.log10(.1), np.log10(10), 50)

    # create placeholder
    # the challenge of creating the placeholder
    # is that the length of the variable change 
    # in each iteration. To handle this method, 
    # my strategy is to create a matrix with 
    # the maxlength of each variable and then, use 
    # nanmean to summary the variables
    
    # get the number of subjects
    num_sub = len(data.keys())
    max_setsize = 5
    
    # create a placeholder
    results = dict()
    summary_Rate_data = np.empty( [ num_sub, max_setsize]) + np.nan
    summary_Val_data  = np.empty( [ num_sub, max_setsize]) + np.nan
    summary_Rate_theo = np.empty( [ num_sub, len(betas), max_setsize,]) + np.nan
    summary_Val_theo  = np.empty( [ num_sub, len(betas), max_setsize,]) + np.nan

    # run Blahut-Arimoto
    for subi, sub in enumerate(data.keys()):

        #print(f'Subject:{subi}')
        sub_data  = data[ sub]
        blocks    = np.unique( sub_data.block) # all blocks for a subject
        setsize   = np.zeros( [len(blocks),])
        Rate_data = np.zeros( [len(blocks),])
        Val_data  = np.zeros( [len(blocks),])
        Rate_theo = np.zeros( [len(blocks), len(betas)])
        Val_theo  = np.zeros( [len(blocks), len(betas)])
        #errors    = np.zeros( [len(blocks),])
        #bias_state= np.zeros( [len(blocks), 6]) 

        # estimate the mutual inforamtion for each block 
        for bi, block in enumerate(blocks):
            idx      = ((sub_data.block == block) & 
                        (sub_data.iter < 10))
            states   = sub_data.state[idx].values
            actions  = sub_data.action[idx].values
            cor_acts = sub_data.correctAct[idx].values
            rewards  = sub_data.reward[idx].values
            
            # estimate some critieria 
            #errors[bi]    = np.sum( actions != cor_acts) / len( actions)
            Rate_data[bi] = MI_from_data( states, actions, prior)

            Val_data[bi] = np.mean( rewards)

            # estimate the theoretical RD curve
            S_card  = np.unique( states)
            A_card  = range(3)
            nS      = len(S_card)
            nA      = len(A_card)
            
            # calculate distortion fn (utility matrix) 
            Q_value = np.zeros( [ nS, nA])
            for i, s in enumerate( S_card):
                a = int(cor_acts[states==s][0]) # get the correct response
                Q_value[ i, a] = 1

            # init p(s) 
            p_s     = np.zeros( [ nS, 1])
            for i, s in enumerate( S_card):
                p_s[i, 0] = np.mean( states==s)
            p_s += np.finfo(float).eps
            p_s = p_s / np.sum( p_s)
            
            # run the Blahut-Arimoto to get the theoretical solution
            for betai, beta in enumerate(betas):
                
                # get the optimal channel for each tradeoff
                pi_a1s, p_a = Blahut_Arimoto( -Q_value, p_s,
                                              beta)
                # calculate the expected distort (-utility)
                # EU = ∑_s,a p(s)π(a|s)Q(s,a)
                theo_util  = np.sum( p_s * pi_a1s * Q_value)
                # Rate = β*EU - ∑_s p(s) Z(s) 
                # Z(s) = log ∑_a p(a)exp(βQ(s,a))  # nSx1
                Zstate     = logsumexp( beta * Q_value + np.log(p_a.T), 
                                    axis=-1, keepdims=True)
                theo_rate  = beta * theo_util - np.sum( p_s * Zstate)

                # record
                Rate_theo[ bi, betai] = theo_rate
                Val_theo[ bi, betai]  = theo_util

            setsize[bi] = len(np.unique( states))

        for zi, sz in enumerate([ 2, 3, 4, 5, 6]):
            summary_Rate_data[ subi, zi] = np.nanmean( Rate_data[ (setsize==sz),])
            summary_Val_data[ subi, zi]  = np.nanmean(  Val_data[ (setsize==sz),])
            summary_Rate_theo[ subi, :, zi] = np.nanmean( Rate_theo[ (setsize==sz), :],axis=0)
            summary_Val_theo[ subi, :, zi]  = np.nanmean(  Val_theo[ (setsize==sz), :],axis=0)

    # prepare for the output 
    results[ 'Rate_theo'] = np.nanmean( summary_Rate_theo, axis=0)
    results[  'Val_theo'] = np.nanmean(  summary_Val_theo, axis=0)
    results[ 'Rate_data'] = summary_Rate_data
    results[  'Val_data'] = summary_Val_data

    return results
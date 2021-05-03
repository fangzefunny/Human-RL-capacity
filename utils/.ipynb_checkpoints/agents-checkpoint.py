import numpy as np 
from scipy.special import softmax, logsumexp 
from utils.analyze import mutual_info, normalize, entropy

# the replay buffer to store the memory 
class simpleBuffer:
    
    def __init__( self):
        self.table = []
        
    def push( self, *args):
        self.table = tuple([ x for x in args]) 
        
    def sample( self ):
        return self.table

def dL_dW( pred, target):
    nd = pred.shape[0]
    pred = pred.reshape([-1, 1])
    target = target.reshape([-1, 1])
    dL_dpred = 1 + np.log( pred+1e-20) - np.log( target+1e-20)
    dpred_dW = pred * np.eye(nd) - np.dot(pred, pred.T)
    return np.dot( dpred_dW, dL_dpred)

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Base agent class    %
%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''

class BaseRLAgent:
    
    def __init__( self, obs_dim, action_dim):
        self.obs_dim = obs_dim 
        self.action_dim = action_dim
        self.action_space = range( self.action_dim)
        self._init_critic()
        self._init_actor()
        self._init_marginal_obs()
        self._init_marginal_action()
        self._init_memory()
        
    def _init_marginal_obs( self):
        self.po = np.ones( [self.obs_dim, 1]) * 1 / self.obs_dim

    def _init_marginal_action( self):
        self.pa = np.ones( [self.action_dim, 1]) * 1 / self.action_dim

    def _init_memory( self):
        self.memory = simpleBuffer()
        
    def _init_critic( self):
        self.q_table = np.ones( [self.obs_dim, self.action_dim]) * 1/self.action_dim

    def _init_actor( self):
        self.pi = np.ones( [ self.obs_dim, self.action_dim]) * 1 / self.action_dim
            
    def q_value( self, obs, action):
        q_obs = self.q_table[ obs, action ]
        return q_obs 
        
    def eval_action( self, obs, action):
        pi_obs_action = self.pi[ obs, action]
        return pi_obs_action
    
    def get_action( self, obs):
        pi_obs = self.pi[ obs, :]
        return np.random.choice( self.action_space, p = pi_obs)
        
    def update(self):
        return NotImplementedError

    def pi_complexity( self):
        po = self.po.reshape( self.obs_dim, 1)
        pi = self.pi.reshape( self.obs_dim, self.action_dim)
        pa = self.pa.reshape( self.action_dim, 1)
        return mutual_info(po, pi, pa)

class RepActBasemodel( BaseRLAgent):
    '''Base model with representation

    Init Ps, Px, Pa as uniform distribution 
    Init Q( s, x, a)
    evaluate of action: pi(a|s) = sum_x psi(x|s)pi(a|x)
    generate action: pi(a|s) = sum_x psi(x|s)pi(a|x)

    bel means belief
    psi means state encoder 
    '''

    def __init__( self, obs_dim, action_dim):
        super().__init__( obs_dim, action_dim)
        self._init_marginal_state()
        self._init_bel_critic()
        
    def _init_psi( self):
        self.psi = np.eye( self.obs_dim) 

    def _init_marginal_state(self):
        self.ps = np.ones( [self.obs_dim, 1]) * 1 / self.obs_dim
        
    def _init_bel_critic( self):
        self.belq_table = np.ones( [ self.obs_dim, self.obs_dim, 
                             self.action_dim]) * 1 / self.action_dim

    def belq_value( self, obs, action):
        q_bel = self.belq_table[ obs, :, action ]
        return q_bel

    def eval_action( self, obs, action):
        pi_obs_action = self.psi @ self.pi 
        pi_ao = pi_obs_action[ obs, action]
        return pi_ao
    
    def get_action( self, obs):
        pi_obs_action = self.psi @ self.pi 
        pi_obs = pi_obs_action[ obs, :]
        return np.random.choice( self.action_space, p = pi_obs)
    
    def pi_complexity( self):
        ps = self.ps.reshape( self.obs_dim, 1)
        pi = self.pi.reshape( self.obs_dim, self.action_dim)
        pa = self.pa.reshape( self.action_dim, 1)
        return mutual_info(ps, pi, pa)

    def rep_complexity( self):
        po  = self.po.reshape( self.obs_dim, 1)
        psi = self.psi.reshape( self.obs_dim, self.obs_dim)
        ps  = self.ps.reshape( self.obs_dim, 1)
        return mutual_info(po, psi, ps) 

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Agents for the paper    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''

class RLbaseline( BaseRLAgent):
    '''Implement the sto-update softmax RL

    st: the current state 
    at: the current action

    Update Q:
        Q(st, at) += lr_q * [reward - Q(st, at)]

    Update Pi: 
        loss(st, :) = max Q(st, :) - Q(st, :) 
        Pi( s_t, :) = exp( -beta * loss(st, :)) / sum_a exp( -beta * loss(st, :))
    '''
    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim,)
        self.lr  = params[0]
        self.tau = params[1]

    def update( self):

        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate q prediction
        q_pred = self.q_value( obs, action)

        # update critic 
        self.q_table[ obs, action] += \
                    self.lr * ( reward - q_pred)

        # update policy
        beta = np.clip( 1/self.tau, 0, 1e10)
        self.pi = softmax( beta * self.q_table, axis=1)

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Agents Collin's paper   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''

class RLWM( RepActBasemodel):

    def __init__(self, obs_dim, action_dim, params):
        super().__init__(obs_dim, action_dim)
        self.lr_q = 1

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Agents for the paper    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''

class fix_tau_fix_psi( RepActBasemodel):
    '''Implement fix temp fix psi model

    problem: 
        max_{pi} E[r] s.t. I^psi(s;x) + I^{pi}(x;a) <= C

    Fix g:
        g(s,x) = 1 - eps     if s=x 
               = eps/(ns-1)  otherwise
 
    Update Q:
        belR(st, x, at) = q(x|st)R(st, at) 
        belQ(st, x, at) += lr_q * [ belR(st, x, at)- belQ(st, x, at)]

    Update Pi:
        for xi = [1:ns]:
            belQx = belQ(xi, :) 
            Pi(xi, :) = exp( -1/tau * loss(xi, :)) / sum_a exp( -beta * loss(xi, :))
        (In reality, I use matrix multiplication)

    Fix tau:
    '''
    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim)
        # paramter for the learning model 
        self.lr_q = params[0]
        self.err  = params[1]
        self.tau  = params[2]
        # init state encoder: psi
        self.psi = np.eye( self.obs_dim) * (
            1 - self.err * (1 + 1 / (self.obs_dim -1)))\
             + self.err / (self.obs_dim - 1)
        self.ps = (self.po.T @ self.psi).T 
       
    def update( self):

        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # make predictions
        q_pred = self.q_value( obs, action)

        # update critic bel q
        self.q_table[ obs, action] += \
                    self.lr_q * ( reward - q_pred)
        
        # believed of the q value
        belq = np.sum( self.psi[ :, :, np.newaxis] * \
                   self.q_table[ :, np.newaxis, :], axis=0) 

        # update policy: optimize towards 
        beta = np.clip( 1/self.tau, 0, 1e10)
        self.pi = softmax( beta * belq 
                  + np.log( self.pa.reshape([-1]) + 1e-20), axis=1)
        

class fix_C_fix_psi( RepActBasemodel):
    '''Implement fix temp fix psi model

    problem: 
        max_{pi} E[r] s.t. I^psi(s;x) + I^{pi}(x;a) <= C

    Fix g:
        g(s,x) = 1 - eps     if s=x 
               = eps/(ns-1)  otherwise
 
    Update Q:
        Q(st,at) = q(x|st)R(st, at) 
        
    Form belQ:
        belQ( ot, s, a) = 

    Update Pi:
        for xi = [1:ns]:
            belQx = belQ(xi, :) 
            Pi(xi, :) = exp( -1/tau * loss(xi, :)) / sum_a exp( -beta * loss(xi, :))
        (In reality, I use matrix multiplication)

    Update tau:
        

    '''
    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim)
        # paramter for the learning model 
        self.lr_q = params[0]
        self.lr_t = params[1]
        self.err  = params[2]
        self.tau  = params[3]
        self.C    = params[4]
        # init state encoder: psi
        self.psi = np.eye( self.obs_dim) * (
            1 - self.err * (1 + 1 / (self.obs_dim -1)))\
             + self.err / (self.obs_dim - 1)
        self.ps = (self.po.T @ self.psi).T 
       
    def update( self):

        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # make predictions
        q_pred = self.q_value( obs, action)

        # update critic bel q
        self.q_table[ obs, action] += \
                    self.lr_q * ( reward - q_pred)
        
        # believed of the q value
        belq = np.sum( self.psi[ :, :, np.newaxis] * \
                   self.q_table[ :, np.newaxis, :], axis=0) 

        # update policy: optimize towards 
        beta = 1/self.tau
        self.pi = softmax( beta * belq 
                  + np.log( self.pa.reshape([-1]) + 1e-20), axis=1)

        # update marginal policy: 
        #self.pa = ( self.ps.T @ self.pi).T # (1 x ns, ns x na).T

        # update temperature 
        self.affordance  = self.rep_complexity() + self.pi_complexity()
        self.free_memory = ( self.C - self.affordance)
        self.tau = np.max( [1e-20, self.tau + self.lr_t * (-self.free_memory) ])

class fix_C( RepActBasemodel):
    '''fix_C model

    The optimization objective for the actor is:

        L(π, ψ, β) = - E(r|π) + τ[I(o;s) + I(s;a)
                   = E_p(o,s,a)[ -Q(s,o) + τ log[ψ(s|o)/p(s)] + τ log[π(a|s)/p(a)]]

    The update schema of this model:

        Q(st,at) = Q(st,at) + α_q * [r(st,at) - Q(st,at)
        πt = argmin L( π, ψt-1, βt-1)
        ψt = argmin L( πt, ψ，βt-1)
        βt = argmin L( πt，ψt, β)

    '''
    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim)
        # paramter for the learning model 
        self.lr_q   = params[0]
        self.lr_pi  = params[1]
        self.lr_a   = params[2]
        self.lr_psi = params[3]
        self.lr_s   = params[4]
        self.lr_t   = params[5]
        self.tau    = params[6]
        self.C      = params[7]
        self.free_memory = self.C
        # init state encoder: psi
        self.theta_psi = np.eye( self.obs_dim) 
        self.psi = np.exp(self.theta_psi - logsumexp( self.theta_psi, axis=-1, keepdims=True))  
        self.theta_pi  = np.ones([ self.obs_dim, self.action_dim])/self.action_dim
        self.pi  = np.exp(self.theta_pi - logsumexp( self.theta_pi,  axis=-1, keepdims=True)) 
       
    def update( self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # make predictions
        q_pred = self.q_value( obs, action)

        # update the q function
        self.q_table[ obs, action] +=  self.lr_q * ( reward - q_pred)

        # get the inference distribution of observation given state 
        # the inference follow the Baye's rule
        # p(o|s) = ψ(s|o)p(o)/p(s) --> nOxnS
        p_o1s = self.po * self.psi / self.ps.T
        
        # update the believed Q value: BelQ(s,a) = p(ot|s)Q(ot,a)
        belq = p_o1s[ obs, :, np.newaxis] * self.q_table[ obs, np.newaxis, :]

        # update policy: 
        # θ_π -= α_π∇θ_π[π(a|s)||π*(a|s)]
        # π(a|s) ∝ p(a)exp( β * BelQ(s,a))
        beta = np.clip( 1/self.tau, 0, 1e10)
        log_pi_a1s = beta * belq + np.log( self.pa.T ) + 1e-20
        pi_a1s = np.exp( log_pi_a1s - logsumexp( log_pi_a1s, axis=-1, keepdims=True))
        for s in range(self.obs_dim):
            self.theta_pi[s,:] -= self.lr_pi * dL_dW( self.pi[s,:], pi_a1s[s,:]).reshape([-1])+1e-20      
        self.pi = np.exp( self.theta_pi - logsumexp( self.theta_pi, axis=-1, keepdims=True)) 
        
        # update the marginal action distribution: p(a) += α_a[ π(a|o) - p(a)]
        p_a1ot = (self.psi @ self.pi)[ obs, :].reshape([-1,1])  
        self.pa += self.lr_a * (p_a1ot - self.pa) + 1e-20
        self.pa = self.pa / np.sum(self.pa) 

        # update state encode. Here we use a gradient based method here
        # θ_ψ -= α_ψ∇θ_ψ[ψ(s|o)||ψ*(s|o)]
        # ψ*(s|o) ∝ p(s)exp(βF)
        # F = EQ(ot,s) - τD[π(a|s)||p(a)]
        # compute EQ(ot,s) = ∑_a π(a|s)Q(o,a) [nS,]
        EQ = np.sum( self.q_table[ obs, np.newaxis, :] 
                   * self.pi[ np.newaxis, :, :], axis=-1)
        # D[π(a|s)||p(a)] [nS,]
        DKL = np.sum( self.pi * np.log( self.pi + 1e-20)
                    - self.pi * np.log( self.pa.T + 1e-20), axis=-1)
        log_psi_s1o = beta * EQ - DKL + np.log( self.ps[ :, 0] + 1e-20) + 1e-20
        psi_s1o = np.exp( log_psi_s1o - logsumexp( log_psi_s1o))
        self.theta_psi[ obs, :] -= self.lr_psi * dL_dW( self.psi[ obs, :], psi_s1o).reshape([-1])+1e-20
        self.psi[ obs, :] = np.exp(self.theta_psi[ obs, :] - logsumexp( self.theta_psi[ obs,:]))

        # update marginal state distribution: p(s) += α_s[ ψ(s|ot) - p(s)]
        psi_s1ot = self.psi[ obs, :].reshape([-1,1])
        self.ps += self.lr_s * ( psi_s1ot - self.ps) + 1e-20
        self.ps = self.ps / np.sum(self.ps) 

        # update temperature 
        self.cog_load  = self.rep_complexity() + self.pi_complexity()
        self.free_memory = ( self.C - self.cog_load)
        self.tau = np.max( [1e-20, self.tau + self.lr_t * (-self.free_memory) ])


class fix_C_heur( RepActBasemodel):

    '''Implement IT model 2

    Problem: 
        max_{pi, beta} E[r]  s.t. I^g(s;x) + I^pi(x;a) <= C

    Optimization objective: 
        max_{pi, beta} E[r] - beta[I^g(s,x) + I^pi(x,a)]

    Update Q:
        R(st, x, at) = q(x|st)R(st, at) 
        Q(st, x, at) += lr_q * [R(st, x, at)- Q(st, x, at)]

    Update G:
        for si = [1:ns]:
            G(si, :) = exp(  1/beta * Q_v(si, :) ) / sum_x exp( beta * Q_v(xi, :)) 

    Update Pi:
        for xi = [1:ns]:
            Pi(xi, :) = exp( 1/beta * Q(xi, :) ) / sum_a exp( -beta * loss(xi, :))
        (In reality, I use matrix multiplication)

    Update beta: projected gradient 
        max( [0, beta + lr_b * (I(s,x) + I(x,a) - C)])
    '''

    def __init__( self, state_dim, action_dim, params):
        super().__init__( state_dim, action_dim)
        # pass the parameter
        self.lr_q = params[0]
        self.lr_g = params[1]
        self.lr_a = params[2]
        self.lr_t = params[3]
        self.tau  = params[4]
        self.C    = params[5]
        self.free_resource = self.C
        self.psi = np.eye( self.obs_dim)
    
    def update( self):

        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # make predictions
        q_pred = self.q_value( obs, action)

        # update critic bel q
        self.q_table[ obs, action] += \
                    self.lr_q * ( reward - q_pred)

        # believed of the q value
        belq = np.sum( self.psi[ :, :, np.newaxis] * \
                   self.q_table[ :, np.newaxis, :], axis=0) 

        # update policy: optimize towards 
        beta = np.clip( 1/self.tau, 0, 1e10)
        self.pi = softmax( beta * belq 
                  + np.log( self.pa.reshape([-1]) + 1e-20), axis=1)

        # update policy bias
        action_obs = (self.psi @ self.pi)[ obs, : ]
        self.pa += self.lr_a * (action_obs.reshape([-1,1]) - self.pa)

        # if self.free_resource<=0:
        s = np.eye(self.obs_dim)[obs].reshape([1, -1])
        sim_matrix = softmax(np.dot( s, np.dot(self.pi, np.log( self.pi.T +1e-20))),axis=1)
        self.psi[ obs, :] = (1 - self.lr_g) * self.psi[ obs, :].copy() \
                                + self.lr_g * sim_matrix

        # update beta
        self.affordance  = self.rep_complexity() + self.pi_complexity()
        self.free_memory = ( self.C - self.affordance)
        self.tau = np.max( [1e-20, self.tau + self.lr_t * (-self.free_memory) ])
        
class gradient_based:

    def __init__( self, obs_dim, action_dim, params):
        self.obs_dim = obs_dim 
        self.action_dim = action_dim
        self.action_space = range( self.action_dim)
        self._init_critic()
        self._init_actor()
        self._init_marginal_obs()
        self._init_marginal_action()
        self._init_memory()
        self.lr_v     = params[0]
        self.lr_theta = params[1]
        self.lr_a     = params[2]
        self.beta     = params[3]

    def _init_critic( self):
        self.v = np.zeros([ self.obs_dim, 1])

    def _init_actor( self):
        self.theta = np.zeros( [ self.obs_dim, self.action_dim]) + 1e-20
        self.pi    = np.ones( [ self.obs_dim, self.action_dim]) / self.action_dim
    
    def _init_marginal_obs( self):
        self.p_s   = np.ones( [ self.obs_dim, 1]) / self.obs_dim
    
    def _init_marginal_action( self):
        self.p_a   = np.ones( [ self.action_dim, 1]) / self.action_dim

    def _init_memory( self):
        self.memory = simpleBuffer()

    def value( self, obs):
        v_obs = self.v[ obs, 0]
        return v_obs 

    def q_value( self, obs, action):
        q_sa = self.v[ obs, 0] * self.theta[ obs, action] 
        return q_sa 
        
    def eval_action( self, obs, action):
        pi_obs_action = self.pi[ obs, action]
        return pi_obs_action
    
    def get_action( self, obs):
        pi_obs = self.pi[ obs, :]
        return np.random.choice( self.action_space, p = pi_obs)
        
    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: V(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + 1e-20) \
                   - np.log( self.p_a[ action, 0] + 1e-20)

        # compute predictioin error: δ = βr(st,at) - C_π(st,at) - V(st) --> scalar
        rpe = self.beta * reward - pi_comp - pred_v_obs 
        
        # update critic: V(st) = V(st) + α_v * δ --> scalar
        self.v[ obs, 0] += self.lr_v * rpe

        # update policy parameter: θ = θ + α_θ * β * I(s=st) * δ *[1- π(at|st)] --> [nS,] 
        I_s = np.zeros([self.obs_dim])
        I_s[obs] = 1.
        self.theta[ :, action] += self.lr_theta * rpe * \
                                  self.beta * I_s * (1 - pi_like) 

        # update policy parameter: π(a|s) ∝ p(a)exp(θ(s,a)) --> nSxnA
        # note that to prevent numerical problem, I add an small value
        # to π(a|s). As a constant, it will be normalized.  
        log_pi = self.beta * self.theta + np.log(self.p_a.T) + 1e-15
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))
    
        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + 1e-20
        self.p_a = self.p_a / np.sum( self.p_a)

    def pi_complexity( self):
        return mutual_info( self.p_s, self.pi, self.p_a)

class gershman_model2( gradient_based):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim, params)

    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: V(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + 1e-20) \
                   - np.log( self.p_a[ action, 0] + 1e-20)

        # compute predictioin error: δ = βr(st,at) - C_π(st,at) - V(st) --> scalar
        rpe = self.beta * reward - pi_comp - pred_v_obs 
        
        # update critic: V(st) = V(st) + α_v * δ --> scalar
        self.v[ obs, 0] += self.lr_v * rpe

        # update policy parameter: θ = θ + α_θ * [β + π(at|st)/(p(at)*N)]* δ *[1- π(at|st)] --> scalar 
        self.theta[ obs, action] += self.lr_theta * rpe * (1 - pi_like) \
                                   * (self.beta + pi_like/self.p_a[action, 0]/self.obs_dim)

        # update policy parameter: π(a|s) ∝ p(a)exp(θ(s,a)) --> nSxnA
        # note that to prevent numerical problem, I add an small value
        # to π(a|s). As a constant, it will be normalized.  
        log_pi = self.beta * self.theta + np.log(self.p_a.T) + 1e-15
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))
    
        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + 1e-20
        self.p_a = self.p_a / np.sum( self.p_a)

class gershman_fixC(gradient_based):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim, params)
        self.lr_v     = params[0]
        self.lr_theta = params[1]
        self.lr_a     = params[2]
        self.lr_t     = params[3]
        self.tau      = params[4]
        self.C        = params[5]
    
    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: V(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + 1e-20) \
                   - np.log( self.p_a[ action, 0] + 1e-20)

        # calculate beta: β = 1/τ
        beta = np.clip( 1/self.tau, 1e-20, 1e10)

        # compute predictioin error: δ = βr(st,at) - C_π(st,at) - V(st) --> scalar
        rpe = beta * reward - pi_comp - pred_v_obs 
        
        # update critic: V(st) = V(st) + α_v * δ --> scalar
        self.v[ obs, 0] += self.lr_v * rpe

        # update policy parameter: θ = θ + α_θ * [β + π(at|st)/(p(at)*N)]* δ *[1- π(at|st)] --> scalar 
        self.theta[ obs, action] += self.lr_theta * rpe * (1 - pi_like) \
                                   * (beta + pi_like/self.p_a[action, 0]/self.obs_dim)

        # update policy parameter: π(a|s) ∝ p(a)exp(θ(s,a)) --> nSxnA
        # note that to prevent numerical problem, I add an small value
        # to π(a|s). As a constant, it will be normalized.  
        log_pi = beta * self.theta + np.log(self.p_a.T) + 1e-15
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))
    
        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + 1e-20
        self.p_a = self.p_a / np.sum( self.p_a)

        # update τ
        self.cog_load  = self.pi_complexity()
        self.free_memory = ( self.C - self.cog_load)
        self.tau = np.max( [1e-20, self.tau + self.lr_t * (-self.free_memory) ])
    




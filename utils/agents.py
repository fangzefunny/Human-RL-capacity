import numpy as np 
from scipy.special import softmax, logsumexp 
from utils.analyze import mutual_info, normalize, entropy

eps_ = np.finfo(float).eps

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
    dL_dpred = 1 + np.log( pred+eps_) - np.log( target+eps_)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    RL baseline model    %
%%%%%%%%%%%%%%%%%%%%%%%%%%% 
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Policy compression model    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

class Grad_model:

    def __init__( self, obs_dim, action_dim):
        self.obs_dim = obs_dim 
        self.action_dim = action_dim
        self.action_space = range( self.action_dim)
        self._init_critic()
        self._init_actor()
        self._init_marginal_obs()
        self._init_marginal_action()
        self._init_memory()
        
    def _init_critic( self):
        self.v = np.ones([ self.obs_dim, 1])

    def _init_actor( self):
        self.theta = np.zeros( [ self.obs_dim, self.action_dim]) + eps_
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
        
    def pi_complexity( self):
        return mutual_info( self.p_s, self.pi, self.p_a)

    def update( self):
        raise NotImplementedError

class Pi_model_1( Grad_model):

    def __init__(self, obs_dim, action_dim, params):
        super().__init__(obs_dim, action_dim)
        self.lr_v     = params[0]
        self.lr_theta = params[1]
        self.lr_a     = params[2]
        self.beta     = params[3]

    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: V(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + eps_) \
                   - np.log( self.p_a[ action, 0] + eps_)

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
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + eps_
        self.p_a = self.p_a / np.sum( self.p_a)

class Pi_model_2( Grad_model):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim)
        self.lr_v     = params[0]
        self.lr_theta = params[1]
        self.lr_a     = params[2]
        self.beta     = params[self.obs_dim+1]

    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: V(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        pi_like    = self.eval_action( obs, action)

        # compute policy compliexy: C_π(s,a)= log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + eps_) \
                   - np.log( self.p_a[ action, 0] + eps_)

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
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a)
        self.p_a = self.p_a / np.sum( self.p_a)

class Pi_model_3( Grad_model):

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
        pi_comp = np.log( self.pi[ obs, action] + eps_) \
                   - np.log( self.p_a[ action, 0] + eps_)

        # calculate beta: β = 1/τ
        beta = np.clip( 1/self.tau, eps_, 1e10)

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
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + eps_
        self.p_a = self.p_a / np.sum( self.p_a)

        # update τ
        self.cog_load  = self.pi_complexity()
        self.free_memory = ( self.C - self.cog_load)
        self.tau = np.max( [eps_, self.tau + self.lr_t * (-self.free_memory) ])
    




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
    dL_dP = (target - pred)
    dP_dW = 1. 
    return dL_dP * 1.

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
        self.p_o = np.ones( [self.obs_dim, 1]) * 1 / self.obs_dim

    def _init_marginal_action( self):
        self.p_a = np.ones( [self.action_dim, 1]) * 1 / self.action_dim

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
        po = self.p_o.reshape( self.obs_dim, 1)
        pi = self.pi.reshape( self.obs_dim, self.action_dim)
        pa = self.p_a.reshape( self.action_dim, 1)
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
        self.p_s = np.ones( [self.obs_dim, 1]) * 1 / self.obs_dim
        
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
        ps = self.p_s.reshape( self.obs_dim, 1)
        pi = self.pi.reshape( self.obs_dim, self.action_dim)
        pa = self.p_a.reshape( self.action_dim, 1)
        return mutual_info(ps, pi, pa)

    def rep_complexity( self):
        po  = self.p_o.reshape( self.obs_dim, 1)
        psi = self.psi.reshape( self.obs_dim, self.obs_dim)
        ps  = self.p_s.reshape( self.obs_dim, 1)
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
        self.beta = params[1]
        self.lr_a = params[2]

    def update( self):

        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate q prediction
        q_pred = self.q_value( obs, action)

        # update critic 
        self.q_table[ obs, action] += \
                    self.lr * ( reward - q_pred)

        # update policy
        self.pi = softmax( self.beta * self.q_table, axis=1)

        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + eps_
        self.p_a = self.p_a / np.sum( self.p_a)

class optimal( RLbaseline):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim, params)
        
    def update( self):

        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate q prediction
        q_pred = self.q_value( obs, action)

        # update critic 
        self.q_table[ obs, action] += \
                    self.lr * ( reward - q_pred)

        # update policy
        self.pi = softmax( self.beta * self.q_table, axis=1)

        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ obs, :].reshape([-1,1]) - self.p_a) + eps_
        self.p_a = self.p_a / np.sum( self.p_a)

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
        I_s = np.zeros([self.obs_dim,])
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

class Psi_Pi_model_3( RepActBasemodel):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim)
        self.lr_q     = params[0]
        self.lr_pi    = params[1]
        self.lr_a     = params[2]
        self.err      = params[3]
        self.beta     = params[self.obs_dim+2]
        # init state encoder
        self.psi      = np.eye( self.obs_dim) * (
                        1 - self.err * (1 + 1 / (self.obs_dim -1))) \
                            + self.err / (self.obs_dim - 1)
        # parameterize the policy
        self.theta    = np.zeros( [ self.obs_dim, self.action_dim]) + eps_
    
    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: Q(st,at) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_q_obs = self.q_value( obs, action)

        # update critic: 
        # Q(st, at) = Q(st,at) + α_q( r - Q(st,at)) 
        self.q_table[ obs, action] += self.lr_q * ( reward - pred_q_obs)

        # calculate the belQ: 
        # belQ(x,at) = ψ(x|st)Q(st,at)  1xnSx1
        belq = np.sum(self.q_table[ :, np.newaxis, :] * 
                      self.psi[ :, :, np.newaxis], axis=0) 
        
        # update policy parameter:
        # θ_π -= α_π ψ(x|st) ∇θ_π[π(a|x)||π*(a|x)]
        # π*(a|s) ∝ p(a)exp( β * BelQ(s,a))
        # π = π + α_π[ π(a|s) - π] 
        log_pi_a1s = self.beta * belq + np.log( self.p_a.T ) + eps_
        pi_a1s = np.exp( log_pi_a1s - logsumexp( log_pi_a1s, axis=-1, keepdims=True)) 
        self.pi += self.lr_pi * ( pi_a1s - self.pi)
        self.pi = self.pi / np.sum( self.pi, axis=-1, keepdims=True)
        
        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( self.pi[ [obs], :].T - self.p_a) + eps_
        self.p_a = self.p_a / np.sum( self.p_a)

    
class Pi_Rep_Grad(Grad_model):

    def __init__( self, obs_dim, action_dim, params):
        super().__init__( obs_dim, action_dim) 
        self.lr_v     = params[0]
        self.lr_theta = params[1]
        self.lr_a     = params[2]
        self.err      = params[3]
        self.beta     = params[self.obs_dim+2]
        self._init_marginal_representation()
        # init state encoder 
        self.psi      = np.eye( self.obs_dim) * (
                        1 - self.err * (1 + 1 / (self.obs_dim -1))) \
                            + self.err / (self.obs_dim - 1)

    def _init_psi( self):
        self.psi = np.eye( self.obs_dim) 

    def _init_marginal_representation(self):
        self.p_x = np.ones( [ self.obs_dim, 1])/ self.obs_dim
    
    def eval_action(self, obs, action):
        pi_obs_action = self.psi @ self.pi 
        pi_ao = pi_obs_action[ obs, action]
        return pi_ao

    def get_action( self, obs):
        pi_obs_action = self.psi @ self.pi 
        pi_obs = pi_obs_action[ obs, :]
        return np.random.choice( self.action_space, p = pi_obs)
    
    def pi_complexity(self):
        return mutual_info( self.p_x, self.pi, self.p_a)

    def psi_complexity( self):
        return mutual_info( self.p_s, self.psi, self.p_x)

    def update(self):
        
        # collect sampeles 
        obs, action, reward = self.memory.sample() 

        # calculate v prediction: V(st) --> scalar
        # and policy likelihood:  π(at|st) --> scalar 
        pred_v_obs = self.value( obs)
        psi_x1st = self.psi[ [obs], :]
        pi_a1st  = (psi_x1st @  self.pi).T # (1xns x nsxna).T = 1xna.T =nax1

        # compute policy compliexy: log( π(at|st)) - log( p(at)) --> scalar  
        pi_comp = np.log( self.pi[ obs, action] + eps_) \
                   - np.log( self.p_a[ action, 0] + eps_)
        # compute policy compliexy: ∑_x ψ(x|st) * log( ψ(x|st)) - log( p(x)) --> scalar 
        psi_comp = np.sum( self.psi[ obs, : ] * 
                    (np.log( self.psi[ obs, : ] + eps_)
                   - np.log( self.p_x.reshape([-1]) + eps_)))
        
        # compute predictioin error: δ = βr(st,at) - C_π(st,at) - V(st) --> scalar
        rpe = self.beta * reward - pi_comp - psi_comp - pred_v_obs 
        
        # update critic: V(st) = V(st) + α_v * δ --> scalar
        self.v[ obs, 0] += self.lr_v * rpe

         # update policy parameter: θ = θ - α_θ * △θ
        I_act = np.zeros_like(self.p_a)
        I_act[ action, 0] = 1.
        dL_dpi = (I_act - pi_a1st) * - rpe
        theta_grad = np.zeros( [ self.obs_dim, self.action_dim])
        for x in range( self.obs_dim):
            theta_grad[ x, :] = psi_x1st[0, x] * self.beta * dL_dpi.reshape([-1])
        self.theta -= self.lr_theta * theta_grad

        # update policy parameter: π(a|s) ∝ p(a)exp(θ(s,a)) --> nSxnA
        # note that to prevent numerical problem, I add an small value
        # to π(a|s). As a constant, it will be normalized.  
        log_pi = self.beta * self.theta +  np.log(self.p_a.T) + eps_
        self.pi = np.exp( log_pi - logsumexp( log_pi, axis=-1, keepdims=True))

        # update the mariginal policy: p(a) = p(a) + α_a * [ π(a|st) - p(a)] --> nAx1
        self.p_a += self.lr_a * ( pi_a1st - self.p_a)
        self.p_a = self.p_a / np.sum( self.p_a)

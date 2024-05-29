import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statistics import fmean
import logging

# f(x,y)=a+2(b−a)min(x,y,1−x,1−y)   

class PruneRatioEnv(object):

  def __init__(self, randomize=True):
    """ 
    Create a Fare Upsell Revenue Simulator. 
    Args:
      randomize (bool): if False, customer behavior is sampled from a fixed 
        model; else the model is resampled at reset() from a hyper-prior.
    """
    self.randomize = randomize
    # number of customers whose behavior is sampled in each step
    self.n_networks = 100
    self.max_gain = 15


  def reset(self):
    # sample (latent) customer behavior from hyper-prior
    self.a_one = 0.6
    self.a_two = 0.7
    self.a_three = 0.5

    self.x = np.linspace(0.01,0.99,41)
    #self.y = np.linspace(0.01,0.99,41)

    self.ratio_one, self.ratio_two, self.ratio_three = np.meshgrid(self.x, self.x, self.x, indexing='ij')
    c = np.linspace(0,self.max_gain,41)
    cost_one, cost_two, cost_three = np.meshgrid(c, c, c, indexing='ij')

    #p_one = self._r_cost_one(self.ratio_one, self.a_one)
    #p_two = self._r_cost_two(self.ratio_two, self.a_two)
    p_one = self._r_cost_one(self.x, self.a_one)
    p_two = self._r_cost_two(self.x, self.a_two)
    p_three = self._r_cost_three(self.x, self.a_three)

    p_two = (1-p_two)*p_one
    p_three = (1-p_three)*p_two
    p_one_joint, p_two_joint, p_three_joint = np.meshgrid(p_one, p_two, p_three, indexing='ij')
    self.revenue = (p_one_joint*cost_one +  p_two_joint*cost_two + p_three_joint*cost_three)/2
    return None

  @staticmethod
  def _r_cost_one(x, a_one):
    return a_one-np.abs(x-a_one)

  @staticmethod
  def _r_cost_two(x, a_two):
    return np.abs(x-a_two)

  @staticmethod
  def _r_cost_three(x, a_three):
    return np.abs(x-a_three)

  def step(self, action):
    """ 
    Simulate aggregate customer behavior within a finite sample
    
    Args:
      action (np.array): vector in R+^2 of upsell prices [BASIC->MEDIUM, BASIC->HIGH]

    Returns:
      observation, reward, done, info (see gym.Env interface)
      observation (np.array): normalized vector in R^3 of proportion of customers that bought [BASIC, MEDIUM, HIGH]
      reward (scalar): total fare upsell revenue
      done (bool): always False (this environment is non-episodic)
      info (object): always None (reserved for debug info, of which this environment provides none currently)
    """

    # Synthetic Ground Truth Customer Price Sensitivity (for comparison)
    ratios = action

    # probability of customer buying HIGH at delta HIGH
    p_one = self._r_cost_one(ratios[0], self.a_one)
    p_two = self._r_cost_two(ratios[1], self.a_two)
    p_three = self._r_cost_three(ratios[2], self.a_three)
    p_two = (1-p_two)*p_one
    p_three = (1-p_three)*p_two
    #np.eye(3)[np.random.choice(2, p=p, size=(self.n_customers ))].mean(axis=0)
    cost_one = np.random.binomial(self.max_gain,p_one,self.n_networks).mean(axis=0)
    cost_two = np.random.binomial(self.max_gain,p_two,self.n_networks).mean(axis=0)
    cost_three = np.random.binomial(self.max_gain,p_three,self.n_networks).mean(axis=0)
    revenue = 0.1 + (p_one*cost_one+p_two*cost_two + p_three*cost_three)/(p_one + p_two + p_three)
    #revenue = (fares[1:] * price).sum()
    
    return [cost_one,cost_two, cost_three], revenue, False, None


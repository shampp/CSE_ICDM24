import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statistics import fmean
import logging

class FareUpsellEnv(object):

  def __init__(self, randomize=True):
    """ 
    Create a Fare Upsell Revenue Simulator. 
    Args:
      randomize (bool): if False, customer behavior is sampled from a fixed 
        model; else the model is resampled at reset() from a hyper-prior.
    """
    self.randomize = randomize
    # number of customers whose behavior is sampled in each step
    self.n_customers = 25


  def reset(self):
    # sample (latent) customer behavior from hyper-prior
    if self.randomize:
      self.a_high = np.random.normal(loc=-0.15, scale=0.07)
      self.c_high = np.random.normal(loc=0, scale=8.)
      self.a_medium = np.random.normal(loc=-0.15, scale=0.07)
      self.c_medium = np.random.normal(loc=0, scale=8.)
    else:
      #self.a_medium, self.a_high = -0.12, -0.08
      #self.c_medium, self.c_high = -0.5, 2.
      self.a_medium, self.a_high = -0.05, -0.05
      self.c_medium, self.c_high = -0.5, 2

    self.x = np.linspace(10,50,41)
    self.y = np.linspace(30,70,41)
    self.price_medium, self.price_high = np.meshgrid(self.x, self.y)
    p_high = self._p_price(self.y, self.a_high, self.c_high)
    p_medium = self._p_price(self.x, self.a_medium, self.c_medium)
    p_medium_joint, p_high_joint = np.meshgrid(p_medium, p_high)
    p_medium_joint = (1-p_high_joint) * p_medium_joint
    self.revenue = p_high_joint * self.price_high + p_medium_joint * self.price_medium

    return None

  @staticmethod
  def _p_price(price, a, c):
      return 1/(1+np.exp(-a*price-c))

  def show_with_elimination(self, cc, prices=None, elm_prices=None, states=None, solution=None):
    fig, ax = plt.subplots(1, 1, figsize=(6,7))
    ax.set_xlim([9.5,50.5])
    ax.set_ylim([29.5,70.5])
    
    def mark_eliminations(axis, observations):
        axis.scatter(x=np.array(observations)[:,0], y=np.array(observations)[:,1], marker='X', c='c', s=4**2, label='eliminated')
        axis.legend()

    def mark_observations(axis, observations):
        axis.scatter(x=np.array(observations)[:,0], y=np.array(observations)[:,1], marker='+', c='k', s=9**2, label='observations')
        axis.legend()

    def mark_optimal_price(axis, label='optimal price setting'):
        y_,x_ = np.unravel_index(np.argmax(self.revenue, axis=None),self.revenue.shape)
        logging.info("optimal arm is {},{} and revenue is: {}".format(self.x[x_], self.y[y_],self.revenue[y_,x_]))
        axis.scatter(x=self.x[x_], y=self.y[y_], marker='+', c='r', s=9**2, label=label)
        axis.plot(axis.get_xlim(), [y_, y_], 'r:', alpha=0.5)
        axis.plot([x_, x_], axis.get_ylim(), 'r:', alpha=0.5)
        axis.legend()

    def contour(axis, surface, cmap='inferno'):
        contours = axis.contourf(self.price_medium, self.price_high, surface, levels=50, cmap=cmap)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(contours, cax=cax)
        axis.set_xlabel('standard delivery')
        axis.set_ylabel('express delivery')

    def show_revenue(axis, observed_prices=None, eliminated_prices=None, optimal_price=True):
        contour(axis, self.revenue)
        if observed_prices is not None:
          mark_observations(axis, observed_prices)
        if eliminated_prices is not None:
          mark_eliminations(axis,eliminated_prices)
        if optimal_price:
          mark_optimal_price(axis)
        axis.set_title('true expected revenue')

    #axis = ax[2]
    show_revenue(ax, observed_prices=prices, eliminated_prices = elm_prices)

    if solution is not None:
      ax.scatter(solution[0],solution[1], marker='o', s=10**2, c='r',label='optimizer solution')
      ax.legend()
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.05,1,0.2),fancybox=True,shadow=True,ncol=4,fontsize=7,mode="expand", borderaxespad=0)
    fname = 'elm_fig_%s.pdf' %(cc)
    fig.savefig(fname)
    plt.clf()
    plt.close('all')

  def show(self, prices=None, states=None, solution=None):
    fig, ax = plt.subplots(1, 3, figsize=(18, 7))

    x = np.linspace(10,50,41)
    y = np.linspace(30,70,41)
    p_high = self._p_price(x, self.a_high, self.c_high)
    p_medium = self._p_price(y, self.a_medium, self.c_medium)

    
    axis = ax[0]
    axis.plot(p_medium, 'r', label='True behavior')
    if prices is not None:
      axis.scatter(prices[:,0], states[:,1]/(1-self._p_price(prices[:,1], self.a_high, self.c_high)), marker='x', c='k', label='Observations')
    axis.legend()
    axis.set_ylabel('probability of purchase')
    axis.set_xlabel('MEDIUM price')
    axis.set_title('MEDIUM customer sensitivity')

    axis = ax[1]
    axis.plot(p_high, 'r', label='True behavior')
    if prices is not None:
      axis.scatter(prices[:,1], states[:,2], marker='x', c='k', label='Observations')
    axis.legend()
    axis.set_ylabel('probability of purchase')
    axis.set_xlabel('HIGH price')
    axis.set_title('HIGH customer sensitivity')

    def mark_observations(axis, observations):
        axis.scatter(x=np.array(observations)[:,0], y=np.array(observations)[:,1], marker='+', c='k', s=9**2, label='observations')
        axis.legend()

    def mark_optimal_price(axis, label='optimal price setting'):
        y_,x_ = np.unravel_index(np.argmax(self.revenue, axis=None), self.revenue.shape)
        #y_,x_ = np.unravel_index(np.argsort(np.abs(self.revenue - fmean(self.revenue.flatten())).flatten())[:2],self.revenue.shape)
        #y_,x_ = np.unravel_index(np.argpartition(np.abs(self.revenue - fmean(self.revenue.flatten())).flatten(),2)[:1],self.revenue.shape)
        axis.scatter(x=x_, y=y_, marker='+', c='r', s=9**2, label=label)
        axis.plot(axis.get_xlim(), [y_, y_], 'r:', alpha=0.5)
        axis.plot([x_, x_], axis.get_ylim(), 'r:', alpha=0.5)
        axis.legend()

    def contour(axis, surface, cmap='inferno'):
        contours = axis.contourf(self.price_medium, self.price_high, surface, levels=50, cmap=cmap)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(contours, cax=cax)
        axis.set_xlabel('MEDIUM price')
        axis.set_ylabel('HIGH price')

    def show_revenue(axis, observed_prices=None, optimal_price=True):
        contour(axis, self.revenue)
        if observed_prices is not None:
          mark_observations(axis, observed_prices)
        if optimal_price:
          mark_optimal_price(axis)
        axis.set_title('True expected revenue per passenger')

    axis = ax[2]
    show_revenue(axis, observed_prices=prices)

    if solution is not None:
      #print("solution: {}".format(solution))
      ax[2].scatter(np.array(solution[0]), np.array(solution[1]), marker='o', s=10**2, c='r', label='Optimizer Solution')
      ax[2].legend()
    fname = 'gp_fig.pdf'
    fig.savefig(fname)
    plt.clf()
    plt.close('all')

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
    price = action

    # probability of customer buying HIGH at delta HIGH
    p_high = self._p_price(price[1], self.a_high, self.c_high) 
    # probability of customer buying MEDIUM at delta MEDIUM
    p_medium = (1-p_high) * self._p_price(price[0], self.a_medium, self.c_medium)
    p_low = 1 - p_high - p_medium
    p = np.array([p_low, p_medium, p_high]).reshape(-1)
    #print("p:", *p, sep='-')
    # one-hot encoding of selected fares
    fares = np.eye(3)[np.random.choice(3, p=p, size=(self.n_customers ))].mean(axis=0)
    revenue = (fares[1:] * price).sum()
    
    return fares, revenue, False, None


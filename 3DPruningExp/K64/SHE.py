#@title Sequential Block Elimination
from itertools import product
import numpy as np
from numpy.random import Generator, PCG64
from collections import defaultdict
import itertools
from operator import itemgetter
import numpy.lib.recfunctions as rf
#from math import log
from pathlib import Path
import logging
from statistics import mean

class SHE(object):
    def __init__(self, n_rounds, bounds):
        self.n_rounds = n_rounds
        self.rg = Generator(PCG64(12345))
        self.cnt1 = 16
        self.cnt2 = 16
        self.cnt3 = 16

        
        self.block_arr = []

        ar1 = np.linspace(bounds[0,0], bounds[0,1], self.cnt1, dtype='float16')
        ar2 = np.linspace(bounds[1,0], bounds[1,1], self.cnt2, dtype='float16')
        ar3 = np.linspace(bounds[2,0], bounds[2,1], self.cnt3, dtype='float16')
        self.ln1 = ar1.shape[0]
        self.ln2 = ar2.shape[0]
        self.ln3 = ar3.shape[0]
        no_actions = self.cnt1*self.cnt2*self.cnt3

        self.full_actions = np.array([*product(ar1,ar2,ar3)]).reshape(self.cnt1*self.cnt2*self.cnt3,3) # set of actions tuples
        self.incl_actions = np.ones(no_actions,dtype=bool)

        self.actions_mean = np.zeros(no_actions, dtype='float32')
        self.means = np.zeros(no_actions, dtype='float32')


        self.revenues = defaultdict(list)
        self.reset()

        log_file = Path('./logs/%s.log' %('she'))  #logging as SuccessiveBlockElimination (sbe)
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        self.log = logging.getLogger('she')
        logging.info("Running Successive Block Elimination algorithm")

        self.M = no_actions # no of blocks
        self.n_r = [4, 4, 4, 4, 2, 2, 2, 2]
        self.n_r_ind = 0
        self.n_r_cnt = 0
        self.round_count = 0

    def reset(self):
        self.max = None # maximum value observed thus far
        self.argmax = None # argument at which maximum value was observed

    def act(self,t):
        pl_arm = None
        pl_arm_ind = None
        incl_indx = np.nonzero(self.incl_actions)[0]
        if (self.n_r_ind == len(self.n_r)):
            self.n_r_ind -= 1
        pl_cnt = self.n_r[self.n_r_ind]
        if (self.round_count == self.M):
            self.round_count = 0    #reset round count
            self.n_r_cnt += 1
        if (self.n_r_cnt == self.n_r[self.n_r_ind]):
            logging.info("running action elimination algorithm")
            self.eliminate_least_rewarding_half()   #this function should update active indices and M
            self.n_r_ind += 1
            self.M = len(incl_indx)

        pl_arm_ind = incl_indx[self.round_count]
        pl_arm = self.full_actions[pl_arm_ind]
        self.round_count+=1
        return list(pl_arm), pl_arm_ind

    def eliminate_least_rewarding_half(self):
        incl_indx = np.nonzero(self.incl_actions)[0]
        logging.info("==== Eliminating the worst half ====")
        if (len(incl_indx) == 1):
            logging.info("only one arm remaining... returning")
            return 0
        median = np.median(self.means[incl_indx])
        logging.info("Median Revenue:{}".format(median))
        eliminated_inds = incl_indx[self.means[incl_indx]<= median]
        logging.info("Eliminated actions are: " +','.join('{}'.format(el) for el in zip(*eliminated_inds)))
        self.incl_actions[eliminated_inds] = False

        logging.info("Number of active arms: {}".format(self.incl_actions.sum()))


    def update(self, state, revenue, ind):
        incl_ind = np.nonzero(self.incl_actions)[0]


        self.revenues[ind].append(revenue)

        arm_mean = mean(self.revenues[ind])
        self.actions_mean[ind] = arm_mean

        self.means[ind] = arm_mean

        reward = 0

        max_lin_ind = np.argmax(self.actions_mean[incl_ind])
        max_ind = incl_ind[max_lin_ind]
        self.argmax = self.full_actions[max_ind].tolist()
        self.max = [self.actions_mean[max_ind]]
        logging.info("optimizer solution:{}, corresponding revenue:{}, index:{}".format(self.argmax, self.max, max_ind))
        return reward

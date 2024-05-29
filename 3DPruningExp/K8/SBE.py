#@title Sequential Block Elimination
from itertools import product
import numpy as np
from numpy.random import Generator, PCG64, beta
from collections import defaultdict
import itertools
from operator import itemgetter
import numpy.lib.recfunctions as rf
from math import log,sqrt,ceil
from pathlib import Path
import logging
from statistics import mean,fmean,stdev,variance
from sklearn.cluster import KMeans

class SBE(object):
    def __init__(self, n_rounds, bounds):
        self.n_rounds = n_rounds
        self.rg = Generator(PCG64(12345))
        self.cnt1 = 16
        self.cnt2 = 16
        self.cnt3 = 16
        self.init_clus_sz = 8
        self.M = self.init_clus_sz
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
        self.incl_clusters = np.ones(self.init_clus_sz,dtype=bool)
        self.actions_mean = np.zeros(no_actions, dtype='float32')
        self.actions_var = np.zeros(no_actions, dtype='float32')
        self.actions_wards = np.zeros(no_actions, dtype='uint16')

        self.revenues = defaultdict(list)
        self.reset()

        log_file = Path('./logs/%s.log' %('sbe'))  #logging as SuccessiveBlockElimination (sbe)
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        self.log = logging.getLogger('sbe')
        logging.info("Running Successive Block Elimination algorithm")

        self.blocks = self.get_block()
        incl_indx = np.nonzero(self.incl_actions)[0]
        for i in np.arange(self.init_clus_sz):
            i_block_indcs =  incl_indx[self.blocks.labels_ == i]
            self.block_arr.append(i_block_indcs)
        self.n_r = [6] + [4]*(self.M-1)
        self.n_r_cnt = 0
        self.play_cnt = self.n_r[self.n_r_cnt]*self.M


    def reset(self):
        self.max = None # maximum value observed thus far
        self.argmax = None # argument at which maximum value was observed

    def get_block(self):
        logging.info("Block the actions into {} clusters".format(self.init_clus_sz))
        return KMeans(n_clusters=self.init_clus_sz,n_init=10,random_state=0).fit(self.full_actions)

    def act(self,t):
        pl_arm = None
        pl_arm_ind = None
        if (t > self.play_cnt):
            self.n_r_cnt = self.n_r_cnt+1 if len(self.n_r)-1 > self.n_r_cnt else len(self.n_r)
            if self.n_r_cnt != len(self.n_r):  #this means we are not yet done with block elimination
                self.eliminate_least_rewarding_block()
                incl_clus_no = np.nonzero(self.incl_clusters)[0]
                actv_star_indcs = [self.rg.choice(self.block_arr[i]) for i in incl_clus_no]
                self.play_cnt += self.M*self.n_r[self.n_r_cnt]
                star_ind = t%self.M
                pl_arm_ind = actv_star_indcs[star_ind]
                pl_arm = self.full_actions[pl_arm_ind]
            else:   #this means we are done with block elimination and now we need to start exploring the individual arms
                incl_indx = np.nonzero(self.incl_actions)[0]
                max_ind = np.argmax(self.actions_mean[incl_indx])
                pl_arm_ind = incl_indx[max_ind]
                pl_arm = self.full_actions[pl_arm_ind]
        else:
            incl_clus_no = np.nonzero(self.incl_clusters)[0]
            actv_star_indcs = [self.rg.choice(self.block_arr[i]) for i in incl_clus_no]
            star_ind = t%self.M
            pl_arm_ind = actv_star_indcs[star_ind]
            pl_arm = self.full_actions[pl_arm_ind]

        return list(pl_arm), pl_arm_ind

    def eliminate_least_rewarding_block(self):
        incl_clus_no = np.nonzero(self.incl_clusters)[0]
        actv_star_indcs = [self.rg.choice(self.block_arr[i]) for i in incl_clus_no]
        min_ind = np.argmin(self.actions_mean[actv_star_indcs])
        self.incl_clusters[incl_clus_no[min_ind]] = False
        self.incl_actions[self.block_arr[incl_clus_no[min_ind]]] = False
        self.M -= 1

    def update(self, state, revenue, ind):
        incl_ind = np.nonzero(self.incl_actions)[0]

        self.revenues[ind].append(revenue)

        arm_mean = mean(self.revenues[ind])
        self.actions_mean[ind] = arm_mean

        reward = 0

        max_lin_ind = np.argmax(self.actions_mean[incl_ind])
        max_ind = incl_ind[max_lin_ind]
        self.argmax = self.full_actions[max_ind].tolist()
        self.max = [self.actions_mean[max_ind]]
        logging.info("optimizer solution:{}, corresponding revenue:{}, index:{}".format(self.argmax, self.max, max_ind))
        return reward

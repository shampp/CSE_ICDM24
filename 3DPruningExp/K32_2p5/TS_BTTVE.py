#@title Recursive Block Elimination with variance dependent arm selection
from itertools import product
import numpy as np
from numpy.random import Generator, PCG64, beta, lognormal, choice
from collections import defaultdict
import itertools
from operator import itemgetter
import numpy.lib.recfunctions as rf
from math import log, sqrt, floor
from pathlib import Path
import logging
from statistics import mean,fmean,stdev, variance
from scipy.stats import sem
from sklearn.cluster import KMeans

class TS_BTTVE(object):
    def __init__(self, bounds):
        self.rg = Generator(PCG64(12345))
        self.cnt1 = 16
        self.cnt2 = 16
        self.cnt3 = 16
        self.no_clus = 32
        self.round = 0

        self.block_array = []
        #self.block_mean = []
        #self.block_var = []
        self.block_pmean = []
        self.block_pvar = []

        #self.block_switch = 1
        #self.block_inc = 1

        ar1 = np.linspace(bounds[0,0], bounds[0,1], self.cnt1, dtype='float16')
        ar2 = np.linspace(bounds[1,0], bounds[1,1], self.cnt2, dtype='float16')
        ar3 = np.linspace(bounds[2,0], bounds[2,1], self.cnt3, dtype='float16')
        self.ln1 = ar1.shape[0]
        self.ln2 = ar2.shape[0]
        self.ln3 = ar3.shape[0]
        no_actions = self.cnt1*self.cnt2*self.cnt3

        self.gamma = 2
        self.p_mean = np.log(2.8)
        self.p_var = 0.5

        self.alpha = 1
        self.beta = 1
        self.n0 = 1
        self.mu0 = 1

        self.full_actions = np.array([*product(ar1,ar2,ar3)]).reshape(self.cnt1*self.cnt2*self.cnt3,3) # set of actions tuples
        self.incl_actions = np.ones(no_actions,dtype=bool)
        self.incl_clusters = np.ones(self.no_clus,dtype=bool)

        self.actions_mean = np.zeros(no_actions, dtype='float32') #to store means of underlying normal distribution
        self.actions_pmean = np.zeros(no_actions, dtype='float32') #to store the log of the means

        self.actions_pvar = np.zeros(no_actions, dtype='float32') #to store the log of sample variance *(n-1))
        self.actions_sem = np.zeros(no_actions, dtype='float32') # to sore standard error
        self.actions_wards = np.zeros(no_actions, dtype='uint16') # number of times each action played

        self.pmeans = np.ones(no_actions, dtype='float32')*self.p_mean
        self.pvars = np.ones(no_actions, dtype='float32')*self.p_var

        self.cpmeans = np.ones(self.no_clus, dtype='float32')*self.p_mean
        self.cpvars = np.ones(self.no_clus, dtype='float32')*self.p_var
        self.cluster_wards = np.zeros(self.no_clus, dtype='uint16') # number of times each action played

        self.revenues = defaultdict(list)

        self.reset()

        log_file = Path('./logs/%s.log' %('bttve'))  #logging as SuccessiveBlockElimination (sbe)
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        self.log = logging.getLogger('bttve')
        logging.info("Running Successive Block Elimination algorithm")
        self.blocks = self.create_clusters(self.full_actions[self.incl_actions])
        for i in range(self.no_clus):
            logging.info("Cluster: {}, Member Indices: {}".format(i, np.where(self.blocks.labels_ == i)[0]))
            logging.info("Cluster: {}, Member Actions: {}".format(i, self.full_actions[np.where(self.blocks.labels_==i)[0]]))

    def create_clusters(self, arr):
        incl_indx = np.nonzero(self.incl_actions)[0]
        self.no_clus = min(self.no_clus, len(incl_indx))
        logging.info("Cluster the actions into {} clusters".format(self.no_clus))
        return KMeans(n_clusters=self.no_clus, n_init=10, random_state=0).fit(arr)

    def reset(self):
        self.max = None # maximum value observed thus far
        self.argmax = None # argument at which maximum value was observed

    def act(self,t):
        self.round = t

        incl_cl_indx = np.nonzero(self.incl_clusters)[0]

        self.pmeans[:] = self.p_mean
        self.cpmeans[:] = self.p_mean
        self.cpvars[:] = self.p_var
        self.pvars[:] = self.p_var
        if self.block_array:    #contains indexes of actions in each block
            for i,j in enumerate(incl_cl_indx):
                self.cpmeans[j] = self.block_pmean[i]
                self.cpvars[j] = self.block_pvar[i]

        self.pmeans[np.nonzero(self.actions_pmean)] = self.actions_pmean[np.nonzero(self.actions_pmean)]
        self.pvars[np.nonzero(self.actions_pvar)] = self.actions_pvar[np.nonzero(self.actions_pvar)]

        elm_ratio = ((~self.incl_actions).sum())/((self.incl_actions).sum())
        incl_indx = np.nonzero(self.incl_actions)[0]

        
        if (elm_ratio < 7): #we sample first from cluster and then sample from the members
            prec = self.rg.gamma(self.alpha + self.cluster_wards/2, self.beta + 0.5*self.cpvars + (0.5*(self.cluster_wards*self.n0)*np.square(self.cpmeans-self.mu0))/(self.cluster_wards+self.n0))
            p_var = np.sqrt(1/prec)
            p_mean = self.rg.normal((self.cluster_wards*self.cpmeans + self.mu0)/(self.cluster_wards+1),(self.cluster_wards+1)*prec )
            logging.info("Included Clusters: {}".format(incl_cl_indx))
            samples = lognormal(p_mean[incl_cl_indx], p_var[incl_cl_indx])
            cl_pl_arm = incl_cl_indx[choice(np.flatnonzero(samples == np.max(samples)))]
            logging.info("Chosen Cluster is: {}".format(cl_pl_arm))
            self.cluster_wards[cl_pl_arm] +=1

            sel_cl_actions = np.where(self.blocks.labels_ == cl_pl_arm)[0]
            logging.info("Actions in the chosen cluster: {}".format(sel_cl_actions))

            prec = self.rg.gamma(self.alpha + self.actions_wards/2, self.beta + 0.5*self.pvars + (0.5*(self.actions_wards*self.n0)*np.square(self.pmeans-self.mu0))/(self.actions_wards+self.n0))
            p_var = np.sqrt(1/prec)
            p_mean = self.rg.normal((self.actions_wards*self.pmeans + self.mu0)/(self.actions_wards+1),(self.actions_wards+1)*prec)
            samples = lognormal(p_mean[sel_cl_actions], p_var[sel_cl_actions])
            pl_arm_ind = sel_cl_actions[choice(samples.argsort()[::-1][:2])]
            pl_arm = self.full_actions[pl_arm_ind]
        else: #we sample from the remaining actions
            prec = self.rg.gamma(self.alpha + self.actions_wards/2, self.beta + 0.5*self.pvars + (0.5*(self.actions_wards*self.n0)*np.square(self.pmeans-self.mu0))/(self.actions_wards+self.n0))
            p_var = np.sqrt(1/prec)
            p_mean = self.rg.normal( (self.actions_wards*self.pmeans + self.mu0)/(self.actions_wards+1), (self.actions_wards+1)*prec )
            samples = lognormal(p_mean[incl_indx], p_var[incl_indx])
            pl_arm_ind = incl_indx[choice(samples.argsort()[::-1][:2])]
            pl_arm = self.full_actions[pl_arm_ind]

        return list(pl_arm), pl_arm_ind


    def eliminate_arms(self):
        logging.info("==== Runnung elimination algorithms ====")
        self.block_array = []
        #self.block_mean = []
        #self.block_var = []
        self.block_pmean = []
        self.block_pvar = []

        block_revenues = []
        incl_indx = np.nonzero(self.incl_actions)[0]
        def update_bounds_get_indcs():    #update the bounds of the blocks
            incl_cl_indx = np.nonzero(self.incl_clusters)[0]
            for i in incl_cl_indx:  #cluster index and cluster labels are same
                clust_memb_indcs = np.where(self.blocks.labels_ == i)[0]
                logging.info("cluster: {} corresponding actions: {}".format(i, clust_memb_indcs))
                self.block_array.append(clust_memb_indcs)
                vals = [ y for x in clust_memb_indcs for y in self.revenues[x] ]
                if not vals:
                    vals = [np.exp(self.p_mean)]
                if len(vals) >= 4:
                  block_revenues.append([min(vals),max(vals),fmean(vals),fmean(vals),sem(vals),variance(vals),vals])
                else:
                  if len(vals) < 2:
                    block_revenues.append([0,1000,0,fmean(vals),0,0,vals])
                  else:
                    block_revenues.append([0,1000,0,fmean(vals),0,variance(vals),vals])

            for i in range(len(block_revenues)):
                nln = len(block_revenues[i][6])
                self.block_pmean.append(fmean(np.log(block_revenues[i][6])))
                if nln >=2:
                    #self.block_pmean.append(fmean(np.log(block_revenues[i][6])))
                    self.block_pvar.append((nln-1)*variance(np.log(block_revenues[i][6])))
                else:
                    self.block_pvar.append(self.p_var)
                    #self.block_pmean.append(fmean(np.log(block_revenues[i][6]))) if block_revenues[i][6][0] != 0 else self.block_pmean.append(block_revenues[i][6][0])

                #self.block_mean.append(block_revenues[i][3])
                #self.block_var.append(block_revenues[i][5])

            unexpl_block_indcs = [i for i,x in enumerate(block_revenues) if x[1] == 1000]
            logging.info("not sufficiently enough explored block indices are: {}".format(unexpl_block_indcs))
            LBs = [ x[2] - self.gamma*x[4] for x in block_revenues ]
            logging.info("Lower bounds are: {}".format(LBs))
            max_LB_ind = np.argmax(LBs)
            logging.info("Max lower bound is:{}".format(block_revenues[max_LB_ind][2]-self.gamma*block_revenues[max_LB_ind][4]))
            for i in unexpl_block_indcs:
              block_revenues[i][2] = block_revenues[max_LB_ind][2]
              block_revenues[i][4] = block_revenues[max_LB_ind][4]

            excl_inds = [i for i,x in enumerate(block_revenues) if (x[2]+self.gamma * x[4] < LBs[max_LB_ind]) ]
            self.incl_clusters[excl_inds] = False
            for i in excl_inds:
                self.incl_actions[self.block_array[i]] = False

            #elm_ratio = ((~self.incl_actions).sum())/((self.incl_actions).sum())
            #logging.info("elimination ratio is: {}".format(elm_ratio))
            #if ( elm_ratio < 7):
            #    logging.info("Enough blocks are not removed.. continue blocking")
            #    self.block_switch = 1
            #    self.block_inc = floor(elm_ratio)+1
            #else:
            #    logging.info("Enough blocks are removed .. Continuing with individual arm selection")
            #    self.block_switch = 0



        elm_ratio = ((~self.incl_actions).sum())/((self.incl_actions).sum())
        if (elm_ratio < 7):
            logging.info("Removing blocks ...")
            update_bounds_get_indcs() #update lower and upper bounds
        else:
            logging.info("Running individual arm elimination")
            arm_revenues = []
            for ind in incl_indx:
                vals = self.revenues[ind]
                if len(vals) >= 3:
                    arm_revenues.append([min(vals), max(vals), fmean(vals), sem(vals), vals])
                else:
                    arm_revenues.append([0, 1000, 0, 0, vals])

            unexpl_arm_indcs = [i for i,x in enumerate(arm_revenues) if x[2] == 0]
            logging.info("Unexplored actions are: {}".format(incl_indx[unexpl_arm_indcs]))
            LBs = [ arm_revenues[i][2]- self.gamma*arm_revenues[i][3] for i,x in enumerate(arm_revenues) ]
            logging.info("Lower bounds are: {}".format(LBs))
            max_LB_ind = np.argmax(LBs)
            logging.info("Max lower bound is: {}".format(arm_revenues[max_LB_ind][2]-self.gamma*arm_revenues[max_LB_ind][3]))
            for i in unexpl_arm_indcs:
              arm_revenues[i][2] = arm_revenues[max_LB_ind][2]
              arm_revenues[i][3] = arm_revenues[max_LB_ind][3]

            excl_inds = [i for i,x  in enumerate(arm_revenues) if ( x[2]+self.gamma * x[3] < LBs[max_LB_ind]) ]
            for i in excl_inds:
                self.incl_actions[incl_indx[i]] = False

        if self.incl_actions.sum() == 0:
            logging.info("All actions are eliminated. Check the code")
            exit(-1)

        logging.info("Number of active actions: {}".format(self.incl_actions.sum()))

        return 0

    def update(self, state, revenue, ind):
        incl_indx = np.nonzero(self.incl_actions)[0]
        elm_rnd = 2.5*self.no_clus

        self.revenues[ind].append(revenue)

        #self.p_mean = fmean(np.log(list(itertools.chain(*self.revenues.values()))))
        #if self.round > 1:
        #    self.p_var = variance(np.log(list(itertools.chain(*self.revenues.values()))))
            #arm_sem = sem(list(itertools.chain(*self.revenues.values())))

        arm_sem = 7 #we keep this high to 
        arm_var = 2
        arm_pvar = self.p_var
        nln = len(self.revenues[ind])
        if nln > 1:
            arm_sem = sem(self.revenues[ind])
            arm_var = variance(self.revenues[ind])
            arm_pvar = (nln-1)*variance(np.log(self.revenues[ind]))

        arm_mean = fmean(self.revenues[ind])
        arm_pmean = fmean(np.log(self.revenues[ind]))
        self.actions_mean[ind] = arm_mean
        self.actions_sem[ind] = arm_sem

        self.actions_pmean[ind] = arm_pmean
        self.actions_pvar[ind] = arm_pvar

        self.actions_wards[ind] += 1
        reward = 0
        #weighted_sum = np.dot(state_weights,state)

        logging.info("round:{}, arm:{}, state:{}, revenues:{}, meanr:{}, sem:{}, ind:{}, count:{}".format(self.round, self.full_actions[ind].tolist(), state, revenue, arm_mean, arm_sem, ind, self.actions_wards[ind]))
        #logging.info("sample mean:{} sample var:{}".format(self.p_mean,self.p_var))

        #logging.info("current bounds: {}".format((self.actions_mean-self.gamma*self.actions_sem).flatten()[self.flt_inds_actv_actions]))
        maxx_lin_ind = (self.actions_mean - self.gamma*self.actions_sem)[incl_indx].argmax()
        maxx_ind = incl_indx[maxx_lin_ind]
        self.argmax = self.full_actions[maxx_ind].tolist()
        self.max = [self.actions_mean[maxx_ind]]
        bnd = [self.actions_mean[maxx_ind] - self.gamma*self.actions_sem[maxx_ind]]
        logging.info("optimizer solution:{}, revenue:{}, bound:{}, index:{}, linear index:{}, count:{}".format(self.argmax, self.max, bnd, maxx_ind, maxx_lin_ind, self.actions_wards[maxx_ind]))

        if not (self.round%elm_rnd):
            self.eliminate_arms()
        return reward

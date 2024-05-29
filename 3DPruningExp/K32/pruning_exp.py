#@title In action
from PruneRatioEnv import *
from itertools import product
from scipy.spatial.distance import euclidean
from TS_BTVE import TS_BTVE
from TS_BTHE import TS_BTHE
from TS_BTTVE import TS_BTTVE
from SHE import SHE
from SBE import SBE
from VBR import VBR
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statistics import mean,stdev

def regret_calculation(seq_err):
    t = len(seq_err)
    regret = [x / y for x, y in zip(seq_err, range(1, t + 1))]
    return regret

def plot_simp_regret(n_rounds,regret):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-v0_8-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {'BESE':'b', 'BEHI':'g', 'BESE-TT':'r', 'SBE':'c', 'VBR':'m', 'SH':'k'}
        sty = {'BESE':'-', 'BEHI':'-', 'BESE-TT':'-', 'SBE':'-', 'VBR':'-', 'SH':'-'}
        labels = {'BESE':'TBE', 'BESE-TT':'TBE (TT)','BEHI':'BEHI', 'SBE':'SBE','VBR':'VBR', 'SH':'SH'}
        regret_file = 'simple_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for bandit in regret.keys():
                val = bandit+','+','.join([str(e) for e in regret[bandit]])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), regret[bandit], c=col[bandit], ls=sty[bandit], label=labels[bandit])
                ax.set_xlabel('rounds',fontsize=20)
                ax.set_ylabel('per-round simple regret',fontsize=20)
                ax.legend(loc='best', fancybox=True, shadow=True,ncol=1,fontsize=20)
        fig.savefig('per_round_simple_regret.pdf', format='pdf')
        f = plt.figure()
        f.clear()
        plt.close(f)


def plot_miss_prob(miss_prob,std_err):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-v0_8-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {'BESE':'b', 'BEHI':'g', 'BESE-TT':'r', 'SBE':'c', 'VBR':'m', 'SH':'k'}
        sty = {'BESE':'-', 'BEHI':'-', 'BESE-TT':'-', 'SBE':'-', 'VBR':'-', 'SH':'-'}
        #labels = {'BESE':'BESE', 'BEHI':'BEHI', 'BESE-TT':'BESE (TT)', 'SBE':'SBE','VBR':'VBR', 'SH':'SH'}
        labels = {'BESE':'TBE', 'BESE-TT':'TBE (TT)','BEHI':'BEHI', 'SBE':'SBE','VBR':'VBR', 'SH':'SH'}
        keys = ['TBE', 'TBE (TT)', 'SBE']
        #ax.bar(list(miss_prob.keys()), list(miss_prob.values()))
        ax.bar(keys, list(miss_prob.values()))
        ax.errorbar(keys, list(miss_prob.values()),std_err.values(),fmt='.',color='Red',elinewidth=2,capthick=2,errorevery=1, alpha=0.5, ms=4, capsize = 2)
        #ax.errorbar(list(miss_prob.keys()),list(miss_prob.values()),std_err.values(),fmt='.',color='Red',elinewidth=2,capthick=2,errorevery=1, alpha=0.5, ms=4, capsize = 2)
        ax.legend(fontsize=25)
        fig.savefig('err_prob.pdf', format='pdf')
        f = plt.figure()
        f.clear()
        plt.close(f)

def plot_cum_regret(n_rounds,regret):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-v0_8-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {'BESE':'b', 'BEHI':'g', 'BESE-TT':'r', 'SBE':'c', 'VBR':'m', 'SH':'k'}
        sty = {'BESE':'-', 'BEHI':'-', 'BESE-TT':'-', 'SBE':'-', 'VBR':'-', 'SH':'-'}
        labels = {'BESE':'BESE', 'BEHI':'BEHI', 'BESE-TT':'BESE (TT)', 'SBE':'SBE','VBR':'VBR', 'SH':'SH'}
        regret_file = 'cumulative_regret.txt'
        with open(regret_file, "w") as regret_fd:
            for bandit in regret.keys():
                val = bandit+','+','.join([str(e) for e in regret[bandit]])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), regret[bandit], c=col[bandit], ls=sty[bandit], label=labels[bandit])
                ax.set_xlabel('rounds')
                ax.set_ylabel('per-round cumulative regret')
                ax.legend(loc='best',bbox_to_anchor=(0.5,1.05),fancybox=True,shadow=True,ncol=2, fontsize=25)
        fig.savefig('per_round_cumulative_regret.pdf', format='pdf')
        f = plt.figure()
        f.clear()
        plt.close(f)


def main():
    #np.random.seed(0)
    env = PruneRatioEnv(randomize=False)
    env.reset()
    bandits = ['BESE', 'BESE-TT', 'BEHI', 'SBE']
    #bandits = ['BESE', 'BESE-TT', 'SBE', 'SH', 'VBR']
    #bandits = ['SBE']
    bandits = ['BESE', 'BESE-TT', 'SBE']
    seq_err = dict()
    smp_err = dict()
    miss_prob = dict()
    pr_regret = dict()
    sm_regret = dict()
    ms_prob = dict()
    std_err = dict()
    n_rounds = 1001
    no_t = 50
    for bandit in bandits:
        seq_err[bandit] = dict()
        smp_err[bandit] = dict()
        miss_prob[bandit] = dict()
        for s in range(no_t):
            seq_err[bandit][s] = list()
            smp_err[bandit][s] = list()
            #miss_prob[bandit][s] = list()

            seq_err[bandit][s].append(1)
            smp_err[bandit][s].append(1)

            np.random.seed(s)
            opt = None
            if bandit == 'BESE':
                opt = TS_BTVE(bounds=np.array([[0.01,0.99],[0.01,0.99],[0.01,0.99]]))
            if bandit == 'BEHI':
                opt = TS_BTHE(bounds=np.array([[0.01,0.99],[0.01,0.99],[0.01,0.99]]))
            if bandit == 'BESE-TT':
                opt = TS_BTTVE(bounds=np.array([[0.01,0.99],[0.01,0.99],[0.01,0.99]]))
            if bandit == 'SBE':
                opt = SBE(n_rounds=n_rounds,bounds=np.array([[0.01,0.99],[0.01,0.99],[0.01,0.99]]))
            if bandit == 'VBR':
                opt = VBR(n_rounds=n_rounds,bounds=np.array([[0.01,0.99],[0.01,0.99]]))
            if bandit == 'SH':
                opt = SHE(n_rounds=n_rounds,bounds=np.array([[0.01,0.99],[0.01,0.99], [0.01,0.99]]))


            args = [] # prices set by the optimizer
            vals = [] # step reward (mean revenue per customer)
            states = [] # states (customer behavior information auxiliary to revenue, NOTE: states do not impact future behavior)
            maxvals = [] # best reward observed thus far by the random search optimizer
            elm_prices = list()
            miss_cnt = 0

            pre_rnd = 100
            opt_revs = 7.53
            epsl = 1
            for i in range(1,n_rounds):
                action, ind = opt.act(i)
                state, revenue, _, _ = env.step(action)
                opt.update(state, revenue, ind)    #we have a reward of 1

                if ((opt_revs-epsl <= revenue) and (revenue <= opt_revs+epsl)):
                    if (i > 0):
                        seq_err[bandit][s].append(seq_err[bandit][s][i-1])
                else:
                    seq_err[bandit][s].append(1) if (i==1) else seq_err[bandit][s].append(seq_err[bandit][s][i-1]+1.0)

                if ((opt_revs-1.5*epsl <= opt.max[0]) and (opt.max[0] <= opt_revs+1.5*epsl)):
                    if (i > 0):
                        smp_err[bandit][s].append(smp_err[bandit][s][i-1])
                else:
                    miss_cnt += 1
                    smp_err[bandit][s].append(1) if (i==1) else smp_err[bandit][s].append(smp_err[bandit][s][i-1]+1.0)

                args.append(action)
                vals.append(revenue)
                states.append(state)
                maxvals.append(opt.max)

                '''if not (i%pre_rnd):
                    elm = set([x for x in elm_prices if x])
                    if len(elm) != 0:
                        env.show_with_elimination('%s-%d' %(bandit,i),ratios=np.array(args), elm_ratios = list(elm),states=np.array(states), solution=opt.argmax)
                '''
            miss_prob[bandit][s] = miss_cnt/n_rounds

    
        pr_regret[bandit] = regret_calculation(list(map(mean,zip(*seq_err[bandit].values()))))
        sm_regret[bandit] = regret_calculation(list(map(mean,zip(*smp_err[bandit].values()))))
        ms_prob[bandit] =  mean(miss_prob[bandit].values())
        std_err[bandit] = 0.3*stdev(miss_prob[bandit].values())
        print("Bandit: {}, mip:{} std_err:{}".format(bandit,ms_prob[bandit],std_err[bandit]))
        #plt.plot(vals, label='step reward')
        #plt.plot(maxvals, label='max reward')
        #plt.legend()
        handlers = opt.log.handlers[:]
        for handler in handlers:
            handler.close()
            self.log.removeHandler(handler)


    plot_cum_regret(n_rounds,pr_regret)
    plot_simp_regret(n_rounds,sm_regret)
    plot_miss_prob(ms_prob,std_err)

if __name__ == '__main__':
    main()

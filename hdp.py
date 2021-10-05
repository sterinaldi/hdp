import numpy as np
import os
import matplotlib.pyplot as plt
import json
from matplotlib import rcParams
from scipy.stats import dirichlet
from scipy.special import logsumexp
from numpy.random import uniform, choice
from collections import Counter

rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=15
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6


class HDP:
    '''
    Class to analyse a set of mass posterior samples and reconstruct the mass distribution using a Hierarchical Dirichlet Process.
    
    Arguments:
        :iterable events:       list of single-event posterior samples
        :int N_bins:            number of bins for the DP. The bigger, the better.
        :int N_mc:              number of Monte Carlo samples for M_i integral
        :int N_draws:           number of mass function draws
        :float m_min:           lower bound of mass prior
        :float m_max:           upper bound of mass prior
        :float alpha0:          prior for mass function concentration parameter
        :float gamma0:          prior for single-event concentration parameter
        :function injected_mf:  python function with simulated density
        
    '''
    def __init__(self, events,
                       N_bins,
                       name = '',
                       N_draws = 1e4,
                       m_min = None,
                       m_max = None,
                       alpha0 = 1.,
                       gamma0 = 1.,
                       injected_mf = None,
                       out_folder = './',
                       ):
        
        self.events = events
        self.name   = name
        self.N_samps_per_ev = np.array([len(ev) for ev in self.events])
        self.N_evs = len(events)
        self.log_N_comb = np.sum([np.log(gamma0 + len(ev)) for ev in self.events])
        self.N_draws = int(N_draws)
        self.N_bins = int(N_bins)
        if m_min is not None:
            self.m_min = m_min
        else:
            self.m_min = np.min([np.min(a) for a in self.events])
        if m_max is not None:
            self.m_max = m_max
        else:
            self.m_max = np.max([np.max(a) for a in self.events])
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        self.injected_mf = injected_mf
        self.bins = np.linspace(self.m_min, self.m_max, self.N_bins+1)
        self.binwidth = self.bins[1]-self.bins[0]
        self.out_folder = out_folder
    
    def count(self):
        counts = np.zeros(self.N_bins)
        for ev in self.events:
            c       = Counter(np.digitize(ev, self.bins))
            heights = np.array([c[i] for i in range(1, self.N_bins+1)]) + self.gamma0/self.N_bins
            counts += heights/(self.gamma0 + len(ev))
        self.counts = counts
        return
    
    def draw_samples(self):
        self.draws = dirichlet(self.alpha0/self.N_bins + self.counts).rvs(self.N_draws)
        return
    
    def save_samples(self):
        with open(self.out_folder + '/mf_samples_{0}.json'.format(self.name), 'w') as f:
            json.dump([list(self.bins[:-1]), [list(d) for d in self.draws]], f)
        return
    
    def compute_stats(self):
        self.percentiles = [50, 5, 16, 84, 95]
        p = {}
        for perc in self.percentiles:
            p[perc] = np.percentile(self.draws.T/self.binwidth, perc, axis = 1)
        for perc in self.percentiles:
            p[perc] = p[perc]/(np.sum(p[50])*self.binwidth)
            
        names = ['m'] + [str(perc) for perc in self.percentiles]
        np.savetxt(self.out_folder + '/rec_prob_{0}.txt'.format(self.name), np.array([self.bins[:-1], p[5], p[16], p[50], p[84], p[95]]).T, header = ' '.join(names))
        self.p = p
        return
        
        
    
    def plot_samples(self):
        fig, ax = plt.subplots()
        
        ax.bar(x = self.bins[:-1], height = self.p[95] - self.p[5], bottom = self.p[5], width = np.diff(self.bins), align = 'edge', lw = 0, color = 'mediumturquoise', alpha = 0.5)
        ax.bar(x = self.bins[:-1], height = self.p[84] - self.p[16], bottom = self.p[16], width = np.diff(self.bins), align = 'edge', lw = 0, color = 'darkturquoise', alpha = 0.5)
        ax.hist(self.bins[:-1], bins = self.bins, weights = self.p[50], histtype = 'step', color = 'steelblue', label = r"\textsc{Reconstructed}", zorder = 100,)
        ax.set_xlabel('$M\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        
        if self.injected_mf is not None:
            mass = np.linspace(self.m_min, self.m_max, 1000)
            inj_prob = np.array([self.injected_mf(mi) for mi in mass])
            inj_prob = inj_prob/(np.sum(inj_prob)*(mass[1] - mass[0]))
            ax.plot(mass, inj_prob, lw = 0.3, color = 'r', label = r"\textsc{Injected}")
        
        ax.grid(True,dashes=(1,3))
        ax.legend(loc=0,frameon=False,fontsize=10)
        fig.savefig(self.out_folder + '/mass_function_{0}.pdf'.format(self.name), bbox_inches = 'tight')
        ax.set_yscale('log')
        fig.savefig(self.out_folder + '/log_mass_function_{0}.pdf'.format(self.name), bbox_inches = 'tight')
        return
        
    def run(self):
        print('Counting...')
        self.count()
        print('Drawing...')
        self.draw_samples()
        print('Saving...')
        self.save_samples()
        print('Plotting...')
        self.compute_stats()
        self.plot_samples()
        return

if __name__ == '__main__':
    
    from mf import injected_density
    
    event_files = ['./events/'+f for f in os.listdir('./events/') if not f.startswith('.')]
    events      = []
    for event in event_files:
        events.append(np.genfromtxt(event))
    
    sampler = HDP(events, 200, injected_mf = injected_density, N_draws = 1e4)
    sampler.run()

import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
from hdp import HDP
from collections import Counter
import deepdish as dd
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15
import astropy.units as u
from scipy.interpolate import interp1d

colours = ["purple", "#DE8F05", "#029E73", "#D55E00"]

labels = ["\\textsc{Truncated}", "\\textsc{Power-law Peak}", "\\textsc{Broken Power-law}", "\\textsc{Multi Peak}"]
filenames = [
    "/Users/stefanorinaldi/Documents/mass_inference/GWTC/o1o2o3_mass_b_iid_mag_two_comp_iid_tilt_powerlaw_redshift_mass_data.h5",
    "/Users/stefanorinaldi/Documents/mass_inference/GWTC/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_mass_data.h5",
    "/Users/stefanorinaldi/Documents/mass_inference/GWTC/o1o2o3_mass_d_iid_mag_two_comp_iid_tilt_powerlaw_redshift_mass_data.h5",
    "/Users/stefanorinaldi/Documents/mass_inference/GWTC/o1o2o3_mass_e_iid_mag_two_comp_iid_tilt_powerlaw_redshift_mass_data.h5",
    ]

R0 = 20.0 # per Gpc^3 per yr

alpha = 1.8 # primary mass pl slope
beta = 0 # secondary mass pl slope
gamma = 2.7 # redshift slope

mmin = 5 # Solar masses
mmax = 41 # Solar masses

m1_norm = (1-alpha)/(mmax**(1-alpha) - mmin**(1-alpha))
#m2_norm = (1+beta)/(m1**(1+beta) - mmin**(1+beta))

def log_dNdm1dm2dzdsz2(m1, m2, z, s1z, s2z):
    
    log_pm1 = -alpha*np.log(m1) + np.log(m1_norm)
    log_pm2 = beta*np.log(m2) + np.log(m2_norm)
    
    # Here we convert from dN/dVdt in the comoving frame to dN/dzdt in the detector frame:
    #
    # dN/dzdt = dN/dVdt * dV/dz * 1/(1+z)
    #
    # The first factor is the Jacobian between comoving volume and redshift
    # (commonly called the "comoving volume element"), and the second factor accounts for
    # time dilation in the source frame relative to the detector frame.
    log_dNdV = np.log(R0) + gamma*np.log1p(z)
    log_dVdz = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value)
    log_time_dilation = -np.log1p(z)
    log_dNdz = log_dNdV + log_dVdz + log_time_dilation
    
    log_p_sz = np.log(0.25) # 1/2 for each spin dimension
    
    return np.where((m2 < m1) & (mmin < m2) & (m1 < mmax), log_pm1 + log_pm2 + log_p_sz + log_dNdz, np.NINF)

def selection_function(file, bins):
#    with h5py.File(sensitivity_file, 'r') as f:
#        Tobs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
#        Ndraw = f.attrs['total_generated']
#        m1             = np.array(f['injections']['mass1_source'])
#        m2             = np.array(f['injections']['mass2_source'])
#        gstlal_ifar    = np.array(f['injections']['ifar_gstlal'])
#        pycbc_bbh_ifar = np.array(f['injections']['ifar_pycbc_bbh'])
#        p_draw         = np.array(f['injections']['sampling_pdf'])
#        z              = np.array(f['injections']['redshift'])
#        s1z            = np.array(f['injections']['spin1z'])
#        s2z            = np.array(f['injections']['spin2z'])
##        m1_det         = m1[np.where(ifar > 1.)]
##        m2_det         = m2[np.where(ifar > 1.)]
#    one_year = 1.0 # IFAR measured in years
#    detection_selector = (gstlal_ifar > one_year) | (pycbc_bbh_ifar > one_year)
##    log_dN = np.where(detection_selector, log_dNdm1dm2dzdsz2(m1, m2, z, s1z, s2z), np.NINF)
##    log_mu = np.log(Tobs) + (log_dN - np.log(p_draw)) - np.log(Ndraw)
##    mu = np.exp(log_mu)
#    m1_det        = m1[np.where(detection_selector)]
#    m2_det        = m2[np.where(detection_selector)]
#    counts_m1     = Counter(np.digitize(m1, bins))
#    counts_m1_det = Counter(np.digitize(m1_det, bins))
#    n_m1          = np.array([counts_m1[i] for i in range(1,len(bins))])*np.exp(-alpha*np.log(bins[:-1]) + np.log(m1_norm))
#    n_m1_obs      = np.array([ci if ci > 0 else 1 for ci in [counts_m1_det[i] for i in range(1,len(bins))]])
#    selfunc_m1    = n_m1_obs/n_m1
#
#    counts_m2     = Counter(np.digitize(m2, bins))
#    counts_m2_det = Counter(np.digitize(m2_det, bins))
#    n_m2          = np.array([counts_m2[i] for i in range(1,len(bins))])*np.exp(beta*np.log(bins[:-1]) + np.log(m2_norm))
#    n_m2_obs      = np.array([ci if ci > 0 else 1 for ci in [counts_m2_det[i] for i in range(1,len(bins))]])
#    selfunc_m2    = n_m2_obs/n_m2

    selfunc_m1 = np.genfromtxt('/Users/stefanorinaldi/Documents/mass_inference/GWTC/seleff.txt', names = True)
    selfunc_m2 = np.genfromtxt('/Users/stefanorinaldi/Documents/mass_inference/GWTCm2/seleff.txt', names = True)
    
    sf_m1      = interp1d(selfunc_m1['m1'], selfunc_m1['pdet'], bounds_error = False, fill_value = (selfunc_m1['pdet'][0], selfunc_m1['pdet'][-1]))
    sf_m2      = interp1d(selfunc_m2['m1'], selfunc_m2['pdet'], bounds_error = False, fill_value = (selfunc_m2['pdet'][0], selfunc_m2['pdet'][-1]))
    
    fig, ax = plt.subplots()
    ax.plot(bins[:-1], sf_m1(bins[:-1]), marker = '', color = 'r', label = r"$S_1$")
    ax.plot(bins[:-1], sf_m2(bins[:-1]), marker = '', color = 'b', label = r"$S_2$")
    ax.set_xlabel('$M\ [M_\\odot]$')
    ax.set_ylabel('$S(M)$')
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    ax.set_yscale('log')
    fig.savefig('./O3a/selection_functions.pdf', bbox_inches = 'tight')
    
    return sf_m1(bins[:-1]), sf_m2(bins[:-1])


def compute_stats(samples, bins, binwidth):
    percentiles = [50, 5, 16, 84, 95]
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(samples.T/binwidth, perc, axis = 1)
    for perc in percentiles:
        p[perc] = (p[perc]).flatten()
    norm = np.sum(p[50]*binwidth)
    for perc in percentiles:
        p[perc] = p[perc]/norm
    
    
    names = ['m'] + [str(perc) for perc in percentiles]
    np.savetxt('./O3a/rec_prob.txt', np.array([bins[:-1], p[5], p[16], p[50], p[84], p[95]]).T, header = ' '.join(names))
    return p
    
    

def plot_samples(p, bins):

    fig, ax = plt.subplots()
    ax.bar(x = bins[:-1], height = p[95] - p[5], bottom = p[5], width = np.diff(bins), align = 'edge', lw = 0, color = 'mediumturquoise', alpha = 0.5)
    ax.bar(x = bins[:-1], height = p[84] - p[16], bottom = p[16], width = np.diff(bins), align = 'edge', lw = 0, color = 'darkturquoise', alpha = 0.5)
    
############
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    dmass_1 = mass_1[1]-mass_1[0]
    for i, ff in enumerate(filenames):
        f = dd.io.load(ff)
        ppd = f["ppd"]
        lines = f["lines"]
        mass_1_ppd = np.trapz(ppd, mass_ratio, axis=0)
        mass_ratio_ppd = np.trapz(ppd, mass_1, axis=-1)
        mean = mass_1_ppd/(np.sum(mass_1_ppd)*dmass_1)
        ax.plot(mass_1, mean, color=colours[i], label=labels[i], lw = 0.5)
############

    ax.hist(bins[:-1], bins = bins, weights = p[50], histtype = 'step', color = 'steelblue', label = r"\textsc{HDP}", zorder = 100,)
    ax.set_xlabel('$M\ [M_\\odot]$')
    ax.set_ylabel('$p(M)$')
    ax.grid(True,dashes=(1,3))
    ax.legend(loc=0,frameon=False,fontsize=10)
    ax.set_ylim(top = 0.2)
    fig.savefig('./O3a/m_astro.pdf', bbox_inches = 'tight')
    ax.set_yscale('log')
    fig.savefig('./O3a/log_m_astro.pdf', bbox_inches = 'tight')
    return


if __name__ == '__main__':

    m_min  = 2
    m_max  = 100
    N_bins = 200

    out_dir = './O3a/'

    dir_m1 = '/Users/stefanorinaldi/Documents/mass_inference/GWTC/events/'
    dir_m2 = '/Users/stefanorinaldi/Documents/mass_inference/GWTCm2/events/'
    dirs = [dir_m1, dir_m2]

    bins = np.linspace(m_min, m_max, N_bins+1)
    binwidth = bins[1] - bins[0]
    
    sensitivity_file = '/Users/stefanorinaldi/Documents/mass_inference/GWTC/sensitivity_estimate.hdf5'

    selfunc_m1, selfunc_m2 = selection_function(sensitivity_file, bins)
    
    for dir, name in zip(dirs, ['m1', 'm2']):
        print(name)
        event_files = [os.path.join(dir, f) for f in os.listdir(dir) if not f.startswith('.')]
        events      = []
        for event in event_files:
            events.append(np.genfromtxt(event))
        
        sampler = HDP(events = events, N_bins = N_bins, name = name, out_folder = out_dir, m_min = m_min, m_max = m_max)
        sampler.run()
    
    with open('./O3a/mf_samples_m1.json', 'r') as f:
        m1_obs = json.load(f)[1:]
    with open('./O3a/mf_samples_m2.json', 'r') as f:
        m2_obs = json.load(f)[1:]
    
    m1_astro = m1_obs/selfunc_m1
    m1_astro = m1_astro/(np.sum(m1_astro)*binwidth)
    m2_astro = m2_obs/selfunc_m2
    m2_astro = m2_astro/(np.sum(m2_astro)*binwidth)
    
    m_astro = (m1_astro + m2_astro)/2.
    m_astro = np.array([f/(np.sum(f)*binwidth) for f in m_astro])
    
#    with open('./O3a/astro_samples.json', 'w') as f:
#        json.dump([list(bins[:-1]), [list(d) for d in m_astro]], f)
        
    p = compute_stats(m_astro, bins, binwidth)
    plot_samples(p, bins)


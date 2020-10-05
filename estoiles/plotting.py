'''
This contains all of the functions needed for plotting.
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import astropy.units as u
import scipy.optimize
from datetime import datetime

import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn

def savefig(fig, figpath, writepdf=False, dpi=450):
    fig.savefig(figpath, dpi=dpi, bbox_inches='tight')
    print('{}: made {}'.format(datetime.now().isoformat(), figpath))

    if writepdf:
        pdffigpath = figpath.replace('.png','.pdf')
        fig.savefig(pdffigpath, bbox_inches='tight', rasterized=True, dpi=dpi)
        print('{}: made {}'.format(datetime.now().isoformat(), pdffigpath))

    plt.close('all')

def paper_plot():
    sns.set_context("paper")
    sns.set_style('ticks')
    sns.set_palette('colorblind')
    plt.rc('font', family='serif', serif='cm10')
    figparams = {
            'text.latex.preamble': [r'\usepackage{amsmath}',
            r'\bf'],
            'text.usetex':True,
            'axes.labelsize':20.,
            'xtick.labelsize':16,
            'ytick.labelsize':16,
            'figure.figsize':[6., 4.],
            'font.family':'DejaVu Sans',
            'legend.fontsize':12}
    plt.rcParams.update(figparams)
    cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_emcee(flatchain, outpath):
    plt.close('all')
    paper_plot()
    up95 = np.percentile(flatchain,68)
    fig = plt.figure()
    plt.hist(flatchain[:, 0], 50, color="k", histtype="step", density=True);
    plt.axvline(up95)
    plt.xlabel(r'$\log_{10} M_s$');
    plt.ylabel(r'$p(M_s)$');
    savefig(fig, outpath, writepdf=0, dpi=100)

def plot_chains(samples, outpath):
    plt.close('all')
    paper_plot()
    fig = plt.figure()
    plt.plot(samples[:,:,0])
    plt.ylabel(r'$\log_{10} M_s$');
    plt.xlabel(r'$N$');
    savefig(fig, outpath, writepdf=0, dpi=100)

def plot_logprob(samples, outpath):
    plt.close('all')
    paper_plot()
    fig = plt.figure()
    plt.plot(np.abs(samples[100:]))
    plt.yscale('log')
    plt.ylabel(r'$\mid \log \mathcal{L} \mid$');
    plt.xlabel(r'$N$');
    savefig(fig, outpath, writepdf=0, dpi=100)

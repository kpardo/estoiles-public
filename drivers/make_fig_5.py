import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import astropy.units as u
import astropy.constants as const
import scipy
from astropy.coordinates import SkyCoord
import healpy as hp
import matplotlib.patches as patches

import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn
from estoiles.plotting import savefig, paper_plot
from estoiles.convenience import make_file_path
from estoiles.paths import *
import os

paper_plot()
fname = DATADIR+'forfigs/allfreqsig.npy'
if not os.path.exists(fname):
    import drivers.make_allfreqsig

sig1 = np.load(DATADIR+'forfigs/allfreqsig.npy',allow_pickle=True)[:,1]
GM = np.load(PNINTERPDIR+'h1PN_Grid.npy',allow_pickle=True)
LMC = GM[:,0,0,0]

fmin = 7.7e-8/u.s
fmax = 5.6e-4/u.s
logfmin = np.log10(fmin.value)
logfmax = np.log10(fmax.value)
freqArray=10.**np.linspace(logfmin,logfmax,15)/u.s
grid_max = GM[-1,-1,-1,:].reshape([3,1])
logMs_max = grid_max[0]-.6*(-3.)
fmax_pf = (const.c**3/(6**1.5*np.pi*const.G)).to(u.kg/u.s)

def logqmin(LMCarray,f_ref):
    qmin = np.empty([])
    for logmc in LMCarray:
        mc = 10.**logmc*u.Msun
        if ISCO_f(mc,1.) <= f_ref:
            qmin = np.append(qmin,0.)
        else:
            qfreq = scipy.optimize.fsolve(calc_fmax,lo_logq(mc,f_ref),args=(f_ref,mc))
            qmin = np.append(qmin,qfreq[0])
    return qmin[1:]
def ISCO_f(Mc,q):
    Mc = Mc.to(1*u.kg)
    return fmax_pf/Mc*q**.6*(1+q)**.3
def calc_fmax(logq,f,Mc):
    Mc = Mc.to(1*u.kg)
    q = 10.**logq
    return abs(fmax_pf/Mc*q**.6*(1+q)**.3-f)
def lo_logq(Mc,freq):
    Mc = Mc.to(1*u.kg)
    return np.log10((Mc*freq/fmax_pf)**(5/3))

logMcmin = []
for fi in freqArray:
    qmin = logqmin(LMC,fi)
    b = np.where(qmin==0.)[0]
    if b.size:
        logMcmin.append(LMC[b[0]]-.6*(-3))
    else: logMcmin.append((grid_max[0]-.6*(-3))[0])

fig,ax = plt.subplots(figsize = (7,4.5))
x = freqArray.value
ax.set_xscale('log')

R = 0.1*u.Mpc
conv = -.6*np.log10(1*u.Mpc/R)
ax.fill_between(x,sig1+conv,logMs_max,color='gold',alpha = .2,label=r'$100~\rm{kpc}$')

R = 1*u.Mpc
conv = -.6*np.log10(1*u.Mpc/R)
ax.fill_between(x,sig1+conv,logMs_max,color='firebrick',alpha=.6,label=r'$1~\rm{Mpc}$')

R = 50*u.Mpc
conv = -.6*np.log10(1*u.Mpc/R)
ax.plot(x,sig1+conv-1.2,label=r'$\rm{Full~signal,}~50~\rm{Mpc}$',color='k',linewidth=2,zorder=1)

R = 10*u.Mpc
conv = -.6*np.log10(1*u.Mpc/R)
ax.plot(x,sig1+conv-0.6,label=r'$0.11~\rm{mas,}~10~\rm{Mpc}$',color='k',linewidth=2,linestyle='dashed',zorder=1)

ax.fill_between(x,np.array(logMcmin)-1.8,LMC[-1],facecolor='darkgray',zorder=2)
ax.text(7e-5,9,r'\bf{ISCO}',fontsize=20)
ax.legend(fontsize=14,facecolor='white',framealpha=1,loc='lower left')
ax.set_xlabel(r'Frequency [Hz]')
ax.set_yticks([6,7,8,9])
ax.set_yticklabels(['6','7','8','9'])
ax.set_ylim(6.2,LMC[-1])
ax.set_xlim(x.min(),x.max())
ax.set_ylabel(r'$\log_{10} \mathcal{M}_c~[M_{\odot}]$')
ax.tick_params(axis='both', which='major', labelsize=25)

outpath = make_file_path(FIGSDIR, {}, extra_string='fullsig',ext='.png')

savefig(fig, outpath, writepdf=True)

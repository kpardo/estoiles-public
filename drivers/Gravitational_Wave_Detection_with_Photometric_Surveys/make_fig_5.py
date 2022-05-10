import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy
from astropy.coordinates import SkyCoord
import matplotlib.patches as patches
import make_sig as ms

import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn
from estoiles.plotting import savefig, paper_plot
from estoiles.convenience import make_file_path
from estoiles.paths import *
import os

paper_plot()

logMs_max = 9.73928365-.6*(-3.)
freqArray = np.geomspace(7.7e-8,5.6e-4,15)
cls = np.array([68,95,99.7])
c = 3e8*u.m/u.s
G = 6.67e-11*u.m**3/u.kg/u.s**2
fmax_pf = c**3/(6**1.5*np.pi*G)

def logqmin(LMCarray,f_ref):
    qmin = np.empty([])
    for logmc in LMCarray:
        mc = 10.**logmc*u.Msun
        if ISCO_f(mc,1.) <= f_ref/u.s:
            qmin = np.append(qmin,0.)
        else:
            qfreq = scipy.optimize.fsolve(calc_fmax,lo_logq(mc,f_ref/u.s),args=(f_ref/u.s,mc))
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

fig,ax = plt.subplots(figsize = (7,4.5))
ax.set_xscale('log')

fname = DATADIR+'forfigs/sig.npy'
if not os.path.exists(fname):
    x = ms.make_sig(freqArray*u.Hz,cls,False,True)
    np.save(fname,x)
sig=np.load(fname,allow_pickle=True)[:,1]

R = 1*u.Mpc
conv = -.6*np.log10(1*u.Mpc/R)
ax.fill_between(freqArray,sig+conv,logMs_max,color='gold',alpha=.2,label=r'$1~\rm{Mpc}$')

R = 10*u.Mpc
conv = -.6*np.log10(1*u.Mpc/R)
ax.fill_between(freqArray,sig+conv,logMs_max,color='firebrick',alpha = .6,label=r'$10~\rm{Mpc}$')

# with mean
fname = DATADIR+'forfigs/sigwm.npy'
if not os.path.exists(fname):
    x = ms.make_sig(freqArray*u.Hz,cls,False,False)
    np.save(fname,x)
sig=np.load(fname,allow_pickle=True)[:,1]
R = 1*u.Gpc
conv = -.6*np.log10(1*u.Mpc/R)
ax.plot(freqArray,sig+conv,label=r'$\rm{Full~signal,}~1~\rm{Gpc}$',color='k',linewidth=2,zorder=1)

# improved sigma
fname = DATADIR+'forfigs/sigsmsig.npy'
if not os.path.exists(fname):
    x = ms.make_sig(freqArray*u.Hz,cls,False,True,sigma=0.11)
    np.save(fname,x)
sig=np.load(fname,allow_pickle=True)[:,1]
R = 100*u.Mpc
conv = -.6*np.log10(1*u.Mpc/R)
ax.plot(freqArray,sig+conv,label=r'$0.11~\rm{mas,}~100~\rm{Mpc}$',color='k',linewidth=2,
        linestyle='dashed',zorder=1)

LMC = np.linspace(5.739283647798979,9.73928364779898,100)
logMcmin = []
for fi in freqArray:
    qmin = logqmin(LMC,fi)
    b = np.where(qmin==0.)[0]
    if b.size:
        logMcmin.append(LMC[b[0]]-.6*(-3))
    else: logMcmin.append(logMs_max)
ax.fill_between(freqArray,np.array(logMcmin)-1.8,LMC[-1],facecolor='darkgray',zorder=2)

ax.text(7e-5,9,r'\bf{ISCO}',fontsize=20)
ax.legend(fontsize=16,facecolor='white',framealpha=1,loc='lower left')
ax.set_xlabel(r'Frequency [Hz]')
ax.set_yticks([6,7,8,9])
ax.set_yticklabels(['6','7','8','9'])
ax.set_ylim(7,LMC[-1])
ax.set_xlim(freqArray[0],freqArray[-1])
ax.set_ylabel(r'$\log_{10} \mathcal{M}_c~[M_{\odot}]$')
ax.tick_params(axis='both', which='major', labelsize=25)

outpath = make_file_path(FIGSDIR, {}, extra_string='figure_5',ext='.png')

savefig(fig, outpath, writepdf=False)

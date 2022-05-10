import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy
from astropy.coordinates import SkyCoord
import matplotlib.patches as patches

import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn
import make_sig as ms
from estoiles.plotting import savefig, paper_plot
from estoiles.convenience import make_file_path
from estoiles.paths import *
import os

paper_plot()
c = 3e8*u.m/u.s
G = 6.67e-11*u.m**3/u.kg/u.s**2
freqs = np.array([7.7e-8,6.6e-6,5.6e-4])
freqArray = np.geomspace(7.7e-8,5.6e-4,15)
cls = np.array([68,95,99.7])
ind = [np.argmin(abs(freqArray-x)) for x in freqs]
colors = ['gold','orange','firebrick']
strings = [r'$f=7.7\times10^{-8}~\rm{Hz}$',r'$f=6.6\times10^{-6}~\rm{Hz}$',
           r'$f=5.6\times10^{-4}~\rm{Hz}$']

for fn in ['sig','sigwm']:
    fname = DATADIR+'forfigs/'+fn+'.npy'
    if os.path.exists(fname):
        print('file exists.')
        x = np.load(fname,allow_pickle=True)
    else:
        if fn == 'sig':
            x = ms.make_sig(freqArray*u.Hz,cls,False,True)
            np.save(DATADIR+'forfigs/'+fn+'.npy',x)
        elif fn == 'sigwm':
            x = ms.make_sig(freqArray*u.Hz,cls,False,False)
            np.save(DATADIR+'forfigs/'+fn+'.npy',x)
    if fn == 'sig':
        sig = x[ind]
    elif fn =='sigwm':
        sigwm = x[ind]

LMC = np.linspace(5.739283647798979,9.73928364779898,100)
fmax_pf = c**3/(6**1.5*np.pi*G)

fig,axs = plt.subplots(2,3,figsize=(11,7*1.1))

for k in range(2):
    if k == 0:
        z = sig
    else: z = sigwm
    for i in range(3):
        ax = axs[k,i]

        for j in range(3):
            y = (LMC-z[i,j])/.6
            ax.tick_params(axis='both', which='major', labelsize=25)
            ax.fill_between(LMC,y,-3,alpha = .2*(j+1),color = colors[j])
        if not i == 0:
            ax.set_yticklabels([])
        if not k == 1:
            ax.set_xticklabels([])
            ax.set_title(strings[i],fontsize=24,pad = 10)

        Mc_ISCO = (fmax_pf*1.5**.3/freqs[i]*u.s).to(1*u.Msun)
        ax.fill_betweenx([-3,2],np.log10(Mc_ISCO.value),LMC[-1],color='darkgray')

for ax in axs.flatten():
    ax.set_xticks([6,7,8,9])
    ax.set_yticks([-2,-1,0,1,2])
    ax.set_ylim(-3,2)
    ax.set_xlim(LMC.min(),LMC.max())
fig.tight_layout()
fig.text(0.5,0,r'$\log_{10} \mathcal{M}_c~[M_{\odot}]$',fontsize=24,va='top',ha='center')
fig.text(0,0.5,r'$\log_{10} D_L~[\rm Mpc]$',fontsize=24,rotation=90,va='center',ha='right')

axs[0,0].add_patch(patches.Rectangle((6,0.5),3.,1,facecolor='w',edgecolor='k'))
axs[0,0].text(6.15,1,r'No Mean Signal',fontsize=24,va='center')
axs[1,0].add_patch(patches.Rectangle((6,0.5),3.4,1,facecolor='w',edgecolor='k'))
axs[1,0].text(6.15,1,r'With Mean Signal',fontsize=24,va='center')

outpath = make_file_path(FIGSDIR, {}, extra_string='figure_4',ext='.png')

savefig(fig, outpath, writepdf=False)

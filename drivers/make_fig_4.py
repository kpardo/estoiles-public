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
import drivers.make_sig as ms
from estoiles.plotting import savefig, paper_plot
from estoiles.convenience import make_file_path
from estoiles.paths import *

paper_plot()
freq = 7.7e-8/u.s

CALCSIG = False
if CALCSIG:
    f = np.array([freq.to(1*u.Hz)])
    cls = np.array([68,95,99.7])
    sig1,sig2,sig3 = ms((f,cls,False,False))
else:
    sig1,sig2,sig3 = np.load(DATADIR+'forfigs/wm8e-08sig.npy',allow_pickle=True).flatten()

name = 'h1PN'
GM = np.load(PNINTERPDIR+'h1PN_Grid.npy',allow_pickle=True)
LMC = GM[:,0,0,0]
fmax_pf = (const.c**3/(6**1.5*np.pi*const.G)).to(u.kg/u.s)

fig,ax = plt.subplots(figsize=(7,7))
ax.tick_params(axis='both', which='major', labelsize=25)
y = (LMC-sig1)/.6
ax.fill_between(LMC,y,-3,alpha = .2,color = 'gold',label=r'$1\sigma$')
y = (LMC-sig2)/.6
ax.fill_between(LMC,y,-3,alpha = .4,color = 'orange',label=r'$2\sigma$')
y = (LMC-sig3)/.6
ax.fill_between(LMC,y,-3,alpha = .6,color = 'firebrick',label=r'$3\sigma$')
ax.set_ylim(-3,2)
ax.set_xlim(LMC.min(),LMC.max())

Mc_ISCO = (fmax_pf*1.5**.3/freq).to(1*u.Msun)
ax.fill_betweenx([-3,2],np.log10(Mc_ISCO.value),LMC[-1],color='darkgray')
size = 50
ax.add_patch(patches.Rectangle((6, 1),3.5,.8,facecolor='w',edgecolor='k',alpha=1))
ax.set_title(r'With Mean Signal',fontsize=40,pad = 20)
ax.text(6.2,1.2,r'$f=7.7\times10^{-8}~\rm{Hz}$',fontsize = 38)

outpath = make_file_path(FIGSDIR, {}, extra_string='mcr',ext='.png')

savefig(fig, outpath, writepdf=False)

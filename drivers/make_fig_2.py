import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
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
fname = DATADIR+'forfigs/dnmag.npy'
if not os.path.exists(fname):
    import make_dnmag

hm_mag = np.asarray(np.load(fname,allow_pickle=True))
conv = ((1*u.rad).to(1*u.uas)).value

xsize=500
ysize=int(xsize/2.)
longitude = np.radians(np.linspace(-180, 180, xsize))
latitude = np.radians(np.linspace(-90, 90, ysize))
theta = np.linspace(np.pi, 0, ysize)
phi   = np.linspace(-np.pi, np.pi, xsize)
PHI, THETA = np.meshgrid(phi, theta)
NSIDE = 48
grid_pix = hp.ang2pix(NSIDE, THETA, PHI)
grid_map = hm_mag[grid_pix]*conv

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection='mollweide')
image = ax.pcolormesh(longitude[::-1], latitude, grid_map, rasterized=True,zorder=0,cmap='OrRd')

ax.set_longitude_grid(60,)
ax.grid(color='darkgray',linewidth=1.4)

cb = fig.colorbar(image, orientation='horizontal', shrink = .6, pad=.05)
unit = r'Mean Deflection $[\mu as]$'
cb.ax.xaxis.set_label_text(unit)
cb.ax.xaxis.labelpad = 1
ax.tick_params(axis='y', labelsize=20)
plt.setp(ax.get_xticklabels(), visible=False)
ax.text(53*np.pi/180,3*np.pi/180,r'$\mathbf{60^\circ}$',fontsize=20,color='w')
ax.text(2*np.pi/180,3*np.pi/180,r'$\mathbf{0^\circ}$',fontsize=20,color='k')
ax.text(110*np.pi/180,3*np.pi/180,r'$\mathbf{120^\circ}$',fontsize=20,color='w')
ax.text(-67*np.pi/180,3*np.pi/180,r'-$\mathbf{60^\circ}$',fontsize=20,color='w')
ax.text(-130*np.pi/180,3*np.pi/180,r'-$\mathbf{120^\circ}$',fontsize=20,color='w')

outpath = make_file_path(FIGSDIR, {}, extra_string='figure_2',ext='.png')

savefig(fig, outpath, writepdf=False)

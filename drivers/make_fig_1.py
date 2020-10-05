import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import astropy.units as u
import scipy
from astropy.coordinates import SkyCoord
import healpy as hp
import matplotlib.patches as patches

import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn
import estoiles.Gaia_stars as GaiaStar
from estoiles.plotting import savefig, paper_plot
from estoiles.convenience import make_file_path
from estoiles.paths import *

paper_plot()

scoord = SkyCoord(l=90*u.deg, b=90*u.deg, frame='galactic')
telcoord = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')
g = gwc.GWcalc(10**9*u.Msun,1,1e-6/u.s,1.*u.Mpc,0.*u.deg,0.*u.deg,scoord,telcoord)
src = g.CoordTr(telcoord,scoord)
theta = (0*u.s,np.array([True,False,False,False,False]))
h_m = g.calc_h(theta,phi_=0.*u.rad)

Gaia_data = GaiaStar.GaiaStars(Nstars=200,minmag=0,maxmag=21,fov_l=telcoord.l,fov_b=telcoord.b)
llist = Gaia_data.l
blist = Gaia_data.b
stc = g.CoordTr(telcoord,SkyCoord(l=llist,b=blist,frame='galactic'))

mock_dn = cdn.dn(h_m, src, stc)
stcxy = stc[:2,:]
mean = np.mean(mock_dn[:2,:],axis=1).reshape([2,1])
MF1 = 1e8
md1 = (mock_dn[:2,:])*MF1
MF2 = 1.5e10
md2 = (mock_dn[:2,:]-mean)*MF2

fig = plt.figure(figsize=(10,5))
ax = fig.add_axes((0,0,1,1))
r2d = 180./np.pi
ax.quiver(stcxy[0,:]*r2d,stcxy[1,:]*r2d,md1[0,:]*r2d,md1[1,:]*r2d,angles='xy', scale_units='xy',scale=1)
ScaleBar = (.05)/MF1*u.deg
ax.set_aspect('equal')

ax.add_patch(patches.Rectangle((-.26, -.25),0.2,0.06,facecolor='w',edgecolor='k'))
ax.plot([-.245,-.195],[-.25+0.025,-.25+0.025],linewidth=2,color='k')

ScaleBar = ScaleBar.to(1*u.uas)
ax.text(-.18,-.25+0.018,'{:.2g}'.format(ScaleBar.value)+r'~$\mu as$',fontsize=20)

ax.set_xlim(-.31,.31)
ax.set_ylim(-.31,.31)
ax.add_patch(patches.Rectangle((-.26, .21),0.41,0.07,facecolor='w',edgecolor='k'))
ax.text(-.245,.23,r'With Mean Signal',fontsize=30)
ax.arrow(.26,-.26,.0,.08,color = 'firebrick',width=.006)
ax.arrow(.26,-.26,-.08,0,color = 'firebrick',width=.006)
ax.text(.245,-.14,r'\bf{N}',color='firebrick',fontsize=20)
ax.text(.11,-.27,r'\bf{E}',color='firebrick',fontsize=20)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)

outpath = make_file_path(FIGSDIR, {}, extra_string='FOV',ext='.png')
savefig(fig, outpath, writepdf=False)

# hemisphere dn orthographic projection
Gaia_data = GaiaStar.GaiaStars(Nstars=500,minmag=0,maxmag=9,fov_l=180*u.deg,fov_b=45*u.deg,b_span=90*u.deg,l_span=360*u.deg)
llist = Gaia_data.l
blist = Gaia_data.b
telcoord = SkyCoord(l = 90*u.deg, b=90*u.deg, frame='galactic')
stc = g.CoordTr(telcoord,SkyCoord(l=llist,b=blist,frame='galactic'))

scoord = SkyCoord(l=90*u.deg, b=90*u.deg, frame='galactic')
g = gwc.GWcalc(10**9*u.Msun,1,1e-6/u.s,1.*u.Mpc,0.*u.deg,0.*u.deg,scoord,telcoord)
src = g.CoordTr(telcoord,scoord)
theta = (0*u.s,np.array([True,False,False,False,False]))
h_m = g.calc_h(theta,phi_=0.*u.rad)
mock_dn_plus = cdn.dn(h_m, src, stc)
h_m = g.calc_h(theta,phi_=np.pi/4.*u.rad)
mock_dn_cross = cdn.dn(h_m, src, stc)

fig,ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
MF = 3.3e10
ax.quiver(stc[0,:],stc[1,:],mock_dn_plus[0,:].value*MF,mock_dn_plus[1,:].value*MF,color='k',pivot='middle',angles='xy', scale_units='xy', scale=1.)
ax.quiver(stc[0,:],stc[1,:],mock_dn_cross[0,:].value*MF,mock_dn_cross[1,:].value*MF,color='firebrick',alpha = 1,pivot='middle',angles='xy', scale_units='xy', scale=1.)

ax.plot([0.5,0.65],[-.97,-.97],linewidth=2,color='k')
ax.plot([0.5,0.65],[-.93,-.93],linewidth=2,color='firebrick')
ScaleBar = (.15)/MF*u.rad
ScaleBar = ScaleBar.to(1*u.uas)
ax.text(0.69,-.97,'{:.2f}'.format(ScaleBar.value)+r'~$\mu as$',fontsize=20)
ax.axis('off')

outpath = make_file_path(FIGSDIR, {}, extra_string='hemisphere_FOV',ext='.png')

savefig(fig, outpath, writepdf=False)

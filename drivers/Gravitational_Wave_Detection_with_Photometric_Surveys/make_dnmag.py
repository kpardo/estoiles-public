import numpy as np
import healpy as hp
import estoiles.calc_dn as cdn
from estoiles.gw_source import GWSource
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.spatial.transform import Rotation as R
from estoiles.paths import *
import os

fname = DATADIR+'forfigs/dnmag.npy'
if os.path.exists(fname):
    print('file exists.')
else:
    NSIDE = 48
    NPIX = hp.nside2npix(NSIDE)
    mag = np.empty([NPIX])
    a = np.degrees(hp.pix2ang(nside = NSIDE,ipix = np.arange(NPIX)))
    g = GWSource(freq = 1e-6*u.Hz,sourcecoord=SkyCoord(l=90*u.deg,b=0*u.deg,frame='galactic'))
    h = g.h
    srcoord = g.srcindet

    for i in range(NPIX):
        stc = SkyCoord(l=a[1,i]*u.deg, b=(90-a[0,i])*u.deg, frame='galactic').cartesian
        starcoord_car = np.array([stc.x.value,stc.y.value,stc.z.value])
        r = R.from_euler('ZY',[-g.telcoord.l.value,90-g.telcoord.b.value],degrees=True)
        starcoord = r.apply(starcoord_car.T,inverse=True).T
        dn = cdn.dn(h, srcoord, starcoord)
        mag[i] = np.linalg.norm(dn)
    np.save(fname,mag)

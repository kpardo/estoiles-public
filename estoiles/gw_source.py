'''
This class defines instances of supermassive blackhole binaries with the given
characteristics.
'''
from dataclasses import dataclass
import numpy as np
import math
import astropy.units as u
from astropy.coordinates import SkyCoord
import emcee
import corner
import datetime

import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn

@dataclass
class GWSource:
    freq: float
    Mc: float = 1.e8*u.Msun
    q: float = 1.
    dl: float = 1.*u.Mpc
    inc: float = 0.*u.deg
    psi: float = 0*u.deg
    phi: float = 0.*u.rad
    time: float = 0.*u.s
    telcoord: SkyCoord = SkyCoord(l=-90*u.deg,b=90*u.deg,frame='galactic')
    sourcecoord: SkyCoord = SkyCoord(l=0*u.deg,b=90*u.deg,frame='galactic')


    def __post_init__(self):
        self.logMs0 = np.log10((self.Mc/self.dl**.6).value)
        g = gwc.GWcalc(self.Mc, self.q, self.freq, self.dl, self.inc, self.psi,
                self.sourcecoord, self.telcoord)
        self.h = g.calc_h((self.time, np.array([True,False,False,False,False])),
              phi_=self.phi)
        self.srcindet = g.CoordTr(self.telcoord,self.sourcecoord)

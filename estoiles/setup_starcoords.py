'''
makes star coords
'''
from dataclasses import dataclass
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

import estoiles.gw_calc as gwc
# import estoiles.Gaia_stars as GaiaStars
from estoiles.paths import *

@dataclass
class StarCoords:
    nstars: float
    telcoord: SkyCoord = SkyCoord(l=-90*u.deg,b=90*u.deg,frame='galactic')
    USEGAIA: bool = False
    USEFOVLIST: bool = False
    USEHEALPY: bool = False

    def __post_init__(self):
        if self.USEGAIA:
            l,b,mag = self.load_gaia()
            gaiakwargs = {'l':l, 'b':b, 'mag':mag}
        else:
            gaiakwargs = None
        self.fov = self.load_fov()
        self.starcoords = self.get_starcoords()

    def load_gaia(self):
        Gaia_data = GaiaStars(fov_l = self.telcoord.l,fov_b = self.telcoord.b, Nstars = self.nstars)
        return Gaia_data.l,Gaia_data.b,Gaia_data.mag

    def load_fov(self):
        if self.USEFOVLIST:
            ##useful for making heatmaps
            if self.USEHEALPY:
                NSIDE = 8
                NPIX = hp.nside2npix(NSIDE)
                fov = np.degrees(hp.pix2ang(nside = NSIDE,ipix = np.arange(NPIX))) # theta,phi in deg
            else:
                NTHE = 20
                NPHI = 40
                THE = np.linspace(0.5,179.5,NTHE)
                PHI = np.linspace(-180,180,NPHI)
                THE,PHI = np.meshgrid(THE,PHI)
                fov =[]
                fov.append(THE.flatten())
                fov.append(PHI.flatten())
        else:
            ls = np.array([0.])
            bs = np.array([0.])
            fov = [90-bs,ls]
        return fov

    def get_lblist(self, nstars = None,USEGRID=False, GETGS = False,**kwargs):
        if nstars is None:
            nstars = self.nstars
        if self.USEGAIA:
            l, b, mag = kwargs['l'], kwargs['b'], kwargs['mag']
            if self.nstars <= len(l):
                ind = np.random.randint(0,GaiaN_max,size=nstars)
                return l[ind],b[ind]
            else: print('Max # of stars in Gaia catalog exceeded')
        elif GETGS:
            ns = np.sqrt(nstars).astype(int)
            if not (nstars-ns**2) == 0:
                raise ValueError('Bad number of stars')
            dl = .53*u.deg
            db = .53*u.deg
            ll = np.linspace(0,dl,ns,endpoint=False)-dl/2
            bl = np.linspace(0,db,ns,endpoint=False)-db/2
            ll,bl = np.meshgrid(ll,bl)
            ll += np.random.rand(ns,ns)*(dl/ns)
            bl += np.random.rand(ns,ns)*(db/ns)
            return ll.flatten(),bl.flatten()
        elif USEGRID:
            ## create regular grid of stars
            dl = .53*u.deg
            db = .53*u.deg
            ns = np.sqrt(nstars).astype(int)
            if not (nstars-ns**2) == 0:
                raise ValueError('Bad number of stars')
            ll = np.linspace(-dl/2,dl/2,ns)
            bl = np.linspace(-db/2,db/2,ns)
            ll,bl = np.meshgrid(ll,bl)
            return ll,bl
        else:
            ## use uniform random field of stars
            dl = .53*u.deg
            db = .53*u.deg
            np.random.seed(0)
            ll = -dl/2.+np.random.rand(nstars)*dl
            np.random.seed(10)
            bl = -db/2.+np.random.rand(nstars)*db
            return ll,bl

    def get_starcoords(self, USEFOVLIST=None):
        if USEFOVLIST is None:
            USEFOVLIST = self.USEFOVLIST
        if USEFOVLIST:
            nstars = 32
            lrel,brel = self.get_lblist(nstars=nstars, USEGRID=True)
            i = 0 ##external loop over fovs
            L = self.fov[1][i]*u.deg+lrel/np.sin(self.fov[0][i]*u.deg)
            B = 90*u.deg-(self.fov[0][i]*u.deg+brel)
            stcoord = SkyCoord(l=L,b=B,frame='galactic')
            stc = gwc.coordtransform(self.telcoord,stcoord)
        else:
            ## consider merging with method above
            lrel,brel = self.get_lblist()
            L = self.fov[1][0]*u.deg+lrel/np.sin(self.fov[0][0]*u.deg)
            B = 90*u.deg-(self.fov[0][0]*u.deg+brel)
            ogcoords = SkyCoord(l=L,b=B,frame='galactic')
            stc = gwc.coordtransform(self.telcoord, ogcoords)
        return stc

    def get_gs(self, nguide):
        l_GS,b_GS= self.get_lblist(nstars=nguide,GETGS=True)
        L = self.fov[1][0]*u.deg+l_GS/np.sin(self.fov[0][0]*u.deg)
        B = 90*u.deg-(self.fov[0][0]*u.deg+b_GS)
        stcoord = SkyCoord(l=L,b=B,frame='galactic')
        stc = gwc.coordtransform(self.telcoord,stcoord)
        return stc

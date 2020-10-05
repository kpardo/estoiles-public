'''
This class returns cataloged Gaia stars coordinates
'''

from dataclasses import dataclass
import numpy as np
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from estoiles.paths import *
import os

@dataclass
class GaiaStars():
    QUERYALL: bool = False
    Nstars: float = 1000
    minmag: float = 3
    maxmag: float = 21
    fov_l: float = 0*u.deg
    fov_b: float = 0*u.deg
    b_span: float = 0.53*u.deg
    l_span: float = 0.53*u.deg

    def __post_init__(self):
        self.TelCoord = SkyCoord(l=self.fov_l,b=self.fov_b,frame='galactic')
        self.minb = (self.fov_b-0.5*self.b_span).value
        self.maxb = (self.fov_b+0.5*self.b_span).value
        if (self.minb < -90) or (self.maxb > 90):
            print('Bad b coordinate.')
        else: self.get_coords()

    def get_coords(self):
        fname = DATADIR+'GAIA/GaiaStars_l'+'{:.0f}'.format(self.fov_l.value)+'b'+'{:.0f}'.format(self.fov_b.value)+'_mag'+str(self.minmag)+'_'+str(self.maxmag)+'.npy'
        if os.path.exists(fname):
            print('file exists.')
            r = np.load(fname,allow_pickle=True)
        else:
            print('file not found, querying now...')
            self.minl = (self.fov_l-0.5*self.l_span).value
            self.maxl = (self.fov_l+0.5*self.l_span).value

            if self.minl < 0:
                l_string = " AND (g.l BETWEEN "+str(360 + self.minl)+" AND 360 \
                             OR g.l BETWEEN 0 AND "+str(self.maxl)+")"
            elif self.maxl > 360:
                l_string = " AND (g.l BETWEEN "+str(self.minl)+" AND 360 \
                             OR g.l BETWEEN 0 AND "+str(self.maxl-360)+")"
            else:
                l_string = " AND g.l BETWEEN "+str(self.minl)+" AND "+str(self.maxl)

            job = Gaia.launch_job_async("SELECT l,b,phot_g_mean_mag \
                    FROM gaiadr2.gaia_source AS g  \
                    WHERE g.phot_g_mean_mag \
                    BETWEEN " + str(self.minmag) + " \
                    AND " + str(self.maxmag) + "\
                    AND g.b BETWEEN "+ str(self.minb) +" \
                    AND "+ str(self.maxb)+l_string, dump_to_file=False)
            r = job.get_results()
            if len(r)!=0:
                print('query finished, '+str(len(r))+' stars were found')
                np.save(fname,r)
            else:
                print('no qualified stars found.')
                return

        if not self.QUERYALL:
            if len(r)<= self.Nstars:
                print('requested star number exceeds catalog size')
            else:
                ind = np.random.randint(len(r),size=self.Nstars)
                r = r[ind]

        self.l = np.array(r['l'])*u.deg
        self.b = np.array(r['b'])*u.deg
        self.mag = np.array(r['phot_g_mean_mag'])
        self.Nstars = len(self.l)
        print('finished loading coordinates of '+str(self.Nstars)+' stars.')

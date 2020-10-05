import numpy as np
import astropy.units as u
from estoiles.paths import *
import drivers.make_sig as ms
import os

fname = DATADIR+'forfigs/allfreqsig.npy'
if os.path.exists(fname):
    print('file exists.')
else:
    fmin = 7.7e-8
    fmax = 5.6e-4
    freqArray = np.logspace(fmin,fmax,15)*u.Hz
    cls = np.array([68,95,99.7])
    sigs = ms([freqArray,cls,False,True])
    np.save(fname,sigs)

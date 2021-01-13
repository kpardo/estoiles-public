import numpy as np
import astropy.units as u
from estoiles.paths import *
import make_sig as ms
import os

freqArray = np.geomspace(7.7e-8,5.6e-4,15)*u.Hz
cls = np.array([68,95,99.7])

for fn in ['sig','sigwm','sigsmsig']:
    fname = DATADIR+'forfigs/'+fn+'.npy'
    if os.path.exists(fname):
        print(fn+' file exists.')
    else:
        if fn =='sig':
            sigs = ms.make_sig(freqArray,cls,False,True)
        elif fn == 'sigwm':
            sigs = ms.make_sig(freqArray,cls,False,False)
        elif fn == 'sigsmsig':
            sigs = ms.make_sig(freqArray,cls,False,False,sigma=0.11)
        np.save(fname,sigs)
        print(fn+' file generated.')

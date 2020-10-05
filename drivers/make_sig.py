'''
This module produce confidence levels at given frequency values
'''

import estoiles.sampler as p
import numpy as np

def make_sig(theta_):
    '''
    Input Params:
    theta_ -- freqs,cls,INJECT,MS
    freqs: an array of frequencies, with the unit of u.Hz
    cls: an array of percentile levels
    INJECT: boonlean for signal injection
    MS: boolean for mean subtraction

    Return:
    res - an array of confidence values, with shape [len(freqs),len(cls)]
    '''
    freqs,cls,INJECT,MS = theta_
    res = np.empty([len(freqs),len(cls)])
    for ifreq,freqval in enumerate(freqs):
        sam = p.Sampler(freq=freqval,OVERWRITE=False,INJECT=INJECT,MEANSUB=MS,nchains=32,ntune=100,nsamples=1000)
        samples = sam.sampler.get_chain(discard=sam.ntune,flat=True)
        res[ifreq] = np.percentile(samples,cls)
    return res

'''
This module runs the mcmc calculation.
'''

from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import math
import astropy.units as u
from astropy.coordinates import SkyCoord
import emcee
import corner
import datetime
from scipy.spatial.transform import Rotation as R

import estoiles.gw_calc as gwc
import estoiles.calc_dn as cdn
import estoiles.plotting as plot
from estoiles.convenience import make_file_path
from estoiles.setup_starcoords import StarCoords
from estoiles.gw_source import GWSource
from estoiles.paths import *

@dataclass
class Sampler():
    freq: float = 1.e-6*u.Hz
    nstars: int = 1000
    nsamples: int = 1000
    nchains: int = 8
    ntune: int = 1000
    ndims: int = 1
    minlogMs: float = 4.539283647798979
    maxlogMs: float = 11.539283647798978
    USEGS: bool = False
    MEANSUB: bool = False
    GAIA: bool = False
    OVERWRITE: bool = True
    SEED: int = 0
    INJECT: bool = False
    injectobj: Any = GWSource(freq=freq)
    nguide: int = 0

    def __post_init__(self):
        self.set_source(self.freq)
        self.calc_sigma()
        self.stars = self.load_stars()
        self.load_model(nguide=self.nguide)
        self.load_data()
        self.run_inference()
        if self.OVERWRITE:
            self.make_diagnostic_plots()

    def set_source(self, freq=None, Mc=1.e7*u.Msun):
        if freq is None:
            freq = self.freq
        self.src = GWSource(freq=freq, Mc=Mc)
        return self.src

    def calc_sigma(self):
        ws = (1.1*u.mas).to(u.rad)/np.sqrt(1.e8/self.nstars)
        T_obs = (6.*72.*u.d).to(1.*u.s)
        Ncycle = T_obs.to(1.*u.s)*(self.freq.to(1./u.s))
        wfirst_sig = (ws/np.sqrt(Ncycle)).value
        self.wfirst_sig = wfirst_sig
        return wfirst_sig

    def load_stars(self):
        s = StarCoords(nstars=self.nstars, USEGAIA=self.GAIA)
        return s

    def load_model(self, nguide=4):
        mock_dn = cdn.dn(self.src.h, self.src.srcindet, self.stars.starcoords)

        if self.MEANSUB:
            if self.USEGS:
                gs = self.stars.get_gs(nguide)
                mockGS = cdn.dn(self.src.h, self.src.srcindet, gs)
                means = np.mean(mockGS,axis=1).reshape([3,1])
            else:
                means = np.mean(mock_dn,axis=1).reshape([3,1])
            mock_dn = mock_dn.value - means.value

        self.model = mock_dn
        return mock_dn

    def load_data(self):
        if self.INJECT == False:
            np.random.seed(0)
            noise_w = np.random.normal(0, self.wfirst_sig, (2,self.nstars))
            noise_w = np.concatenate([np.zeros([1,self.nstars]),noise_w],axis=0)
            r = R.from_euler('ZY',[-self.stars.fov[1][0],90-self.stars.fov[0][0]],degrees=True)
            noise_w = r.apply(noise_w.T,inverse=True).T
            data = noise_w
            self.data = data
        else:
            data = cdn.dn(self.injectobj.h, self.injectobj.srcindet, self.stars.starcoords)
            self.data = data

        return data

    def run_inference(self):
        npar, nwalkers = self.ndims, self.nchains
        initial = np.array([(self.minlogMs+self.maxlogMs)/2.])
        np.random.seed(self.SEED)
        p0 = [np.array(initial) + 1e-2 * np.random.randn(npar) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnprob,
                args=[self.data])
        sampler.run_mcmc(p0, self.ntune+self.nsamples, progress=True)

        samples = sampler.get_chain(discard=self.ntune)

        if self.OVERWRITE:
            ## save results
            pklpath = make_file_path(CHAINSDIR, [np.log10(self.freq.value),
                np.log10(self.nstars), np.log10(self.nsamples)],
                extra_string=f'samples_{self.MEANSUB}_{self.USEGS}',
                ext='.pkl')
            with open(pklpath, 'wb') as buff:
                pickle.dump(samples, buff)
            print('Wrote {}'.format(pklpath))

        ## save samples to class
        self.sampler = sampler

    def lnlike(self, theta,data_):
        model = self.model*(10.**(theta-self.src.logMs0))**(5./3.)
        S = np.sum(-(data_- model)**2/(2*self.wfirst_sig**2))
        if ~np.isfinite(S):
            return -np.inf
        return S

    def lnprior(self, theta_):
        if self.minlogMs<=theta_<self.maxlogMs:
            return 0.0
        else:
            return -np.inf

    def lnprob(self, theta_, data_):
        lp = self.lnprior(theta_)
        if ~np.isfinite(lp):
            return -np.inf
        ll = self.lnlike(theta_,data_)
        return lp + ll



    def make_diagnostic_plots(self):
        outpath = make_file_path(DIAGNOSTICDIR,
                [np.log10(self.freq.value),np.log10(self.nstars),
                    np.log10(self.nsamples)],
                extra_string=f'posteriorplot_{self.MEANSUB}_{self.USEGS}',
                ext='.png')
        plot.plot_emcee(self.sampler.get_chain(flat=True, discard=self.ntune), outpath)
        outpath = make_file_path(DIAGNOSTICDIR,
                [np.log10(self.freq.value),np.log10(self.nstars),
                    np.log10(self.nsamples)],
                extra_string=f'chainsplot_{self.MEANSUB}_{self.USEGS}',
                ext='.png')
        plot.plot_chains(self.sampler.get_chain(), outpath)
        outpath = make_file_path(DIAGNOSTICDIR,
                [np.log10(self.freq.value),np.log10(self.nstars),
                    np.log10(self.nsamples)],
                extra_string=f'logprobplot_{self.MEANSUB}_{self.USEGS}',
                ext='.png')
        plot.plot_logprob(self.sampler.get_log_prob(), outpath)

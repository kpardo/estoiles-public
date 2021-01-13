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

import estoiles.calc_dn as cdn
import estoiles.plotting as plot
import estoiles.population as gen_pop

from estoiles.convenience import make_file_path
from estoiles.setup_starcoords import StarCoords
from estoiles.gw_source import GWSource
from estoiles.paths import *

@dataclass
class Sampler():
    freq: float = 1.e-6*u.Hz
    nstars: int = 1000
    nsamples: int = 1500
    nchains: int = 8
    ntune: int = 50
    ndims: int = 1
    minlogMs: float = 4.539283647798979
    maxlogMs: float = 11.539283647798978
    minN: int = 10000
    maxN: int = 140000
    popmass: float = 1.e9*u.Msun
    USEGS: bool = False
    MEANSUB: bool = True
    GAIA: bool = False
    OVERWRITE: bool = True
    USEPOP: bool = False
    SEED: int = 0
    INJECT: bool = False
    CATALOG: bool = False
    sigma: float = 1.1

    injectobj: Any = GWSource(freq=freq)
    nguide: int = 0

    def __post_init__(self):
        self.stars = self.load_stars()
        self.calc_sigma()
        if self.USEPOP:
            self.prior_l = self.minN
            self.prior_u = self.maxN
        else:
            self.prior_l = self.minlogMs
            self.prior_u = self.maxlogMs
            self.set_source(self.freq)
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
        ws = (self.sigma*u.mas).to(u.rad)/np.sqrt(1.e8/self.nstars)
        T_obs = (6.*72.*u.d).to(1.*u.s)
        Nm = T_obs/(15*u.min).to(1*u.s)
        wfirst_sig = (ws/np.sqrt(Nm)).value
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
        np.random.seed(0)
        noise_w = np.random.normal(0, self.wfirst_sig, (2,self.nstars))
        noise_w = np.concatenate([np.zeros([1,self.nstars]),noise_w],axis=0)
        r = R.from_euler('ZY',[-self.stars.fov[1][0],90-self.stars.fov[0][0]],degrees=True)
        noise_w = r.apply(noise_w.T,inverse=True).T
        data = noise_w
        if self.INJECT:
            dn = cdn.dn(self.injectobj.h, self.injectobj.srcindet, self.stars.starcoords)
            if self.MEANSUB:
                dn = np.mean(dn,axis=1).reshapße([3,1])
                dn = dn - means
            data += dn
        self.data = data

        # temporary:
        if self.USEPOP and self.CATALOG:
            file = DATADIR+'catalogs/Sloan_dn.npy'
            if os.path.exists(file):
                self.popdn = np.load(file,allow_pickle=True)
            else:
                maxf = gen_pop.Population.get_ISCO(self.popmass)
                if self.freq is not None:
                    if self.freq > maxf:
                        self.freq = maxf
                pop = gen_pop.Population(nsource=145155,sloan=True,Mc=self.popmass,freq=self.freq)
                self.popdn = np.array([cdn.dn(pop.harray[i],pop.srcindetarray[i],self.stars.starcoords) for i in range(pop.nsource)])
                np.save(file,self.popdn)

        return data

    def run_inference(self):
        npar, nwalkers = self.ndims, self.nchains
        initial = np.array([(self.prior_l+self.prior_u)/2.])
        np.random.seed(self.SEED)
        if self.USEPOP:
            p0 = [initial + 1000*np.random.rand(npar) for i in range(nwalkers)]
        else:
            p0 = [initial + 1.e-2 * np.random.randn(npar) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnprob,
                args=[self.data])
        sampler.run_mcmc(p0, self.ntune+self.nsamples, progress=True)
        samples = sampler.get_chain(discard=self.ntune)

        if self.OVERWRITE:
            pklpath = make_file_path(CHAINSDIR, [
                np.log10(self.nstars), np.log10(self.nsamples)],
                extra_string=f'samples_{self.MEANSUB}_{self.USEGS}',
                ext='.pkl')
            with open(pklpath, 'wb') as buff:
                pickle.dump(samples, buff)
            print('Wrote {}'.format(pklpath))

        self.sampler = sampler

    def lnlike(self, theta,data_):
        if self.USEPOP:
            n = round(theta[0]).astype(int)
            if self.CATALOG:
                # pop = gen_pop.Population(nsource=n,twomass=True,Mc=self.popmass)
                ind = np.random.choice(np.arange(145155),size=n,replace=False)
                dn = self.popdn[ind]
            else:
                freqs = np.ones(n)*self.freq
                ls = np.random.rand(n)*360*u.deg
                bs = np.random.rand(n)*180.*u.deg - 90.*u.deg
                sourcecoord = SkyCoord(l=ls,b=bs,frame = 'galactic')
                Mcs = np.ones(n)*self.popmass
                poppar = {'freq':freqs,
                    'sourcecoord':sourcecoord,
                    'Mc':Mcs}
                pop = gen_pop.Population(nsource=n,randomvars=poppar)
                dn = np.array([cdn.dn(pop.harray[i],pop.srcindetarray[i],self.stars.starcoords) for i in range(pop.nsource)])
            model = np.sum(dn,axis=0)
            if self.MEANSUB:
                means = np.mean(model,axis=1).reshape([3,1])
                model = model - means

        else:
            model = self.model*(10.**(theta-self.src.logMs0))**(5./3.)
        S = np.sum(-(data_- model)[1:]**2/(2*self.wfirst_sig**2))
        if ~np.isfinite(S):
            return -np.inf
        return S

    def lnprior(self, theta_):
        if self.prior_l<=theta_<self.prior_u:
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
        if self.USEPOP:
            parval = [np.log10(self.popmass.value)]
        else:
            parval = [np.log10(self.freq.value),np.log10(self.nstars),
            np.log10(self.nsamples)]
        outpath = make_file_path(DIAGNOSTICDIR,
                parval,
                extra_string=f'posteriorplot_{self.MEANSUB}_{self.USEGS}',
                ext='.png')
        ##FIXME: should also thin out samples by half the autocorr time.
        plot.plot_emcee(self.sampler.get_chain(flat=True, discard=self.ntune), outpath)
        outpath = make_file_path(DIAGNOSTICDIR,
                parval,
                extra_string=f'chainsplot_{self.MEANSUB}_{self.USEGS}',
                ext='.png')
        plot.plot_chains(self.sampler.get_chain(), outpath)
        outpath = make_file_path(DIAGNOSTICDIR,
                parval,
                extra_string=f'logprobplot_{self.MEANSUB}_{self.USEGS}',
                ext='.png')
        plot.plot_logprob(self.sampler.get_log_prob(), outpath)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import astropy.units as u
import scipy

from estoiles.plotting import savefig, paper_plot
from estoiles.convenience import make_file_path
from estoiles.paths import *

paper_plot()

# Sensitivity curves for various surveys
c = 3e8*u.m/u.s
G = 6.67e-11*u.m**3/u.kg/u.s**2
H0 = (70*u.km/u.s*1./u.Mpc).to(1./u.s)
strain_prefac = 2*G**(5/3)/c**4*np.pi**(2/3)
fmax_prefac = c**3/(6**1.5*np.pi*G)

def Mc_maxISCO(q,f):
    return fmax_prefac*q**0.6*(1+q)**.3/f
def hc_SMBHB(f):
    f = f.to(1*u.Hz)
    h0 = 1.93e-15
    f0 = 3.72e-8*u.Hz
    g = -1.08
    return h0*(f/f0)**(-2/3)*(1+f/f0)**g
def df(Mc,f):
    Mc = Mc.to(1*u.kg)
    return 96/5*(2*np.pi)**(8/3)*(G*Mc/c**3)**(5/3)*f**(11/3)
def hcf(theta):
    Mc,R,f = theta
    df_ = df(Mc,f)
    return np.sqrt(2*f**2/df_)*h((R,Mc,1,f))
def h(theta):
    R,Mc,q,f = theta
    R = R.to(1*u.m)
    Mc = Mc.to(1*u.kg)
    return strain_prefac*f**(2./3.)*Mc**(5./3.)/R

## Roman EML survey
fmin_Roman = 7.7e-8/u.s
fmax_Roman = 5.6e-4/u.s
obs_time = 6*72*u.d
f_Roman = np.linspace(fmin_Roman, fmax_Roman, 500)
pf = np.sqrt(3/(np.pi**2*2*1e8))
hc_Roman = (1.1*u.mas/(np.sqrt(f_Roman*obs_time))).to(1*u.rad)*pf

fname = DATADIR+'forfigs/upperlimMs.npy'
if not os.path.exists(fname):
    import drivers.make_upperlimMs

logMsMax = np.load(DATADIR+'forfigs/upperlimMs.npy',allow_pickle=True)
logfmin = np.log10(fmin_Roman.value)
logfmax = np.log10(fmax_Roman.value)
freqArray = 10.**np.linspace(logfmin,logfmax,15)
Ms_func = scipy.interpolate.interp1d(freqArray,logMsMax-1.8,bounds_error=False,fill_value=logMsMax.max()-1.8)

## Gaia
fmin_Gaia = 6e-9/u.s
fmax_Gaia = 4.4e-7/u.s
f_Gaia = np.linspace(fmin_Gaia,fmax_Gaia, 500)
pf = np.sqrt(3/(np.pi**2*2*1e9))
hc_Gaia = (0.7*u.mas/(np.sqrt(f_Gaia*5*obs_time))).to(1*u.rad)*pf

## LISA
fmin_LISA = 1e-5*u.Hz
fmax_LISA = 1e-2*u.Hz
L = (2.5*u.Gm).to(1*u.m)
fs = 19.09*u.mHz
f = np.linspace(fmin_LISA,fmax_LISA,100).to(1*u.mHz)
f_LISA = f.to(1*u.Hz)
P_oms = (1.5e-11*u.m)**2*(1+(2*u.mHz/f)**4)/u.Hz
P_acc = (3e-15*u.m/u.s**2)**2*(1+(.4*u.mHz/f)**2)*(1+(f/(8*u.mHz))**4)/u.Hz
Sn = 10/(3*L**2)*(P_oms+4*P_acc/(2*np.pi*f)**4)*(1+.6*(f/fs)**2)
hc_LISA = np.sqrt(Sn*f_LISA)
f = np.linspace(3e-4/u.s,f_LISA[-1],50)
hLISA = hcf((Mc_maxISCO(1,f).to(1*u.Msun),(3*c/H0).to(1*u.Mpc),f))

## IPTA
fmin_PTA = (1/(5*u.yr)).to(1*u.Hz)
fmax_PTA = (17/u.yr).to(1*u.Hz)
f_PTA = np.linspace(fmin_PTA,fmax_PTA,100).to(1/u.s)
dt = (1/fmax_PTA).to(1*u.s)
sig = (250*u.ns).to(1*u.s)
hc_PTA = np.sqrt(24*np.pi**2*dt)*sig*f_PTA**1.5

## SMBH background
fmin_sb = 5e-9*u.Hz
fmax_sb = 2e-7*u.Hz
f_sb = np.linspace(fmin_sb,fmax_sb,100)
hc_sb = hc_SMBHB(f_sb)

fig,ax = plt.subplots(figsize=(9,6))
## Roman EML plot
ax.loglog(f_Roman,hc_Roman,linewidth=2,color='k')
ax.text(3e-7,5e-15,r'\bf{Roman EML survey}',fontsize=20,rotation = -10,horizontalalignment='left',verticalalignment='bottom')

## Roman EML f range fiducial source
fms = f_Roman[:-1]
maxMclim = hcf((10.**(Ms_func(fms))*u.Msun,10*u.Mpc,fms))
a = -18
da = np.log10(maxMclim.max())-a
for i in range(0,50):
    ax.fill_between(fms,10**(a+i*da/50)*np.ones_like(maxMclim),maxMclim,color='firebrick',alpha = 0.01,where=maxMclim>10**(a+i*da/50)*np.ones_like(maxMclim))
ax.fill_between([],[],[],color='firebrick',label='10 Mpc',alpha=.6)

## Gaia
ax.loglog(f_Gaia,hc_Gaia,linestyle='-.',linewidth=2,color='k')
ax.text(7e-9,4e-14,r'\textit{Gaia}',fontsize=20,rotation = -10)

## LISA
ax.loglog(f_LISA,hc_LISA,linestyle='--',linewidth=2,color='k')
ax.text(5e-4,6e-21,'LISA',fontsize=20,rotation=-30)
f = np.linspace(3e-4/u.s,f_LISA[-1],50)

## LISA f range fiducial source
a = -22
da = np.log10(hLISA.max())-a
for i in range(0,50):
    ax.fill_between(f,10**(a+i*da/50)*np.ones_like(hLISA),hLISA,color='tab:blue',alpha = 0.01,where=hLISA>10**(a+i*da/50)*np.ones_like(hLISA))
ax.fill_between([],[],[],color='tab:blue',label=r'$z=3$',alpha=.2)

## PTA
ax.loglog(f_PTA,hc_PTA,linestyle=':',linewidth=2,color='k')
ax.text(8e-9,5e-16,'IPTA',fontsize=20,rotation=30)

## SMBH background
a = -21
da = np.log10(hc_sb.max())-a
for i in range(0,50):
    ax.fill_between(f_sb,10**(a+i*da/50)*np.ones_like(hc_sb),hc_sb,color='orange',alpha = 0.01,where=hc_sb>10**(a+i*da/50)*np.ones_like(hc_sb))
ax.fill_between([],[],[],color='orange',label='SMBHMB',alpha=.3)
_,ymax = ax.get_ylim()
ax.vlines(f_PTA[0].value,hc_PTA[0].value,ymax,linestyle=':',linewidth=2,color='k')
ax.vlines(f_PTA[-1].value,hc_PTA[-1].value,ymax,linestyle=':',linewidth=2,color='k')

ax.legend(loc='upper right',fontsize=16)
ax.set_ylabel(r'Characteristic Strain $h_c$',fontsize=24)
ax.set_xlabel(r'Frequency [Hz]',fontsize=24)

outpath = make_file_path(FIGSDIR, {}, extra_string='sens',ext='.png')

savefig(fig, outpath, writepdf=True)

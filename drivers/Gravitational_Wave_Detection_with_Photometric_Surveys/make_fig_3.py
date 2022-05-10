import matplotlib.pyplot as plt
import numpy as np
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
fmax_pf = c**3/(6**1.5*np.pi*G)

def ISCO_f(Mc,q):
    Mc = Mc.to(1*u.kg)
    return fmax_pf/Mc*q**.6*(1+q)**.3
def hc_SMBHB(f):
    f = f.to(1*u.Hz)
    h0 = 1.93e-15
    f0 = 3.72e-8*u.Hz
    g = -1.08
    return h0*(f/f0)**(-2/3)*(1+f/f0)**g
def h(theta):
    R,Mc,q,f = theta
    R = R.to(1*u.m)
    Mc = Mc.to(1*u.kg)
    return strain_prefac*f**(2./3.)*Mc**(5./3.)/R
def df(Mc,f):
    Mc = Mc.to(1*u.kg)
    return 96/5*(2*np.pi)**(8/3)*(G*Mc/c**3)**(5/3)*f**(11/3)
def hcf(theta):
    Mc,R,f = theta
    df_ = df(Mc,f)
    return np.sqrt(2*f**2/df_)*h((R,Mc,1,f))

#generate data
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

## RST
fmin_Roman = 7.7e-8/u.s
fmax_Roman = 5.6e-4/u.s
obs_time = 6*72*u.d
f_Roman = np.linspace(fmin_Roman, fmax_Roman, 500)
pf = np.sqrt(3/(np.pi**2*2*1e8))
N_obs = obs_time.to(1*u.min)/(15*u.min)
hc_Roman = 1.1*u.mas.to(1*u.rad)*pf*np.ones_like(f_Roman)/np.sqrt(N_obs)

## Gaia
fmin_Gaia = 6e-9/u.s
fmax_Gaia = 4.4e-7/u.s
f_Gaia = np.linspace(fmin_Gaia,fmax_Gaia, 500)
pf = np.sqrt(3/(np.pi**2*2*1e9))
hc_Gaia = 0.7*u.mas.to(1*u.rad)*pf*np.ones_like(f_Gaia)/np.sqrt(70)

fig,ax = plt.subplots(figsize=(9,6))
ax.loglog(f_PTA,hc_PTA,linewidth=2,color='gray')
ax.text(1e-8,2e-14,'IPTA',fontsize=20,rotation=33)

ax.loglog(f_Roman,hc_Roman*100,linewidth=2,color='k')
ax.text(4e-7,2e-13,r'\bf Roman EML Survey',fontsize=20,horizontalalignment='left',verticalalignment='bottom')
ax.loglog(f_Roman,hc_Roman,linewidth=2,linestyle='dashed',color='k')
ax.text(6e-7,2e-15,r'\bf (Roman EML, full signal)',fontsize=16,horizontalalignment='left',
        verticalalignment='bottom')

ax.loglog(f_Gaia,hc_Gaia,linewidth=2,color='gray')
ax.text(6e-8,1.e-14,r'\textit{Gaia}',fontsize=20)

ax.loglog(f_LISA,hc_LISA,linewidth=2,color='gray')
ax.text(5e-4,6e-21,'LISA',fontsize=20,rotation=-33)

#color block
a = -21
da = np.log10(f_sb.max().value)-a
nlayer = 30
layer_alp = 0.02
for i in range(0,nlayer):
    y = 10**(a+i*da/nlayer)*np.ones_like(hc_sb)
    ax.fill_between(f_sb,y,hc_sb,color='gold',alpha = layer_alp,where=hc_sb>y)
ax.fill_between([],[],[],color='gold',label=r'$\tilde{h}_c$(SMBHBB)',alpha=layer_alp*nlayer/1.5)

a = -19
mc = 10**9.7*u.Msun
DL = 50*u.Mpc
fmax = ISCO_f(mc,1)
f = np.linspace(fmin_Roman,fmax,100)
x = h((DL,mc,1,f))
da = np.log10(x.max().value)-a
nlayer = 30
layer_alp = 0.02
for i in range(0,nlayer):
    y = 10**(a+i*da/nlayer)*np.ones_like(f*u.s)
    ax.fill_between(f,y,x.value,color='lightcoral',alpha = layer_alp,where=x>y)
ax.fill_between([],[],[],color='lightcoral',label=r'$h(10^{9.7}~M_\odot,$ 50 Mpc)',alpha=layer_alp*nlayer)

mc = 10**7.5*u.Msun
DL = 50*u.Mpc
fmax = ISCO_f(mc,1)
f = np.linspace(fmin_Roman,fmax,100)
x = h((DL,mc,1,f))
da = np.log10(x.max().value)-a
nlayer = 30
layer_alp = 0.02
for i in range(0,nlayer):
    y = 10**(a+i*da/nlayer)*np.ones_like(f*u.s)
    ax.fill_between(f,y,x.value,color='plum',alpha = layer_alp,where=x>y)
ax.fill_between([],[],[],color='plum',label=r'$h(10^{7}~M_\odot,$ 50 Mpc)',alpha=layer_alp*nlayer)

fmax = ISCO_f(1e6*u.Msun,1)
fm_LISA = np.linspace(3e-4/u.s,fmax,50)
hLISA = hcf((1e6*u.Msun,25*u.Gpc,fm_LISA))

a = -22
da = np.log10(hLISA.max())-a
nlayer = 30
layer_alp = 0.02
for i in range(0,nlayer):
    y = 10**(a+i*da/nlayer)*np.ones_like(hLISA)
    ax.fill_between(fm_LISA,y,hLISA,color='tab:blue',alpha = layer_alp,where=hLISA>y)
ax.fill_between([],[],[],color='tab:blue',label=r'$\tilde{h}_c(10^6~M_\odot,~z=3$)',alpha=layer_alp*nlayer)

_,ymax = ax.get_ylim()
ax.vlines(f_PTA[0].value,hc_PTA[0].value,ymax,linewidth=2,color='gray')
ax.vlines(f_PTA[-1].value,hc_PTA[-1].value,ymax,linewidth=2,color='gray')

ax.legend(loc='lower left',fontsize=18)
ax.set_ylabel(r'Strain',fontsize=24)
ax.set_xlabel(r'Frequency [Hz]',fontsize=24)

outpath = make_file_path(FIGSDIR, {}, extra_string='figure_3',ext='.png')

savefig(fig, outpath, writepdf=False)

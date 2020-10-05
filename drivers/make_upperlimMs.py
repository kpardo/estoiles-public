import numpy as np
import scipy.optimize
import astropy.constants as const
import astropy.units as u

GM = np.load(PNINTERPDIR+'h1PN_Grid.npy',allow_pickle=True)
LMC = GM[:,0,0,0]
grid_max = GM[-1,-1,-1,:].reshape([3,1])
grid_min = GM[0,0,0,:].reshape([3,1])
fmin = 7.7e-8/u.s
fmax = 5.6e-4/u.s
freqArray = np.logspace(fmin,fmax,15)
fmax_pf = (const.c**3/(6**1.5*np.pi*const.G)).to(u.kg/u.s)

def logqmin(LMCarray,f_ref):
    qmin = np.empty([])
    for logmc in LMCarray:
        mc = 10.**logmc*u.Msun
        if ISCO_f(mc,1.) <= f_ref:
            qmin = np.append(qmin,0.)
        else:
            qfreq = scipy.optimize.fsolve(calc_fmax,lo_logq(mc,f_ref),args=(f_ref,mc))
            qmin = np.append(qmin,qfreq[0])
    return qmin[1:]
def ISCO_f(Mc,q):
    Mc = Mc.to(1*u.kg)
    return fmax_pf/Mc*q**.6*(1+q)**.3
def calc_fmax(logq,f,Mc):
    Mc = Mc.to(1*u.kg)
    q = 10.**logq
    return abs(fmax_pf/Mc*q**.6*(1+q)**.3-f)
def lo_logq(Mc,freq):
    Mc = Mc.to(1*u.kg)
    return np.log10((Mc*freq/fmax_pf)**(5/3))

logMcmin = []
for fi in freqArray:
    qmin = logqmin(LMC,fi)
    b = np.where(qmin==0.)[0]
    if b.size:
        logMcmin.append(LMC[b[0]]-.6*grid_min[2])
    else: logMcmin.append(grid_max[0]-.6*grid_min[2])

np.save(DATADIR+'forfigs/upperlimMs',logMcmin)

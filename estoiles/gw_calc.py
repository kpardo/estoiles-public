'''
This module defines the GWcalc class, which models GW from an inspiraling binary, as is observed from a specific telescope position.
'''

## import necessary packages
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.constants as const

## set constants
c = const.c
G = const.G
Gc3 = G/c**3


class GWcalc:


    def __init__(self,Mc,q,f0,R,inc,psi,sourceCoord,detCoord,
            phip=0*u.rad,phic=0*u.rad):
        '''Initialize a GWcalc instance.

        Keyword arguments:
        Mc -- chirp mass
        q -- mass ratio m1/m2
        f0 -- GW frequency
        R -- distance from telescope to GW source
        inc -- inclination angle between the binary's orbital angular momentum and the vector from the orbital center to the telescope
        psi -- angle between the major axis of the binary's projected orbit on
        the celestial sphere counterclockwise to the longitudinal line through
        orbital center, in the telescope frame
        sourceCoord -- GW source coordinate, SkyCoord object
        detCoord -- z-axis of the telescope frame, SkyCoord object
        phip -- initial phase of hp (default 0*u.rad)
        phic -- initial phase of hc (default 0*u.rad)
        '''

        self.Mc = Mc
        self.m = Mc*((1+q)**2/q)**(3/5)
        self.dm = (q-1)/(q+1)
        self.wo = f0*np.pi    # orbital angular frequency
        self.R = (R).to(1*u.m)
        self.inc = (inc).to(1*u.rad)
        self.psi = (psi).to(1*u.rad)
        self.q = self.CoordTr(detCoord,sourceCoord)
        self.phip = phip
        self.phic = phic
        self.setup()

    def setup(self):
        '''Calculate constants specific to a GWcalc instance.'''
        i,j,k = self.q
        a = SkyCoord(i,j,k, representation_type='cartesian')
        a.representation_type = 'unitspherical'
        self.Hp,self.Hc = self.H(a,self.psi)
        self.tau = (self.m).to(1*u.kg)*5*Gc3
        self.eta = (1-self.dm**2)/4
        self.tc = (self.tau/self.eta)*(0.625/(self.tau*self.wo))**(8/3)
        self.eta1 = 743/4032 + 11/48*self.eta
        self.eta2 = 19583/254016 + 24401/193536*self.eta + 31/288*self.eta**2
        self.eta3 = 3715/1008 + 55/12*self.eta
        self.eta4 = 15293365/1016064 + 27145/1008*self.eta + 3085/144*self.eta**2
        self.ho = .4*self.tau*self.eta*c/self.R
        self.s1 = np.sin(self.inc)
        self.s2 = np.sin(2*self.inc)
        self.s4 = np.sin(4*self.inc)
        self.c1 = np.cos(self.inc)
        self.c2 = np.cos(2*self.inc)
        self.c4 = np.cos(4*self.inc)
        self.c6 = np.cos(6*self.inc)

    def CoordTr(self,detCoord,sourceCoord):
        '''Convert sourceCoord into the frame of detCoord.

        The telescope frame x-axis is chosen to be parallel to the equitorial
        plane, i.e., b = 0 in Galactic coordinate or dec = 0 in ICRS. The
        y-axis is calculated accordingly to produce a right-handed coordinate
        system.

        Keyword arguments:
        detCoord -- z-axis of the telescope frame, SkyCoord object
        sourceCoord -- GW source coordinate, SkyCoord object

        Returns:
        projection of sourceCoord in the telescope's frame, in cartesian representation
        '''

        detCoord.representation_type = 'unitspherical'
        beta = np.pi/2*u.rad - detCoord.b
        alpha = np.pi/2*u.rad + detCoord.l
        dnC = np.array([sourceCoord.cartesian.x.value, sourceCoord.cartesian.y.value, sourceCoord.cartesian.z.value])
        s = np.sin
        c = np.cos
        TrMat = np.array([[c(alpha),-s(alpha)*c(beta),s(beta)*s(alpha)],
            [s(alpha),c(alpha)*c(beta),-s(beta)*c(alpha)],
            [0,s(beta),c(beta)]])
        invTrMat = np.linalg.inv(TrMat)
        return np.matmul(invTrMat,dnC)

    def H(self,sourceCoord,psi_):
        '''Compute plus and cross polarization basis matrices.

        Keyword arguments:
        sourceCoord -- GW source coordinate in the telescope's frame, SkyCoord
        object in ICRS frame psi_ -- angle between the major axis of the
        binary's projected orbit on the celestial sphere counterclockwise to
        the longitudinal line through orbital center, in the telescope frame

        Returns:
        Hp, Hc -- plus and cross polarization matrices, in cartesian basis of the telescope frame
        '''

        phi_ = sourceCoord.ra
        theta_ = 90*u.deg - sourceCoord.dec
        P = np.array([[np.sin(psi_)*np.sin(phi_)-np.cos(psi_)*np.cos(theta_)*np.cos(phi_)],
            [-np.sin(psi_)*np.cos(phi_)-np.cos(theta_)*np.sin(phi_)*np.cos(psi_)],
            [np.cos(psi_)*np.sin(theta_)]])
        Q = np.array([[-np.cos(psi_)*np.sin(phi_)-np.sin(psi_)*np.cos(theta_)*np.cos(phi_)],
            [np.cos(psi_)*np.cos(phi_)-np.cos(theta_)*np.sin(phi_)*np.sin(psi_)],
            [np.sin(psi_)*np.sin(theta_)]])
        Hp = np.array(np.matmul(P,P.T) - np.matmul(Q,Q.T))
        Hc = np.array(np.matmul(P,Q.T) + np.matmul(Q,P.T))
        return Hp,Hc

    def calcphi(self,t_):
        '''Calculate orbital phase, assuming zero at coalescence.'''
        Th = self.eta/self.tau*(self.tc-t_)
        coeff = .25*Th**(-.25)
        x = coeff*(1 + self.eta1*Th**(-.25) - np.pi*.2*Th**(-3/8) + self.eta2*Th**(-.5))
        coeff = -x**(-2.5)/(32*self.eta)
        self.phi = (coeff*(1 + self.eta3*x -10*np.pi*x**(3/2) + self.eta4*x**2))*u.rad
        return self.phi+self.phip

    def calcA(self,theta_):
        '''Calculate waveform amplitudes accurate up to 2PN.

        Keyword arguments:
        theta_[0] -- time
        theta_[1] -- a boolean array to specify to which post-Newtonian term the waveform is calculated
        theta_[2] -- orbital phase, assuming 0 at coalescence

        Returns:
        Ap, Ac -- plus and cross polarization wave amplitude
        '''
        t_,orderList,phi_ = theta_
        ## need to make 2 copies of these cp and sp functions if we ever want
        ##to use different initial phases (phip,phic) for two polarizations.
        cp1 = np.cos(phi_)
        cp2 = np.cos(2*phi_)
        cp3 = np.cos(3*phi_)
        cp4 = np.cos(4*phi_)
        cp5 = np.cos(5*phi_)
        cp6 = np.cos(6*phi_)
        sp1 = np.sin(phi_)
        sp2 = np.sin(2*phi_)
        sp3 = np.sin(3*phi_)
        sp4 = np.sin(4*phi_)
        sp5 = np.sin(5*phi_)
        sp6 = np.sin(6*phi_)

        Th = self.eta/self.tau*(self.tc-t_)
        coeff = .25*Th**(-.25)
        x = coeff*(1 + self.eta1*Th**(-.25) - np.pi*.2*Th**(-3/8) + self.eta2*Th**(-.5))

        y1 = x**(.5)
        y2 = y1*y1
        y3 = y2*y1
        y4 = y3*y1
        y5 = y4*y1
        y6 = y5*y1

        # H functions
        # 0th-order
        if orderList[0]:
            h0p = -(1+self.c2) * cp2
            h0c = -2 * self.c1 * sp2
        else:
            h0p = 0
            h0c = 0

        # 1/2
        if orderList[1]:
            h1p1 = -self.dm * 0.125 * (5+self.c2)
            h1p3 = self.dm * 1.125 * (1+self.c2)
            h1p = self.s1 * (h1p1 + cp3*h1p3)

            h1c = -0.75 * self.dm * self.s1 * self.c1 * (sp1-3*sp3)
        else:
            h1p = 0
            h1c = 0

        # 2/2
        if orderList[2]:
            h2p2 = (19 + 9*self.c2 - 2*self.c4 - self.eta*(19-11*self.c2-6*self.c4))/6
            h2p4 = -(4/3)*self.s2*(1+self.c2)*(1-3*self.eta)
            h2p = h2p2*cp2 + h2p4*cp4

            h2c2 = ((17-4*self.c2) - self.eta*(13-12*self.c2))/3
            h2c4 = -(8/3)*(1-3*self.eta)*self.c1*self.s2
            h2c = self.c1*h2c2*sp2 + h2c4*sp4
        else:
            h2p = 0
            h2c = 0

        # 3/2
        if orderList[3]:
            h3p1 = (57 + 60*self.c2 - self.c4 - self.eta*(98-24*self.c2-2*self.c4))/192
            h3p3 = (73 + 40*self.c2 - 9*self.c4 - self.eta*(50 - 16*self.c2 - 18*self.c4))*(-27/384)
            h3p5 = (625/384)*(1 - 2*self.eta)*(self.s2 + self.s2*self.c2)
            h3p2 = -2*np.pi*(1 + self.c2)
            h3p = self.s1*self.dm*(h3p1*cp1 + h3p3*cp3 + h3p5*cp5) + h3p2*cp2

            b = 2*self.eta
            h3c1 = (63 - 5*self.c2 - b*(23 - 5*self.c2))/96
            h3c3 = -27*(67 - 15*self.c2 - b*(19 - 15*self.c2))/192
            h3c5 = 625*(1 - b)*self.s2/192
            h3c = self.s1*self.c1*self.dm*(h3c1*sp1 + h3c3*sp3 + h3c5*sp5) - 4*np.pi*self.c1*sp2
        else:
            h3p = 0
            h3c = 0

        # 4/2
        if orderList[4]:
            a = (5/3)*self.eta
            b = 5*self.eta*self.eta
            h4p2 = (22 + 396*self.c2 + 145*self.c4 - 5*self.c6 + a*(706-216*self.c2 -251*self.c4 + 15*self.c6) - b*(98 - 108*self.c2 + 7*self.c4 + 5*self.c6))/120
            h4p4 = (59 + 35*self.c2 - 8*self.c4 - a*(131 + 59*self.c2 - 24*self.c4) + b*(21 - 3*self.c2 - 8*self.c4))*self.s2/7.5
            h4p6 = -81*(1 - 5*self.eta + b)*(self.s4 + self.s4*self.c2)/40
            h4ps = self.s1*(11 + 7*self.c2 + (50 + 10*self.c2)*np.log(2))/40
            h4pc = -5*np.pi*self.s1*(5 + self.c2)/40
            h4p3s = (270*np.log(1.5) - 189)*(self.s1 + self.s1*self.c2)/40
            h4p3c = 135*np.pi*(self.s1 + self.s1*self.c2)/40
            h4p = h4p2*cp2 + h4p4*cp4 + h4p6*cp6 + self.dm*(h4ps*sp1 + h4pc*cp1 + h4p3s*sp3 + h4p3c*cp3)

            h4c2 = (68 + 226*self.c2 - 15*self.c4 + a*(572 - 490*self.c2 + 45*self.c4) - b*(56 - 70*self.c2 + 15*self.c4))*self.c1/60
            h4c4 = (55 - 12*self.c2 - a*(119 - 36*self.c2) + b*(17 - 12*self.c2))*4*self.c1*self.s2/15
            h4c6 = -4.05*(1 - 3*a + b)*self.c1*self.s4
            h4cc = (3 + 10*np.log(2))*(3/20)
            h4c3 = (-63 + 90*np.log(1.5))*(3/20)
            h4c = h4c2*sp2 + h4c4*sp4 + h4c6*sp6 - self.dm*self.s1*self.c1*(h4cc*cp1 + 0.75*np.pi*sp1 + h4c3*cp3 - 6.75*np.pi*sp3)
        else:
            h4p = 0
            h4c = 0

        self.Ap = self.ho*(y2*h0p + y3*h1p + y4*h2p + y5*h3p + y6*h4p)
        self.Ac = self.ho*(y2*h0c + y3*h1c + y4*h2c + y5*h3c + y6*h4c)

        return self.Ap,self.Ac

    def calc_h(self,theta_,phi_=None,A=None):
        '''Compute waveform tensor.

        Keyword arguments:
        theta_[0] -- time of evaluation
        theta_[1] -- a boolean array to specify to which post-Newtonian term the waveform is calculated
        phi_ -- orbital phase (default None)
        A -- a tuple of wave amplitudes (Ap,Ac) (default None)
        '''
        t_, orderList_ = theta_
        if t_ < self.tc:
            if phi_ is not None:
                self.phi = phi_
            else:
                self.phi = self.calcphi(t_)
            if A is not None:
                self.Ap,self.Ac = A
            else:
                Theta = t_,orderList_,self.phi
                self.Ap, self.Ac = self.calcA(Theta)
            term1 = self.Ap*self.Hp*np.exp(self.phip.value*1j)
            term2 = self.Ac*self.Hc*np.exp(self.phic.value*1j)
            h = (term1 + term2)*np.exp(self.phi.value*1j)
        else:
            self.Ap = 0
            self.Ac = 0
            h = 0.
        return h


def coordtransform(detCoord,sourceCoord):
    ##FIXME: should probably take the one in the class out?? can just be a
    ## generic function. doesn't need class variables.
    '''Convert sourceCoord into the frame of detCoord.

    The telescope frame x-axis is chosen to be parallel to the equitorial
    plane, i.e., b = 0 in Galactic coordinate or dec = 0 in ICRS. The y-axis is
    calculated accordingly to produce a right-handed coordinate system.

    Keyword arguments:
    detCoord -- z-axis of the telescope frame, SkyCoord object
    sourceCoord -- GW source coordinate, SkyCoord object

    Returns:
    projection of sourceCoord in the telescope's frame, in cartesian representation
    '''

    detCoord.representation_type = 'unitspherical'
    beta = np.pi/2*u.rad - detCoord.b
    alpha = np.pi/2*u.rad + detCoord.l
    dnC = np.array([sourceCoord.cartesian.x.value, sourceCoord.cartesian.y.value, sourceCoord.cartesian.z.value])
    s = np.sin
    c = np.cos
    TrMat = np.array([[c(alpha),-s(alpha)*c(beta),s(beta)*s(alpha)],
        [s(alpha),c(alpha)*c(beta),-s(beta)*c(alpha)],
        [0,s(beta),c(beta)]])
    invTrMat = np.linalg.inv(TrMat)
    return np.matmul(invTrMat,dnC)

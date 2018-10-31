# Author: Taylor Bell
# Last Update: 2018-10-31

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import scipy.optimize

class KeplerOrbit(object):
    def __init__(self, Porb=None, t0=0, a=const.au.value, inc=90, e=0, Omega=270, argp=90, m1=const.M_sun.value, m2=0):
        
        self.e = e
        self.a = a
        self.inc = np.pi/2-inc*np.pi/180
        self.Omega = Omega*np.pi/180
        self.argp = argp*np.pi/180
        self.t0 = t0
        self.m1 = m1
        self.m2 = m2
        
        if Porb == None:
            self.Porb = self.get_period()
        else:
            self.Porb = Porb
    
    # Find the keplerian orbital period
    def get_period(self):
        return 2*np.pi*self.a**(3/2)/np.sqrt(const.G.value*(self.m1+self.m2))/(24*3600)
    
    # Get the mean motion
    def get_meanMotion(self):
        return 2*np.pi/self.Porb
    
    # Convert true anomaly to eccentric anomaly
    def ta_to_ea(self, ta):
        return 2.*np.arctan(np.sqrt((1.-self.e)/(1.+self.e))*np.tan(ta/2.))
    
    # Convert eccentric anomaly to mean anomaly
    def ea_to_ma(self, ea):
        return ea - self.e*np.sin(ea)
    
    # Convert true anomaly to mean anomaly
    def ta_to_ma(self, ta):
        return self.ea_to_ma(self.ta_to_ea(ta))
    
    # Get the time of periastron
    def get_peri_time(self):
        return self.t0-self.ta_to_ma(np.pi/2.-self.argp)/(2*np.pi)*self.Porb
    
    # Get the time of transit
    def get_trans_time(self):
        return self.t0
    
    # Get the time of secondary eclipse
    def get_ecl_time(self):
        return (self.t0 + (self.ta_to_ma(3.*np.pi/2.-self.argp)-self.ta_to_ma(1.*np.pi/2.-self.argp))/(2*np.pi)*self.Porb)
    
    # Convert time to mean anomaly
    def get_meanAnomaly(self, t):
        return (t-self.get_peri_time()) * self.get_meanMotion()
    
    # Convert time to eccentric anomaly numerically
    def get_eccentricAnomaly(self, t, xtol=1e-10):
        M = self.get_meanAnomaly(t)
        f = lambda E: E - self.e*np.sin(E) - M
        if self.e < 0.8:
            E0 = M
        else:
            E0 = np.pi*np.ones_like(M)
        E = scipy.optimize.fsolve(f, E0, xtol=xtol)
        return E
    
    # Convert time to true anomaly
    def get_trueAnomaly(self, t):
        return 2*np.arctan(np.sqrt((1+self.e)/(1-self.e))*np.tan(self.get_eccentricAnomaly(t)/2))
    
    # Find the host--planet separation at time t
    def get_distance(self, t):
        return self.a*(1-self.e**2)/(1+self.e*np.cos(self.get_trueAnomaly(t)))
    
    # Find the position of the planet at time t
    def get_xyz(self, t):
        E = self.get_eccentricAnomaly(t)
        
        # The following code is roughly based on:
        # https://space.stackexchange.com/questions/8911/determining-orbital-position-at-a-future-point-in-time
        P = self.a*(np.cos(E)-self.e)
        Q = self.a*np.sin(E)*np.sqrt(1-self.e**2)
        
        # Rotate by argument of periapsis
        x = (np.cos(self.argp-np.pi/2.)*P-np.sin(self.argp-np.pi/2.)*Q)
        y = np.sin(self.argp-np.pi/2.)*P+np.cos(self.argp-np.pi/2.)*Q
        
        # Rotate by inclination
        z = -np.sin(self.inc)*x
        x = np.cos(self.inc)*x
        
        # Rotate by longitude of ascending node
        xtemp = x
        x = -(np.sin(self.Omega)*xtemp+np.cos(self.Omega)*y)
        y = (np.cos(self.Omega)*xtemp-np.sin(self.Omega)*y)
        
        return x, y, z
    
    # A Convenience Routine Used to Visualize the Planet's Orbit
    def show_orbit(self):
        t = np.linspace(0,self.Porb,100, endpoint=False)

        x, y, z = np.array(self.get_xyz(t))/const.au.value

        tPeri = self.get_peri_time()
        tTrans = self.get_trans_time()
        tEcl = self.get_ecl_time()

        xTrans, yTrans, zTrans = np.array(self.get_xyz(tTrans))/const.au.value
        xEcl, yEcl, zEcl = np.array(self.get_xyz(tEcl))/const.au.value
        xPeri, yPeri, zPeri = np.array(self.get_xyz(tPeri))/const.au.value

        plt.plot(y, x, '.', c='k', ms=2)
        plt.plot(0,0, '*', c='r', ms=15)
        plt.plot(yTrans, xTrans, 'o', c='b', ms=10, label='Transit')
        plt.plot(yEcl, xEcl, 'o', c='k', ms=7, label='Eclipse')
        if self.e != 0:
            plt.plot(yPeri, xPeri, 'o', c='r', ms=5, label='Periastron')
        plt.xlabel('y')
        plt.ylabel('x')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.legend(loc=6, bbox_to_anchor=(1,0.5))
        plt.show()

        plt.plot(y, z, '.', c='k', ms=2)
        plt.plot(0,0, '*', c='r', ms=15)
        plt.plot(yTrans, zTrans, 'o', c='b', ms=10)
        plt.plot(yEcl, zEcl, 'o', c='k', ms=7)
        if self.e != 0:
            plt.plot(yPeri, zPeri, 'o', c='r', ms=5)
        plt.gca().set_aspect('equal')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.show()

        plt.plot(x, z, '.', c='k', ms=2)
        plt.plot(0,0, '*', c='r', ms=15)
        plt.plot(xTrans, zTrans, 'o', c='b', ms=10)
        plt.plot(xEcl, zEcl, 'o', c='k', ms=7)
        if self.e != 0:
            plt.plot(xPeri, zPeri, 'o', c='r', ms=5)
        plt.xlabel('x')
        plt.ylabel('z')
        plt.gca().set_aspect('equal')
        plt.show()

# Author: Taylor Bell
# Last Update: 2018-11-19

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import scipy.optimize

class KeplerOrbit(object):
    """A Keplerian orbit.
    
    Attributes:
        a (float): The semi-major axis in m.
        Porb (float): The orbital period in days.
        inc (float): The orbial inclination (in degrees above face-on)
        t0 (float): The linear ephemeris in days.
        e (float): The orbital eccentricity.
        Omega (float): The longitude of ascending node (in degrees CCW from line-of-sight).
        argp (float): The argument of periastron (in degrees CCW from Omega).
        m1 (float): The mass of body 1 in kg.
        m2 (float): The mass of body 2 in kg.
    
    """
    
    def __init__(self, a=const.au.value, Porb=None, inc=90, t0=0, e=0, Omega=270, argp=90, m1=const.M_sun.value, m2=0):
        """Initialization function.
        
        Args:
            a (float, optional): The semi-major axis in m.
            Porb (float, optional): The orbital period in days.
            inc (float, optional): The orbial inclination (in degrees above face-on)
            t0 (float, optional): The linear ephemeris in days.
            e (float, optional): The orbital eccentricity.
            Omega (float, optional): The longitude of ascending node (in degrees CCW from line-of-sight).
            argp (float, optional): The argument of periastron (in degrees CCW from Omega).
            m1 (float, optional): The mass of body 1 in kg.
            m2 (float, optional): The mass of body 2 in kg.
        
        """
        
        self.e = e
        self.a = a
        self.inc = np.pi/2-inc*np.pi/180
        self.Omega = Omega*np.pi/180
        self.argp = argp*np.pi/180
        self.t0 = t0
        self.m1 = m1
        self.m2 = m2
        
        if Porb is None:
            self.Porb = self.period()
        else:
            self.Porb = Porb
    
    def period(self):
        """Find the keplerian orbital period.
        
        Returns:
            float: The keplerian orbital period.
        
        """
        
        return 2*np.pi*self.a**(3/2)/np.sqrt(const.G.value*(self.m1+self.m2))/(24*3600)
    
    def meanMotion(self):
        """Get the mean motion.
        
        Returns:
            float: The mean motion in radians.
            
        """
        
        return 2*np.pi/self.Porb
    
    def ta_to_ea(self, ta):
        """Convert true anomaly to eccentric anomaly.
        
        Args:
            ta (ndarray): The true anomaly in radians.
        
        Returns:
            ndarray: The eccentric anomaly in radians.
        
        """
        
        return 2.*np.arctan(np.sqrt((1.-self.e)/(1.+self.e))*np.tan(ta/2.))
    
    def ea_to_ma(self, ea):
        """Convert eccentric anomaly to mean anomaly.
        
        Args:
            ea (ndarray): The eccentric anomaly in radians.
        
        Returns:
            ndarray: The mean anomaly in radians.
        
        """
        
        return ea - self.e*np.sin(ea)
    
    def ta_to_ma(self, ta):
        """Convert true anomaly to mean anomaly.
        
        Args:
            ta (ndarray): The true anomaly in radians.
        
        Returns:
            ndarray: The mean anomaly in radians.
        
        """
        
        return self.ea_to_ma(self.ta_to_ea(ta))
    
    def peri_time(self):
        """Get the time of periastron.
        
        Returns:
           float: The time of periastron.
           
        """
        
        return self.t0-self.ta_to_ma(np.pi/2.-self.argp)/(2*np.pi)*self.Porb
    
    def trans_time(self):
        """Get the time of transit.
        
        Returns:
           float: The time of transit.
           
        """
        
        return self.t0
    
    def ecl_time(self):
        """Get the time of secondary eclipse.
        
        Returns:
           float: The time of secondary eclipse.
           
        """
        
        return (self.t0 + (self.ta_to_ma(3.*np.pi/2.-self.argp)-self.ta_to_ma(1.*np.pi/2.-self.argp))/(2*np.pi)*self.Porb)
    
    def meanAnomaly(self, t):
        """Convert time to mean anomaly.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            ndarray: The mean anomaly in radians.
        
        """
        
        return (t-self.peri_time()) * self.meanMotion()
    
    def eccentricAnomaly(self, t, xtol=1e-10):
        """Convert time to eccentric anomaly, numerically.
        
        Args:
            t (ndarray): The time in days.
            xtol (float): tolarance on error in eccentric anomaly.
        
        Returns:
            ndarray: The eccentric anomaly in radians.
        
        """
        
        M = self.meanAnomaly(t)
        f = lambda E: E - self.e*np.sin(E) - M
        if self.e < 0.8:
            E0 = M
        else:
            E0 = np.pi*np.ones_like(M)
        E = scipy.optimize.fsolve(f, E0, xtol=xtol)
        return E
    
    def trueAnomaly(self, t, xtol=1e-10):
        """Convert time to true anomaly, numerically.
        
        Args:
            t (ndarray): The time in days.
            xtol (float): tolarance on error in eccentric anomaly (calculated along the way).
        
        Returns:
            ndarray: The true anomaly in radians.
        
        """
        
        return 2*np.arctan(np.sqrt((1+self.e)/(1-self.e))*np.tan(self.eccentricAnomaly(t, xtol=xtol)/2))
    
    def distance(self, t, xtol=1e-10):
        """Find the separation between the two bodies.
        
        Args:
            t (ndarray): The time in days.
            xtol (float): tolarance on error in eccentric anomaly (calculated along the way).
        
        Returns:
            ndarray: The separation between the two bodies.
        
        """
        
        return self.a*(1-self.e**2)/(1+self.e*np.cos(self.trueAnomaly(t, xtol=xtol)))
    
    # Find the position of the planet at time t
    def xyz(self, t, xtol=1e-10):
        """Find the coordinates of body 2 with respect to body 1.
        
        Args:
            t (ndarray): The time in days.
            xtol (float): tolarance on error in eccentric anomaly (calculated along the way).
        
        Returns:
            list: A list of 3 ndarrays containing the x,y,z coordinate of body 2 with respect to body 1.
            
                The x coordinate is along the line-of-sight.
                The y coordinate is perpendicular to the line-of-sight and in the orbital plane.
                The z coordinate is perpendicular to the line-of-sight and above the orbital plane
        
        """
        
        E = self.eccentricAnomaly(t, xtol=xtol)
        
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
    
    def show_orbit(self):
        """A convenience routine to visualize the orbit
        
        """
        
        t = np.linspace(0,self.Porb,100, endpoint=False)

        x, y, z = np.array(self.xyz(t))/const.au.value

        tPeri = self.peri_time()
        tTrans = self.trans_time()
        tEcl = self.ecl_time()

        xTrans, yTrans, zTrans = np.array(self.xyz(tTrans))/const.au.value
        xEcl, yEcl, zEcl = np.array(self.xyz(tEcl))/const.au.value
        xPeri, yPeri, zPeri = np.array(self.xyz(tPeri))/const.au.value

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

        return

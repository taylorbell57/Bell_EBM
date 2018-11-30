# Author: Taylor Bell
# Last Update: 2018-11-30

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
        obliq (float, optional): The obliquity (axial tilt) of body 2 (in degrees toward body 1).
        argobliq (float, optional): The reference orbital angle used for obliq (in degrees from inferior conjunction).
        wWind (float, optional): Body 2's wind angular velocity in revolutions/s.
        t_peri (float): Time of body 2's closest approach to body 1.
        t_ecl (float): Time of body 2's eclipse by body 1.
        t_trans (float): Time of body 1's eclipse by body 2.
    
    """
    
    def __init__(self, a=const.au.value, Porb=None, inc=90, t0=0, e=0, Omega=270, argp=90, # orbital parameters
                 obliq=0, argobliq=0, Prot=None, wWind=0, # spin parameters
                 m1=const.M_sun.value, m2=0): # mass parameters
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
            obliq (float, optional): The obliquity (axial tilt) of body 2 (in degrees toward body 1).
            argobliq (float, optional): The reference orbital angle used for obliq (in degrees from inferior conjunction).
            wWind (float, optional): The body 2's wind angular velocity in revolutions/s.
        
        """
        
        self.e = e
        self.a = a
        self.inc = inc
        self.Omega = Omega
        self.argp = argp
        self.t0 = t0
        self.m1 = m1
        self.m2 = m2
        
        #Orbital Attributes
        self.Porb_input = Porb
        self.Porb = Porb
        
        # Obliquity Attributes
        self.obliq = obliq                 # degrees toward star
        self.argobliq = argobliq           # degrees from t0
        if -90 <= self.obliq <= 90:
            self.ProtSign = 1
        else:
            self.ProtSign = -1
        
        #Rotation Rate Attributes
        if Prot is not None:
            self.Prot_input = Prot*self.ProtSign               # days
        elif self.Porb is not None:
            self.Prot_input = self.Porb*self.ProtSign
        else:
            self.Prot_input = None
        
        if wWind is None:
            self.wWind = 0
        else:
            self.wWind = wWind                 # revolutions/s
            
        if self.Prot_input is not None:
            self.wRot = 1/(self.Prot_input*24*3600) # m/s
        elif self.Porb is not None:
            self.wRot = 1/(self.Porb*24*3600) # m/s
        else:
            self.wRot = None
        
        if self.wRot is not None:
            self.Prot = 1/((self.wWind+self.wRot)*(24*3600)) # days
        
        
        self.t_trans = self.t0
        if self.Porb is not None:
            self.t_peri = self.t0-self.ta_to_ma(np.pi/2.-self.argp*np.pi/180)/(2*np.pi)*self.Porb
            if self.t_peri < 0:
                self.t_peri = self.Porb + self.t_peri

            self.t_ecl = (self.t0 + (self.ta_to_ma(3.*np.pi/2.-self.argp*np.pi/180)
                                     - self.ta_to_ma(1.*np.pi/2.-self.argp*np.pi/180))/(2*np.pi)*self.Porb)
            if self.t_ecl < 0:
                self.t_ecl = self.Porb + self.t_ecl
            
        return
    
    
    def solve_period(self):
        """Find the Keplerian orbital period.
        
        Returns:
            float: The Keplerian orbital period.
        
        """
        
        return 2*np.pi*self.a**(3/2)/np.sqrt(const.G.value*(self.m1+self.m2))/(24*3600)
    
    
    def set_Porb(self, Porb=None):
        """Set the orbital period.
        
        Args:
            Porb (float, optional): The orbital period in days. If Porb==None, solve for the Keplerian orbital period.
        
        Returns:
            None
        
        """
        
        if Porb == None:
            self.Porb = self.solve_period()
        else:
            self.Porb = Porb
            
        # Update self.Prot
        if self.Prot_input is None:
            self.wRot = 1/(self.Porb*self.ProtSign*24*3600) # m/s
        else:
            self.wRot = 1/(self.Prot_input*24*3600) # m/s
        
        self.Prot = 1/((self.wWind+self.wRot)*(24*3600)) # days
    
        self.t_peri = self.t0-self.ta_to_ma(np.pi/2.-self.argp*np.pi/180)/(2*np.pi)*self.Porb
        if self.t_peri < 0:
            self.t_peri = self.Porb + self.t_peri

        self.t_ecl = (self.t0 + (self.ta_to_ma(3.*np.pi/2.-self.argp*np.pi/180)
                                 - self.ta_to_ma(1.*np.pi/2.-self.argp*np.pi/180))/(2*np.pi)*self.Porb)
        if self.t_ecl < 0:
            self.t_ecl = self.Porb + self.t_ecl
    
        return
    
    def set_Prot(self, Prot):
        """Set body 2's rotational period.
        
        Args:
            Prot (float): The rotational period in days.
        
        Returns:
            None
        
        """
        
        self.Prot_input = Prot
        
        # Update self.Prot
        if self.Prot_input is None:
            self.wRot = 1/(self.Porb*self.ProtSign*24*3600) # m/s
        else:
            self.wRot = 1/(self.Prot_input*24*3600) # m/s
        
        self.Prot = 1/((self.wWind+self.wRot)*(24*3600)) # days
    
        return
    
    def mean_motion(self):
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
    
    
    def mean_anomaly(self, t):
        """Convert time to mean anomaly.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            ndarray: The mean anomaly in radians.
        
        """
        
        return (t-self.t_peri) * self.mean_motion()
    
    
    def eccentric_anomaly(self, t, xtol=1e-10):
        """Convert time to eccentric anomaly, numerically.
        
        Args:
            t (ndarray): The time in days.
            xtol (float): tolarance on error in eccentric anomaly.
        
        Returns:
            ndarray: The eccentric anomaly in radians.
        
        """
        
        if type(t)!= np.ndarray:
            t = np.array([t])
        tShape = t.shape
        t = t.flatten()
        
        M = self.mean_anomaly(t)
        f = lambda E: E - self.e*np.sin(E) - M
        if self.e < 0.8:
            E0 = M
        else:
            E0 = np.pi*np.ones_like(M)
        E = scipy.optimize.fsolve(f, E0, xtol=xtol)
        
        # Make some commonly used values exact
        E[np.abs(E)<xtol] = 0
        E[np.abs(E-2*np.pi)<xtol] = 2*np.pi
        E[np.abs(E-np.pi)<xtol] = np.pi
        
        return E.reshape(tShape)
    
    def true_anomaly(self, t, xtol=1e-10):
        """Convert time to true anomaly, numerically.
        
        Args:
            t (ndarray): The time in days.
            xtol (float): tolarance on error in eccentric anomaly (calculated along the way).
        
        Returns:
            ndarray: The true anomaly in radians.
        
        """
        
        return 2*np.arctan(np.sqrt((1+self.e)/(1-self.e))*np.tan(self.eccentric_anomaly(t, xtol=xtol)/2))
    
    def distance(self, t, xtol=1e-10):
        """Find the separation between the two bodies.
        
        Args:
            t (ndarray): The time in days.
            xtol (float): tolarance on error in eccentric anomaly (calculated along the way).
        
        Returns:
            ndarray: The separation between the two bodies.
        
        """
        
        if type(t)!=np.ndarray or len(t.shape)!=1:
            t = np.array([t]).reshape(-1)
        
        distance = self.a*(1-self.e**2)/(1+self.e*np.cos(self.true_anomaly(t, xtol=xtol)))
        return distance.reshape(-1,1)
    
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
        
        E = self.eccentric_anomaly(t, xtol=xtol)
        
        # The following code is roughly based on:
        # https://space.stackexchange.com/questions/8911/determining-orbital-position-at-a-future-point-in-time
        P = self.a*(np.cos(E)-self.e)
        Q = self.a*np.sin(E)*np.sqrt(1-self.e**2)
        
        # Rotate by argument of periapsis
        x = (np.cos(self.argp*np.pi/180-np.pi/2.)*P-np.sin(self.argp*np.pi/180-np.pi/2.)*Q)
        y = np.sin(self.argp*np.pi/180-np.pi/2.)*P+np.cos(self.argp*np.pi/180-np.pi/2.)*Q
        
        # Rotate by inclination
        z = -np.sin(np.pi/2-self.inc*np.pi/180)*x
        x = np.cos(np.pi/2-self.inc*np.pi/180)*x
        
        # Rotate by longitude of ascending node
        xtemp = x
        x = -(np.sin(self.Omega*np.pi/180)*xtemp+np.cos(self.Omega*np.pi/180)*y)
        y = (np.cos(self.Omega*np.pi/180)*xtemp-np.sin(self.Omega*np.pi/180)*y)
        
        return x, y, z
    
    
    def get_phase_periastron(self):
        """Get the orbital phase of periastron.
        
        Returns:
            float: The orbital phase of periastron.
            
        """
        
        return self.get_phase(self.t_peri)
    
    
    def get_phase_transit(self):
        """Get the orbital phase of transit.
        
        Returns:
            float: The orbital phase of transit.
            
        """
        
        return 0
    
    
    def get_phase_eclipse(self):
        """Get the orbital phase of eclipse.
        
        Returns:
            float: The orbital phase of eclipse.
            
        """
        
        return self.get_phase(self.t_ecl)
    
    
    def get_phase(self, t):
        """Get the orbital phase.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            ndarray: The orbital phase.
            
        """
        
        phase = (self.true_anomaly(t)-self.true_anomaly(self.t0))/(2*np.pi)
        phase = phase + 1*(phase<0).astype(int)
        return phase
    
    
    def get_ssp(self, t):
        """Calculate the sub-stellar longitude and latitude.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            list: A list of 2 ndarrays containing the sub-stellar longitude and latitude.
                
                Each ndarray is in the same shape as t.
        
        """
        
        if type(t)!=np.ndarray:
            t = np.array([t])
            tshape = t.shape
        elif len(t.shape)!=1:
            tshape = t.shape
            t = t.reshape(-1)
        else:
            tshape = t.shape
        
        sspLon = self.true_anomaly(t)*180/np.pi - (t-self.t0)/self.Prot*360 + self.Omega+self.argp
        sspLon = sspLon%180+(-180)*(np.rint(np.floor(sspLon%360/180) > 0))
        sspLat = self.obliq*np.cos(self.get_phase(t)*2*np.pi-self.argobliq*np.pi/180)
        return sspLon.reshape(tshape), sspLat.reshape(tshape)

    def get_sop(self, t):
        """Calculate the sub-observer longitude and latitude.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            list: A list of 2 ndarrays containing the sub-observer longitude and latitude.
            
                Each ndarray is in the same shape as t.
        
        """
        
        if type(t)!=np.ndarray:
            t = np.array([t])
        sopLon = 180-((t-self.t0)/self.Prot)*360
        sopLon = sopLon%180+(-180)*(np.rint(np.floor(sopLon%360/180) > 0))
        sopLat = 90-self.inc-self.obliq
        return sopLon, sopLat
    
    
    def plot_orbit(self):
        """A convenience routine to visualize the orbit
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        t = np.linspace(0,self.Porb,100, endpoint=False)

        x, y, z = np.array(self.xyz(t))/const.au.value

        xTrans, yTrans, zTrans = np.array(self.xyz(self.t0))/const.au.value
        xEcl, yEcl, zEcl = np.array(self.xyz(self.t_ecl))/const.au.value
        xPeri, yPeri, zPeri = np.array(self.xyz(self.t_peri))/const.au.value

        plt.plot(y, x, '.', c='k', ms=2)
        plt.plot(0,0, '*', c='r', ms=15)
        plt.plot(yTrans, xTrans, 'o', c='b', ms=10, label=r'$\rm Transit$')
        plt.plot(yEcl, xEcl, 'o', c='k', ms=7, label=r'$\rm Eclipse$')
        if self.e != 0:
            plt.plot(yPeri, xPeri, 'o', c='r', ms=5, label=r'$\rm Periastron$')
        plt.xlabel('$y$')
        plt.ylabel('$x$')
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
        plt.xlabel('$y$')
        plt.ylabel('$z$')
        plt.show()

        plt.plot(x, z, '.', c='k', ms=2)
        plt.plot(0,0, '*', c='r', ms=15)
        plt.plot(xTrans, zTrans, 'o', c='b', ms=10)
        plt.plot(xEcl, zEcl, 'o', c='k', ms=7)
        if self.e != 0:
            plt.plot(xPeri, zPeri, 'o', c='r', ms=5)
        plt.xlabel('$x$')
        plt.ylabel('$z$')
        plt.gca().set_aspect('equal')
        
        return plt.gcf()

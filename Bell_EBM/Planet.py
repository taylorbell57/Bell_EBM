# Author: Taylor Bell
# Last Update: 2018-11-02

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.constants as const

from .KeplerOrbit import KeplerOrbit
from .Map import Map
from . import H2_Dissociation_Routines as h2

class Planet(object):
    """A planet.

    Attributes:
        a (float): The planet's semi-major axis in m.
        albedo (float): The planet's Bond albedo.
        argobliq (float): The reference orbital angle used for the obliquity (in degrees from inferior conjunction).
        argp (float): The planet's argument of periastron (in degrees CCW from Omega).
        C (float, optional): The planet's heat capacity in J/m^2/K.
        cp (float or callable): The planet's isobaric specific heat capacity in J/kg/K.
        cpParams (iterable, optional): Any parameters to be passed to cp if using the bell2018 LTE H2+H mix cp
        e (float): The planet's orbital eccentricity.
        g (float): The planet's surface gravity in m/s^2.
        inc (float): The planet's orbial inclination (in degrees above face-on)
        map (Map): The planet's temperature map.
        mass (float): The planet's mass in kg.
        mlDensity (float): The density of the planet's mixed layer.
        mlDepth (float): The depth of the planet's mixed layer.
        obliq (float): The planet's obliquity (axial tilt) (in degrees toward star).
        Omega (float): The planet's longitude of ascending node (in degrees CCW from line-of-sight).
        orbit (KeplerOrbit): The planet's orbit.
        plType (str): The planet's composition.
        Porb (float): The planet's orbital period in days.
        Prot (float): The planet's rotational period in days.
        rad (float): The planet's radius in m.
        t0 (float): The planet's linear ephemeris in days.
        useHealpix (bool): Whether the planet's map uses a healpix grid.
        vWind (float): The planet's wind velocity in m/s.
    
    """
    
    def __init__(self, plType='gas', rad=1, mass=1, a=0.03, Porb=None, Prot=None, vWind=0, albedo=0,
                 inc=90, t0=0, e=0, Omega=270, argp=90, obliq=0, argobliq=0, nside=16, useHealpix=False):
        """Initialization function.
        
        Args:
            plType (str, optional): The planet's composition.
            rad (float, optional): The planet's radius in m.
            mass (float, optional): The planet's mass in kg.
            a (float, optional): The planet's semi-major axis in m.
            Porb (float, optional): The planet's orbital period in days.
            Prot (float, optional): The planet's rotational period in days.
            vWind (float, optional): The planet's wind velocity in m/s.
            albedo (float, optional): The planet's Bond albedo.
            inc (float, optional): The planet's orbial inclination (in degrees above face-on)
            t0 (float, optional): The planet's linear ephemeris in days.
            e (float, optional): The planet's orbital eccentricity.
            Omega (float, optional): The planet's longitude of ascending node (in degrees CCW from line-of-sight).
            argp (float, optional): The planet's argument of periastron (in degrees CCW from Omega).
            obliq (float, optional): The planet's obliquity (axial tilt) (in degrees toward star).
            argobliq (float, optional): The reference orbital angle used for obliq (in degrees from inferior conjunction).
            nside (int, optional): A parameter that sets the resolution of the map.
            useHealpix (bool, optional): Whether the planet's map uses a healpix grid.
        
        """
        
        #Planet Attributes
        self.plType = plType
        self.rad = rad*const.R_jup.value   # m
        self.mass = mass*const.M_jup.value # kg
        self.g = const.G.value*self.mass/self.rad**2 # m/s^2
        self.albedo = albedo               # None
        
        # Planet's Thermal Attributes
        if self.plType.lower()=='water':
            #water
            self.cp = 4.1813e3             # J/kg/K
            self.mlDepth = 50              # m
            self.mlDensity = 1e3           # kg/m^3
            self.C = self.mlDepth*self.mlDensity*self.cp
        elif self.plType.lower() == 'rock':
            #basalt
            self.cp = 0.84e3                # J/kg/K
            self.mlDepth = 0.5              # m
            self.mlDensity = 3e3            # kg/m^3
            self.C = self.mlDepth*self.mlDensity*self.cp
        elif self.plType.lower() == 'gas':
            # H2 atmo
            self.cp = 14.31e3              # J/kg/K
            self.mlDepth = 0.1e5           # Pa
            self.mlDensity = 1/self.g      # s^2/m
            self.C = self.mlDepth*self.mlDensity*self.cp
        elif self.plType.lower() == 'bell2018':
            # LTE H2+H atmo
            self.cp = h2.true_cp
            self.cpParams = None
            self.mlDepth = 0.1e5           # Pa
            self.mlDensity = 1/self.g      # s^2/m
        else:
            print('Planet type not accepted!')
            return False
        
        #Map Attributes
        self.map = Map(nside, useHealpix=useHealpix)
                
        #Orbital Attributes
        self.a = a*const.au.value          # m
        self.Porb = Porb                   # days (if None, will be set to Kepler expectation when loaded into system)
        self.e = e                         # None
        self.Omega = Omega                 # degrees ccw from line-of-sight
        self.inc = inc                     # degrees above face-on
        self.argp = argp                   # degrees ccw from Omega
        self.obliq = obliq                 # degrees toward star
        if self.obliq <= 90:
            self.ProtSign = 1
        else:
            self.ProtSign = -1
        self.argobliq = argobliq           # degrees from t0
        self.t0 = t0                       # days
        if self.Porb is not None:
            self.orbit = KeplerOrbit(a=self.a, Porb=self.Porb, inc=self.inc, t0=self.t0, 
                                     e=self.e, Omega=self.Omega, argp=self.argp, m2=self.mass)
        else:
            self.orbit = None
        
        #Rotation Rate Attributes
        if Prot is None:
            self.Prot_input = self.Porb
        else:
            self.Prot_input = Prot               # days
        if self.Prot_input is None:
            self.vRot = 0
        else:
            self.vRot = 2*np.pi*self.rad/(self.Prot_input*24*3600) # m/s
        self.vWind = vWind                 # m/s
        if self.vWind == 0:
            self.Prot = self.Prot_input
        else:
            self.Prot = 2*np.pi*self.rad/((self.vWind+self.vRot)*(24*3600))
    
    # Used to propogate any changes to the planet's attributes through the other attributes
    def update(self):
        """Update the planet's properties
        
        Used to propogate any manual changes to the planet's attributes through the other, dependent attributes.
        
        """
        
        self.g = const.G.value*self.mass/self.rad**2
        
        if self.plType.lower() == 'gas':
            # H2 atmo
            self.mlDensity = 1/self.g        # s^2/m
            self.C = self.mlDepth*self.mlDensity*self.cp
        elif self.plType.lower() == 'bell2018':
            # LTE H2+H atmo
            self.mlDensity = 1/self.g        # s^2/m
        else:
            self.C = self.mlDepth*self.mlDensity*self.cp
        
        if self.Porb is not None and self.Prot is None:
            self.Prot_input = self.Porb
        
        if self.Prot_input is not None:
            self.vRot = 2*np.pi*self.rad/(self.Prot_input*24*3600) # m/s
            if self.vWind == 0:
                self.Prot = self.Prot_input
            else:
                self.Prot = 2*np.pi*self.rad/((self.vWind+self.vRot)*(24*3600))
        
        if self.Porb is not None:
            self.orbit = KeplerOrbit(a=self.a, Porb=self.Porb, inc=self.inc, t0=self.t0, 
                                     e=self.e, Omega=self.Omega, argp=self.argp, m2=self.mass)
    
    def SSP(self, t):
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
        trueAnom = (self.orbit.trueAnomaly(t)*180/np.pi)
        trueAnom[trueAnom<0] += 360
        sspLon = (trueAnom-t/self.Prot*360)
        sspLon = sspLon%180+(-180)*(np.rint(np.floor(sspLon%360/180) > 0))
        sspLat = self.obliq*np.cos(t/self.Porb*2*np.pi-self.argobliq*np.pi/180)
        return sspLon.reshape(tshape), sspLat.reshape(tshape)

    def SOP(self, t):
        """Calculate the sub-observer longitude and latitude.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            list: A list of 2 ndarrays containing the sub-observer longitude and latitude.
            
                Each ndarray is in the same shape as t.
        
        """
        
        if type(t)!=np.ndarray:
            t = np.array([t])
        sopLon = 180-(t/self.Prot)*360
        sopLon = sopLon%180+(-180)*(np.rint(np.floor(sopLon%360/180) > 0))
        sopLat = 90-self.inc-self.obliq*np.cos(t/self.Porb*2*np.pi-self.argobliq*np.pi/180)
        return sopLon, sopLat

    def Fout(self, T=None, bolo=True, wav=1e-6):
        """Calculate the instantaneous total outgoing flux.
        
        Args:
            T (ndarray): The temperature (if None, use self.map.values).
            bolo (bool, optional): Determines whether computed flux is bolometric
                (True, default) or wavelength dependent (False).
            wav (float, optional): The wavelength to use if bolo==False.
        
        Returns:
            ndarray: The emitted flux in the same shape as T.
        
        """
        
        if T is None:
            T = self.map.values
        elif type(T)!=np.ndarray:
            T = np.array([T])
        
        if bolo:
            return const.sigma_sb.value*T**4
        else:
            a = (2*const.h.value*const.c.value**2/wav**5)
            b = (const.h.value*const.c.value)/(wav*const.k_B.value)
            return a/np.expm1(b/T)
        
    def weight(self, t, refPos='SSP'):
        """Calculate the weighting of map pixels.
        
        Weight flux by visibility/illumination kernel, assuming the star/observer are infinitely far away for now.
        
        Args:
            t (ndarray): The time in days.
            refPos (str, optional): The reference position (SSP or SOP).
        
        Returns:
            ndarray: The weighting of map mixels at time t. Has shape (t.size, self.map.npix).
        
        """
        
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array([t]).reshape(-1,1)
        
        if refPos == 'SSP':
            refLon, refLat = self.SSP(t)
        elif refPos == 'SOP':
            refLon, refLat = self.SOP(t)
        else:
            print('Reference point "'+str(refPos)+'" not understood!')
            return False
        
        lonWeight = np.max(np.append(np.zeros((1,t.size,self.map.npix)), 
                                     np.cos((self.map.lonGrid-refLon)*np.pi/180)[np.newaxis,:,:], axis=0), axis=0)
        latWeight = np.cos((self.map.latGrid-refLat)*np.pi/180)[0]
        return lonWeight*latWeight
    
    def Fp_vis(self, t, T=None, bolo=True, wav=4.5e-6):
        """Calculate apparent outgoing planetary flux (used for making phasecurves).
        
        Weight flux by visibility/illumination kernel, assuming the star/observer are infinitely far away for now.
        
        Args:
            t (ndarray): The time in days.
            T (ndarray): The temperature (if None, use self.map.values).
            bolo (bool, optional): Determines whether computed flux is bolometric
                (True, default) or wavelength dependent (False).
            wav (float, optional): The wavelength to use if bolo==False
        
        Returns:
           ndarray: The apparent emitted flux. Has shape (t.size, self.map.npix).
        
        """
        
        if T is None:
            T = self.map.values
        
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array(t).reshape(-1,1)
        
        weights = self.weight(t, 'SOP')
        flux = self.Fout(T, bolo, wav)*self.map.pixArea*self.rad**2
        # used to try to remove wiggles from finite number of pixels coming in and out of view
        weightsNormed = weights*(4*np.pi/self.map.npix)/np.pi
        
        return np.sum(flux*weights, axis=1)/np.sum(weightsNormed, axis=1)

    def showMap(self, tempMap=None, time=None):
        """A convenience routine to plot the planet's temperature map.
        
        Args:
            tempMap (ndarray): The temperature map (if None, use self.map.values).
            time (float, optional): The time corresponding to the map used to de-rotate the map.
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        if tempMap is None:
            if time is None:
                subStellarLon = self.SSP(self.map.time)[0].flatten()
            else:
                subStellarLon = self.SSP(time)[0].flatten()
        else:
            self.map.set_values(tempMap, time)
            if time is not None:
                subStellarLon = self.SSP(time)[0].flatten()
            else:
                subStellarLon = None
        return self.map.plot_map(subStellarLon)
    
    def showDissociation(self, tempMap=None, time=None):
        """A convenience routine to plot the planet's H2 dissociation map.
        
        Args:
            tempMap (ndarray, optional): The temperature map (if None, use self.map.values).
            time (float, optional): The time corresponding to the map used to de-rotate the map.
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        if tempMap is None:
            if time is None:
                subStellarLon = self.SSP(self.map.time)[0].flatten()
            else:
                subStellarLon = self.SSP(time)[0].flatten()
        else:
            self.map.set_values(tempMap, time)
            if time is not None:
                subStellarLon = self.SSP(time)[0].flatten()
            else:
                subStellarLon = None
        self.map.plot_dissociation(subStellarLon)
        return plt.gcf()

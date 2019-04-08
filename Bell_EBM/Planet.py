# Author: Taylor Bell
# Last Update: 2019-02-15

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
        albedo (float): The planet's Bond albedo.
        cpParams (iterable, optional): Any parameters to be passed to cp if using the bell2018 LTE H2+H mix cp
        C (float, optional): The planet's heat capacity in J/m^2/K.
        emissivity (float): The emissivity of the emitting layer (between 0 and 1).
        g (float): The planet's surface gravity in m/s^2.
        instRedistFrac (float): The fraction of flux that is instantly redistributed across the entire planet.
        internalFlux (float): The planet's internal heating flux.
        orbit (Bell_EBM.KeplerOrbit): The planet's orbit.
        plType (str): The planet's composition.
        trasmissivity (float): The trasmissivity of the emitting layer (between 0 and 1).
        T_exponent (float): The exponent which determinges the rate at which the planet cools (4 for blackbody cooling,
            1 for Newtonian cooling) when calculating Fout with bolo=True.
        
    """
    
    
    @property
    def absorptivity(self):
        """float: The fraction of flux absorbed by the planet.
        
        Read-only.
        
        """
        return 1-self.albedo-self.trasmissivity
    
    @property
    def cp(self):
        """float or callable: The planet's isobaric specific heat capacity in J/kg/K.
        
        Changing the planet's cp may update planet.C
        
        """
        return self._cp
    
    @property
    def Porb(self):
        """float: The planet's orbital period in days."""
        return self.orbit.Porb
    
    @property
    def Prot(self):
        """float: The planet's rotational period in days."""
        return self.orbit.Prot
    
    @property
    def t0(self):
        """float: The planet's linear ephemeris in days."""
        return self.orbit.t0
    
    @property
    def map(self):
        """Bell_EBM.Map: The planet's temperature map."""
        return self._map
    
    @property
    def mass(self):
        """float: The planet's mass in kg.
        
        Changing the planet's mass will update planet.g and possibly planet.mlDensity and planet.C
        
        """
        return self._mass
    
    @property
    def mlDepth(self):
        """float: The depth of the planet's mixed layer.
        
        In units of m for rock/water models, or Pa for gas/bell2018 models.
        
        Changing the planet's mlDepth may update planet.C
        
        """
        return self._mlDepth
    
    @property
    def mlDensity(self):
        """float: The density of the planet's mixed layer
        
        In kg/m^3 if rock/water models or the inverse of the planet's surface gravity (in s^2/m) for gas/bell2018 models.
        
        Changing the planet's mlDensity may update planet.C
        
        """
        return self._mlDensity
    
    @property
    def rad(self):
        """float: The planet's radius if m.
        
        Changing the planet's radius will update planet.g and possibly planet.mlDensity and planet.C
        
        """
        return self._rad
    
    @property
    def vWind(self):
        """float: The planet's windspeed in m/s."""
        return self._vWind
    
    
    
    
    
    
    
    
    def __init__(self, plType='gas', rad=const.R_jup.value, mass=const.M_jup.value,
                 a=0.03*const.au.value, Porb=None, Prot=None, inc=90., t0=0., e=0., Omega=270., argp=90, obliq=0., argobliq=0.,
                 vWind=0., albedo=0., cp=None, cpParams=None, mlDepth=None, mlDensity=None, T_exponent=4.,
                 emissivity=1., trasmissivity=0., internalFlux=0., instRedistFrac=0.,
                 nlat=16, nlon=None):
        """Initialization function.
        
        Args:
            plType (str, optional): The planet's composition.
            rad (float, optional): The planet's radius in m.
            mass (float, optional): The planet's mass in kg.
            a (float, optional): The planet's semi-major axis in m.
            Porb (float, optional): The planet's orbital period in days.
            Prot (float, optional): The planet's rotational period in days.
            inc (float, optional): The planet's orbial inclination (in degrees above face-on)
            t0 (float, optional): The planet's linear ephemeris in days.
            e (float, optional): The planet's orbital eccentricity.
            Omega (float, optional): The planet's longitude of ascending node (in degrees CCW from line-of-sight).
            argp (float, optional): The planet's argument of periastron (in degrees CCW from Omega).
            obliq (float, optional): The planet's obliquity (axial tilt) (in degrees toward star).
            argobliq (float, optional): The reference orbital angle used for obliq (in degrees from inferior conjunction).
            vWind (float, optional): The planet's wind velocity in m/s.
            albedo (float, optional): The planet's Bond albedo.
            cp (float or callable, optional): The planet's isobaric specific heat capacity in J/kg/K.
            cpParams (iterable, optional): Any parameters to be passed to cp if using the bell2018 LTE H2+H mix cp
            mlDepth (float, optional): The depth of the planet's mixed layer (in units of m for rock/water models,
                or Pa for gas/bell2018 models).
            mlDensity (float, optional): The density of the planet's mixed layer (in kg/m^3) if rock/water models,
                or the inverse of the planet's surface gravity (in s^2/m) for gas/bell2018 models.
            T_exponent (float): The exponent which determinges the rate at which the body cools (4 for blackbody cooling,
                1 for Newtonian cooling).
            emissivity (float, optional): The emissivity of the emitting layer (between 0 and 1).
            trasmissivity (float, optional): The trasmissivity of the emitting layer (between 0 and 1).
            internalFlux (float, optional): The planet's internal heating flux.
            instRedistFrac (float): The fraction of flux that is instantly redistributed across the entire planet.
            nlat (int, optional): The number of latitudinal cells to use for rectangular maps.
            nlon (int, optional): The number of longitudinal cells to use for rectangular maps.
                If nlon==None, uses 2*nlat.
            
        """
        
        #Planet Attributes
        self.plType = plType.lower()
        self._rad = rad   # m
        self._mass = mass # kg
        self.g = const.G.value*self.mass/self.rad**2 # m/s^2
        self.albedo = albedo               # None
        
        self._cp = cp
        self.cpParams = cpParams
        self._mlDepth = mlDepth
        self._mlDensity = mlDensity
        self.T_exponent = T_exponent
        
        if emissivity > 1:
            self.emissivity = 1.
        elif emissivity < 0:
            self.emissivity = 0.
        else:
            self.emissivity = emissivity
        
        if trasmissivity > 1:
            self.emissivity = 1.
        elif trasmissivity < 0:
            self.trasmissivity = 0.
        else:
            self.trasmissivity = trasmissivity
        
        self.internalFlux = internalFlux
        self.instRedistFrac = instRedistFrac
        
        # Planet's Thermal Attributes
        if self.plType=='water':
            #water
            if self.cp == None:
                self._cp = 4.1813e3             # J/kg/K
            if self.mlDepth == None:
                self._mlDepth = 50.              # m
            if self.mlDensity == None:
                self._mlDensity = 1e3           # kg/m^3
            self.C = self.mlDepth*self.mlDensity*self.cp
        elif self.plType == 'rock':
            #basalt
            if self.cp == None:
                self._cp = 0.84e3                # J/kg/K
            if self.mlDepth == None:
                self._mlDepth = 0.5              # m
            if self.mlDensity == None:
                self._mlDensity = 3e3            # kg/m^3
            self.C = self.mlDepth*self.mlDensity*self.cp
        elif self.plType == 'gas':
            # H2 atmo
            if self.cp == None:
                self._cp = 14.31e3              # J/kg/K
            if self.mlDepth == None:
                self._mlDepth = 0.1e5           # Pa
            if self.mlDensity == None:
                self._mlDensity = 1/self.g      # s^2/m
            self.C = self.mlDepth*self.mlDensity*self.cp
        elif self.plType == 'bell2018':
            # LTE H2+H atmo
            if self.cp == None:
                self._cp = h2.true_cp
            if self.mlDepth == None:
                self._mlDepth = 0.1e5           # Pa
            self._mlDensity = 1./self.g      # s^2/m
            if self.cpParams is None:
                self.cpParams = h2.getSahaApproxParams(self.mlDepth)
        else:
            print('Planet type not accepted!')
            return False
        
        self._vWind = vWind
        
        #Map Attributes
        self.map = Map(nlat=nlat, nlon=nlon)
        
        self.orbit = KeplerOrbit(a=a, Porb=Porb, inc=inc, t0=t0, e=e, Omega=Omega, argp=argp,
                                 obliq=obliq, argobliq=argobliq, Prot=Prot,
                                 m2=self.mass)
    
    
    @Porb.setter
    def Porb(self, Porb):
        self.orbit.Porb = Porb
        return
    
    @t0.setter
    def t0(self, t0):
        self.orbit.t0 = t0
        return
    
    @Prot.setter
    def Prot(self, Prot):
        self.orbit.Prot = Prot
        return
    
    @map.setter
    def map(self, map):
        self._map = map
        
        if self.vWind == None:
            self.wind_dlon = 0.
        else:
            omegaWind = self.vWind/self.rad # radians/s
            self.wind_dlon = omegaWind/(self.map.dlon*np.pi/180)
            upwindIndex = np.roll(np.arange(self.map.nlon), int(np.sign(self.wind_dlon)))
            self.upwindLonIndex = (np.ones(self.map.nlat).reshape(-1,1)*upwindIndex).astype(int)
            self.upwindLatIndex = (np.arange(self.map.nlat).reshape(-1,1)*np.ones(self.map.nlon).reshape(1,-1)).astype(int)
        
        return
    
    @mass.setter
    def mass(self, mass):
        self._mass = mass
        
        # Update dependent attributes
        g_old = self.g
        self.g = const.G.value*self.mass/self.rad**2
        
        if self.plType.lower() == 'gas' or self.plType.lower() == 'bell2018':
            if self.mlDensity == 1./g_old:
                self.mlDensity = 1./self.g        # s^2/m
            self.C = self.mlDepth*self.mlDensity*self.cp
                
        return
    
    @rad.setter
    def rad(self, rad):
        self._rad = rad
        
        # Update dependent attributes
        g_old = self.g
        self.g = const.G.value*self.mass/self.rad**2
        
        if self.plType.lower() == 'gas' or self.plType.lower() == 'bell2018':
            if self.mlDensity == 1./g_old:
                self.mlDensity = 1./self.g        # s^2/m
            self.C = self.mlDepth*self.mlDensity*self.cp
                
        return
    
    @mlDepth.setter
    def mlDepth(self, mlDepth):
        self._mlDepth = mlDepth
        
        # Update dependent properties
        if not callable(self.cp):
            self.C = self.mlDepth*self.mlDensity*self.cp
        return
    
    @mlDensity.setter
    def mlDensity(self, mlDensity):
        self._mlDensity = mlDensity
        
        # Update dependent properties
        if not callable(self.cp):
            self.C = self.mlDepth*self.mlDensity*self.cp
        return
    
    @cp.setter
    def cp(self, cp):
        self._cp = cp
        
        # Update dependent properties
        if not callable(self.cp):
            self.C = self.mlDepth*self.mlDensity*self.cp
        
        return
    
    @vWind.setter
    def vWind(self, vWind):
        self._vWind = vWind
        
        if self.vWind == None:
            self.wind_dlon = 0.
        else:
            omegaWind = self.vWind/self.rad # radians/s
            self.wind_dlon = omegaWind/(self.map.dlon*np.pi/180)
            upwindIndex = np.roll(np.arange(self.map.nlon), int(np.sign(self.wind_dlon)))
            self.upwindLonIndex = (np.ones(self.map.nlat).reshape(-1,1)*upwindIndex).astype(int)
            self.upwindLatIndex = (np.arange(self.map.nlat).reshape(-1,1)*np.ones(self.map.nlon).reshape(1,-1)).astype(int)
        
        return
    
    
    
    
    
    def Fout(self, T=None, bolo=True, wav=1e-6):
        """Calculate the instantaneous outgoing flux.
        
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
        
        if bolo:
            return self.emissivity*const.sigma_sb.value*T**self.T_exponent
        else:
            a = (2*const.h.value*const.c.value**2/wav**5)
            b = (const.h.value*const.c.value)/(wav*const.k_B.value)
            return self.emissivity*a/np.expm1(b/T)
        
    def weight(self, t, TA=None, refPos='SSP'):
        """Calculate the weighting of map pixels.
        
        Weight flux by visibility/illumination kernel, assuming the star/observer are infinitely far away for now.
        
        Args:
            t (ndarray): The time in days.
            EA (ndarray, optional): The eccentric anomaly in radians.
            refPos (str, optional): The reference position; SSP (sub-stellar point) or SOP (sub-observer point).
        
        Returns:
            ndarray: The weighting of map mixels at time t. Has shape (t.size, self.map.npix).
        
        """
        
        if type(t)!=np.ndarray or len(t.shape)<3:
            t = np.array(t).reshape(-1,1,1)
        if TA is not None and (type(TA)!=np.ndarray or len(TA.shape)<3):
            TA = np.array(TA).reshape(-1,1,1)
        
        if refPos.lower() == 'ssp':
            refLon, refLat = self.orbit.get_ssp(t, TA)
        elif refPos.lower() == 'sop':
            refLon, refLat = self.orbit.get_sop(t)
        else:
            print('Reference point "'+str(refPos)+'" not understood!')
            return False
        
        weight = (np.cos(self.map.latGrid_radians[np.newaxis,:])
                  *np.cos(refLat*np.pi/180.)
                  *np.cos((self.map.lonGrid_radians[np.newaxis,:]-refLon*np.pi/180.))
                  + np.sin(self.map.latGrid_radians[np.newaxis,:])
                    *np.sin(refLat*np.pi/180.))
        
        weight = np.max(np.append(np.zeros_like(weight[np.newaxis,:]), weight[np.newaxis,:], axis=0), axis=0)
        
        return weight

    
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
            T = self.map.values[np.newaxis,:]
        
        weights = self.weight(t, refPos='SOP')
        flux = self.Fout(T, bolo, wav)*self.map.pixArea*self.rad**2
        # used to try to remove wiggles from finite number of pixels coming in and out of view
        # weightsNormed = weights*(4.*np.pi/self.map.npix)/np.pi
        
        return np.sum(flux*weights, axis=(1,2))#/np.sum(weightsNormed, axis=1)

    def plot_map(self, tempMap=None, refLon=None):
        """A convenience routine to plot the planet's temperature map.
        
        Args:
            tempMap (ndarray): The temperature map (if None, use self.map.values).
            refLon (float, optional): The centre longitude used to rotate the map.
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        oldValues = self.map.values
        if tempMap is not None:
            self.map.set_values(tempMap)
        
        fig = self.map.plot_map(refLon)
        
        self.map.set_values(oldValues)
        
        return fig
    
    def plot_H2_dissociation(self, dissMap=None, refLon=None):
        """A convenience routine to plot the planet's H2 dissociation map.
        
        Args:
            dissMap (ndarray, optional): The H2 dissociation fraction values for the map (if None,
                use self.map.dissValues).
            refLon (float, optional): The centre longitude used to rotate the map.
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        oldValues = self.map.values
        oldDissValues = self.map.dissValues
        if dissMap is not None:
            self.map.set_values(oldValues, dissValues=dissMap)
        
        fig = self.map.plot_H2_dissociation(refLon)
        
        self.map.set_values(oldValues, dissValues=oldDissValues)
        
        return fig

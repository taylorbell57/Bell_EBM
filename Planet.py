# Author: Taylor Bell
# Last Update: 2018-10-31

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.constants as const

from KeplerOrbit import KeplerOrbit
from Map import Map
import H2_Dissociation_Routines as h2

class Planet(object):
    def __init__(self, plType='gas', rad=1, mass=1, a=0.03, Porb=None, Prot=None, vWind=0,
                 e=0, Omega=270, inc=90, argp=90, obliq=0, argobliq=0, t0=0, albedo=0, useHealpix=False, nside=16):
        
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
            self.density = 1e3             # kg/m^3
            self.C = self.mlDepth*self.density*self.cp
        elif self.plType.lower() == 'rock':
            #basalt
            self.cp = 0.84e3                # J/kg/K
            self.mlDepth = 0.5              # m
            self.density = 3e3              # kg/m^3
            self.C = self.mlDepth*self.density*self.cp
        elif self.plType.lower() == 'gas':
            # H2 atmo
            self.cp = 14.31e3              # J/kg/K
            self.mlDepth = 0.1e5           # Pa
            self.density = 1/self.g        # s^2/m
            self.C = self.mlDepth*self.density*self.cp
        elif self.plType.lower() == 'bell2018':
            # LTE H2+H atmo
            self.cp = h2.true_cp
            self.cpParams = None
            self.mlDepth = 0.1e5           # Pa
            self.density = 1/self.g        # s^2/m
        else:
            print('Planet type not accepted!')
            return False
        
        #Map Attributes
        self.nside = nside
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
        if self.Porb != None:
            self.orbit = KeplerOrbit(Porb=self.Porb, t0=self.t0, a=self.a,
                                  inc=self.inc, e=self.e, Omega=self.Omega,
                                  argp=self.argp)
        else:
            self.orbit = None
        
        #Rotation Rate Attributes
        if Prot == None:
            self.Prot_input = self.Porb
        else:
            self.Prot_input = Prot               # days
        if self.Prot_input == None:
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
        
        self.g = const.G.value*self.mass/self.rad**2
        
        if self.plType.lower() == 'gas':
            # H2 atmo
            self.density = 1/self.g        # s^2/m
            self.C = self.mlDepth*self.density*self.cp
        elif self.plType.lower() == 'bell2018':
            # LTE H2+H atmo
            self.density = 1/self.g        # s^2/m
        else:
            self.C = self.mlDepth*self.density*self.cp
        
        if self.Porb != None and self.Prot==None:
            self.Prot_input = self.Porb
        
        if self.Prot_input != None:
            self.vRot = 2*np.pi*self.rad/(self.Prot_input*24*3600) # m/s
            if self.vWind == 0:
                self.Prot = self.Prot_input
            else:
                self.Prot = 2*np.pi*self.rad/((self.vWind+self.vRot)*(24*3600))
        
        if self.Porb != None:
            self.orbit = KeplerOrbit(Porb=self.Porb, t0=self.t0, a=self.a,
                                  inc=self.inc, e=self.e, Omega=self.Omega,
                                  argp=self.argp)
    
    # Calculate the sub-stellar longitude and latitude
    def SSP(self, t):
        if type(t)!=np.ndarray or len(t.shape)!=1:
            t = np.array([t]).reshape(-1)
        trueAnom = (self.orbit.get_trueAnomaly(t)*180/np.pi)
        trueAnom[trueAnom<0] += 360
        ssp = (trueAnom-t/self.Prot*360)
        ssp = ssp%180+(-180)*(np.rint(np.floor(ssp%360/180) > 0))
        sspLat = self.obliq*np.cos(t/self.Porb*2*np.pi-self.argobliq*np.pi/180)
        return ssp.reshape(-1,1), sspLat.reshape(-1,1)

    # Calculate the sub-observer longitude and latitude
    def SOP(self, t):
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array(t).reshape(-1,1)
        sop = 180-(t/self.Prot)*360
        sop = sop%180+(-180)*(np.rint(np.floor(sop%360/180) > 0))
        sopLat = 90-self.inc-self.obliq*np.cos(t/self.Porb*2*np.pi-self.argobliq*np.pi/180)
        return sop, sopLat

    # Calculate the instantaneous total outgoing flux
    def Fout(self, T, bolo=True, wav=1e-6):
        if bolo:
            return const.sigma_sb.value*T**4
        else:
            a = (2*const.h.value*const.c.value**2/wav**5)
            b = (const.h.value*const.c.value)/(wav*const.k_B.value)
            return a/np.expm1(b/T)
        
    # Weight flux by visibility/illumination kernel (assuming the host star is infinitely far away for now)
    def weight(self, t, refPos='SSP'):
        if refPos == 'SSP':
            refLon, refLat = self.SSP(t)
        elif refPos == 'SOP':
            refLon, refLat = self.SOP(t)
        else:
            print('Reference point "'+str(refPos)+'" not understood!')
            return False
        
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array(t).reshape(-1,1)
        
        lonWeight = np.max(np.append(np.zeros((1,t.size,self.map.npix)), 
                                     np.cos((self.map.lonGrid-refLon)*np.pi/180)[np.newaxis,:,:], axis=0), axis=0)
        latWeight = np.cos((self.map.latGrid-refLat)*np.pi/180)[0]
        return lonWeight*latWeight
    
    # Calculate Fp_apparent (used for making phasecurves)
    def Fp_vis(self, t, T, bolo=True, wav=4.5e-6):
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array(t).reshape(-1,1)
        
        weights = self.weight(t, 'SOP')
        flux = self.Fout(T, bolo, wav)*self.map.pixArea*self.rad**2
        # used to remove wiggles from finite number of pixels coming in and out of view
        weightsNormed = weights*(4*np.pi/self.map.npix)/np.pi
        
        return np.sum(flux*weights, axis=1)/np.sum(weightsNormed, axis=1)

    # A Convenience Routine to Plot the Planet's Temperature Map
    def showMap(self, tempMap, time=0):     
        subStellarLon = self.SSP(time)[0].flatten()
        self.map.set_map(tempMap)
        self.map.plot_map(subStellarLon)
        return plt.gcf()
    
    # A Convenience Routine to Plot the Planet's Dissociation Map
    def showDissociation(self, tempMap, time=0):
        subStellarLon = self.SSP(time)[0].flatten()
        self.map.set_map(tempMap)
        self.map.plot_dissociation(subStellarLon)
        return plt.gcf()

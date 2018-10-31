# Author: Taylor Bell
# Last Update: 2018-10-31

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.constants as const
import healpy as hp
import numbers

from KeplerOrbit import *
import H2_Dissociation_Routines as h2

class Planet(object):
    def __init__(self, plType='rock', rad=1, mass=1, a=0.03, Porb=None, Prot=None, vWind=0,
                 e=0, Omega=270, inc=90, argp=90, t0=0, albedo=0, useHealpix=True, nside=5):
        
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
        self.useHealpix = useHealpix
        if self.useHealpix:
            self.nside = np.rint(nside).astype(int)
            self.npix = hp.nside2npix(self.nside)
            self.pixArea = hp.nside2pixarea(self.nside)*self.rad**2
            
            coords = np.empty((self.npix,2))
            for i in range(self.npix):
                coords[i,:] = np.array(hp.pix2ang(self.nside, i))*180/np.pi
            lon = coords[:,1]
            lat = coords[:,0]-90
            self.lat = lat.reshape(1,-1)
            self.lon = lon.reshape(1,-1)
        else:
            self.nside = np.rint(nside).astype(int)
            self.npix = self.nside*(2*self.nside+1)
            
            dlat = 180/self.nside
            lat = np.linspace(-90+dlat/2, 90-dlat/2, self.nside)
            latTop = lat+dlat/2
            latBot = lat-dlat/2

            dlon = 360/(self.nside*2+1)
            lon = np.linspace(-180+dlon/2, 180-dlon/2, self.nside*2+1)
            lonRight = lon+dlon/2
            lonLeft = lon-dlon/2
            
            latArea = np.abs(2*np.pi*self.rad**2*(np.sin(latTop*np.pi/180)-np.sin(latBot*np.pi/180)))
            areas = latArea.reshape(1,-1)*(np.abs(lonRight-lonLeft)/360).reshape(-1,1)
            lat, lon = np.meshgrid(lat, lon)

            self.pixArea = areas.reshape(1, -1)
            self.lat = lat.reshape(1,-1)
            self.lon = lon.reshape(1, -1)
            
                
        #Orbital Attributes
        self.a = a*const.au.value          # m
        self.Porb = Porb                   # days (if None, will be set to Kepler expectation)
        self.e = e                         # None
        self.Omega = Omega                 # degrees
        self.inc = inc                     # degrees
        self.argp = argp                   # degrees 
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
        
        if self.useHealpix:
            self.nside = np.rint(self.nside).astype(int)
            self.npix = hp.nside2npix(self.nside)
            self.pixArea = hp.nside2pixarea(self.nside)*self.rad**2
            
            coords = np.empty((self.npix,2))
            for i in range(self.npix):
                coords[i,:] = np.array(hp.pix2ang(self.nside, i))*180/np.pi
            lon = coords[:,1]
            lat = coords[:,0]-90
            self.lat = lat.reshape(1,-1)
            self.lon = lon.reshape(1,-1)
        else:
            self.nside = np.rint(self.nside).astype(int)
            self.npix = self.nside*(2*self.nside+1)
            
            dlat = 180/self.nside
            lat = np.linspace(-90+dlat/2, 90-dlat/2, self.nside)
            latTop = lat+dlat/2
            latBot = lat-dlat/2

            dlon = 360/(self.nside*2+1)
            lon = np.linspace(-180+dlon/2, 180-dlon/2, self.nside*2+1)
            lonRight = lon+dlon/2
            lonLeft = lon-dlon/2
            
            latArea = np.abs(2*np.pi*self.rad**2*(np.sin(latTop*np.pi/180)-np.sin(latBot*np.pi/180)))
            areas = latArea.reshape(1,-1)*(np.abs(lonRight-lonLeft)/360).reshape(-1,1)
            lat, lon = np.meshgrid(lat, lon)

            self.pixArea = areas.reshape(1, -1)
            self.lat = lat.reshape(1,-1)
            self.lon = lon.reshape(1, -1)
        
        if self.Porb != None:
            self.orbit = KeplerOrbit(Porb=self.Porb, t0=self.t0, a=self.a,
                                  inc=self.inc, e=self.e, Omega=self.Omega,
                                  argp=self.argp)
    
    # Calculate the sub-stellar longitude
    def SSP(self, t):
        if type(t)!=np.ndarray or len(t.shape)!=1:
            t = np.array([t]).reshape(-1)
        trueAnom = (self.orbit.get_trueAnomaly(t)*180/np.pi)
        trueAnom[trueAnom<0] += 360

        ssp = (trueAnom-t/self.Prot*360)
        ssp = ssp%180+(-180)*(np.rint(np.floor(ssp%360/180) > 0))
        return ssp.reshape(-1,1)

    # Calculate the sub-observer longitude
    def SOP(self, t):
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array(t).reshape(-1,1)
        
        sop = 180-(t/self.Prot)*360
        sop = sop%180+(-180)*(np.rint(np.floor(sop%360/180) > 0))
        return sop

    # Calculate the instantaneous total outgoing flux
    def Fout(self, T, bolo=True, wav=1e-6):
        if bolo:
            return const.sigma_sb.value*T**4
        else:
            a = (2*const.h.value*const.c.value**2/wav**5)
            b = (const.h.value*const.c.value)/(wav*const.k_B.value)
            return a/np.expm1(b/T)
        
    # Weight flux by visibility/illumination kernel (assuming the host star is infinitely far away for now)
    def weight(self, t, refLon='SSP', refLat=0):
        if refLon == 'SSP':
            refLon = self.SSP(t)
        elif refLon == 'SOP':
            refLon = self.SOP(t)
        elif not isinstance(refLon, numbers.Number) or type(refLon)==np.ndarray:
            print('Reference longitude "'+str(refLon)+'" not understood!')
            return False
        
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array(t).reshape(-1,1)
        
        lonWeight = np.max(np.append(np.zeros((1,t.size,self.lon.size)), 
                                     np.cos((self.lon-refLon)*np.pi/180)[np.newaxis,:,:], axis=0), axis=0)
        latWeight = np.cos((self.lat-refLat)*np.pi/180)[0]
        return lonWeight*latWeight
    
    # Calculate Fp_apparent (used for making phasecurves)
    def Fp_vis(self, t, T, bolo=True, wav=4.5e-6):
        refLat = (90-self.inc)

        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array(t).reshape(-1,1)
        
        weights = self.weight(t, 'SOP', refLat)
        flux = self.Fout(T, bolo, wav)*self.pixArea
        # used to remove wiggles from finite number of pixels coming in and out of view
        weightsNormed = weights*(4*np.pi/self.npix)/np.pi
        
        return np.sum(flux*weights, axis=1)/np.sum(weightsNormed, axis=1)

    # A Convenience Routine to Plot the Planet's Temperature Map
    def showMap(self, tempMap, time=0):
        current_cmap = matplotlib.cm.get_cmap('inferno')
        current_cmap.set_bad(color='white')
        if self.useHealpix:
            im = hp.orthview(tempMap, flip='geo', cmap='inferno',
                             rot=(self.SSP(time).flatten(), 0, 0), return_projected_map=True)
            plt.clf()
            plt.imshow(im, cmap='inferno')
            plt.xticks([])
            plt.yticks([])
            plt.setp(plt.gca().spines.values(), color='none')
        else:
            tempMap = tempMap.reshape((self.nside, int(2*self.nside+1)), order='F')
            
            ssp = self.SSP(time).flatten()
            dlon = 360/(self.nside*2+1)
            lon = np.linspace(-180+dlon/2, 180-dlon/2, self.nside*2+1)
            rotInd = -(np.where(np.abs(lon-ssp) < dlon/2+1e-6)[0][0]-(self.nside))
            tempMap = np.roll(tempMap, rotInd, axis=1)
            
            plt.imshow(tempMap, cmap='inferno', extent=(-180,180,-90,90))
            plt.xticks([-180,-90,0,90,180])
            plt.yticks([-90,-45,0,45,90])
            
        cbar = plt.colorbar(orientation="horizontal",fraction=0.075)
        cbar.set_label('Temperature (K)', fontsize='x-large')
        
        return plt.gcf()
    
    # A Convenience Routine to Plot the Planet's Dissociation Map
    def showDissociation(self, tempMap, time=0):
        current_cmap = matplotlib.cm.get_cmap('inferno')
        current_cmap.set_bad(color='white')
        if self.useHealpix:
            im = hp.orthview(h2.dissFracApprox(tempMap)*100., flip='geo', cmap='inferno', min=0,
                             rot=(self.SSP(time).flatten(), 0, 0), return_projected_map=True)
            plt.clf()
            plt.imshow(im, cmap='inferno', vmin=0)
            plt.xticks([])
            plt.yticks([])
            plt.setp(plt.gca().spines.values(), color='none')
        else:
            dissMap = h2.dissFracApprox(tempMap.reshape((self.nside, int(2*self.nside+1)), order='F'))*100.
            
            ssp = self.SSP(time).flatten()
            dlon = 360/(self.nside*2+1)
            lon = np.linspace(-180+dlon/2, 180-dlon/2, self.nside*2+1)
            rotInd = -(np.where(np.abs(lon-ssp) < dlon/2+1e-6)[0][0]-(self.nside))
            dissMap = np.roll(dissMap, rotInd, axis=1)
            
            plt.imshow(dissMap, cmap='inferno', extent=(-180,180,-90,90), vmin=0)
            plt.xticks([-180,-90,0,90,180])
            plt.yticks([-90,-45,0,45,90])
        
        cbar = plt.colorbar(orientation="horizontal",fraction=0.075)
        cbar.set_label('Dissociation Fraction (%)', fontsize='x-large')
        
        return plt.gcf()

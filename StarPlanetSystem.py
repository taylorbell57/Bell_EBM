# Author: Taylor Bell
# Last Update: 2018-10-31

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import scipy.integrate

from Star import Star
from Planet import Planet
from KeplerOrbit import KeplerOrbit

class System(object):
    def __init__(self, star=None, planet=None):
        if star == None:
            self.star = Star()
        else:
            self.star = star
        
        if planet == None:
            self.planet = Planet()
        else:
            self.planet = planet
            
        updatePlanet = False
        if self.planet.Porb == None:
            self.planet.orbit = KeplerOrbit(t0=self.planet.t0, a=self.planet.a, inc=self.planet.inc,
                                         e=self.planet.e, Omega=self.planet.Omega, argp=self.planet.argp,
                                         m1=self.star.mass, m2=self.planet.mass)
            self.planet.Porb = self.planet.orbit.Porb
            updatePlanet = True
        if self.planet.Prot_input == None:
            self.planet.Prot_input = self.planet.Porb
            self.planet.Prot = self.planet.Porb
            updatePlanet = True
            
        if updatePlanet:
            planet.update()
        
    # Get the orbital phase of periastron
    def get_phase_periastron(self):
        return (self.planet.orbit.get_peri_time()/self.planet.Porb) % 1
    
    # Get the orbital phase of transit
    def get_phase_transit(self):
        return 0
    
    # Get the orbital phase of eclipse
    def get_phase_eclipse(self):
        return (self.planet.orbit.get_ecl_time()/self.planet.Porb) % 1
    
    # Get the orbital phase of the planet at time(s) t
    def get_phase(self, t):
        return ((t-self.planet.t0)/self.Porb) % 1
    
    # Get the x,y,z coordinate(s) of the planet at time(s) t
    def get_xyzPos(self, t):
        return self.planet.orbit.get_xyz(t)
            
    # Calculate the instantaneous separation between star and planet at time(s) t
    def distance(self, t):
        if type(t)!=np.ndarray or len(t.shape)!=1:
            t = np.array([t]).reshape(-1)
        return self.planet.orbit.get_distance(t).reshape(-1,1)

    # Calculate the instantaneous irradiation at time(s) t
    def Firr(self, t):
        return self.star.Fstar()/(np.pi*self.distance(t)**2)

    # Calculate the instantaneous incident flux at time(s) t
    def Finc(self, t):
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array(t).reshape(-1,1)
        return self.Firr(t)*self.planet.weight(t)

    # Calculate the planet's lightcurve (ignoring any occultations)
    def lightcurve(self, t, T, bolo=True, tStarBright=None, wav=4.5e-6):
        return self.planet.Fp_vis(t, T, bolo, wav)/self.star.Fstar(bolo, tStarBright, wav)
    
    # Invert the fp/fstar phasecurve into an apparent temperature phasecurve
    def invert_lc(self, fp_fstar, bolo=True, tStarBright=None, wav=4.5e-6):
        if bolo:
            return (fp_fstar*self.star.Fstar(bolo=True)/(np.pi*self.planet.rad**2)/const.sigma_sb.value)**0.25
        else:
            if tStarBright==None:
                tStarBright = self.star.teff
            a = const.h.value*const.c.value/(const.k_B.value*wav)
            b = np.expm1(a/tStarBright)
            c = 1  +  b/(fp_fstar/(self.planet.rad/self.star.rad**2))
            return a*np.log(c)**-1
    
    #dT/dt - used by scipy.integrate.ode
    def ODE(self, t, T):
        CdT_dt = (24*3600)*(self.Finc(t)-self.planet.Fout(T))
        if not callable(self.planet.cp):
            return CdT_dt/self.planet.C
        else:
            if self.planet.cpParams == None:
                return (CdT_dt/(self.planet.mlDepth*self.planet.density*self.planet.cp(T)))
            else:
                return (CdT_dt/(self.planet.mlDepth*self.planet.density
                                *self.planet.cp(T, *self.planet.cpParams)))

    # Run the model - can be used to burn in temperature map or make a phasecurve
    def runModel(self, T0, t0, t1, dt, verbose=True):
        r = scipy.integrate.ode(self.ODE)
        r.set_initial_value(T0, t0)

        if verbose:
            print('Starting Run')
        times = []
        maps = []
        while r.successful() and r.t <= t1-dt:
            times.append(r.t+dt)
            maps.append(np.max(np.append(np.zeros((self.planet.map.npix,1)),
                                         r.integrate(r.t+dt).reshape(-1,1), axis=1), axis=1))
        times = np.array(times)
        maps = np.array(maps)

        if len(times) < 10:
            if verbose:
                print('Failed: Trying a smaller time step!')
            dt /= 10
            times = []
            maps = []
            while r.successful() and r.t <= t1-dt:
                times.append(r.t+dt)
                maps.append(np.max(np.append(np.zeros((self.planet.map.npix,1)),
                                             r.integrate(r.t+dt).reshape(-1,1), axis=1), axis=1))
            times = np.array(times)
            maps = np.array(maps)

        if len(times) < 10:
            print('Failed to run the model!')
        if verbose:
            print('Done!')
        
        return times, maps

    # A convenience plotting routine to show the planet's phasecurve
    def plot_lightcurve(self, t, T, bolo=True, tStarBright=None, wav=4.5e-6):
        x = t/self.planet.Porb - np.rint(t[0]/self.planet.Porb)
        t = np.append(t[x>=0], t[x<0])
        T = np.append(T[x>=0], T[x<0], axis=0)
        x = np.append(x[x>=0], x[x<0]+1)
        if self.planet.e != 0:
            x *= self.planet.Porb
        
        lc = self.lightcurve(t, T, bolo=bolo, tStarBright=tStarBright, wav=wav)*1e6
        
        plt.plot(x, lc)
        plt.gca().axvline(self.get_phase_eclipse()*np.max(x), c='k', ls='--', label='Eclipse')
        if self.planet.e != 0 and self.get_phase_eclipse()!=self.get_phase_periastron():
            plt.gca().axvline(self.get_phase_periastron()*np.max(x), c='red', ls='--', label='Periastron')

        plt.legend(loc=8, bbox_to_anchor=(0.5,1), fontsize='x-large', ncol=2)
        plt.ylabel(r'$F_p/F_*$ (ppm)', fontsize='xx-large')
        if self.planet.e == 0:
            plt.xlabel('Orbital Phase', fontsize='xx-large')
        else:
            plt.xlabel('Time from Transit (days)', fontsize='xx-large')
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(0)
        plt.setp(plt.gca().get_xticklabels(), fontsize='x-large')
        plt.setp(plt.gca().get_yticklabels(), fontsize='x-large')
        return plt.gcf()
    
    # A convenience plotting routine to show the planet's phasecurve in units of temperature
    def plot_tempcurve(self, t, T, bolo=True, tStarBright=None, wav=4.5e-6):
        x = t/self.planet.Porb - np.rint(t[0]/self.planet.Porb)
        t = np.append(t[x>=0], t[x<0])
        T = np.append(T[x>=0], T[x<0], axis=0)
        x = np.append(x[x>=0], x[x<0]+1)
        if self.planet.e != 0:
            x *= self.planet.Porb
        
        lc = self.lightcurve(t, T, bolo=bolo, tStarBright=tStarBright, wav=wav)
        tc = self.invert_lc(lc, bolo=bolo, tStarBright=tStarBright, wav=wav)
        
        plt.plot(x, tc)
        plt.gca().axvline(self.get_phase_eclipse()*np.max(x), c='k', ls='--', label='Eclipse')
        if self.planet.e != 0 and self.get_phase_eclipse()!=self.get_phase_periastron():
            plt.gca().axvline(self.get_phase_periastron()*np.max(x), c='red', ls='--', label='Periastron')

        plt.legend(loc=8, bbox_to_anchor=(0.5,1), fontsize='x-large', ncol=2)
        if bolo:
            plt.ylabel(r'$T_{\rm eff, hemi, apparent}$ (K)', fontsize='xx-large')
        else:
            plt.ylabel(r'$T_{\rm b, hemi, apparent}$ (K)', fontsize='xx-large')
        if self.planet.e == 0:
            plt.xlabel('Orbital Phase', fontsize='xx-large')
        else:
            plt.xlabel('Time from Transit (days)', fontsize='xx-large')
        plt.xlim(np.min(x), np.max(x))
        plt.ylim(0)
        plt.setp(plt.gca().get_xticklabels(), fontsize='x-large')
        plt.setp(plt.gca().get_yticklabels(), fontsize='x-large')
        return plt.gcf()

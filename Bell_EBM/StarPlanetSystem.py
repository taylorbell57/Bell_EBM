# Author: Taylor Bell
# Last Update: 2018-11-01

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import scipy.integrate

from .Star import Star
from .Planet import Planet
from .KeplerOrbit import KeplerOrbit



class System(object):
    """A Star+Planet System.

    Attributes:
        star (Star): The host star.
        planet (Planet): The planet.
    
    """

    def __init__(self, star=None, planet=None):
        """Initialization function.
        
        Attributes:
            star (Star, optional): The host star.
            planet (Planet, optional): The planet.

        """
        
        if star is None:
            self.star = Star()
        else:
            self.star = star
        
        if planet is None:
            self.planet = Planet()
        else:
            self.planet = planet
            
        updatePlanet = False
        if self.planet.Porb is None:
            self.planet.orbit = KeplerOrbit(t0=self.planet.t0, a=self.planet.a, inc=self.planet.inc,
                                         e=self.planet.e, Omega=self.planet.Omega, argp=self.planet.argp,
                                         m1=self.star.mass, m2=self.planet.mass)
            self.planet.Porb = self.planet.orbit.Porb
            updatePlanet = True
        if self.planet.Prot_input is None:
            self.planet.Prot_input = self.planet.Porb
            self.planet.Prot = self.planet.Porb
            updatePlanet = True
            
        if updatePlanet:
            planet.update()
    
    def get_phase_periastron(self):
        """Get the orbital phase of periastron.
        
        Returns:
            float: The orbital phase of periastron.
            
        """
        
        return (self.planet.orbit.peri_time()/self.planet.Porb) % 1
    
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
        
        return (self.planet.orbit.ecl_time()/self.planet.Porb) % 1
    
    def get_phase(self, t):
        """Get the orbital phase.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            ndarray: The orbital phase.
            
        """
        
        return ((t-self.planet.t0)/self.Porb) % 1
    
    def get_xyzPos(self, t):
        """Get the x,y,z coordinate(s) of the planet.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            list: A list of 3 ndarrays containing the x,y,z coordinate of the planet with respect to the star.
            
                The x coordinate is along the line-of-sight.
                The y coordinate is perpendicular to the line-of-sight and in the orbital plane.
                The z coordinate is perpendicular to the line-of-sight and above the orbital plane
            
        """
        
        return self.planet.orbit.xyz(t)
    
    def distance(self, t):
        """Calculate the instantaneous separation between star and planet.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            ndarray: The separation between the star and planet in m.
            
        """
        
        if type(t)!=np.ndarray or len(t.shape)!=1:
            t = np.array([t]).reshape(-1)
        return self.planet.orbit.distance(t).reshape(-1,1)
    
    def Firr(self, t):
        """Calculate the instantaneous irradiation.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            ndarray: The instantaneous irradiation.
            
        """
        
        return self.star.Fstar()/(np.pi*self.distance(t)**2)

    def Finc(self, t):
        """Calculate the instantaneous incident flux.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            ndarray: The instantaneous incident flux.
            
        """
        
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array([t]).reshape(-1,1)
        return self.Firr(t)*self.planet.weight(t)

    def lightcurve(self, t, T=None, bolo=True, tStarBright=None, wav=4.5e-6, debug=False):
        """Calculate the planet's lightcurve (ignoring any occultations).
        
        Args:
            t (ndarray): The time in days.
            T (ndarray): The temperature map (either shape (1, self.planet.map.npix) and
                constant over time or shape is (t.shape, self.planet.map.npix). If None,
                use self.planet.map.values instead (default).
            bolo (bool, optional): Determines whether computed flux is bolometric
                (True, default) or wavelength dependent (False).
            tBright (ndarray): The brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
        
        Returns:
            ndarray: The observed planetary flux normalized by the stellar flux.
            
        """
        
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array([t]).reshape(-1,1)
        
        if T is None:
            T = self.planet.map.values
        
        if len(T.shape)==1:
            T = T.reshape(1,-1)
        
        return self.planet.Fp_vis(t, T, bolo, wav, debug=debug)/self.star.Fstar(bolo, tStarBright, wav)
    
    def invert_lc(self, fp_fstar, bolo=True, tStarBright=None, wav=4.5e-6):
        """Invert the fp/fstar phasecurve into an apparent temperature phasecurve.
        
        Args:
            fp_fstar (ndarray): The observed planetary flux normalized by the stellar flux.
            bolo (bool, optional): Determines whether computed flux is bolometric (True, default)
                or wavelength dependent (False).
            tBright (ndarray): The brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
            
        Returns:
            ndarray: The apparent, disk-integrated temperature.
            
        """
        
        if bolo:
            return (fp_fstar*self.star.Fstar(bolo=True)/(np.pi*self.planet.rad**2)/const.sigma_sb.value)**0.25
        else:
            if tStarBright is None:
                tStarBright = self.star.teff
            a = const.h.value*const.c.value/(const.k_B.value*wav)
            b = np.expm1(a/tStarBright)
            c = 1  +  b/(fp_fstar/(self.planet.rad/self.star.rad**2))
            return a*np.log(c)**-1
    
    def ODE(self, t, T):
        """The derivative in temperature with respect to time.
        
        Used by scipy.integrate.ode to update the map
        
        Args:
            t (ndarray): The time in days.
            T (ndarray): The temperature map with shape (self.planet.map.npix).
        
        Returns:
            ndarray: The derivative in temperature with respect to time.
            
        """
        
        CdT_dt = (24*3600)*(self.Finc(t)-self.planet.Fout(T))
        
        if not callable(self.planet.cp):
            C = self.planet.C
        else:
            if self.planet.cpParams is None:
                C = (self.planet.mlDepth*self.planet.mlDensity*self.planet.cp(T))
            else:
                C = (self.planet.mlDepth*self.planet.mlDensity*self.planet.cp(T, *self.planet.cpParams))
        return CdT_dt/C

    # Run the model - can be used to burn in temperature map or make a phasecurve
    def runModel(self, T0=None, t0=0, t1=None, dt=None, verbose=True):
        """Evolve the planet's temperature map with time.
        
        Args:
            T0 (ndarray): The initial temperature map with shape (self.planet.map.npix).
                If None, use self.planet.map.values instead (default).
            t0 (float, optional): The time corresponding to T0 (default is 0).
            t1 (float, optional): The end point of the run (default is 1 orbital period later).
            dt (float, optional): The time step used to evolve the map (default is 1/100 of the orbital period).
            verbose (bool, optional): Output comments of the progress of the run (default = False).
        
        Returns:
            list: A list of 2 ndarrays containing the time and map of each time step.
            
        """
        
        if T0 is None:
            T0 = self.planet.map.values
        if t1 is None:
            t1 = t0+self.planet.Porb
        if dt is None:
            dt = self.planet.Porb/100.
        
        r = scipy.integrate.ode(self.ODE)
        r.set_initial_value(T0, t0)

        if verbose:
            print('Starting Run')
        times = np.array([t0]).reshape(1,1)
        maps = np.array([T0]).reshape(1,-1)
        while r.successful() and r.t <= t1-dt:
            times = np.append(times, np.array(r.t+dt).reshape(1,1), axis=0)
            maps = np.append(maps, np.max(np.append(np.zeros((self.planet.map.npix,1)),
                                          r.integrate(r.t+dt).reshape(-1,1), axis=1), axis=1).reshape(1,-1),
                             axis=0)
        
        if len(times) < np.floor((t1-t0)/dt):
            if verbose:
                print('Failed: Trying a smaller time step!')
            dt /= 10
            times = np.array([t0]).reshape(1,1)
            maps = np.array([T0]).reshape(1,-1)
            while r.successful() and r.t <= t1-dt:
                times = np.append(times, np.array(r.t+dt).reshape(1,1), axis=0)
                maps = np.append(maps, np.max(np.append(np.zeros((self.planet.map.npix,1)),
                                              r.integrate(r.t+dt).reshape(-1,1), axis=1), axis=1).reshape(1,-1),
                                 axis=0)

        if len(times) < np.floor((t1-t0)/dt):
            print('Failed to run the model!')
        if verbose:
            print('Done!')
        
        self.planet.map.set_values(maps[-1], times[-1,0])
        
        return times, maps

    def plot_lightcurve(self, t=None, T=None, bolo=True, tStarBright=None, wav=4.5e-6):
        """A convenience plotting routine to show the planet's phasecurve.
        
        Args:
            t (ndarray, optional): The time in days with shape (t.size,1). If none, use
                [self.planet.t0,self.planet.t0+self.planet.Porb].
            T (ndarray, optional): The temperature map in K with shape (1, self.planet.map.npix)
                if the map is constant or (t.size,self.planet.map.npix). If None, use
                self.planet.map.values instead.
            bolo (bool, optional): Determines whether computed flux is bolometric (True, default)
                or wavelength dependent (False).
            tBright (ndarray): The brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
        
        Returns:
            figure: The figure containing the plot.
            
        """
        
        if t is None:
            # Use Prot instead as map would rotate
            t = self.planet.t0+np.linspace(0, self.planet.Prot, 1000)
            x = t/self.planet.Prot - np.rint(t[0]/self.planet.Prot)
        else:
            x = t/self.planet.Porb - np.rint(t[0]/self.planet.Porb)
        
        if T is None:
            T = self.planet.map.values.reshape(1,-1)
        elif type(T)!=np.ndarray:
            T = np.array([T]).reshape(1,-1)
        elif len(T.shape)==1:
            T = T.reshape(1,-1)
        
        lc = self.lightcurve(t, T, bolo=bolo, tStarBright=tStarBright, wav=wav)*1e6
        
        t = np.append(t[x>=0], t[x<0])
        lc = np.append(lc[x>=0], lc[x<0])
        x = np.append(x[x>=0], x[x<0]+1)
        if self.planet.e != 0:
            x *= self.planet.Porb
        
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
    
    def plot_tempcurve(self, t=None, T=None, bolo=True, tStarBright=None, wav=4.5e-6):
        """A convenience plotting routine to show the planet's phasecurve in units of temperature.
        
        Args:
            t (ndarray, optional): The time in days with shape (t.size,1). If none, use
                [self.planet.t0,self.planet.t0+self.planet.Porb].
            T (ndarray, optional): The temperature map in K with shape (1, self.planet.map.npix) if
                the map is constant or (t.size,self.planet.map.npix). If None, use
                self.planet.map.values instead.
            bolo (bool, optional): Determines whether computed flux is bolometric (True, default)
                or wavelength dependent (False).
            tBright (ndarray): The brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
        
        Returns:
            figure: The figure containing the plot.
            
        """
        
        if t is None:
            # Use Prot instead as map would rotate
            t = self.planet.t0+np.linspace(0, self.planet.Prot, 1000)
            x = t/self.planet.Prot - np.rint(t[0]/self.planet.Prot)
        else:
            x = t/self.planet.Porb - np.rint(t[0]/self.planet.Porb)
        
        if T is None:
            T = self.planet.map.values.reshape(1,-1)
        elif type(T)!=np.ndarray:
            T = np.array([T]).reshape(1,-1)
        elif len(T.shape)==1:
            T = T.reshape(1,-1)
        
        lc = self.lightcurve(t, T, bolo=bolo, tStarBright=tStarBright, wav=wav)
        tc = self.invert_lc(lc, bolo=bolo, tStarBright=tStarBright, wav=wav)
        
        t = np.append(t[x>=0], t[x<0])
        tc = np.append(tc[x>=0], tc[x<0])
        x = np.append(x[x>=0], x[x<0]+1)
        if self.planet.e != 0:
            x *= self.planet.Porb
        
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

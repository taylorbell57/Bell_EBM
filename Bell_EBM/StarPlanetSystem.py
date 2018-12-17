# Author: Taylor Bell
# Last Update: 2018-12-12

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import scipy.integrate
import scipy.optimize as spopt
import warnings

from .Star import Star
from .Planet import Planet
from .KeplerOrbit import KeplerOrbit
from . import H2_Dissociation_Routines as h2


class System(object):
    """A Star+Planet System.

    Attributes:
        star (Bell_EBM.Star): The host star.
        planet (Bell_EBM.Planet): The planet.
    
    """

    def __init__(self, star=None, planet=None, neq=False):
        """Initialization function.
        
        Attributes:
            star (Bell_EBM.Star, optional): The host star.
            planet (Bell_EBM.Planet, optional): The planet.
            neq (bool, optional): Whether or not to use non-equilibrium ODE.

        """
        
        if star is None:
            self.star = Star()
        else:
            self.star = star
        
        if planet is None:
            self.planet = Planet()
        else:
            self.planet = planet
        
        self.neq = neq
        if self.planet.plType == 'bell2018' and neq:
            self.ODE = self.ODE_NEQ
        else:
            self.ODE = self.ODE_EQ
        
        self.planet.orbit.m1 = self.star.mass
    
    def get_phase_periastron(self):
        """Get the orbital phase of periastron.
        
        Returns:
            float: The orbital phase of periastron.
            
        """
        
        return self.planet.orbit.phase_periastron
    
    
    def get_phase_transit(self):
        """Get the orbital phase of transit.
        
        Returns:
            float: The orbital phase of transit.
            
        """
        
        return 0.
    
    
    def get_phase_eclipse(self):
        """Get the orbital phase of eclipse.
        
        Returns:
            float: The orbital phase of eclipse.
            
        """
        
        return self.planet.orbit.phase_eclipse
    
    
    def get_phase(self, t):
        """Get the orbital phase.
        
        Args:
            t (ndarray): The time in days.
        
        Returns:
            ndarray: The orbital phase.
            
        """
        
        return self.planet.orbit.get_phase(t)
    
    def get_teq(self, t=0):
        """Get the planet's equilibrium temperature.
        
        Args:
            t (ndarray, optional): The time in days.
        
        Returns:
            ndarray: The planet's equilibrium temperature at time(s) t.
            
        """
        return 0.25**0.25*self.get_tirr(t)
    
    def get_tirr(self, t=0.):
        """Get the planet's irradiation temperature.
        
        Args:
            t (ndarray, optional): The time in days.
        
        Returns:
            ndarray: The planet's irradiation temperature at time(s) t.
            
        """
        
        if self.planet.orbit.e == 0:
            dist = self.planet.orbit.a*np.ones_like(t)
        else:
            dist = self.planet.orbit.distance(t)
        
        if type(t) == float or type(t) == int:
            dist = float(dist)
        
        return self.star.teff*np.sqrt(self.star.rad/dist)
    
    def Firr(self, t=0., TA=None, bolo=True, tStarBright=None, wav=4.5e-6):
        """Calculate the instantaneous irradiation.
        
        Args:
            t (ndarray, optional): The time in days.
            TA (ndarray, optional): The true anomaly in radians.
            bolo (bool, optional): Determines whether computed flux is bolometric
                (True, default) or wavelength dependent (False).
            tStarBright (ndarray): The stellar brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
        
        Returns:
            ndarray: The instantaneous irradiation.
            
        """
        
        # Just grab semi-major axis for circular orbits to speed things up
        if self.planet.orbit.e == 0:
            dist = self.planet.orbit.a*np.ones_like(t)
        else:
            dist = self.planet.orbit.distance(t, TA)
            
        firr = self.planet.absorptivity*self.star.Fstar(bolo, tStarBright, wav)/(np.pi*dist**2)
        
        return firr

    def Fin(self, t=0, TA=None, bolo=True, tStarBright=None, wav=4.5e-6):
        """Calculate the instantaneous incident flux.
        
        Args:
            t (ndarray, optional): The time in days.
            TA (ndarray, optional): The true anomaly in radians.
            bolo (bool, optional): Determines whether computed flux is bolometric
                (True, default) or wavelength dependent (False).
            tStarBright (ndarray): The stellar brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
        
        Returns:
            ndarray: The instantaneous incident flux.
            
        """
        
        return self.Firr(t, TA, bolo, tStarBright, wav)*self.planet.weight(t, TA)

    def lightcurve(self, t=None, T=None, bolo=True, tStarBright=None, wav=4.5e-6, allowReflect=True, allowThermal=True):
        """Calculate the planet's lightcurve (ignoring any occultations).
        
        Args:
            t (ndarray, optional): The time in days. If None, will use 1000 time steps around orbit.
            T (ndarray, optional): The temperature map (either shape (1, self.planet.map.npix) and
                constant over time or shape is (t.shape, self.planet.map.npix). If None,
                use self.planet.map.values instead (default).
            bolo (bool, optional): Determines whether computed flux is bolometric
                (True, default) or wavelength dependent (False).
            tStarBright (ndarray): The stellar brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
            allowReflect (bool, optional): Account for the contribution from reflected light.
            allowThermal (bool, optional): Account for the contribution from thermal emission.
        
        Returns:
            ndarray: The observed planetary flux normalized by the stellar flux.
            
        """
        
        if t is None:
            # Use Prot instead as map would rotate
            t = self.planet.orbit.t0+np.linspace(0., self.planet.orbit.Prot, 1000)
            x = t/self.planet.orbit.Prot - np.rint(t[0]/self.planet.orbit.Prot)
        
        if type(t)!=np.ndarray or len(t.shape)==1:
            t = np.array([t]).reshape(-1,1)
        
        if T is None:
            T = self.planet.map.values
        
        if len(T.shape)==1:
            T = T.reshape(1,-1)
        
        if allowThermal:
            fp = self.planet.Fp_vis(t, T, bolo, wav)
        else:
            fp = np.zeros_like(t.flatten())
        
        if allowReflect:
            fRefl = self.Fin(t, None, bolo, tStarBright, wav)
            fRefl *= self.planet.albedo/self.planet.absorptivity # Get only the reflected portion
            fRefl = fRefl*self.planet.weight(t, refPos='SOP')*self.planet.map.pixArea*self.planet.rad**2
            fRefl = np.sum(fRefl, axis=1)
            fp += fRefl
        
        return fp/self.star.Fstar(bolo, tStarBright, wav)
    
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
            return (fp_fstar*self.star.Fstar(bolo)/(np.pi*self.planet.rad**2.)/const.sigma_sb.value)**0.25
        else:
            if tStarBright is None:
                tStarBright = self.star.teff
            a = const.h.value*const.c.value/(const.k_B.value*wav)
            b = np.expm1(a/tStarBright)
            c = 1  +  b/(fp_fstar/(self.planet.rad/self.star.rad)**2)
            return a*np.log(c)**-1
    
    def ODE_EQ(self, t, T, dt, TA=None):
        """The derivative in temperature with respect to time.
        
        Used by scipy.integrate.ode to update the map
        
        Args:
            t (ndarray): The time in days.
            T (ndarray): The temperature map with shape (self.planet.map.npix).
            TA (ndarray, optional): The true anomaly in radians (much faster to compute if provided).
        
        Returns:
            ndarray: The derivative in temperature with respect to time.
            
        """
        
        dt *= 24.*3600.
        
        dEs = (self.Fin(t, TA)-self.planet.Fout(T))*dt
        
        if not callable(self.planet.cp):
            C = self.planet.C
        else:
            if self.planet.cpParams is None:
                C = (self.planet.mlDepth*self.planet.mlDensity*self.planet.cp(T))
            else:
                C = (self.planet.mlDepth*self.planet.mlDensity*self.planet.cp(T, *self.planet.cpParams))
        
        return dEs/C
    
    def _find_dT(self, dT, dE, T0, chi0, plug, cp):
        """The error function to minimize to find the energy partitioning between dT and dDiss.
        
        """
        
        dDiss = h2.dissFracApprox(T0+dT)-chi0
        dT_diss = dDiss*h2.dissE*plug
        return (dE-(dT*cp*plug+dT_diss))**2
    
    def ODE_NEQ(self, t, T, dt, TA=None):
        """The derivative in temperature with respect to time.
        
        Used by scipy.integrate.ode to update the map
        
        Args:
            t (ndarray): The time in days.
            T (ndarray): The temperature map with shape (self.planet.map.npix).
            dt (float): The timestep in days.
            TA (ndarray, optional): The true anomaly in radians (much faster to compute if provided).
        
        Returns:
            ndarray: The derivative in temperature with respect to time.
            
        """
        
        dt *= 24.*3600.
        
        plug = self.planet.mlDepth*self.planet.mlDensity
        cp = h2.lte_cp(T)
        
        dEs = (self.Fin(t, TA)-self.planet.Fout(T)).flatten()*dt
        
        dTs = []
        for i in range(dEs.size):
            dTs.append(spopt.minimize(self._find_dT, x0=dEs[i]/2./plug/cp[i],
                                      args=(dEs[i], T[i], self.planet.map.dissValues[i], plug, cp[i]),
                                      tol=0.001*plug*cp[i]).x[0])
        dTs = np.array(dTs)
        dDiss = h2.dissFracApprox(T+dTs)-self.planet.map.dissValues
        
        bad = np.where(dDiss > dt/h2.tau_chem(self.planet.mlDepth,T))
        dDiss[bad] = dt/h2.tau_chem(self.planet.mlDepth,T)
        dTs[bad] = dDiss[bad]*h2.dissE/cp[bad]-dEs[bad]/cp[bad]/plug
        
        self.planet.map.dissValues += dDiss
        
        return dTs

    def run_model(self, T0=None, t0=0., t1=None, dt=None, verbose=True, intermediates=False):
        """Evolve the planet's temperature map with time.
        
        Args:
            T0 (ndarray): The initial temperature map with shape (self.planet.map.npix).
                If None, use self.planet.map.values instead (default).
            t0 (float, optional): The time corresponding to T0 (default is 0).
            t1 (float, optional): The end point of the run (default is 1 orbital period later).
            dt (float, optional): The time step used to evolve the map (default is 1/100 of the orbital period).
            verbose (bool, optional): Output comments of the progress of the run (default = False)?
            intermediates (bool, optional): Output the map from every time step? Otherwise just returns the last step.
        
        Returns:
            list: A list of 2 ndarrays containing the time and map of each time step.
            
        """
        
        if T0 is None:
            T0 = self.planet.map.values
        if t1 is None:
            t1 = t0+self.planet.orbit.Porb
        if dt is None:
            dt = self.planet.orbit.Porb/100.
        
        times = (t0 + np.arange(int(np.rint((t1-t0)/dt)))*dt)[:,np.newaxis]
        TAs = list(self.planet.orbit.true_anomaly(times)[:,:,np.newaxis])
        
        if verbose:
            print('Starting Run')
        maps = np.array([T0]).reshape(1,-1)
        
        # Soften the blow on the NEQ ODE
        if self.planet.plType == 'bell2018' and self.neq and np.all(planet.map.dissValues) == 0.:
            planet.map.dissValues = h2.dissFracApprox(T0.flatten())
        
        for i in range(1, len(times)):
            newMap = (maps[-1]+self.ODE(times[i], maps[-1], dt, TAs[i]).flatten()).reshape(1,-1)
            newMap[newMap<0] = 0
            if intermediates:
                maps = np.append(maps, newMap, axis=0)
            else:
                maps = newMap
        
        self.planet.map.set_values(maps[-1], times[-1,0])
        if self.planet.plType == 'bell2018' and not self.neq:
            self.planet.map.dissValues = h2.dissFracApprox(self.planet.map.values)
        
        if not intermediates:
            times = times[-1]
        
        if verbose:
            print('Done!')
        
        return times, maps
        
    def plot_lightcurve(self, t=None, T=None, bolo=True, tStarBright=None, wav=4.5e-6, allowReflect=True, allowThermal=True):
        """A convenience plotting routine to show the planet's phasecurve.

        Args:
            t (ndarray, optional): The time in days with shape (t.size,1).  If None, will use 1000
                time steps around orbit.
            T (ndarray, optional): The temperature map in K with shape (1, self.planet.map.npix)
                if the map is constant or (t.size,self.planet.map.npix). If None, use
                self.planet.map.values instead.
            bolo (bool, optional): Determines whether computed flux is bolometric (True, default)
                or wavelength dependent (False).
            tBright (ndarray): The brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
            allowReflect (bool, optional): Account for the contribution from reflected light.
            allowThermal (bool, optional): Account for the contribution from thermal emission.

        Returns:
            figure: The figure containing the plot.

        """
        
        if self.planet.orbit.e != 0. and (T is None or t is None):
            print('Warning: Maps and times must be entered for eccentric planets. Failing to do so'+
                  ' will result in non-sensical lightcurves.')
            return None
        
        if t is None:
            # Use Prot instead as map would rotate
            tMax = planet.orbit.Porb-(planet.orbit.Prot_input-planet.orbit.Prot)
            t = self.planet.map.time+np.linspace(0., tMax, 1000)
            x = self.get_phase(t)*self.planet.orbit.Porb/tMax
        else:
            t = t.flatten()
            x = self.get_phase(t)
        
        if self.planet.orbit.e != 0:
            x *= self.planet.orbit.Porb
        
        order = np.argsort(x)
        x = x[order]
        t = t[order]
        
        if T is None:
            T = self.planet.map.values.reshape(1,-1)
        elif type(T)!=np.ndarray or len(T.shape)==1:
            T = np.array([T]).reshape(1,-1)

        lc = self.lightcurve(t, T, bolo=bolo, tStarBright=tStarBright, wav=wav, allowReflect=allowReflect,
                             allowThermal=allowThermal)*1e6
        
        plt.plot(x, lc)
        plt.gca().axvline(self.planet.orbit.t_ecl, c='k', ls='--', label=r'$\rm Eclipse$')
        if self.planet.orbit.e != 0:
            plt.gca().axvline(self.planet.orbit.t_peri,
                              c='red', ls='-.', lw=2, label=r'$\rm Periastron$')

        plt.legend(loc=8, bbox_to_anchor=(0.5,1), ncol=2)
        plt.ylabel(r'$F_p/F_*\rm~(ppm)$')
        if self.planet.orbit.e == 0:
            plt.xlabel(r'$\rm Orbital~Phase$')
        else:
            plt.xlabel(r'$\rm Time~from~Transit~(days)$')
        if self.planet.orbit.e != 0:
            plt.xlim(0, self.planet.Porb)
        else:
            plt.xlim(0, 1)
        plt.ylim(0)
        return plt.gcf()
    
    def plot_tempcurve(self, t=None, T=None, bolo=True, tStarBright=None, wav=4.5e-6, allowReflect=False, allowThermal=True):
        """A convenience plotting routine to show the planet's phasecurve in units of temperature.
        
        Args:
            t (ndarray, optional): The time in days with shape (t.size,1).  If None, will use 1000
                time steps around orbit. Must be provided for eccentric planets.
            T (ndarray, optional): The temperature map in K with shape (1, self.planet.map.npix) if
                the map is constant or (t.size,self.planet.map.npix). If None, use
                self.planet.map.values instead. Must be provided for eccentric planets.
            bolo (bool, optional): Determines whether computed flux is bolometric (True, default)
                or wavelength dependent (False).
            tBright (ndarray): The brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
            allowReflect (bool, optional): Account for the contribution from reflected light.
            allowThermal (bool, optional): Account for the contribution from thermal emission.
        
        Returns:
            figure: The figure containing the plot.
            
        """
        
        if self.planet.orbit.e != 0. and (T is None or t is None):
            print('Warning: Maps and times must be entered for eccentric planets. Failing to do so'+
                  ' will result in non-sensical lightcurves.')
            return None
        
        if t is None:
            # Use Prot instead as map would rotate
            tMax = planet.orbit.Porb-(planet.orbit.Prot_input-planet.orbit.Prot)
            t = self.planet.map.time+np.linspace(0., tMax, 1000)
            x = self.get_phase(t)*self.planet.orbit.Porb/tMax
        else:
            t = t.flatten()
            x = self.get_phase(t)
        
        if self.planet.orbit.e != 0:
            x *= self.planet.orbit.Porb
        
        order = np.argsort(x)
        x = x[order]
        t = t[order]
        
        if T is None:
            T = self.planet.map.values.reshape(1,-1)
        elif type(T)!=np.ndarray or len(T.shape)==1:
            T = np.array([T]).reshape(1,-1)
        
        lc = self.lightcurve(t, T, bolo=bolo, tStarBright=tStarBright, wav=wav,
                             allowReflect=allowReflect, allowThermal=allowThermal)
        tc = self.invert_lc(lc, bolo=bolo, tStarBright=tStarBright, wav=wav)
        
        plt.plot(x, tc)
        plt.gca().axvline(self.planet.orbit.t_ecl, c='k', ls='--', label=r'$\rm Eclipse$')
        if self.planet.orbit.e != 0:
            plt.gca().axvline(self.planet.orbit.t_peri,
                              c='red', ls='-.', lw=2, label=r'$\rm Periastron$')

        plt.legend(loc=8, bbox_to_anchor=(0.5,1), ncol=2)
        if bolo:
            plt.ylabel(r'$T_{\rm eff, hemi, apparent}\rm~(K)$')
        else:
            plt.ylabel(r'$T_{\rm b, hemi, apparent}\rm~(K)$')
        if self.planet.orbit.e == 0:
            plt.xlabel(r'$\rm Orbital~Phase$')
        else:
            plt.xlabel(r'$\rm Time~from~Transit~(days)$')
        if self.planet.orbit.e != 0:
            plt.xlim(0, self.planet.Porb)
        else:
            plt.xlim(0, 1)
        plt.ylim(0)
        return plt.gcf()

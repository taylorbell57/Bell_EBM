# Author: Taylor Bell
# Last Update: 2018-11-30

import numpy as np
import astropy.constants as const

class Star(object):
    """A star.

    Attributes:
        teff (float): The star's effective temperature in K.
        rad (float): The star's radius in solar radii.
        mass (float): The star's mass in solar masses.
    
    """
    
    def __init__(self, teff=5778., rad=1., mass=1.):
        """Initialization function.

        Args:
            teff (float, optional): The star's effective temperature in K.
            rad (float, optional): The star's radius in solar radii.
            mass (float, optional): The star's mass in solar masses.

        """
        
        self.teff = teff                   # K
        self.rad = rad*const.R_sun.value   # m
        self.mass = mass*const.M_sun.value
    
    def Fstar(self, bolo=True, tBright=None, wav=4.5e-6):
        """Calculate the stellar flux for lightcurve normalization purposes.
        
        Args:
            bolo (bool, optional): Determines whether computed flux is bolometric
                (True, default) or wavelength dependent (False).
            tBright (ndarray): The brightness temperature to use if bolo==False.
            wav (float, optional): The wavelength to use if bolo==False.
        
        Returns:
            ndarray: The emitted flux in the same shape as T.
        
        """
        
        if bolo:
            return const.sigma_sb.value*self.teff**4 * np.pi*self.rad**2
        else:
            if tBright is None:
                tBright = self.teff
            a = (2.*const.h.value*const.c.value**2/wav**5)
            b = (const.h.value*const.c.value)/(wav*const.k_B.value)
            return a/np.expm1(b/tBright) * np.pi*self.rad**2

# Author: Taylor Bell
# Last Update: 2018-10-31

import numpy as np
import astropy.constants as const

class Star(object):
    def __init__(self, teff=5778, rad=1, mass=1):
        self.teff = teff                   # K
        self.rad = rad*const.R_sun.value   # m
        self.mass = mass*const.M_sun.value
        
    # Calculate the stellar flux for lightcurve normalization purposes
    def Fstar(self, bolo=True, tBright=None, wav=4.5e-6):
        if bolo:
            return const.sigma_sb.value*self.teff**4 * np.pi*self.rad**2
        else:
            if tBright==None:
                tBright = self.teff
            a = (2*const.h.value*const.c.value**2/wav**5)
            b = (const.h.value*const.c.value)/(wav*const.k_B.value)
            return a/np.expm1(b/tBright) * np.pi*self.rad**2

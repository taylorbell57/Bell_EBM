# Author: Taylor Bell
# Last Update: 2019-02-15
# Based on the work presented in:
# Bell, T. J., & Cowan, N. B. (2018). Increased Heat Transport in Ultra-hot Jupiter Atmospheres through H2 Dissociation and Recombination. The Astrophysical Journal Letters, 857(2), L20.

import numpy as np
import astropy.constants as const
import scipy.special

cp_H = 20.78603/const.N_A.value/const.u.value # From NIST

mass = 2*const.m_p.value

dissE = 436e3 #J/mol (Bond-dissociation energy at 298 K)
dissE = dissE - 3./2.*const.R.value*(298.)
ionE = dissE/const.N_A.value
dissE = dissE/(mass*const.N_A.value) #converting to J/kg

omegaRot = 85.4 #K

def tau_diss(P, T):
    """Calculate the dissociation timescale.
    
    These are rough estimates based on Shui (1973).
    
    Args:
        T (ndarray): The temperature in K.
        P (ndarray): The pressure in Pa.
    
    Returns:
        ndarray: The dissociation timescale in s^-1.
    
    """
    
    kd = 10**((np.log10(3e-15) - np.log10(2e-11))/(3-1)*(T**-1*1e4-1) + np.log10(2e-11))
    numberDensity = P/(const.k_B.value*T)/1e6 # molecules/cm^3
    return (kd*numberDensity) # s^-1

def tau_recomb(P, T):
    """Calculate the recombination timescale.
    
    These are rough estimates based on Shui (1973).
    
    Args:
        T (ndarray): The temperature in K.
        P (ndarray): The pressure in Pa.
    
    Returns:
        ndarray: The recombination timescale in s^-1.
    
    """
    
    kd = 2e-32/np.exp((T/7.44e3)**5)
    numberDensity = P/(const.k_B.value*T)/1e6 # molecules/cm^3
    return (kd*numberDensity**2) # s^-1

def nQ(mu, T):
    """Calculate the quantum concentration.

    Args:
        mu (ndarray): The mean molecular weight in units of u.
        T (ndarray): The temperature in K.

    Returns:
        ndarray: The quantum concentration.

    """
    
    return (2.*np.pi*mu*const.m_p.value*const.k_B.value*T)**(3./2.)/const.h.value**3.

def dissFracSaha(T, P):
    """Calculate the dissociation fraction of H2 using the Saha Equation.

    Args:
        T (ndarray): The temperature in K.
        P (ndarray): The pressure in Pa.

    Returns:
        ndarray: The dissociation fraction of H2.

    """
    
    if type(T)!=np.ndarray:
        T = np.array([T])
    
    if np.any(T<1000):
        good = T>1000
        dFrac = np.zeros_like(T)
        z_r = T[good]/(2.*omegaRot)
        Y = (nQ(2.,T[good])/nQ(1,T[good])**2*z_r*np.exp(ionE/(const.k_B.value*T[good]))
             *P/(const.k_B.value*T[good]))
        dFrac[good] = 2.*(1.+np.sqrt(1+4*Y))**-1
        return dFrac
    else:
        z_r = T/(2*omegaRot)
        Y = nQ(2,T)/nQ(1,T)**2*z_r*np.exp(ionE/(const.k_B.value*T))*P/(const.k_B.value*T)
        return 2*(1+np.sqrt(1+4*Y))**-1

def dDissFracSaha(T, P):
    """Calculate the derivative of the dissociation fraction of H2 using the Saha Equation.

    Args:
        T (ndarray): The temperature in K.
        P (ndarray): The pressure in Pa.

    Returns:
        ndarray: The derivative of the dissociation fraction of H2.

    """
    
    if type(T)!=np.ndarray:
        T = np.array([T])
    
    if np.any(T<1000):
        good = T>1000
        dDFrac = np.zeros_like(T)
        z_r = T[good]/(2.*omegaRot)
        Y = (nQ(2,T[good])/nQ(1,T[good])**2*z_r*np.exp(ionE/(const.k_B.value*T[good]))
             *P/(const.k_B.value*T[good]))
        dY = Y*(-3./2.*T[good]**-1-dissE/const.k_B.value*T[good]**-2)
        dchi_H_dY = 2.*(-1.)*(1+np.sqrt(1+4.*Y))**-2*(1./2.)*np.sqrt(1.+4.*Y)**-1*(4.)
        dDFrac[good] = dchi_H_dY*dY
        return dDFrac
    else:
        z_r = T/(2.*parcel.omegaRot)
        Y = nQ(2.,T)/nQ(1.,T)**2*z_r*np.exp(ionE/(const.k_B.value*T))*P/(const.k_B.value*T)
        dY = Y*(-3./2.*T**-1-ionE/const.k_B.value*T**-2)
        dchi_H_dY = 2.*(-1.)*(1+np.sqrt(1+4.*Y))**-2*(1/2)*np.sqrt(1+4.*Y)**-1*(4.)
        return dchi_H_dY*dY
        
def dissFracApprox(T, mu=3320.680532597579, std=471.38088012739126):
    """Calculate the dissociation fraction of H2 using an erf approximation.

    Args:
        T (ndarray): The temperature in K.
        mu (float, optional): The mean for the error function.
        std (float, optional): The standard deviation for the error function.

    Returns:
        ndarray: The dissociation fraction of H2.

    """
    
    return scipy.special.erf((T-mu)/std/np.sqrt(2.))/2.+0.5

def dDissFracApprox(T, mu=3320.680532597579, std=471.38088012739126):
    """Calculate the derivative in the dissociation fraction of H2 using an erf approximation.

    Args:
        T (ndarray): The temperature in K.
        mu (float, optional): The mean for the Gaussian function.
        std (float, optional): The standard deviation for the Gaussian function.

    Returns:
        ndarray: The derivative in the dissociation fraction of H2.

    """
    
    return np.exp(-(T-mu)**2/(2.*std**2))/(std*np.sqrt(2.*np.pi))

def getSahaApproxParams(P = 0.1*const.atm.value):
    """Get the Gaussian and erf parameters used to approximate the Saha equation.

    Args:
        P (ndarray): The pressure in Pa.

    Returns:
        list: 2 floats containing the mean and the standard deviatio for the Gaussian/erf functions.

    """
    
    a = (np.pi*const.m_p.value*const.k_B.value*const.h.value**-2)**(-3/2)/(2*omegaRot)/const.k_B.value
    b = ionE/const.k_B.value
    c = 4812.88 # found numerically using Mathematica
    d = 1151.47 # found numerically using Mathematica
    
    if P>1e2*const.atm.value:
        print('Warning: Your pressure depth is too deep - using the approximation to '+
              'the Saha equation will diverge from the values from Saha by >5% dissociation fraction')
    mu  = (2/3*b)/scipy.special.lambertw(c*(2./3.)*b*P**(-2./3.)).real
    std = (2/3*b)/scipy.special.lambertw(d*(2./3.)*b*P**(-2./3.)).real-mu

    return mu, std

def cp_H2(T):
    """Get the isobaric specific heat capacity of H2 as a function of temperature.

    Args:
        T (ndarray): The temperature.

    Returns:
        ndarray: The isobaric specific heat capacity of H2.

    """
    
    if type(T)!=np.ndarray:
        T = np.array([T])
    else:
        T = T.copy()
    
    #Below ~71 K, cp_H2 equation gives negative values
    T[T<=100] = 100
    
    A = np.array([33.066178, 18.563083, 43.413560]) # From NIST
    B = np.array([-11.363417, 12.257357, -4.293079]) # From NIST
    C = np.array([11.432816, -2.859786, 1.272428]) # From NIST
    D = np.array([-2.772874, 0.268238, -0.096876]) # From NIST
    E = np.array([-0.158558, 1.977990, -20.533862]) # From NIST
    
    temp = T/1000. #to get the right units to match Chase cp equation
    
    indices = np.zeros_like(T, dtype=int)
    indices[np.logical_and(1000 < T, T < 2500)] = 1
    indices[2500. <= T] = 2
    
    cp_H2 = (A[indices] + B[indices]*temp + C[indices]*temp**2 + D[indices]*temp**3 + E[indices]/temp**2)
    cp_H2 = cp_H2/const.N_A.value/(2*const.u.value)
    
    return cp_H2

def delta_cp_H2(T):
    """Get the derivative of the isobaric specific heat capacity of H2 as a function of temperature.
    
    Pretty sure cp_H2 should already include this factor...
    
    Args:
        T (ndarray): The temperature.

    Returns:
        ndarray: The derivative of theisobaric specific heat capacity of H2.

    """
    
    #
    if type(T)!=np.ndarray:
        T = np.array([T])
    else:
        T = T.copy()
    
    #Below ~71 K, cp_H2 equation gives negative values
    T[T<=100] = 100

    B = np.array([-11.363417, 12.257357, -4.293079]) # From NIST
    C = np.array([11.432816, -2.859786, 1.272428]) # From NIST
    D = np.array([-2.772874, 0.268238, -0.096876]) # From NIST
    E = np.array([-0.158558, 1.977990, -20.533862]) # From NIST
    
    temp = T/1000. #to get the right units to match Chase cp equation
    
    indices = np.zeros_like(T, dtype=int)
    indices[np.logical_and(1000 < T, T < 2500)] = 1
    indices[2500. <= T] = 2
    
    #d(cp)/dT = d(cp)/d(temp)*d(temp)/d(T)
    dcp_H2 = (B[indices] + C[indices]*(2*temp) + D[indices]*(3*temp**2) + E[indices]*(-2/temp**3))/(1000.) 
    dcp_H2 = dcp_H2/const.N_A.value/(2*const.u.value)
    
    #Below ~71 K, cp_H2 equation gives negative values
    dcp_H2[T<=100] = 0.

    return dcp_H2

# The LTE Heat Capacity of Hydrogen Gas as a Function of Temperature
def lte_cp(T, mu=3320.680532597579, std=471.38088012739126):
    """Get the isobaric specific heat capacity of an LTE mix of H2+H as a function of temperature.
    
    Does not account for the energy of H2 dissociation/recombination.

    Args:
        T (ndarray): The temperature.
        mu (float): The mean for the Gaussian/erf approximations to the Saha equation.
        std (float): The standard deviation for the Gaussian/erf approximations to the Saha equation.

    Returns:
        ndarray: The isobaric specific heat capacity of an LTE mix of H2+H.

    """
    
    chi = dissFracApprox(T, mu, std)
    return chi*cp_H + (1-chi)*cp_H2(T)

# The LTE Heat Capacity of Hydrogen Gas as a Function of Temperature
def true_cp(T, mu=3320.680532597579, std=471.38088012739126):
    """Get the isobaric specific heat capacity of an LTE mix of H2+H as a function of temperature.
    
    Accounts for the energy of H2 dissociation/recombination.

    Args:
        T (ndarray): The temperature.
        mu (float): The mean for the Gaussian/erf approximations to the Saha equation.
        std (float): The standard deviation for the Gaussian/erf approximations to the Saha equation.

    Returns:
        ndarray: The isobaric specific heat capacity of an LTE mix of H2+H.

    """
    
    chi = dissFracApprox(T, mu, std)
    dChi = dDissFracApprox(T, mu, std)
    return chi*cp_H + (1-chi)*cp_H2(T) + dissE*dChi

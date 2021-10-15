
# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np
from ..utils.constants import *

__all__ = ["SgrA"]


class SgrA:

    def __init__(self):
        self.distance = 2.543e22 #8.429 kpc to cm  2020 
        self.mass = 4.261e6*M_Sun #M_s from GRAIVTY 2020
        
    @property
    def Rs(self):
        return 2*G*self.mass/c**2

    @property
    def submmSED(self):
        """
        add references here!
        8.68e11 Hz (Bower et al. 2019, Alma band 10)
        """

        nu                      = np.array([1.50e+10,   2.20e+10,  4.30e+10,  9.29e+10,  1.05e+11,  1.50e+11,  4.86e11, 6.92e11, 8.68e11])
        nuLnu                   =  np.array([1.14e+33,  1.93e+33, 5.70e+33, 1.81e+34, 2.21e+34, 3.85e+34, 1.45e+35, 1.54e+35, self.Fnu(8.68e11, 1.864)])
        errnuLnu                =  np.array([7.46e+31,  1.09e+32, 7.13e+32, 1.23e+33, 1.48e+33, 4.97e+33, 2.90e+34, 3.10e+34,  self.Fnu(8.68e11, 1.864*0.2)])
        return nu, nuLnu, errnuLnu
    
    @property
    def firSED(self):
        """
        250micron from Stone 16
        160 micron from von Fellenberg 18
        100 micron from von Fellenberg 18
        """
        FIRnu                   =  np.array([1.2e12, 1.87e12, 3.0e12, 1.2e12, 1.87e12, 3.0e12])
        FIRflux                 =  np.array([4.97e+34,   4.26e+34,  4.01e+34, 4.97e+34/.25,   4.26e+34/.25,  4.01e+34/.25])
        FIRerror                =  np.array([3.98e+34,   8.97e+33,  2.54e+34, 3.98e+34/.25,   8.97e+33/.25,  2.54e+34/.25])
        return FIRnu, FIRflux, FIRerror
    
    # YD 02/2021 - Gravity flux can be changed from 1 to 1.1
    @property
    def nirSED(self):
        """
        Using NIR K band median flux based on GRAVITY + 2020
        """
        nu                      = np.array([1.36e14])
        nuLnu                   = np.array([self.Fnu(1.36e14, 1.1)])*1e-3 #Jy
        errnuLnu                = np.array([self.Fnu(1.36e14, 1.1*0.1)])*1e-3 #Jy
        return nu, nuLnu, errnuLnu
        
    @property
    def references(self):
        print("CM - not yet implemented")
        print("MM - not yet implemented")
        print("Sub-MM - Bower+ 2019/2020")
        print("FIR Stone+ 2016, von Fellenberg + 2018")
        print("NIR GRAVITY+ 2020")
        
    def Fnu(self, v,flux):
        """
        return: nu*F_nu [erg/s]
        """
        return (4*np.pi*(self.distance)**2)*(v)*(flux*10**(-23))


# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np
from .constants import *
from .cfuncs import compton_emissivity

__all__ = ["DummyBooster", "DopplerBooster", "nu_B", "nu_crit", "compton_emissivity_from_fun"]

class DummyBooster:

    def __init__(self):
        pass

    def __call__(self, fun):
        return fun

class DopplerBooster:
    """
    Doppler boost decorator for a given luminosity function.

    Parameters
    ----------
    beta : float
        v/c
    phi : float
        Angle wrt line of sight in radian. Approaching towards observer 0, moving away from observer np.pi
    n : float
        Geometry dependent boosting parameter. Spherical source with bulk velocity 3, jet-like 2 etc.
    """

    def __init__(self, beta, phi, n):
        self.beta = beta
        self.phi = phi
        self.n = n

    def _doppler_f(self):
        return np.sqrt(1-self.beta**2)/(1-self.beta*np.cos(self.phi))

    def __call__(self, fun):
        def boosted_f(nu, *args, **kwargs):
            doppler_f = self._doppler_f()
            nu_p = nu / doppler_f
            return doppler_f**self.n*fun(nu_p, *args, **kwargs)
        return boosted_f

def nu_B(B):
    return e*B/(2*np.pi*M_e*c)

def nu_crit(gamma, B, phi):
    sin_phi = np.pi/4. if phi == -1 else np.sin(phi)
    return 3/2.*nu_B(B)*sin_phi*gamma**2

def compton_emissivity_from_fun(nu, n_ph_fun, e_dist_fun, gamma_min=1, gamma_max=1e5, gamma_steps=50,nu_min=1e8, nu_max=1e17, nu_steps=25):
    """    
    Calculate IC luminosity per volume for each nu. Scattering from n_ph_fun to e_dist_fun
    Note: Changed epsilons to nu all around as input so we don't have multiply with h in inputs.
    
    Parameters
    ----------
    nu : np.ndarray
        1-D frequencies to calculate results.
    n_ph_fun : function
        Function to calculate seed photons n_p(nu) [cm-3 Hz-1] 
    e_dist_fun : function
        Function to calculate electron distribution [cm-3 gamma-1]
    gamma_min : float, default=1
        Lower bound in gamma for integration
    gamma_max : float, default=1e5
        Upper bound in gamma for integration
    gamma_steps : int, default=50
        Gamma steps per decade in logspace
    nu_min : float, default=1e8
        Lower bound in nu (epsilon/h) for integration [Hz]
    nu_max : float, default=1e17
        Upper bound in nu (epsilon/h) for integration [Hz]
    nu_steps : int, default=25
        nu steps per decade in logspace 

    Returns
    -------
    j_nu : np.array
        [erg s-1 cm-3 Hz-1 sr-1]
    """
    
    e_steps                 = np.int((np.log10(nu_max) - np.log10(nu_min))*nu_steps)
    epsilon                 = np.logspace(np.log10(nu_min), np.log10(nu_max), e_steps)*h
    g_steps                 = np.int((np.log10(gamma_max) - np.log10(gamma_min))*gamma_steps)
    gamma                   = np.logspace(np.log10(gamma_min), np.log10(gamma_max), g_steps)

    n_ph                    = n_ph_fun(epsilon/h)
    e_dist                  = e_dist_fun(gamma)

    return compton_emissivity(epsilon, gamma, nu, n_ph, e_dist)

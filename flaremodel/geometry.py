# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np
import warnings

from .utils import *
from .utils.gfuncs import GPU_ENABLED

class Geometry:
    """
    Base Geometry class to extend for other geometries

    Parameters
    ----------
    edist: str
        Name of the electron distribution for the geometry. Refer to docs.
    method: str
        Reserved for later use.
    booster: class
        A boosting implementation, ie. :class:`DopplerBooster`
    """

    def __init__(self, edist="thermal", method="brute", booster=None):
        self.name = self.__class__.__name__
        self.edist = edist
        if method == "brute":
            self.j_nu_fun = j_nu_brute
            self.a_nu_fun = a_nu_brute
        else:
            raise ValueError("% method is not implemented.")
        self.booster = DopplerBooster if booster is None else booster

    def _compute_synchrotron(self, nu, ne, g_params, B, params, **kwargs):
        """
        Definition of synchrotron luminosity for the geometry

        Parameters
        ----------
        nu: np.ndarray
            1-D frequencies to calculate synchrotron luminosity [Hz]
        ne: float
            Electron density [cm^-3]
        g_params: list
            Parameters for the implemented geometry
        B: float
            Magnetic Field [G]
        params: list
            List of parameters for the electron distribution
        **kwargs:
            Reserved for additional parameters

        Returns
        -------
        L_nu: np.ndarray
            [erg s-1 Hz-1]

        """
        raise NotImplementedError()

    def _compute_photon_density(self, nu, ne, g_params, B, params):
        """
        Definition of photon density due to internal synchrotron.
        Note that this can return anything for any specific SSC calculation and
        typically not to be used for other purposes.

        Parameters
        ----------
        nu: np.ndarray
            1-D frequencies to calculate synchrotron luminosity [Hz]
        ne: float
            Electron density [cm^-3]
        g_params: list
            Parameters for the implemented geometry
        B: float
            Magnetic Field [G]
        params: list
            List of parameters for the electron distribution
        **kwargs:
            Reserved for additional parameters

        """
        raise NotImplementedError()

    def _compute_IC(self, nu, ne, n_ph_fun, g_params, B, params, **kwargs):
        """
        Definition of SSC luminosity for the geometry

        Parameters
        ----------
        nu: np.ndarray
            1-D frequencies to calculate synchrotron luminosity [Hz]
        ne: float
            Electron density [cm^-3]
        n_ph_fun: function
            Function to calculate photon densities nph(nu) [cm-3 Hz-1]
        g_params: list
            Parameters for the implemented geometry
        B: float
            Magnetic Field [G]
        params: list
            List of parameters for the electron distribution
        **kwargs:
            Reserved for additional parameters

        Returns
        -------
        L_nu: np.ndarray
            [erg s-1 Hz-1]        
        """
        raise NotImplementedError()


    def _compute_SSC(self, nu, ne, g_params, B, params, **kwargs):
        """
        Definition of SSC luminosity for the geometry

        Parameters
        ----------
        nu: np.ndarray
            1-D frequencies to calculate synchrotron luminosity [Hz]
        ne: float
            Electron density [cm^-3]
        g_params: list
            Parameters for the implemented geometry
        B: float
            Magnetic Field [G]
        params: list
            List of parameters for the electron distribution
        **kwargs:
            Reserved for additional parameters

        Returns
        -------
        L_nu: np.ndarray
            [erg s-1 Hz-1]        
        """
        raise NotImplementedError()

    def compute_IC(self, nu, ne, n_ph_fun, g_params, B, params, boost_p=None, **kwargs):
        """
        Same as :func:`Geometry._compute_IC` but accepts ``boost_p`` keyword. See class :class:`DopplerBooster`
        """
        if boost_p is None:
            return self._compute_IC(nu, ne, n_ph_fun, g_params, B, params, **kwargs)
        else:
            return self.booster(*boost_p)(self._compute_IC)(nu, ne, n_ph_fun, g_params, B, params, **kwargs)

    def compute_SSC(self, nu, ne, g_params, B, params, boost_p=None, **kwargs):
        """
        Same as :func:`Geometry._compute_SSC` but accepts ``boost_p`` keyword. See class :class:`DopplerBooster`
        """
        if boost_p is None:
            return self._compute_SSC(nu, ne, g_params, B, params, **kwargs)
        else:
            return self.booster(*boost_p)(self._compute_SSC)(nu, ne, g_params, B, params, **kwargs)

    def compute_synchrotron(self, nu, ne, g_params, B, params, boost_p=None, **kwargs):
        """
        Same as :func:`Geometry._compute_synchrotron` but accepts ``boost_p`` keyword. See class :class:`DopplerBooster`
        """

        if boost_p is None:
            return self._compute_synchrotron(nu, ne, g_params, B, params, **kwargs)
        else:
            return self.booster(*boost_p)(self._compute_synchrotron)(nu, ne, g_params, B, params, **kwargs)


class HomogeneousSphere(Geometry):
    """
    Exact homogeneous sphere.

    g_params for this class are:
        - R, size of the sphere [cm]
        - incang, Magnetic field configuration. -1 for tangled or uniform field lines at an angle to observer [rad]

    Parameters
    ----------
    edist: str
        Name of the electron distribution for the geometry. Refer to docs.
    method: str
        Reserved for later use.
    booster: class
        A boosting implementation, ie. :class:`DopplerBooster`
    """
    def _rtrans_homogeneous_sphere(self, nu, jnu, anu, R):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.select([anu*R < 1e-4, anu*R > 10],
                        [4*np.pi*jnu*R*R*R/3, np.pi*jnu*R*R/anu],
                        default=np.pi*jnu*R*R/anu*(1-1./(2.*anu*anu*R*R)*(1-np.exp(-2*anu*R)*(2*anu*R+1))))

    def _compute_synchrotron(self, nu, ne, g_params, B, params, **kwargs):
        jnu = self.j_nu_fun(nu, ne, B, params, edist=self.edist, incang=g_params[1], **kwargs)
        anu = self.a_nu_fun(nu, ne, B, params, edist=self.edist, incang=g_params[1], **kwargs)
        return 4*np.pi*self._rtrans_homogeneous_sphere(nu, jnu, anu, g_params[0])

    def _compute_photon_density(self, nu, ne, g_params, B, params, **kwargs):
        return self._compute_synchrotron(nu, ne, g_params, B, params, **kwargs)/(h**2*nu*c*4*np.pi*g_params[0]**2)

    def _compute_SSC(self, nu, ne, g_params, B, params, **kwargs):

        n_ph_fun = lambda nu: self._compute_photon_density(nu, ne, g_params, B, params, **kwargs)
        
        return self._compute_IC(nu, ne, n_ph_fun, g_params, B, params, **kwargs)

    def _compute_IC(self, nu, ne, n_ph_fun, g_params, B, params, **kwargs):

        e_dist_fun = lambda gamma: eDist(gamma, ne, params, edist=self.edist)
        j_nu_ic = compton_emissivity_from_fun(nu, n_ph_fun, e_dist_fun, **kwargs)
        
        return (4*np.pi*j_nu_ic)*4*np.pi*g_params[0]**3/3.

class HomogeneousSphereUserdist(Geometry):
    """
    Special geometry class with different interface to work with given user electron distribution.

    Parameters
    ----------
    booster: class
        A boosting implementation, ie. :class:`DopplerBooster`   
    """
    def __init__(self, booster=None):
        self.booster = DopplerBooster if booster is None else booster

    def _rtrans_homogeneous_sphere(self, nu, jnu, anu, R):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # since this computes all then selects wrt proper index, throws bunch of division errors.
            return np.select([anu*R < 1e-4, anu*R > 10],
                        [4*np.pi*jnu*R*R*R/3, np.pi*jnu*R*R/anu],
                        default=np.pi*jnu*R*R/anu*(1-1./(2.*anu*anu*R*R)*(1-np.exp(-2*anu*R)*(2*anu*R+1))))

    def _compute_synchrotron(self, nu, Ne, g_params, B, gamma, **kwargs):
        """
        Computes synchrotron luminosity

        Parameters
        ----------
        nu : np.ndarray
            Frequency array [Hz]
        Ne : np.ndarray
            Electron distribution on gamma grid. Note this is not density total electrons 
        g_params : list
            Geometry parameters. same as :class:`HomogeneousSphere`
        B : float
            Magnetic field [G]
        gamma : np.ndarray
            gamma corresponding to Ne

        Returns
        -------
        L_nu : np.ndarray
            [erg s-1 Hz-1]
        """
        V = 4./3.*np.pi*g_params[0]**3
        jnu = j_nu_userdist(nu, B, gamma, Ne/V, incang=g_params[1], **kwargs)
        anu = a_nu_userdist(nu, B, gamma, Ne/V, incang=g_params[1], **kwargs)
        return 4*np.pi*self._rtrans_homogeneous_sphere(nu, jnu, anu, g_params[0])

    def compute_synchrotron(self, nu, Ne, g_params, B, gamma, boost_p=None, **kwargs):
        if boost_p is None:
            return self._compute_synchrotron(nu, Ne, g_params, B, gamma, **kwargs)
        else:
            return self.booster(*boost_p)(self._compute_synchrotron)(nu, Ne, g_params, B, gamma, **kwargs)


class RadialSphere(Geometry):
    """
    Numerical computation of spherical model where it is possible to set radial profiles in electron density or magnetic field.

    g_params for this class are:
        - R, size of the sphere [cm]
        - R_in, inner radius of the sphere. 0 if not necessary.
        - incang, Magnetic field configuration. -1 for tangled or uniform field lines at an angle to observer [rad]

    Parameters
    ----------
    n_r_fun: function
        Radial profile of n_e
    B_r_fun: function
        Radial profile of magnetic field
    edist: str
        Electron distribution tag
    method: str
        Reserved for later
    bh: boolean, default=False
        Max absorption below R_in
    rsteps: int
        Specify number of radial grid points
    target: str, default="cpu"
        How to perform inverse Compton scattering, either "cpu" or "gpu".
    """

    def __init__(self, n_r_fun=lambda x: 1., B_r_fun=lambda x: 1.,
                        edist="thermal", method="brute", bh=False, rsteps=64, target="cpu"):

        self.rsteps = rsteps
        if target == "gpu" and GPU_ENABLED == False:
            warnings.warn("Setting target to cpu. GPU functions not available...")
            self.target = "cpu"
        else:
            self.target = target
        self.bh = bh
        self.n_r_fun = n_r_fun
        self.B_r_fun = B_r_fun

        super().__init__(edist=edist, method=method)

    def n_r_geom(self, r, R, R_in):
        return np.piecewise(r, [np.logical_and(R_in < r, r < R)], [1., 0])

    def abs_r_geom(self, r, R, R_in):
        if self.bh == True:
            return np.piecewise(r, [r < R_in, np.logical_and(R_in < r, r < R)], 
                                    [1e99, 1., 0.])
        else:
            return self.n_r_geom(r, R, R_in)

    def _compute_grid(self, R):
        rsteps = self.rsteps
        xx = np.linspace(0, R, rsteps)+R/(rsteps-1)*0.5
        xx_sq = xx**2
        RR = np.sqrt(np.add.outer(xx_sq, xx_sq))
        RR = np.concatenate((RR[:,::-1], RR[:,:]), axis=1)
        dx = R/(rsteps-1)
        return dx, xx, RR

    def _radiative_transfer(self, nu, ne, g_params, B, params, **kwargs):
        rsteps = self.rsteps
        R, R_in = g_params[:2]
        dx, xx, RR = self._compute_grid(R)

        j_nu = np.zeros([nu.shape[0], rsteps, 2*rsteps])
        a_nu = np.zeros([nu.shape[0], rsteps, 2*rsteps])

        for i in range(rsteps):
            j_nu[:,i,0] = self.j_nu_fun(nu, ne*self.n_r_fun(xx[i]), B*self.B_r_fun(xx[i]), params, 
                            edist=self.edist, incang=g_params[-1], **kwargs)
            a_nu[:,i,0] = self.a_nu_fun(nu, ne*self.n_r_fun(xx[i]), B*self.B_r_fun(xx[i]), params, 
                            edist=self.edist, incang=g_params[-1], **kwargs)

        for i in range(nu.shape[0]):

            j_nu[i,:,:] = np.interp(RR, xx[:-1], j_nu[i,:-1,0])*self.n_r_geom(RR, R, R_in)
            a_nu[i,:,:] = np.interp(RR, xx[:-1], a_nu[i,:-1,0])*self.abs_r_geom(RR, R, R_in)
            
        return ray_tracing(j_nu, a_nu, dx), xx, RR


    def _compute_synchrotron(self, nu, ne, g_params, B, params, **kwargs):
        integrand_r, xx, _ = self._radiative_transfer(nu, ne, g_params, B, params, **kwargs)
        return  4*np.pi*np.array([np.trapz(integrand_r[i,:,-1]*2*np.pi*xx, xx) for i in range(nu.shape[0])])

    def _compute_photon_density(self, nu, ne, g_params, B, params, **kwargs):
        rsteps = self.rsteps
        R, R_in = g_params[:2]
        S_nu, xx, RR = self._radiative_transfer(nu, ne, g_params, B, params, **kwargs)
        n_ph = np.zeros([rsteps-1, nu.shape[0]])
        bins = xx[1:] - R/(rsteps-1)*0.5
        dcost = (np.tile(xx, (2*rsteps, 1)).T/RR)
        cnts, _ = np.histogram(RR, bins=rsteps-1, range=[0,R], weights=dcost)

        for i in range(nu.shape[0]):
            n_ph[:,i], _ = np.histogram(RR, bins=rsteps-1, range=[0,R], weights=S_nu[i,:,:]*dcost)
            n_ph[:,i] = n_ph[:,i]/cnts*2*np.pi/(h**2*nu[i]*c)

        return  bins, n_ph

    def _compute_IC(self, nu, ne, n_ph_fun, g_params, B, params,
            gamma_min=1, gamma_max=1e5, gamma_steps=50,nu_min=1e8, nu_max=1e17, nu_steps=25):

        rsteps = self.rsteps
        R, R_in = g_params[:2]
        _, Rs, _ = self._compute_grid(R) 
        Rs = Rs[1:] - R/(rsteps-1)*0.5 # This is necessary for proper grid.

        e_steps                 = np.int((np.log10(nu_max) - np.log10(nu_min))*nu_steps)
        epsilon                 = np.logspace(np.log10(nu_min), np.log10(nu_max), e_steps)*h
        g_steps                 = np.int((np.log10(gamma_max) - np.log10(gamma_min))*gamma_steps)
        gamma                   = np.logspace(np.log10(gamma_min), np.log10(gamma_max), g_steps)
    
        n_ph                    = n_ph_fun(epsilon/h)
        e_dist                  = eDist(gamma, ne, params, edist=self.edist)

        integral                = np.zeros([nu.shape[0], rsteps-1])
        
        if self.target == "gpu":
            g_compton_emissivity = gComptonEmissivity(nu, epsilon, gamma)
            for i in range(rsteps-1):
                integral[:,i] = 4*np.pi*g_compton_emissivity(n_ph, e_dist*self.n_r_fun(Rs[i])*self.n_r_geom(Rs[i], R, R_in))
        else:
            for i in range(rsteps-1):
                integral[:,i] = 4*np.pi*compton_emissivity(epsilon, gamma, nu, n_ph, e_dist*self.n_r_fun(Rs[i])*self.n_r_geom(Rs[i], R, R_in))
    
        Rs = Rs[np.newaxis,:]

        return np.trapz(integral[:,:]*4*np.pi*Rs[:,:]**2, Rs[:,:], axis=1)

    def _compute_SSC(self, nu, ne, g_params, B, params,
            gamma_min=1, gamma_max=1e5, gamma_steps=50,nu_min=1e8, nu_max=1e17, nu_steps=25,
            **kwargs):

        R, R_in = g_params[:2]

        rsteps                  = self.rsteps
        e_steps                 = np.int((np.log10(nu_max) - np.log10(nu_min))*nu_steps)
        epsilon                 = np.logspace(np.log10(nu_min), np.log10(nu_max), e_steps)*h
        g_steps                 = np.int((np.log10(gamma_max) - np.log10(gamma_min))*gamma_steps)
        gamma                   = np.logspace(np.log10(gamma_min), np.log10(gamma_max), g_steps)
    
        Rs, n_ph                = self._compute_photon_density(epsilon/h, ne, g_params, B, params, **kwargs)
        e_dist                  = eDist(gamma, ne, params, edist=self.edist)

        integral                = np.zeros([nu.shape[0], rsteps-1])
        
        if self.target == "gpu":
            g_compton_emissivity = gComptonEmissivity(nu, epsilon, gamma)
            for i in range(rsteps-1):
                integral[:,i] = 4*np.pi*g_compton_emissivity(n_ph[i,:], e_dist*self.n_r_fun(Rs[i])*self.n_r_geom(Rs[i], R, R_in))
        else:
            for i in range(rsteps-1):
                integral[:,i] = 4*np.pi*compton_emissivity(epsilon, gamma, nu, n_ph[i,:], e_dist*self.n_r_fun(Rs[i])*self.n_r_geom(Rs[i], R, R_in))
    
        Rs = Rs[np.newaxis,:]

        return np.trapz(integral[:,:]*4*np.pi*Rs[:,:]**2, Rs[:,:], axis=1)

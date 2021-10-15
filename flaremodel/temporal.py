
# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np

from .utils import *

class vdLEdist:
    """
    Numerical simulation of adiabatic cooling for a spherical source. 
    Gamma array can be obtained with vdLEdist._gamma.

    Parameters
    ----------
    g_min : float, default=1
        Minimum gamma of the grid
    g_max : float, default=1e5
        Maximum gamma of the grid
    g_steps : float, default=50
        Gamma steps per decade in logspace
    sync_cool : boolean, default=True
        Whether to enable/disable synchrotron cooling
    """

    def __init__(self, g_min=1., g_max=1e5, g_steps=50, sync_cool=True):

        self.sync_cool = sync_cool
        self._gamma = np.logspace(np.log10(g_min), np.log10(g_max), 
                                np.int(np.log10(g_max/g_min)*g_steps))
        self.e_dist_inj = np.zeros_like(self._gamma)

    def e_dist_fun(self, gamma, Ne):
        return np.zeros_like(gamma)

    def _run(self, ts, dt):
        
        res = np.zeros([ts.shape[0], self._gamma.shape[0]])
        Rs = np.zeros([ts.shape[0]])
        Bs = np.zeros([ts.shape[0]])
        e_dist = self.e_dist_fun(self._gamma, self.N_e)
        B = self.B
        R = self.R

        tsteps = np.int(ts[-1]/dt)+2

        cntr = 0

        if self.sync_cool:

            for i in range(tsteps):
                B, R = cool_onestep(e_dist, self._gamma, self.e_dist_inj, B, R, dt, self.exp_b, 1e99, 0)
                if i*dt > ts[cntr]:
                    res[cntr, :] = e_dist
                    Rs[cntr] = R
                    Bs[cntr] = B
                    cntr += 1

        else:

            R0 = self.R
            for i in range(tsteps):
                _, R = cool_onestep(e_dist, self._gamma, self.e_dist_inj, 0, R, dt, self.exp_b, 1e99, 0)
                if i*dt > ts[cntr]:
                    res[cntr, :] = e_dist
                    Rs[cntr] = R
                    Bs[cntr] = B*(R/R0)**-2
                    cntr += 1


        return Rs, Bs, res

    def get_ngamma(self, ts, N_e, B, R, exp_b, dt):
        """
        Return electron distribution for array of ts

        Parameters
        ----------
        ts : np.ndarray
            1-D time stamps to request electron distribution [s]
        N_e : float
            Initial number of electrons
        B : float
            Initial magnetic field [G]
        R : float
            Initial size of the "blob" [cm]
        exp_b : float
            Fixed expansion speed in units of c
        dt : float
            Time resolution [s]
        Returns
        -------
        Rs : np.ndarray
            1-D array of R(ts) [cm]
        Bs : np.ndarray
            1-D array of B(ts) with B(R)~R^-2 [G]
        Ne_t : np.ndarray
            2-D array Ne_t(ts, gamma)
        """
        self.B = B
        self.R = R
        self.exp_b = exp_b
        self.N_e = N_e

        return self._run(ts, dt)

class GaussianInjectionEdist:
    """
    Numerical simulation of particle injection with Gaussian profile including synchrotron cooling and particle escape. 
    Gamma array can be obtained with GaussianInjectionEdist._gamma.

    Parameters
    ----------
    sig_inj : float
        Sigma of the particle injection profile [s]
    t_peak_inj : float
        Peak time of the injection profile [s]
    g_min : float, default=1
        Minimum gamma of the grid
    g_max : float, default=1e5
        Maximum gamma of the grid
    g_steps : float, default=50
        Gamma steps per decade in logspace
    """

    def __init__(self, sig_inj, t_peak_inj, t_esc=1e99, g_min=1, g_max=1e5, g_steps=50):

        self.t_esc = t_esc
        self._gamma = np.logspace(np.log10(g_min), np.log10(g_max), 
                                np.int(np.log10(g_max/g_min)*g_steps))
        self.inj_r_fun = lambda t: np.exp(-(t-t_peak_inj)**2*0.5/sig_inj**2)/(np.sqrt(2*np.pi)*sig_inj)

    def e_dist_fun(self, gamma, Ne):
        return np.zeros_like(gamma)

    def e_dist_inj_fun(self, gamma, Ne):
        return np.zeros_like(gamma)

    def _run(self, ts, dt):
        
        res = np.zeros([ts.shape[0], self._gamma.shape[0]])
        e_dist = self.e_dist_fun(self._gamma, self.N_e)
        e_dist_inj = self.e_dist_inj_fun(self._gamma, 1.)

        B = self.B
        R = self.R

        tsteps = np.int(ts[-1]/dt)+2

        cntr = 0

        for i in range(tsteps):
            _, _ = cool_onestep(e_dist, self._gamma, e_dist_inj, B, R, dt, 0, self.t_esc, self.inj_r_fun(i*dt)*self.N_e)
            if i*dt > ts[cntr]:
                res[cntr, :] = e_dist
                cntr += 1

        return res

    def get_ngamma(self, ts, N_e, B, dt):
        """
        Return electron distribution for array of ts

        Parameters
        ----------
        ts : np.ndarray
            1-D time stamps to request electron distribution [s]
        N_e : float
            Total number of injected electrons
        B : float
            Initial magnetic field [G]
        dt : float
            Time resolution [s]
        Returns
        -------
        Ne_t : np.ndarray
            2-D array Ne_t(ts, gamma)
        """

        self.B = B
        self.R = 1.
        self.N_e = N_e

        return self._run(ts, dt)


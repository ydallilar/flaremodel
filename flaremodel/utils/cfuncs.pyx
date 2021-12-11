# cython: language_level = 3
# cython: boundscheck = False
# cython: embedsignature = True

# Licensed under a 3-clause BSD style license - see LICENSE

from cpython cimport array
import array

cimport numpy as np
import numpy as np

from cython.parallel import prange

cdef extern from "synchrotron.h":
    int c_j_nu_brute    "j_nu_brute"    (...) nogil
    int c_a_nu_brute    "a_nu_brute"    (...) nogil
    int c_j_nu_userdist "j_nu_userdist" (...) nogil
    int c_a_nu_userdist "a_nu_userdist" (...) nogil
    int c_synchF        "synchF"        (...) nogil
    int c_b_synchF      "b_synchF"      (...) nogil
    int c_synchG        "synchG"        (...) nogil
    int c_synchH        "synchH"        (...) nogil
    int c_b_synchH      "b_synchH"      (...) nogil

cdef extern from "edist.h":
    double powerlaw (double, void*)
    double powerlawexpcutoff (double, void*)
    double thermal (double, void*)
    double kappa (double, void*)
    double bknpowerlaw (double, void*)
    double bknpowerlawexpcutoff (double, void*)
    double powerlaw_norm (void*)
    double thermal_norm (void*)
    double kappa_norm (void*)
    double bknpowerlaw_norm (void*)
    int c_eDist         "eDist"         (...)

cdef extern from "common.h":

    cpdef enum stokes:
        STOKES_I,
        STOKES_Q,
        STOKES_V

    struct Source:
        double B
        double ne
        double R
        int d_type
        stokes pol
        double (*d_func) (double, void*)
        double (*n_func) (void*)
        double* params
        double incang
        double gamma_min
        double gamma_max   
        double gamma_steps

cdef extern from "temporal.h":
    int c_cool_onestep "cool_onestep"  (...)

cdef extern from "rtrace.h":
    void rtrace (...) nogil

cdef extern from "compton.h":
    int c_compton_emissivity "compton_emissivity"  (...) nogil

cdef _set_source_params(Source *source_t, str edist):

    cdef double *params = source_t.params

    if edist == "powerlaw":
        source_t.gamma_min = params[1]
        source_t.gamma_max = params[2]
        source_t.d_func = &powerlaw
        source_t.n_func = &powerlaw_norm
    elif edist == "thermal":
        source_t.d_func = &thermal
        source_t.n_func = &thermal_norm
    elif edist == "kappa":
        source_t.d_func = &kappa
        source_t.n_func = &kappa_norm
    elif edist == "bknpowerlaw":
        source_t.gamma_min = params[3]
        source_t.gamma_max = params[4]
        source_t.d_func = &bknpowerlaw
        source_t.n_func = &bknpowerlaw_norm
    elif edist == "powerlawexpcutoff":
        source_t.gamma_min = params[1]
        source_t.gamma_max = params[2]*10
        source_t.d_func = &powerlawexpcutoff
        source_t.n_func = &powerlaw_norm
    elif edist == "bknpowerlawexpcutoff":
        source_t.gamma_min = params[3]
        source_t.gamma_max = params[4]*10
        source_t.d_func = &bknpowerlawexpcutoff
        source_t.n_func = &bknpowerlaw_norm
    else:
        raise ValueError("%s not implemented" % edist)


def j_nu_brute(double[::1] nu, double ne, double B, list params, str edist, double incang=-1, int steps=50, 
                                                                    double gamma_min=1.1, double gamma_max=1e7,
                                                                    stokes pol=STOKES_I):
    """
    Numerical calculation of synchrotron emissivity for a given (pre-defined) electron distribution.

    Parameters
    ----------
    nu : np.ndarray
        C contiguous 1-D numpy array of frequencies to calculate the coefficient.
    ne : float
        electron density [1/cm3]
    B : float
        Magnetic fied [G]
    params : list
        Set of parameters for the electron distribution
    edist : str
        Name of the electron distribution
    incang : float, default=-1
        Inclination angle, -1 for angle averaged [rad]
    steps : int, default=50
        Steps per decade in gamma for integration
    gamma_min : float, default=1.1
        Lower limit of gamma range used for integration. If the distribution has the parameters, the value is taken from distribution parameters
    gamma_max : float, default=1e7
        Same as gamma_min but upper limit
    Returns
    -------
    j_nu : np.ndarray
        Synchrotron emissivity, same size as nu [ergs cm-3 s-1 Hz-1 sr-1]
    """

    cdef int sz = nu.shape[0]
    cdef double[::1] res = np.empty_like(nu)
    cdef double[::1] t_par = array.array('d', params)

    cdef Source source_t
    source_t.B = B
    source_t.ne = ne
    source_t.gamma_steps = steps
    source_t.gamma_min = gamma_min
    source_t.gamma_max = gamma_max
    source_t.params = &t_par[0]
    source_t.incang = incang
    source_t.pol = pol

    _set_source_params(<Source*> &source_t, edist)
    c_j_nu_brute(&res[0], sz, &nu[0], <Source*> &source_t)

    return np.asarray(res)

def a_nu_brute(double[::1] nu, double ne, double B, list params, str edist, double incang=-1, int steps=50, 
                                                                    double gamma_min=1.1, double gamma_max=1e7,
                                                                    stokes pol=STOKES_I):
    """
    Numerical calculation of synchrotron absorption coefficient for a given (pre-defined) electron distribution.

    Parameters
    ----------
    nu : np.ndarray
        C contiguous 1-D numpy array of frequencies to calculate the coefficient.
    ne : float
        electron density [1/cm3]
    B : float
        Magnetic fied [G]
    params : list
        Set of parameters for the electron distribution
    edist : str
        Name of the electron distribution
    incang : float, default=-1
        Inclination angle, -1 for angle averaged [rad]
    steps : int, default=50
        Steps per decade in gamma for integration
    gamma_min : float, default=1.1
        Lower limit of gamma range used for integration. If the distribution has the parameters, the value is taken from distribution parameters
    gamma_max : float, default=1e7
        Same as gamma_min but upper limit
    Returns
    -------
    a_nu : np.ndarray
        Synchrotron absorption coefficient, same size as nu [cm-1]
    """

    cdef int sz = nu.shape[0]
    cdef double[::1] res = np.empty_like(nu)
    cdef double[::1] t_par = array.array('d', params)

    cdef Source source_t
    source_t.B = B
    source_t.ne = ne
    source_t.gamma_steps = steps
    source_t.gamma_min = gamma_min
    source_t.gamma_max = gamma_max
    source_t.params = &t_par[0]
    source_t.incang = incang
    source_t.pol = pol

    _set_source_params(<Source*> &source_t, edist)
    c_a_nu_brute(&res[0], sz, &nu[0], <Source*> &source_t)

    return np.asarray(res)

def j_nu_userdist(double[::1] nu, double B, double[::1] gamma, double[::1] e_dist, double incang=-1):
    """
    Numerical calculation of synchrotron emissivity for a given numerical electron distribution.

    Parameters
    ----------
    nu : np.ndarray
        C contiguous 1-D numpy array of frequencies to calculate the coefficient.
    B : float
        Magnetic fied [G]
    gamma : np.ndarray
        C contiguous grid points in gamma for corresponding e_dist
    e_dist : np.ndarray
        C contiguous grid points for dN/dgamma [cm-3]
    incang : float, default=-1
        Inclination angle, -1 for angle averaged [rad]
    Returns
    -------
    j_nu : np.ndarray
        Synchrotron emissivity, same size as nu [ergs cm-3 s-1 Hz-1 sr-1]
    """

    cdef int sz = nu.shape[0]
    cdef double[::1] res = np.empty_like(nu)
    cdef int len_gamma = gamma.shape[0]

    cdef Source source_t
    source_t.B = B
    source_t.incang = incang

    c_j_nu_userdist(&res[0], sz, &nu[0], len_gamma, &gamma[0], &e_dist[0], <Source*> &source_t)

    return np.asarray(res)

def a_nu_userdist(double[::1] nu, double B, double[::1] gamma, double[::1] e_dist, double incang=-1):
    """
    Numerical calculation of synchrotron absorption coefficient for a given numerical electron distribution.

    Parameters
    ----------
    nu : np.ndarray
        C contiguous 1-D numpy array of frequencies to calculate the coefficient.
    B : float
        Magnetic fied [G]
    gamma : np.ndarray
        C contiguous grid points in gamma for corresponding e_dist
    e_dist : np.ndarray
        C contiguous grid points for dN/dgamma [cm-3]
    incang : float, default=-1
        Inclination angle, -1 for angle averaged [rad]
    Returns
    -------
    a_nu : np.ndarray
        Synchrotron absorption coefficient, same size as nu [cm-1]
    """
    cdef int sz = nu.shape[0]
    cdef double[::1] res = np.empty_like(nu)
    cdef int len_gamma = gamma.shape[0]

    cdef Source source_t
    source_t.B = B
    source_t.incang = incang

    c_a_nu_userdist(&res[0], sz, &nu[0], len_gamma, &gamma[0], &e_dist[0], <Source*> &source_t)

    return np.asarray(res)

def eDist(double[::1] gamma, double ne, list params, str edist):
    """
    Common function for built-in electron distributions

    Parameters
    ----------
        gamma: np.ndarray
            1-D gamma array (C contiguous)
        ne: float
            Number of electrons
        params: list of floats
            Electron distributions parameters varies based on edist
                - "thermal" : theta_E
                - "kappa" : kappa, kappa_width
                - "powerlaw" : p, g_min, g_max
                - "bknpowerlaw" : p1, p2, g_break, g_min, g_max
                - "powerlawexpcutoff" : p, g_min, g_max
                - "bknpowerlawexpcutoff" : p1, p2, g_break, g_min, g_max
        edist: str
            see params

    Returns
    -------
        dN/dgamma: np.ndarray
    """
    cdef int sz = gamma.shape[0]
    cdef double[::1] res = np.empty_like(gamma)
    cdef double[::1] t_par = array.array('d', params)

    cdef Source source_t
    source_t.ne = ne
    source_t.params = &t_par[0]

    _set_source_params(<Source*> &source_t, edist)
    c_eDist(&res[0], sz, &gamma[0], <Source*> &source_t)

    return np.asarray(res)

cpdef synchF(double[::1] x):

    cdef int sz = x.shape[0]
    cdef double[::1] res = np.empty_like(x)

    c_synchF(&res[0], sz, &x[0])

    return np.asarray(res)

cpdef synchG(double[::1] x):

    cdef int sz = x.shape[0]
    cdef double[::1] res = np.empty_like(x)

    c_synchG(&res[0], sz, &x[0])

    return np.asarray(res)

cpdef synchH(double[::1] x):

    cdef int sz = x.shape[0]
    cdef double[::1] res = np.empty_like(x)

    c_synchH(&res[0], sz, &x[0])

    return np.asarray(res)

cpdef b_synchF(double[::1] x):

    cdef int sz = x.shape[0]
    cdef double[::1] res = np.empty_like(x)

    c_b_synchF(&res[0], sz, &x[0])

    return np.asarray(res)

cpdef b_synchH(double[::1] x):

    cdef int sz = x.shape[0]
    cdef double[::1] res = np.empty_like(x)

    c_b_synchH(&res[0], sz, &x[0])

    return np.asarray(res)

def compton_emissivity(double[::1] epsilon, double[::1] gamma, double[::1] nu,
                        double[::1] n_ph, double[::1] e_dist):
    """
    Integrates KN cross-section for scattering of n_ph(epsilon) seed photons from
    electron distribution e_dist(gamma) with frequency nu.

    Parameters
    ----------
    epsilon: np.ndarray
        Input incident photon energies, note h*nu [ergs]
    gamma: np.ndarray
        Gamma of electron distribution
    nu: np.ndarray
        Requested scattered photon frequencies [Hz]
    n_ph: np.ndarray
        Photon densities at corresponding epsilons [cm-3]
    e_dist: np.ndarray
        Electron densities at corresponding gammas [cm-3]

    Returns
    -------
    res: np.ndarray
        Compton emissivity [erg cm-3 Hz-1 sr-1]
    """
    cdef int eps_sz = epsilon.shape[0]
    cdef int gamma_sz = gamma.shape[0]
    cdef int nu_sz = nu.shape[0]

    cdef double[::1] res = np.empty_like(nu)

    c_compton_emissivity(&res[0], &epsilon[0], &gamma[0], &nu[0],
                        &n_ph[0], &e_dist[0],
                        eps_sz, gamma_sz, nu_sz)

    return np.asarray(res)

def cool_onestep(double[::1] e_dist, double[::1] gamma, double[::1] e_dist_inj,
                    double B, double R,
                    double dt, double exp_b, double t_esc, double inj_r):
    """
    One step generic temporal evolution function for a "blob". Adiabatic and synchrotron cooling
    are included. It is also possible to inject electrons and define an escpace time.

    .. note::

        This function doesn't include any sanity check for given ``dt``. If it is too fast
        for given set of parameters, the function will return garbage results without a warning.

    Parameters
    ----------
    e_dist: np.ndarray
        1-D electron distribution to evaluated for the next step (Note: changed in place!)
    gamma: np.ndarray
        1-D gamma array same size of e_dist
    e_dist_inj: np.ndarray
        Normalized 1-D injected electron distribution same size of e_dist
    B: float
        Magnetic field [G]
    R: float
        Radius of the blob [cm]
    dt: float
        Time resolution [s], ie. the function returns the state after dt.
    exp_b: float
        Expansion speed of the blob [c](speed of light)
    t_esc: float
        Escape timescale [s]
    inj_r: float
        Injection rate [e-/s], e_dist_inj multiplied by this.

    Returns
    -------
    B: float
        Updated B after time dt (if exp_b > 0)
    R: float
        Updated R after time dt (if exp_b > 0)
    """
    cdef int sz = gamma.shape[0]

    cdef Source source_t
    source_t.B = B
    source_t.R = R

    c_cool_onestep(&e_dist[0], sz, &gamma[0], &e_dist_inj[0], 
                dt, exp_b, t_esc, inj_r, <Source*> &source_t)

    return source_t.B, source_t.R

def ray_tracing(double[:,:,::1] j_nu, double[:,:,::1] a_nu, double dx):
    """
    This is a pythonic radiative transfer computation function in the sense that 
    it operates pre-constructed j_nu and a_nu grid.
    The function expects 3-D arrays.
    Ray tracing is performed along the last axis.
    First two axis are reserved for nu and iteration of traces.

    Parameters
    ----------
    j_nu: np.ndarray
        3-D grid of emissivities [erg s-1 cm-3 sr-1 Hz-1]
    a_nu: np.ndarray
        3-D grid of absorption coefficienties [cm-1]
    dx: float
        Distance between each grid point along the traces

    Returns
    -------
    I_nu: np.ndarray
        3-D array of I_nu [erg s-1 cm-2 sr-1 Hz-1]
    """
    cdef double[:,:,::1] res = np.empty_like(j_nu)

    cdef int sz1 = j_nu.shape[0]
    cdef int sz2 = j_nu.shape[1]
    cdef int sz3 = j_nu.shape[2]
    cdef int i

    for i in prange(sz1, nogil=True):
        rtrace(&res[i,0,0], sz2, sz3, &j_nu[i,0,0], &a_nu[i,0,0], dx)

    return np.asarray(res)

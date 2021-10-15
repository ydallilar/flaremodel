# Licensed under a 3-clause BSD style license - see LICENSE

try:
    import pyopencl as cl
    import pyopencl.array as cl_array

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    GPU_ENABLED=True
except:
    GPU_ENABLED=False

import numpy as np

__all__ = ["gTemplate", "gComptonEmissivity", "GPU_ENABLED"]

class gTemplate:

    _clcode = ""

    def __init__(self, *args):
        assert GPU_ENABLED, "GPU functions are not available..."
        self._build()
        self._init_vars(*args)

    def _build(self):
        self.prg = cl.Program(ctx, self._clcode)
        try:
            self.prg.build()
        except:
            print("Error:")
            print(self.prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
            raise

    def _init_vars(self, *args):
        pass

    def __call__(self, *args):
        pass

class gComptonEmissivity(gTemplate):
    """
    PyOpenCL implementation of :func:`compton_emissivity`. Note that this is a class not a function. It needs to be executed as:

    >>> gComptonIntegral(nu, epsilon, gamma)(n_ph, e_dist)

    The reason is to bypass the overhead of compiling OpenCL code and setting up the calculation grid. So, it can be used for repetitive calculations,

    >>> g_compton_emissivity = gComptonEmissivity(nu, epsilon, gamma)
    >>> for i in range(n):
    ...     res_i = g_compton_emissivity(n_ph_i, e_dist_i)

    Parameters
    ----------
    nu: np.ndarray
        Requested scattered photon frequencies [Hz]
    epsilon: np.ndarray
        Input incident photon energies, note h*nu
    gamma: np.ndarray
        Gamma of electron distribution
    """

    _clcode = """
    __constant double M_e=9.10938188e-28;
    __constant double c=2.99792458e10;
    __constant double h=6.62606896e-27;
    __constant double sigma_T=6.65245893699e-25;
    
    double KN(double epsilon, double gamma, double nu){
    
        double epsilon1 = nu*h;
        double gamma_e = 4*epsilon*gamma/(M_e*c*c); 
        double q = epsilon1/(gamma_e*(gamma*M_e*c*c - epsilon1)); 
        double res = 3*sigma_T*c/(4*gamma*gamma*epsilon) *(
                                2*q*log(q) + 
                                (1+2*q)*(1-q) + 
                                (gamma_e*q)*(gamma_e*q)*(1-q)/
                                (2*(1 + gamma_e*q))
                                );
    
        if (isnan(res)) return 0;
        if (q > 1) return 0;
        if (res < 0) return 0;
    
        return res;
    }

    __kernel void compton_fun(__global double *tmp, __global double *epsilon, 
                                                    __global double *gamma,
                                                    __global double *nu,
                                                    __global double *n_ph,
                                                    __global double *e_dist,
                                                    int eps_sz){
    
        int i = get_global_id(0); // nu
        int j = get_global_id(1); // gamma
        int k; // epsilon
        int stride = get_global_size(1);
        
        double nu_i = nu[i];
        double gamma_j = gamma[j];
        double e_dist_j = e_dist[j];
        
        double sum = 0;
        double last = n_ph[0]*e_dist_j*KN(epsilon[0], gamma_j, nu_i);
        double next;
        for(k=0; k<eps_sz-1; k++){
            next = n_ph[k+1]*e_dist_j*KN(epsilon[k+1], gamma_j, nu_i);
            sum += (last+next)*(epsilon[k+1]-epsilon[k])*0.5;
            last = next;
        }
        tmp[i*stride+j] = h*h*nu_i*sum/(4*M_PI);
    
    }
    """

    def _init_vars(self, nu, epsilon, gamma):

        self.gamma = np.ascontiguousarray(gamma, dtype=np.double)
        self.nu = np.ascontiguousarray(nu, dtype=np.double)
        self.epsilon = np.ascontiguousarray(epsilon, dtype=np.double)
        self.eps_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.epsilon)
        self.gamma_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.gamma)
        self.nu_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.nu)

        self.n_ph_dev = cl.Buffer(ctx, mf.READ_ONLY, size=self.epsilon.nbytes)
        self.edist_dev = cl.Buffer(ctx, mf.READ_ONLY, size=self.gamma.nbytes)

        self.tmp = np.empty([self.nu.shape[0], self.gamma.shape[0]])
        self.tmp_dev = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.tmp.nbytes)

    def __call__(self, n_ph, e_dist):
        """
        Parameters
        ----------
        n_ph: np.ndarray
            Photon densities at corresponding epsilons
        e_dist: np.ndarray
            Electron densities at corresponding gammas

        Returns
        -------
        res: np.ndarray
            Compton emissivity [erg s-1 cm-3 sr-1]
        """

        cl.enqueue_copy(queue, self.n_ph_dev, np.ascontiguousarray(n_ph, dtype=np.double))
        cl.enqueue_copy(queue, self.edist_dev, np.ascontiguousarray(e_dist, dtype=np.double))

        self.prg.compton_fun(queue, (self.nu.shape[0], self.gamma.shape[0]), None,
                    self.tmp_dev, self.eps_dev, self.gamma_dev,
                    self.nu_dev, self.n_ph_dev, self.edist_dev, np.int32(self.epsilon.shape[0])).wait()   

        cl.enqueue_copy(queue, self.tmp, self.tmp_dev)

        return np.trapz(self.tmp, self.gamma, axis=1)

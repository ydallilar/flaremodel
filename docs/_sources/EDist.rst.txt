Electron Distributions
======================

.. currentmodule:: flaremodel

.. autofunction:: eDist

.. _EDIST_HOWTO:

How to add new electron distributions
-------------------------------------

Unfortunately, we are not able to provide a convenient way to add new electron distributions in python as electron distributions
are tightly integrated in C utilities at lower level. But, C interface is modular and it is possible to add new distributions on the source code.

Our powerlaw implementation at C level is given below:

.. code-block:: C
    
    // cfuncs/edist.c
    //...

    // params : p, gamma_min, gamma_max
    double powerlaw(double gamma, void *params){
        double *p = (double*) params;
        return pow(gamma, -p[0]);
    }
    
    //...

    double powerlaw_norm(void* params){
        double *p = (double*) params;
        return (P_THRESH > fabs(p[0] - 1)) ? (log(p[2])-log(p[1])) : (pow(p[1], -p[0]+1)-pow(p[2], -p[0]+1))/(p[0]-1); 
    }
    //...

    // cfuncs/edist.h
    //...
    // params : p, gamma_min, gamma_max
    double powerlaw(double gamma, void *params);
    //...
    double powerlaw_norm(void *params);
    //...

Then, the distribution needs to be registered with a name tag in Cython interface. This makes the distribution available to the rest of the code with the given name tag.

.. code-block:: Cython

    # flaremodel/utils/cfuncs.pyx

    # ...

    cdef extern from "edist.h":
        double powerlaw (double, void*) # <--
        double powerlawexpcutoff (double, void*)
        double thermal (double, void*)
        double kappa (double, void*)
        double bknpowerlaw (double, void*)
        double bknpowerlawexpcutoff (double, void*)
        double powerlaw_norm (void*) # <--
        double thermal_norm (void*)
        double kappa_norm (void*)
        double bknpowerlaw_norm (void*)
        int c_eDist         "eDist"         (...)

    # ...

    cdef _set_source_params(Source *source_t, str edist):

        cdef double *params = source_t.params

        if edist == "powerlaw": # <--
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

    # ...

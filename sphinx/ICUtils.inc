Inverse Compton Utilities
-------------------------

.. currentmodule:: flaremodel

Inverse Compton calculations are the most computationally expensive routines by far in this package. Hence, we provide two alternatives; 
one written in C and parallelized with OpenMP, alternative version targets GPUs with PyOpenCL implementation. Depending on hardware or on a specific problem,
one may be more optimal than the other. While the GPU version will thrive on high end hardware, it is hardware agnostic and can be ran on integrated GPUs as well. 

.. autofunction:: compton_emissivity

.. autoclass:: gComptonEmissivity
    :members: __call__

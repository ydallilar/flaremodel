Synchrotron Utilities
---------------------

.. currentmodule:: flaremodel

We provide two methods to calculate synchrotron emissivity and absorption coefficent:

    - `brute` (:func:`j_nu_brute`, :func:`a_nu_brute`): Numerical computation using built-in electron distributions
    - `userdist` (:func:`j_nu_userdist`, :func:`a_nu_userdist`) Numerical computation from an arbitrary numerical distribution on a grid. It is often better to provide a grid equidistant in logspace. 

.. autofunction:: j_nu_brute
.. autofunction:: a_nu_brute

.. autofunction:: j_nu_userdist
.. autofunction:: a_nu_userdist

Geometries
==========

.. currentmodule:: flaremodel

Base class to implement new geometries is :class:`Geometry`. Methods to be implemented are:

    - :func:`Geometry._compute_synchrotron`
    - :func:`Geometry._compute_compute_IC`
    - :func:`Geometry._compute_compute_SSC`
    - :func:`Geometry._compute_photon_density`, only for SSC calculations

It is not necessary to implement all functions, just the ones desired. The equivalent ones without underscore in front are reserved for user interface. 
This is for including additional functionality such as Doppler boosting. (see :class:`DopplerBooster`).

There are two geometries prodived with the code:

    - :class:`HomogeneousSphere`, a basic homogeneous sphere
    - :class:`RadialSphere`, spherical geometry in which it is possible to specify radial profile in electron density and magnetic field.

.. autoclass:: Geometry
    :members: _compute_synchrotron, _compute_photon_density, _compute_SSC, _compute_IC, compute_synchrotron, compute_SSC, compute_IC

.. autoclass:: HomogeneousSphere

.. autoclass:: RadialSphere

.. autoclass:: HomogeneousSphereUserdist
    :members: _compute_synchrotron

.. autoclass:: DopplerBooster


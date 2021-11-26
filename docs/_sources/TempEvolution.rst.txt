Temporal Evolution
==================

.. currentmodule:: flaremodel

We provide two special cases of temporal evolution for electrons. These are built around :func:`cool_onestep`.
Although the function itself fast, temporal evolution requires many iterations using it.
But, python is very slow and iterations requiring python operations can introduce huge performance hit.
Source code for these avialable in `flaremodel/temporal.py` for guidance.

.. note:: If ``dt`` is not small enough, these will produce garbage results. Need to add a warning mechanism when it is the case (TBD).

.. autoclass:: vdLEdist
    :members: get_ngamma

.. autoclass:: GaussianInjectionEdist
    :members: get_ngamma
    
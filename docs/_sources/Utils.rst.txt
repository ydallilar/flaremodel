Low Level Utilities
===================

Here, we introduce low level utilities provided with the package to develop variety of models.
The functions given here is a bit more picky than normal python functions. 
In some cases, we note that it is necessary to provide numpy arrays in C contiguous form.
We will not discuss the details here, but in case of relevant errors, following can be used:

.. code-block:: python

    arr = np.ascontiguousarray(arr)

.. include:: SyncUtils.inc
.. include:: ICUtils.inc
.. include:: TemporalUtil.inc
.. include:: RTransUtils.inc

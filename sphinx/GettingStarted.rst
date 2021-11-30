Getting Started
===============

Flaremodel: A simple one zone code that can do many different flavours of flares! 

Installation
------------

The source code of `flaremodel` can be obtained from our github `repository`_.
Then, the package can be installed manually. 
This is useful if one, for example, may want to add new :ref:`electron distributions<EDIST_HOWTO>`.

.. code-block:: none

    git clone https://github.com/ydallilar/flaremodel
    # or to obtain a specific tag
    # https://github.com/ydallilar/flaremodel@vX.Y.Z
    cd flaremodel
    python setup.py install

Alternatively, the code can be directly installed from `PyPI`_.

.. code-block:: none

    pip install flaremodel

Open an issue on the github `repository`_ in case of errors. Probably, it won't compile on Windows for the moment.

.. _repository: https://github.com/ydallilar/flaremodel
.. _PyPI: https://pypi.org/

C extensions
------------

C extensions are the core of this package and provides fast utility functions. 
A C compiler and GSL headers are necessary to compile the code. 
OpenMP is optional (but needs to disabled manually) and enables multithreading within C extension.

.. code-block:: none

    # System-wide installations will also work. 
    # But, in case, it is not available, it is provided conda-forge channel of anaconda.
    conda install -c conda-forge gsl intel-openmp

If OpenMP is not available, the compile will fail.

.. code-block:: none

    # To disable as a workaround for the moment.
    export FLAREMODEL_OPENMP=0
    pip -v install flaremodel-0.2.1.tar.gz

Note that OpenMP will use optimal number of threads. Typically mean all the threads available.
There are three different ways to limit:

.. code-block:: none

    # In bash shell,
    export OMP_NUM_THREADS=n

    # In python shell before importing flaremodel
    import os; os.environ["OMP_NUM_THREADS"]=str(n)

    # During runtime
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=n)

There is a change in the Numpy C-API at version 1.20 and not compatible with previous versions of numpy. Hence, the following error may occur.
For the moment, we restrict numpy version to <1.20.0.

.. code-block:: none

    ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject

GPU Acceleration
----------------

GPU acceleration is aviable with OpenCL implemenation and only for inverse Compton scattering calculations.
It is optional relevant functions won't load if there is no proper setup avialable.

Instructions to setup opencl environment are given below:

.. code-block:: none

    # Install pyopencl from conda-forge
    conda install -c conda-forge pyopencl

    # There are two routes after this. Either:
    # 1. Use system-wide ocl runtime installed for a selected hardware:
    conda install -c conda-forge ocl-icd-system

    # 2. Or, install relevant runtime from conda-forge, eg. for intel graphics:
    conda install -c conda-forge intel-compute-runtime

Then, relevant hardware should show up (obviously will be different for other hardware).

.. code-block:: ipython

    In [1]: import pyopencl as cl                                                                                                                                                                        

    In [2]: ctx = cl.create_some_context()                                                                                                                                                               
    Choose platform:
    [0] <pyopencl.Platform 'Intel(R) OpenCL HD Graphics' at 0x56356e996720>
    Choice [0]:

At this point, hardware should be ready to use. Depending on the system, there may be more than one option.
To select the correct hardware, ``PYOPENCL_CTX`` environment variable needs to point to the appropriate hardware index otherwise the first index will be used by default.


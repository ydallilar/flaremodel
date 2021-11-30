# flaremodel

## Overview

*flaremodel* is a `Python` package designed for one-zone modeling of astrophysical synchrotron sources.

The code aims to provide modular interface for the modeling of synchrotron sources with efficient low level utility functions written in `C` - the `cfuncs` module.
These functions constitute the core of the *flaremodel*. The `Python` interface is built on top of these functions for convenience and guidance to formulate new models.
Multithreading support is implemented with `OpenMP` at a lower level in the `cfuncs` module where it is found useful.
At the same time, these low level functions bypass the infamous *Global Interpreter Lock (GIL)* of `Python`.
In simple words, users can implement efficient high level `multithreading` on python side when low level C functions are used and avoid the costly overheads of `multiprocessing`.  

Our [documentation](https://ydallilar.github.io/flaremodel/) covers the installation, module references and examples.

The documentation mostly covers the usage of the code. The detailed methodology/physics are covered in our [paper (to be added*)](Oops). 
The examples introduced in the paper are provided as jupyter notebooks in `notebooks/` directory of the tarball and can also serve as a test of the compiled code.

Please cite the paper when the code is used in academical studies. Note that the paper refers to [our first stable release](https://github.com/ydallilar/flaremodel/releases/tag/v1.0.0). 



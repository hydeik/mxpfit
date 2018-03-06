# mxpfit

mxpfit -- a C++ template library for Multi-eXPonential FIT

## Description

`mxpfit` is a library for finding optimal approximation of a function by a
multi-exponential sum, which is given as

<!-- $$                                                                             -->
<!-- f(t) = \sum_{j=1}^{M} c_{j} e^&{-a_{j}t}, \, (t > 0)                           -->
<!-- $$                                                                             -->
<!-- where $a_{j}\in\mathbb{C},\, \mathrm{Re}(a_{j})>0$ and $c_{j} \in \mathbb{C}.$ -->

<img src=https://latex.codecogs.com/gif.latex?f(t)&space;=&space;\sum_{j=1}^{M}&space;c_{j}&space;e^&{-a_{j}t},&space;\,&space;(t&space;>&space;0) />

where
<img src=https://latex.codecogs.com/gif.latex?a_{j}\in\mathbb{C},\,&space;\mathrm{Re}(a_{j})>0 />
and
<img src=https://latex.codecogs.com/gif.latex?c_{j}&space;\in&space;\mathbb{C} />

The library provides mainly two application programming interfaces (APIs) for 1)
recovering exponents and weights in exponential sum from sampled data on a
uniform grid via modified fast ESPRIT algorithm, and 2) reducing the number of
terms of a given exponential sum via modified balanced truncation algorithm. The
library is written in C++ with templates, which enable us to perform the
simulation using various scalar type in good performance.


## Requirement

 - A modern C++ compiler that supports the C++11 standard, 
   such as GCC (>= 4.8.0) and Clang (>= 3.2).
 - [Eigen](http://eigen.tuxfamily.org/) -- linear algebra library
 - [FFTW3](http://www.fftw.org/) -- fast Fourier transform library
 - [CMake](https://cmake.org/) (cross-platform make) for building examples and tests
 - [Doxygen](http://doxygen.org/) for generating source code documents [optional]

## Description of Files

- include/
    - fftw3/
        - shared_plan.hpp: thread-safe wrapper classes for one-dimensional FFT plans
    - mxpfit/
        - balanced_truncation.hpp: Implements modified balanced truncation
          algorithm for finding exponential sum with smaller order.
        - exponential_sum.hpp: Container classes holding parameters of exponential
          sum function.
        - fast_esprit.hpp: Implements the fast ESPRIT algorithm for finding
          exponential sum approximation from sampled data on a uniform grid.
        - esprit.hpp: Implements the original ESPRIT algorithm for finding exponential
          sum approimxation from sampled data on a uniform grid. This algorithm is much
          slower than fast ESPRIT method described above. This has been implemented only
          for testing purpose.
        - prony_like_method_common.hpp: utility functions internally used for implementing
          Prony-like methods.
        - hankel_matrix.hpp: A rectangular Hankel matrix class with fast Hankel
          matrix-vector product.
        - jacobi_svd.hpp: One-sided Jacobi singular value decomposition algorithm.
        - matrix_free_gemv.hpp: provides interface overloading `operator*` for
          product of Hankel/Vandermonde matrix and any vector type in Eigen.
        - partial_lanczos_bidiagonalization.hpp: Find a low-rank approximation of a
          matrix with partial Lanczos bidiagonalization with full reorthogonalization.
        - quasi_cauchy_rrd.hpp: Computes a rank-revealing Cholesky decomposition of
          a self-adjoint quasi-Cauchy matrix with high-relative accuracy.
        - self_adjoint_coneigensolver.hpp: Computes con-eigenvalue decomposition of
          self-adjoint matrix having rank-revealing decomposition.
        - vandermonde_least_squares.hpp: Solve least square solution of
          overdetermined Vandermonde system.
        - vandermonde_matrix.hpp: Wonderment matrix class with matrix-vector
          product interface class with matrix-vector product interface.
- examples
    - balanced_truncation.cpp: an example program for `BalancedTruncation` class
    - fast_esprit.cpp: an example program for `FastESPRIT` class
    - esprit_gauss.cpp: another example program for `FastESPRIT` class
    - esprit_compare.cpp: compare the performance of `ESPRIT` and `FastESPRIT` classes.
- tests/: unit tests


## Installation

`mxpfit` is a header only library. You can use it by including header files
under `include` directory.

For building example programs, type the following commands on terminal:

```
$ cd {mxpfit_root_dir}
$ mkdir build
$ cd build
$ cmake -DBUILD_EXAMPLES=on -DBUILD_TEST=on ..
$ make
```

## Usage:
See `examples/fast_esprit.cpp`, `examples/esprit_gaussian.cpp` and
`examples/balanced_truncation.cpp`.


## Licence
Copyright (c) 2017-2018 Hidekazu Ikeno

Released under the [MIT license](http://opensource.org/licenses/mit-license.php)

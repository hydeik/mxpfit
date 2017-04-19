# mxpfit 

mxpfit -- a C++ template library for Multi-eXPonential FIT

## Description

`mxpfit` is a library for finding optimal approximation of a function by a
multi-exponential sum, which is given as

$$
f(t) = \sum_{j=1}^{M} c_{j} e^&{-a_{j}t}, \, (t > 0)
$$

where $a_{j}\in\mathbb{C},\, \mathrm{Re}(a_{j})>0$ and $c_{j} \in \mathbb{C}.$ 

<!-- The library provides the interface for evaluating exponents and weights from -->
<!-- sampling data on a uniform grid using the fast ESPRIT (Estimation of Signal -->
<!-- Parameters via Rotational Invariance Techniques) algorithm originally proposed -->
<!-- by Potts and Tasche [1]. The parameters can be estimated efficiently even from -->
<!-- the large number of sampling data. The modified balanced truncation algorithm to -->
<!-- find the multi-exponential sum with smaller order is also provided. This feature -->
<!-- would be useful for finding optimal exponential sum approximations of analytic -->
<!-- functions. -->

## Requirement

 - A modern C++ compiler that supports the C++11 standard, 
   such as GCC (>= 4.8.0) and Clang (>= 3.2).
 - [Eigen](http://eigen.tuxfamily.org/) -- linear algebra library
 - [FFTW3](http://www.fftw.org/) -- fast Fourier transform library
 - [CMake](https://cmake.org/) (cross-platform make) for building examples and tests
 - [Doxygen](http://doxygen.org/) for generating source code documents [optional]

## Files

+ include/
  - fftw3/
    - shared_plan.hpp: thread-safe wrapper classes for one-dimensional FFT plans
  - mxpfit/
    - balanced_truncation.hpp: Implements modified balanced truncation
      algorithm for finding exponential sum with smaller order.
    - exponential_sum.hpp: Container classes holding parameters of exponential
      sum function.
    - fast_esprit.hpp: Implements the fast ESPRIT algorithm for finding
      exponential sum approximation from sampled data on a uniform grid.
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
+ examples
  - balanced_truncation.cpp: an example program for `BalancedTruncation` class
  - fast_esprit.cpp: an example program for `FastESPRIT` class
+ tests/: unit tests


## Installation

`mxpfit` is a header only library. You can use it by including header files
under `include` directory.

For building example programs, type the following commands on terminal:

 $ mkdir build
 $ cmake {mxpfit_root_dir}
 $ make


## Usage: 
See `examples/fast_esprit.cpp` and `examples/balanced_truncation.cpp`.


## Licence
Copyright (c) 2017 Hidekazu Ikeno

Released under the [MIT license](http://opensource.org/licenses/mit-license.php)

/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2017 Hidekazu Ikeno
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef MXPFIT_POW_KERNEL_HPP
#define MXPFIT_POW_KERNEL_HPP

#include <mxpfit/exponential_sum.hpp>
#include <mxpfit/modified_prony_reduction.hpp>

namespace mxpfit
{

namespace detail
{

/// \internal
///
/// ### newton
///
/// Solve an equation \f$f(x)=0\f$ by Newton's method
///
/// \tparam T value type for function argument \f$x\f$
/// \tparam NewtonFunctor a unwary function that takes \f$x\f$ as an argument
/// and
///         returns pair of function value \f$f(x)\f$ and its first derivative
///         \f$f^{\prime}(x).\f$
///
/// \param[in] guess     initial guess of the solution
/// \param[in] tol       tolerance for convergence
/// \param[in] fn        an instance of `NewtonFunctor`
/// \param[in] max_iter  maximum number of iterations
///
template <typename T, typename NewtonFunctor>
T newton(T guess, T tol, NewtonFunctor fn, std::size_t max_iter = 1000)
{
    using std::abs;
    auto counter = max_iter;
    auto x       = guess;

    while (counter--)
    {
        // We assume df(x) never to be too small.
        const auto f_and_df = fn(x);
        const auto delta    = std::get<0>(f_and_df) / std::get<1>(f_and_df);
        x -= delta;
        if (abs(delta) < abs(x) * tol)
        {
            break;
        }
    }

    return x;
}

} // namespace detail

///
/// ### computePowKernel
///
/// \brief Compute parameters for the exponential sum approximation of power
/// funciton, \f$r^{-\beta}\,(\beta>0).\f$
///
/// \tparam T Real scalar type for the parameters of the exponential sum
///
/// This funtion computes parameters to approximate the power functions
/// \f$r^{-\beta}\,(\beta>0)\f$ with a linear combination of exponential
/// functions,
///
/// \f[
///   \|r^{-\beta}-sum_{m=1}^{M} w_{m} e^{-a_{m} x} \| < r^{-\beta}\epsilon
/// \f]
///
/// for any given accuracy \f$\epsilon > 0\f$ and distance to singularity
/// \f$\delta>0\f$ real axis `(x)`.
///
/// The multi-exponential function is obtained by discretize the integral
/// representation of spherical Bessel function
///
/// #### References
///
/// 1. G. Beylkin and L. Monz\'{o}n, "Approximation by exponential sums
///    revisited", Appl. Comput. Harmon. Anal. 28 (2010) 131-149.
///    [DOI: https://doi.org/10.1016/j.acha.2009.08.011]
/// 2. W. McLean, "Exponential sum approximations for \f$t^{-\beta}\f$",
///    arXiv:1606.00123 [math]
///
template <typename T>
ExponentialSum<T, T> computePowKernel(T beta, T delta, T eps)
{
    using Index      = Eigen::Index;
    using ResultType = ExponentialSum<T, T>;

    using Eigen::numext::log;
    using Eigen::numext::exp;

    const T eps_d     = eps / T(3); // upper bound of discretization error
    const T eps_t     = eps / T(3); // upper bound of truncation error
    const T log_eps_t = log(eps_t);

    ResultType ret;

    return ret;
}

} // namespace: mxpfit

#endif /* MXPFIT_POW_KERNEL_HPP */

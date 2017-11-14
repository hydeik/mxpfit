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

///
/// \file approx_sph_bessel.hpp
/// Exponential sum approximations of the spherical Bessel functions.
///

#ifndef MXPFIT_APPROX_SPH_BESSEL_HPP
#define MXPFIT_APPROX_SPH_BESSEL_HPP

#include <mxpfit/math/legendre.hpp>
#include <mxpfit/quad/tanh_sinh_rule.hpp>

#include <mxpfit/balanced_truncation.hpp>
#include <mxpfit/exponential_sum.hpp>

namespace mxpfit
{
///
/// ### SphBesselKernel
///
/// Compute the multi-exponential function that approximates the spherical
/// Bessel function of the first kind such that
///
/// \f[
///   \|j_{n}(x)-sum_{m=1}^{M} w_{m} e^{-a_{m} x} \| < \epsilon
/// \f]
///
/// for integer `n` on real axis `(x)`, where \f$ \epsilon \f$ is an arbitrary
/// small positive number.
///
/// The multi-exponential function is obtained by discretize the integral
/// representation of spherical Bessel function
///
/// \f[
///   j_{n}(x)=\int_{-1}^{1} e^{ixt} P_n(t) dt
/// \f]
///
/// The integrand in above expression is highly oscillating when `x` becomes
/// large, thus the direct application of some quadrature rule is inefficient.
/// To avoid this problem, we extend the integral on the complex plane and
/// select the contour as the steepest descent paths, as follows.
///
/// \f{eqnarray*}{
///   j_{n}(x) &=& I_1 + I_2 + (-1)^{n}(I_1^{\ast} + I_{2}^{\ast}), \\
///   I_{1} &=& \frac{(-i)^{n+1}}{2} \int_{0}^{R} e^{-(y-i)x} P_{n}(1+iy) dy, \\
///   I_{2} &=& \frac{(-i)^{n}}{2} \int_{-1}^{0} e^{-(R-ix)y} P_{n}(y+iR) dy, \\
/// \f}
///
/// The integrals \f$ I_1 \f$ and \f$ I_2 \f$ are discretized using appropriate
/// quadrature rule, then the balanced truncation method is applied to reduce
/// the number of terms.
///
template <typename T>
ExponentialSum<std::complex<T>> approxSphBessel(Eigen::Index n, T threshold)
{
    using Real       = T;
    using Complex    = std::complex<T>;
    using Index      = Eigen::Index;
    using ResultType = ExponentialSum<std::complex<T>>;

    using Eigen::numext::conj;
    using Eigen::numext::log;
    using Eigen::numext::cos;
    using Eigen::numext::sin;
    using Eigen::numext::exp;
    // ----- Constants
    constexpr const auto zero = Real();
    constexpr const auto one  = Real(1);
    constexpr const auto half = Real(0.5);
    constexpr const auto eps  = std::numeric_limits<Real>::epsilon();
    // (-i)^n
    constexpr const Complex pow_i[4] = {Complex(one, zero), Complex(zero, -one),
                                        Complex(-one, zero),
                                        Complex(zero, one)};

    //----- adjustable parameters
    const auto R       = Real(4) / std::max(Index(1), n);
    const auto n_quad1 = Index(200);
    const auto tiny1   = eps / n_quad1;

    const auto n_quad2 = Index(120 + 2 * n);
    const auto tiny2   = eps / n_quad2;
    //---------------------------

    //
    // Find R_shift such that j_n(R_shift) < eps
    //
    Real R_shift = Real();
    if (n > Index())
    {
        Real log_dfact = Real(); // log[(2n+1)!!]
        for (Index k = 1; k <= n; ++k)
        {
            log_dfact += log(Real(2 * k + 1));
        }
        const auto thresh = std::min(threshold, sqrt(eps));
        R_shift           = exp((log_dfact + log(thresh)) / Real(n));
        if (R_shift < Real(0.1))
        {
            R_shift = Real();
        }
        // std::cout << "*** l = " << n << ", R_shift = " << R_shift << '\n';
    }

    //
    // Discretization of integral I_1 on the path `-1 + i y`  for
    // `0 <= y <= R`
    //
    quad::DESinhTanhRule<T> rule1(n_quad1, tiny1);
    ResultType es1(n_quad1);

    const auto R_half = R / 2;

    for (Index k = 0; k < n_quad1; ++k)
    {
        const auto yk = R_half * rule1.distanceFromLower(k); // node
        const auto wk = R_half * rule1.w(k);                 // weight
        const auto ak = Complex(yk, one);
        const auto pk = Complex(zero, wk) * exp(ak * R_shift) *
                        math::legendre_p(n, Complex(-one, yk));

        es1.exponent(k) = ak;
        es1.weight(k)   = pk;
    }

    //
    // Discretization of integral I_2 on the path `x + iR` for
    // `0 <= x <= 1`
    //
    quad::DESinhTanhRule<T> rule2(n_quad2, tiny2);
    ResultType es2(n_quad2);

    for (Index k = 0; k < n_quad2; ++k)
    {
        const auto xk = half * rule2.distanceFromLower(k); // node
        const auto wk = half * rule2.w(k);                 // weight
        const auto ak = Complex(R, -xk);
        const auto pk =
            wk * exp(ak * R_shift) * math::legendre_p(n, Complex(xk, R));

        es2.exponent(k) = ak;
        es2.weight(k)   = pk;
    }

    //
    // Truncation for I_1 and I_2 separately
    //
    // mxpfit::BalancedTruncation<Complex> trunc1;
    // trunc1.setThreshold(threshold);
    // es1 = trunc1.compute(es1);

    // mxpfit::BalancedTruncation<Complex> trunc2;
    // trunc2.setThreshold(threshold);
    // es2 = trunc2.compute(es2);

    //
    // Merge two sums
    //
    ResultType es_merged(es1.size() + es2.size());
    es_merged.exponents().head(es1.size()) = es1.exponents();
    es_merged.exponents().tail(es2.size()) = es2.exponents();
    es_merged.weights().head(es1.size())   = es1.weights();
    es_merged.weights().tail(es2.size())   = es2.weights();

    //
    // Truncation for I_1 and I_2 simultaneously
    //
    mxpfit::BalancedTruncation<Complex> trunc1;
    es_merged = trunc1.compute(es_merged, threshold);

    ResultType es_result(2 * es_merged.size());

    const auto pre1 = pow_i[n % 4] / Real(2); // (-i)^n / 2
    const auto pre2 = (n & 1) ? -pre1 : pre1;

    for (Index i = 0; i < es_merged.size(); ++i)
    {
        const auto ai                 = es_merged.exponent(i);
        es_result.exponent(2 * i + 0) = ai;
        es_result.exponent(2 * i + 1) = conj(ai);

        const auto wi               = es_merged.weight(i) * exp(-ai * R_shift);
        es_result.weight(2 * i + 0) = pre1 * wi;
        es_result.weight(2 * i + 1) = pre2 * conj(wi);
    }

    return es_result;
}

} // namespace mxpfit
#endif /* MXPFIT_APPROX_SPH_BESSEL_HPP */

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

#include <iomanip>
#include <iostream>

#include "constants.hpp"
#include "legendre.hpp"
#include "tanh_sinh_rule.hpp"

#include <mxpfit/aak_reduction.hpp>
#include <mxpfit/balanced_truncation.hpp>
#include <mxpfit/exponential_sum.hpp>

#include <boost/math/special_functions/bessel.hpp>

template <typename T>
constexpr T pi()
{
    return T(
        3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679);
}
///
/// ### BesselKernel
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
class BesselKernel
{
public:
    using Index   = Eigen::Index;
    using Real    = T;
    using Complex = std::complex<Real>;

    using ExponentialSumType = mxpfit::ExponentialSum<Complex, Complex>;
    using ExponentsArray     = typename ExponentialSumType::ExponentsArray;
    using WeightsArray       = typename ExponentialSumType::WeightsArray;

    static ExponentialSumType compute(Index order, Real threshold);
};

template <typename T>
typename BesselKernel<T>::ExponentialSumType
BesselKernel<T>::compute(Index n, Real threshold)
{
    using Eigen::numext::conj;
    using Eigen::numext::exp;
    using Eigen::numext::log;
    using Eigen::numext::cos;
    using Eigen::numext::acos;
    using Eigen::numext::sin;
    using Eigen::numext::sqrt;
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
    const auto R = Real(3) / std::max(Index(1), n);

    const auto n_quad1 = Index(500);
    const auto tiny1   = sqrt(eps) * threshold / n_quad1;

    const auto n_quad2 = Index(150 + 2 * n);
    const auto tiny2   = eps / n_quad2;
    //---------------------------

    //
    // Find R_shift such that j_n(R_shift) < eps
    //
    Real R_shift = Real();
    if (n > Index())
    {
        Real log_dfact = Real(); // log[2^n * n!]
        for (Index k = 1; k <= n; ++k)
        {
            log_dfact += log(Real(2 * k));
        }
        const auto thresh = std::min(threshold, sqrt(eps));
        R_shift           = exp((log_dfact + log(thresh)) / Real(n));
        if (R_shift < Real(0.1))
        {
            R_shift = Real();
        }
    }

    ExponentialSumType es_merged(n_quad1 + n_quad2);

    //
    // Discretization of integral I_2 on the path `z = x + iR` for `0<=x<=1`
    //
    quad::DESinhTanhRule<T> rule2(n_quad2, tiny2);

    // std::cout << "*** quadrature rule\n";
    // Real xsum = Real();
    // for (Index k = 0; k < n_quad2; ++k)
    // {
    //     xsum += rule2.adjacentDifference(k);
    //     std::cout << rule2.x(k) << '\t' << rule2.adjacentDifference(k) <<
    //     '\t'
    //               << xsum << '\t' << std::abs(rule2.x(k) - xsum) << '\n';
    // }

    for (Index k = 0; k < n_quad2; ++k)
    {
        const auto xk = half * rule2.distanceFromLower(k); // node
        const auto uk = half * rule2.distanceFromUpper(k); // 1 - x
        const auto wk = half * rule2.w(k);                 // weight

        const auto ak = Complex(R, -xk);
        const auto pk = wk * exp(ak * R_shift) *
                        cos(Real(n) * acos(Complex(xk, R))) /
                        sqrt(Complex(one + xk, R) * Complex(uk, -R));

        es_merged.exponent(k) = ak;
        es_merged.weight(k)   = pk;
    }

    //
    // Discretization of integral I_1 on the path `1 + i y`  for
    // `0 <= y <= R`
    //
    quad::DESinhTanhRule<T> rule1(n_quad1, tiny1);

    const auto R_half = R / 2;
    Index ipos        = n_quad2;
    for (Index k = n_quad1; k > 0; --k)
    {
        const auto yk = R_half * rule1.distanceFromLower(k - 1); // node
        const auto wk = R_half * rule1.w(k - 1);                 // weight
        const auto ak = Complex(yk, -one);
        const auto pk = Complex(zero, -wk) * exp(ak * R_shift) *
                        cos(Real(n) * acos(Complex(one, yk))) /
                        (sqrt(yk * Complex(yk, Real(-2))));

        es_merged.exponent(ipos) = ak;
        es_merged.weight(ipos)   = pk;
        ++ipos;
    }

    // Remove terms with same exponents
    es_merged.uniqueExponents(Real());

    //
    // Truncation for I_1 and I_2 separately
    //
    // mxpfit::BalancedTruncation<Complex> trunc1;
    // es1 = trunc1.compute(es1, threshold * eps);

    // mxpfit::BalancedTruncation<Complex> trunc2;
    // es2 = trunc2.compute(es2, threshold);

    //
    // Truncation for I_1 and I_2 simultaneously
    //
    mxpfit::BalancedTruncation<Complex> trunc1;
    es_merged = trunc1.compute(es_merged, threshold * eps * eps);

    // mxpfit::AAKReduction<Complex> reduction;
    // reduction.setThreshold(threshold);
    // reduction.compute(es_merged);

    ExponentialSumType es_result(2 * es_merged.size());
    const auto pre1 = pow_i[n % 4] / pi<Real>(); // (-i)^n / pi
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

// template <typename T>
// typename BesselKernel<T>::ExponentialSumType
// BesselKernel<T>::compute(Index n, Real threshold)
// {
//     using Eigen::numext::conj;
//     using Eigen::numext::exp;
//     using Eigen::numext::log;
//     using Eigen::numext::cos;
//     using Eigen::numext::acos;
//     using Eigen::numext::sin;
//     using Eigen::numext::sqrt;
//     // ----- Constants
//     constexpr const auto zero = Real();
//     constexpr const auto one  = Real(1);
//     constexpr const auto half = Real(0.5);
//     constexpr const auto eps  = std::numeric_limits<Real>::epsilon();
//     // (-i)^n
//     constexpr const Complex pow_i[4] = {Complex(one, zero), Complex(zero,
//     -one),
//                                         Complex(-one, zero),
//                                         Complex(zero, one)};

//     //----- adjustable parameters
//     const auto R       = Real(3) / std::max(Index(1), n);
//     const auto n_quad1 = Index(500);
//     // const auto tiny1   = eps * eps;
//     const auto tiny1 = eps * sqrt(eps) / n_quad1;

//     const auto n_quad2 = Index(150 + 2 * n);
//     const auto tiny2   = eps / n_quad2;
//     //---------------------------

//     //
//     // Find R_shift such that j_n(R_shift) < eps
//     //
//     Real R_shift = Real();
//     if (n > Index())
//     {
//         Real log_dfact = Real(); // log[2^n * n!]
//         for (Index k = 1; k <= n; ++k)
//         {
//             log_dfact += log(Real(2 * k));
//         }
//         const auto thresh = std::min(threshold, sqrt(eps));
//         R_shift           = exp((log_dfact + log(thresh)) / Real(n));
//         if (R_shift < Real(0.1))
//         {
//             R_shift = Real();
//         }
//     }

//     //
//     // Discretization of integral I_1 on the path `-1 + i y`  for
//     // `0 <= y <= R`
//     //
//     quad::DESinhTanhRule<T> rule1(n_quad1, tiny1);
//     ExponentialSumType es1(n_quad1);

//     const auto R_half = R / 2;
//     for (Index k = 0; k < n_quad1; ++k)
//     {
//         const auto yk = R_half * rule1.distanceFromLower(k); // node
//         const auto wk = R_half * rule1.w(k);                 // weight
//         const auto ak = Complex(yk, one);
//         const auto pk = Complex(zero, wk) * exp(ak * R_shift) *
//                         cos(Real(n) * acos(Complex(-one, yk))) /
//                         (sqrt(yk * Complex(yk, Real(2))));

//         es1.exponent(k) = ak;
//         es1.weight(k)   = pk;
//     }

//     //
//     // Discretization of integral I_2 on the path `z = x + iR` for `-1<=x<=0`
//     //
//     quad::DESinhTanhRule<T> rule2(n_quad2, tiny2);
//     ExponentialSumType es2(n_quad2);

//     for (Index k = 0; k < n_quad2; ++k)
//     {
//         const auto xk = -half * rule2.distanceFromUpper(k); // node
//         const auto lk = half * rule2.distanceFromLower(k);  // x + 1
//         const auto wk = half * rule2.w(k);                  // weight

//         const auto ak = Complex(R, -xk);
//         const auto pk = wk * exp(ak * R_shift) *
//                         cos(Real(n) * acos(Complex(xk, R))) /
//                         sqrt(Complex(lk, R) * Complex(one - xk, -R));

//         es2.exponent(k) = ak;
//         es2.weight(k)   = pk;
//     }
//     //
//     // Truncation for I_1 and I_2 separately
//     //
//     // mxpfit::BalancedTruncation<Complex> trunc1;
//     // es1 = trunc1.compute(es1, threshold * eps);

//     // mxpfit::BalancedTruncation<Complex> trunc2;
//     // es2 = trunc2.compute(es2, threshold);

//     //
//     // Merge two sums
//     //
//     ExponentialSumType es_merged(es1.size() + es2.size());
//     es_merged.exponents().head(es1.size()) = es1.exponents();
//     es_merged.exponents().tail(es2.size()) = es2.exponents();
//     es_merged.weights().head(es1.size())   = es1.weights();
//     es_merged.weights().tail(es2.size())   = es2.weights();

//     //
//     // Truncation for I_1 and I_2 simultaneously
//     //
//     // mxpfit::BalancedTruncation<Complex> trunc1;
//     // es_merged = trunc1.compute(es_merged, threshold * eps * eps);

//     ExponentialSumType es_result(2 * es_merged.size());
//     const auto pre1 = pow_i[n % 4] / pi<Real>(); // (-i)^n / pi
//     const auto pre2 = (n & 1) ? -pre1 : pre1;

//     for (Index i = 0; i < es_merged.size(); ++i)
//     {
//         const auto ai                 = es_merged.exponent(i);
//         es_result.exponent(2 * i + 0) = ai;
//         es_result.exponent(2 * i + 1) = conj(ai);

//         const auto wi               = es_merged.weight(i) * exp(-ai *
//         R_shift); es_result.weight(2 * i + 0) = pre1 * wi; es_result.weight(2
//         * i + 1) = pre2 * conj(wi);
//     }

//     return es_result;
// }

//==============================================================================
// Main
//==============================================================================

using Index              = Eigen::Index;
using Real               = double;
using Complex            = std::complex<Real>;
using RealArray          = Eigen::Array<Real, Eigen::Dynamic, 1>;
using ComplexArray       = Eigen::Array<Complex, Eigen::Dynamic, 1>;
using ExponentialSumType = BesselKernel<Real>::ExponentialSumType;

void bessel_j_kernel_error(int l, const RealArray& x,
                           const ExponentialSumType& ret)
{
    RealArray exact(x.size());
    RealArray approx(x.size());

    for (Index i = 0; i < x.size(); ++i)
    {
        exact(i)  = boost::math::cyl_bessel_j(l, x(i));
        approx(i) = std::real(ret(x(i)));
    }

    RealArray abserr(Eigen::abs(exact - approx));

    for (Index i = 0; i < x.size(); ++i)
    {
        std::cout << std::setw(24) << x(i) << ' '      // point
                  << std::setw(24) << exact(i) << ' '  // exact value
                  << std::setw(24) << approx(i) << ' ' // approximation
                  << std::setw(24) << abserr(i) << '\n';
    }

    Index imax;
    abserr.maxCoeff(&imax);

    std::cout << "\n  abs. error in interval [" << x(0) << ","
              << x(x.size() - 1) << "]\n"
              << "    maximum : " << abserr(imax) << '\n'
              << "    averaged: " << abserr.sum() / x.size() << std::endl;
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const Real threshold = 1.0e-14;
    const Real eps       = Eigen::NumTraits<Real>::epsilon();
    const Index lmax     = 4;
    const Index N        = 2000; // # of sampling points

    std::cout
        << "# Approximation of spherical Bessel function by exponential sum\n";

    RealArray x = Eigen::exp(RealArray::LinSpaced(N, -20.0, 20.0));
    ExponentialSumType ret;
    for (Index l = 0; l <= lmax; ++l)
    {
        std::cout << "\n# --- order " << l;
        ret                      = BesselKernel<Real>::compute(l, threshold);
        const auto thresh_weight = std::max(eps, threshold) / Real(ret.size());
        ret                      = mxpfit::removeIf(
            ret, [=](const Complex& /*exponent*/, const Complex& wi) {
                return std::abs(std::real(wi)) < thresh_weight &&
                       std::abs(std::imag(wi)) < thresh_weight;
            });
        std::cout << " (" << ret.size() << " terms approximation)\n"
                  << ret << '\n';

        // std::cout << " (" << ret.size() << " terms approximation)\n";
        // std::cout
        //     << "# real(exponent), imag(exponent), real(weight),
        //     imag(weight)\n";
        // for (Index i = 0; i < ret.size(); ++i)
        // {
        //     std::cout << std::setw(24) << std::real(ret.exponent(i)) << '\t'
        //               << std::setw(24) << std::imag(ret.exponent(i)) << '\t'
        //               << std::setw(24) << std::real(ret.weight(i)) << '\t'
        //               << std::setw(24) << std::imag(ret.weight(i)) << '\n';
        // }
        // std::cout << '\n' << std::endl;

        std::cout << "\n# errors in small x\n";
        bessel_j_kernel_error(l, x, ret);
    }

    return 0;
}

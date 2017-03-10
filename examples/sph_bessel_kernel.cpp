#include <iomanip>
#include <iostream>

#include "constants.hpp"
#include "legendre.hpp"
#include "tanh_sinh_rule.hpp"

#include <mxpfit/balanced_truncation.hpp>
#include <mxpfit/exponential_sum.hpp>

#include <boost/math/special_functions/bessel.hpp>

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
class SphBesselKernel
{
public:
    using Index   = Eigen::Index;
    using Real    = T;
    using Complex = std::complex<Real>;

    using ExponentialSumType = mxpfit::ExponentialSum<Complex, Complex>;

    static ExponentialSumType compute(Index order, Real threshold);
};

template <typename T>
typename SphBesselKernel<T>::ExponentialSumType
SphBesselKernel<T>::compute(Index n, Real threshold)
{
    using Eigen::numext::conj;
    using Eigen::numext::log;
    using Eigen::numext::cos;
    using Eigen::numext::sin;
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
    const auto R       = Real(5) / (n + 1);
    const auto n_quad1 = Index(200);
    const auto tiny1   = eps * std::sqrt(eps);

    const auto n_quad2 = Index(120 + 2 * n);
    const auto tiny2   = eps * std::sqrt(eps);
    //---------------------------

    //
    // Discretization of integral I_1 on the path `-1 + i y`  for
    // `0 <= y <= R`
    //
    quad::DESinhTanhRule<T> rule1(n_quad1, tiny1);
    ExponentialSumType es1(n_quad1);

    const auto pre1   = pow_i[n % 4] / Real(2); // (-i)^n / 2
    const auto pre2   = (n & 1) ? -pre1 : pre1;
    const auto R_half = R / 2;

    for (Index k = 0; k < n_quad1; ++k)
    {
        const auto yk = R_half * rule1.distanceFromLower(k); // node
        const auto wk = R_half * rule1.w(k);                 // weight
        const auto pk =
            Complex(zero, wk) * math::legendre_p(n, Complex(-one, yk));
        const auto ak = Complex(yk, one);

        es1.exponent(k) = ak;
        es1.weight(k)   = pk;
    }

    //
    // Discretization of integral I_2 on the path `x + iR` for
    // `0 <= x <= 1`
    //
    quad::DESinhTanhRule<T> rule2(n_quad2, tiny2);
    ExponentialSumType es2(n_quad2);

    for (Index k = 0; k < n_quad2; ++k)
    {
        const auto xk = half * rule2.distanceFromLower(k); // node
        const auto wk = half * rule2.w(k);                 // weight
        const auto pk = wk * math::legendre_p(n, Complex(xk, R));
        const auto ak = Complex(R, -xk);

        es2.exponent(k) = ak;
        es2.weight(k)   = pk;
    }

    // //
    // // Truncation for I_1 and I_2 separately
    // //
    // mxpfit::BalancedTruncation<Complex> trunc1;
    // trunc1.setThreshold(threshold);
    // es1 = trunc1.compute(es1);

    // mxpfit::BalancedTruncation<Complex> trunc2;
    // trunc2.setThreshold(threshold);
    // es2 = trunc2.compute(es2);

    //
    // Merge two sums
    //
    ExponentialSumType es_merged(es1.size() + es2.size());
    es_merged.exponents().head(es1.size()) = es1.exponents();
    es_merged.exponents().tail(es2.size()) = es2.exponents();
    es_merged.weights().head(es1.size())   = es1.weights();
    es_merged.weights().tail(es2.size())   = es2.weights();

    //
    // Truncation for I_1 and I_2 simultaneously
    //
    // mxpfit::BalancedTruncation<Complex> trunc1;
    // trunc1.setThreshold(threshold);
    // es_merged = trunc1.compute(es_merged);

    ExponentialSumType es_result(2 * es_merged.size());

    for (Index i = 0; i < es_merged.size(); ++i)
    {
        const auto ai                 = es_merged.exponent(i);
        es_result.exponent(2 * i + 0) = ai;
        es_result.exponent(2 * i + 1) = conj(ai);

        const auto wi               = es_merged.weight(i);
        es_result.weight(2 * i + 0) = pre1 * wi;
        es_result.weight(2 * i + 1) = pre2 * conj(wi);
    }

    return es_result;
}

//==============================================================================
// Main
//==============================================================================

using Index              = Eigen::Index;
using Real               = double;
using Complex            = std::complex<Real>;
using RealArray          = Eigen::Array<Real, Eigen::Dynamic, 1>;
using ComplexArray       = Eigen::Array<Complex, Eigen::Dynamic, 1>;
using ExponentialSumType = SphBesselKernel<Real>::ExponentialSumType;

void sph_bessel_kernel_error(int l, const RealArray& x,
                             const ExponentialSumType& ret)
{
    RealArray exact(x.size());
    RealArray approx(x.size());

    for (Index i = 0; i < x.size(); ++i)
    {
        exact(i)  = boost::math::sph_bessel(l, x(i));
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
              << "    averaged: " << abserr.sum() / x.size() << '\n';
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const double threshold = 1.0e-15;
    const Index lmax       = 20;
    const Index N          = 20000; // # of sampling points

    std::cout
        << "# Approximation of spherical Bessel function by exponential sum\n";

    RealArray x = Eigen::exp(RealArray::LinSpaced(N, -20.0, 20.0));
    ExponentialSumType ret;
    for (Index l = 0; l <= lmax; ++l)
    {
        std::cout << "\n# --- order " << l << '\n';
        ret = SphBesselKernel<Real>::compute(l, threshold);
        ret = removeSmallTerms(ret, threshold / ret.size());
        std::cout << "# no. of terms and (exponents, weights)\n" << ret << '\n';
        std::cout << "# sum of weights: " << ret.weights().sum() << '\n';

        std::cout << "\n# errors in small x\n";
        sph_bessel_kernel_error(l, x, ret);
    }

    return 0;
}

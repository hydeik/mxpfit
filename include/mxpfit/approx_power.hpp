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

#include <algorithm>
#include <iosfwd>
#include <iterator>
#include <stdexcept>

#include <boost/math/special_functions/gamma.hpp>

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
        // std::cout << "(newton): " << x << '\t'     // t
        //           << std::get<0>(f_and_df) << '\t' // f(t)
        //           << std::get<1>(f_and_df) << '\t' // f'(t)
        //           << delta << '\n';
        x -= delta;
        if (abs(delta) < abs(x) * tol)
        {
            break;
        }
    }

    return x;
}

} // namespace: detail

///
/// ### ApproxPowerFunction
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
class ApproxPowerFunction
{
public:
    using Index      = Eigen::Index;
    using Real       = T;
    using ResultType = ExponentialSum<T, T>;

    ApproxPowerFunction()                           = default;
    ApproxPowerFunction(const ApproxPowerFunction&) = default;
    ApproxPowerFunction(ApproxPowerFunction&&)      = default;

    ///
    /// Create an instance with arguments.
    ///
    /// \param[in] beta the power factor \f$\beta > 0\f$
    /// \param[in] eps  the target accuracy of the exponential sum approximation
    /// \param[in] rmin the lower bound of the interval
    /// \param[in] rmax the upper bound of the interval
    /// \pre `beta > 0 && eps > 0 && rmin > 0 && rmax > rmin`
    ///
    ApproxPowerFunction(Real beta, Real eps, Real rmin, Real rmax)
        : m_beta(beta), m_eps(eps), m_rmin(rmin), m_rmax(rmax)
    {
        check_args(beta, eps, rmin, rmax);
        compute_extra_params();
    }

    ~ApproxPowerFunction() = default;

    ApproxPowerFunction& operator=(const ApproxPowerFunction&) = default;
    ApproxPowerFunction& operator=(ApproxPowerFunction&&) = default;

    /// Get the value of power factor \f$ \beta \f$
    Real power_factor() const
    {
        return m_beta;
    }

    /// Get the target accuracy \f$ \epsilon \f$
    Real tolerance() const
    {
        return m_eps;
    }

    ///
    /// Get the lower bound of the interval in which the exponential sum
    /// approximation is constructed.
    ///
    Real rmin() const
    {
        return m_rmin;
    }
    ///
    /// Get the upper bound of the interval in which the exponential sum
    /// approximation is constructed.
    ///
    Real rmax() const
    {
        return m_rmax;
    }
    ///
    /// Set parameters
    ///
    void set_params(Real beta, Real eps, Real rmin, Real rmax)
    {
        check_args(beta, eps, rmin, rmax);
        m_beta = beta;
        m_eps  = eps;
        m_rmin = rmin;
        m_rmax = rmax;
        compute_extra_params();
    }

    ///
    /// Compute the exponential sum approximation with the parameters set in
    /// advance via constructor or `set_params` method.
    ///
    ResultType compute() const;

    ///
    /// Compute the exponential sum approximation with the given parameters.
    ///
    ResultType compute(Real beta, Real eps, Real rmin, Real rmax)
    {
        set_params(beta, eps, rmin, rmax);
        return compute();
    }
    ///
    /// Print parameters to the given ostream
    ///
    template <typename Ch, typename Tr>
    void print(std::basic_ostream<Ch, Tr>& os) const;

private:
    constexpr static const Real safety = Real(10);

    static void check_args(Real beta, Real eps, Real rmin, Real rmax);
    void compute_extra_params();

    Real m_beta;
    Real m_eps;
    Real m_rmin;
    Real m_rmax;

    Real m_h;        // step size for discretization
    Index m_n_minus; // number of terms for trapezoidal rule
    Index m_n_plus;  // number of terms for trapezoidal rule
};

// ostream operator
template <typename Ch, typename Tr, typename T>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os,
                                       const ApproxPowerFunction<T>& es)
{
    es.print(os);
    return os;
}

//
// --- Implementations of member functions
//

template <typename T>
typename ApproxPowerFunction<T>::ResultType
ApproxPowerFunction<T>::compute() const
{
    ResultType es(m_n_minus + m_n_plus + 1);

    const Real scale = m_h / std::tgamma(m_beta);

    for (Index n = -m_n_minus; n <= m_n_plus; ++n)
    {
        const Real tn              = Real(n) * m_h;
        const Real en              = std::exp(-tn);
        es.exponent(n + m_n_minus) = std::exp(tn - en);
        es.weight(n + m_n_minus) =
            scale * (Real(1) + en) * std::exp(m_beta * (tn - en));
    }

    //
    // Reduce number of terms with small exponents (a[i] < 1) via the modified
    // Prony method.
    //
    ModifiedPronyReduction<Real> reduction;
    const Index n_target = static_cast<Index>(std::distance(
        es.exponents().data(),
        std::upper_bound(es.exponents().data(),
                         es.exponents().data() + es.exponents().size(),
                         Real(1))));

    ResultType trunc = reduction.compute(es, n_target, m_eps);

    const Real rmax_tmp = m_rmax * safety;
    trunc.exponents() /= rmax_tmp;
    trunc.weights() *= std::pow(rmax_tmp, -m_beta);

    return trunc;
}

template <typename T>
template <typename Ch, typename Tr>
void ApproxPowerFunction<T>::print(std::basic_ostream<Ch, Tr>& os) const
{
    os << "# Exponential sum approximation of power function, r^{-beta}\n"
       << "#   beta    : " << m_beta << '\n'                    //
       << "#   interval: [" << m_rmin << ',' << m_rmax << "]\n" //
       << "#   relative accuracy: " << m_eps << '\n'            //
       << "#   step size: " << m_h << '\n'                      //
       << "#   number of terms for discritization: [" << -m_n_minus << ", "
       << m_n_plus << "]\n";
}

template <typename T>
void ApproxPowerFunction<T>::check_args(Real beta, Real eps, Real rmin,
                                        Real rmax)
{
    if (!(beta > Real()))
    {
        std::ostringstream msg;
        msg << "Invalid value for the argument `beta': "
               "beta > 0 expected, but beta = "
            << beta << " is given";
        throw std::invalid_argument(msg.str());
    }

    if (!(Real() < eps && eps < std::exp(Real(-1))))
    {
        std::ostringstream msg;
        msg << "Invalid value for the argument `eps': "
               "0 < eps < 1/e expected, but "
            << eps << " is given";
        throw std::invalid_argument(msg.str());
    }

    if (!(Real() < rmin && rmin < rmax))
    {
        std::ostringstream msg;
        msg << "Invalid value for the argument `rmin/rmax': "
               "0 < rmin < rmax expected, but rmin = "
            << rmin << " and rmax = " << rmax << " are given";
        throw std::invalid_argument(msg.str());
    }
}

template <typename T>
void ApproxPowerFunction<T>::compute_extra_params()
{
    constexpr const T pi = boost::math::constants::pi<T>();
    //
    // Construct approximation for interval with extended rmax: otherwise, the
    // relative error of Gaussian sum approximation for `r` close to `rmax`
    // exceeds the prescribed accuracy.
    //
    const Real rmax = m_rmax * safety;

    //
    // Some constants related to given arguments
    //
    const T eps_d = m_eps / T(3); // upper bound of discretization error
    const T eps_t = m_eps / T(3); // upper bound of truncation error
    const T delta = m_rmin / rmax;

    //
    // ----- Spacing of discritization (See eq. (14) in [Beylkin2010])
    //
    // The step size h is determined to satisfy
    //
    //   \sum_{n=1}^{\infty}
    //     \frac{2|\Gamma(\beta + 2 \pi i n / h)|}{\Gamma(\beta)} < \epsilon_d.
    //
    // For \beta = 1/2, the equation above can be written as,
    //
    //   \sum_{n=1}^{\infty} 2 / sqrt(cosh(2 \pi^2 n / h)) < \epsilon_d
    //
    // In practice, \f$|\Gamma(\beta + 2 \pi i n / h)|\f$ decay so rapidly with
    // \f$n\f$ that only the first term in the left-hand-side is significant.
    //
    if (m_beta == T(0.5))
    {
        // solve  2 / sqrt(cosh(2 * pi**2 / h)) == eps_d  with h
        m_h = T(2) * pi * pi / std::acosh(T(4) / (eps_d * eps_d));
    }
    else
    {
        //
        // |Gamma(beta + yi)| <= 1+(y/beta)**2 * exp(-y * atan(y/beta))
        // so we solve (log(r.h.s) == log(eps_d)) with y = 2 * pi / h
        //

        //
        // Initial guess of step size (See eq. (15) in [Beylkin2010])
        //
        // This yields a lower bound of step size h, which is not optimal.
        //
        m_h = T(2) * pi /
              (std::log(T(3)) - m_beta * std::log(std::cos(T(1))) -
               std::log(eps_d));

        const T beta_half = m_beta / T(2);
        const T log_eps_d = std::log(eps_d);
        const T y         = detail::newton(
            // Initial guess of y
            T(2) * pi / m_h,
            // relative accuracy
            eps_d,
            // function value and its derivative
            [=](T t) {
                const T x      = t / m_beta;
                const T atan_x = std::atan2(t, m_beta);
                const T f =
                    beta_half * std::log1p(x * x) - t * atan_x - log_eps_d;
                const T df = (beta_half - x) / (T(1) + x * x) - atan_x;
                return std::make_pair(f, df);
            });

        m_h = T(2) * pi / y;
    }

    //
    // Find lower bound `t_lower` such that the truncation error of integral is
    // bounded up to `eps_t`
    //
    // `t_lower` satisfies
    //
    //   C_2 \exp(-\beta e^{t}) < \epsilon_t
    //
    // with \f$ C_2 = 2e/\Gamma(\beta)\f$ (See eq. (25) in [McLean2017]).
    //
    const T c2 =
        T(2) * boost::math::constants::e<T>() / std::tgamma(m_beta + 1);
    const T t_lower = -std::log(-std::log(eps_t / c2) / m_beta);

    //
    // Find upper bound `t_upper` such that the truncation error of integral is
    // bounded up to `eps_t`
    //
    // This is obtained by solving (see eq.(32) in [Beylkin2010])
    //
    //   \Gamma(beta, \delta \exp(t)) = \epsilon_t
    //
    // For beta = 1/2, the equation becomes
    //
    //   \erfc(\delta exp(t/2)) = \epsilon_t
    //
    const T t_upper =
        std::log(boost::math::gamma_q_inv(m_beta, eps_t)) - std::log(delta);

    //
    // Set the step size and number of terms for trapezoidal rule
    //
    m_n_minus = static_cast<Index>(std::floor((-t_lower) / m_h));
    m_n_plus  = static_cast<Index>(std::ceil((t_upper) / m_h));
}

///
/// ### approx_power
///
template <typename T>
ExponentialSum<T, T> approx_power(T beta, T eps, T rmin, T rmax)
{
    ApproxPowerFunction<T> pow;
    return pow.compute(beta, eps, rmin, rmax);
}

} // namespace: mxpfit

#endif /* MXPFIT_POW_KERNEL_HPP */

/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2017  Hidekazu Ikeno
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
/// \file aak_reduction.hpp
///
/// Sparse approximation of exponential sum using AAK theory.
///

#ifndef MXPFIT_AAK_REDUCTION_HPP
#define MXPFIT_AAK_REDUCTION_HPP

#include <cmath>

#include <algorithm>
#include <tuple>

#include <mxpfit/exponential_sum.hpp>
#include <mxpfit/quasi_cauchy_rrd.hpp>
#include <mxpfit/self_adjoint_coneigensolver.hpp>

namespace mxpfit
{

namespace detail
{

//
// Compare two floating point numbers
//
template <typename T>
bool close_enough(T x, T y, T tol)
{
    using Eigen::numext::abs;
    const auto diff = abs(x - y);
    return diff <= tol * abs(x) || diff <= tol * abs(y);
}
//
// Compare two complex numbers
//
template <typename T>
bool close_enough(const std::complex<T>& x, const std::complex<T>& y, T tol)
{
    // using Eigen::numext::real;
    // using Eigen::numext::imag;
    // return close_enough(real(x), real(y), tol) &&
    //        close_enough(imag(x), imag(y), tol);
    using Eigen::numext::abs;
    const auto diff = abs(x - y);
    return diff <= tol * abs(x) || diff <= tol * abs(y);
}

//
// Newton method
//
template <typename T, typename Real, typename Functor>
T find_root_impl(const T& guess, Real tol, Functor func)
{
    using Eigen::numext::abs;

    T f, df;
    int iter = 100;
    T x      = guess;

    while (--iter)
    {
        std::tie(f, df) = func(x);
        auto delta = f / df;
        // std::cout << "***** " << x << '\t' << delta << '\t' << f << '\t' <<
        // df
        //           << std::endl;
        if (abs(delta) < tol * abs(x))
        {
            break;
        }
        x -= delta;
    }
    return x;
}

//
// From a given finite sequence z[i]=exp(-t[i]) and function values y[i] =
// f(z[i]), formulate and evaluate continued fraction interpolation of the form
//
//                          a[0]
// g(t) = ------------------------------------------
//              a[1] * (exp(-t) - exp(-t[0]))
//        1 + --------------------------------------
//                   a[2] * (exp(-t) - exp(-t[1]))
//            1 + ----------------------------------
//                     a[3] * (exp(-t) - exp(-t[2]))
//                1 + ------------------------------
//                     1 + ...
//
// where g(t[i]) = f(exp(-t[i])) = y[i].
//
// Let us define R[i](t) recursively as,
//
// R [1](t) = 1 / (1 + a[1] * expm1(-t+t[0]) * R[2](t))
// R[i](t) = 1 / (1 + a[i]* expm1(-t+t[i-1]) * R[i+1](t))  for i=1,2,...,n-3
// R[n-1](t) = 1 / (1 +  a[n-1] * expm1(-t+t[n-2]))
//
// Note that R[i+1](t[i]) = 1. Then,
//
// g(t) = a[0] * R[1](t);
//
// The derivative of f(t) can also be evaluated as
//
// R'[n-1](t) = a[n-1] * exp(-t+t[n-2]) / [1 + a[n-1] * expm1(-t+t[n-2])]^2
// R'[i](t) = (a[i] * exp(-t+t[i-1]) * R[i+1](t)
//              - a[i] * expm1(-t+t[i-1]) * R'[i+1](t))
//          / (1 + a[i]* expm1(-t+t[i-1]) * R[i+1](t))^2  for i=1,2,...,n-3
// g'(t) = a[0] * R'[1](t)
//

template <typename VecX, typename VecY, typename VecA>
void continued_fraction_interp_coeffs(const VecX& tau, const VecY& y, VecA& a)
{
    using ResultType = typename VecY::Scalar;
    using Real       = typename Eigen::NumTraits<ResultType>::Real;
    using Index      = Eigen::Index;
    using Eigen::numext::exp;

    static const Real tiny = Eigen::NumTraits<Real>::lowest();

    const auto a0 = y[0];
    a[0]          = a0;

    for (Index i = 1; i < tau.size(); ++i)
    {
        auto v          = a0 / y[i];
        const auto taui = tau[i];
        for (Index j = 1; j < i - 1; ++j)
        {
            v -= Real(1);
            if (v == ResultType())
            {
                v = tiny;
            }
            v = (a[j] * expm1(-taui + tau[j - 1])) / v;
        }
        v -= Real(1);
        a[i] = v / expm1(-taui + tau[i - 1]);
    }
}

//
// Compute value of the continued fraction interpolation g(t) and its
// derivative g'(x).
//
template <typename VecX, typename VecA>
std::tuple<typename VecA::Scalar, typename VecA::Scalar>
continued_fraction_interp_eval_with_derivative(typename VecX::Scalar t,
                                               const VecX& ti, const VecA& ai)
{
    using ResultType = typename VecA::Scalar;
    using Real       = typename Eigen::NumTraits<ResultType>::Real;
    using Index      = Eigen::Index;

    using Eigen::numext::exp;

    static const Real tiny   = Eigen::NumTraits<Real>::lowest();
    constexpr const Real one = Real(1);

    const auto n = ai.size();

    auto d  = one + ai[n - 1] * expm1(-t + ti[n - 2]);
    auto f  = one / d;                                 // R_{n-1}(x)
    auto df = ai[n - 1] * f * exp(-t + ti[n - 2]) / d; // R_{n-1}'(x)

    for (Index k = n - 2; k > 0; --k)
    {
        const auto uk = expm1(-t + ti[k - 1]);
        d             = one + ai[k] * uk * f;
        if (d == ResultType())
        {
            d = tiny;
        }
        const auto f_pre = f;

        f  = one / d;
        df = ai[k] * f * (exp(-t + ti[k - 1]) * f_pre - uk * df) * f;
    }

    // now f = f(x), df = f'(x)
    return std::make_tuple(ai[0] * f, ai[0] * df);
}

} // namespace: detail

///
/// ### AAKReduction
///
/// \brief Find a truncated exponential sum function with smaller number of
///        terms using AAK theory.
///
/// \tparam T  Scalar type of exponential sum function.
///
/// For a given exponential sum function
/// \f[
///   f(t)=\sum_{j=1}^{N} c_{j}^{} e^{-p_{j}^{} t}, \quad
///   (\mathrm{Re}(a_{j}) > 0).
/// \f]
///
/// and prescribed accuracy \f$\epsilon > 0,\f$ this class calculates truncated
/// exponential \f$\tilde{f}(t)\f$ sum such that
///
/// \f[
///   \tilde{f}(t)=\sum_{j=1}^{M} \tilde{c}_{j}^{}e^{-\tilde{p}_{j}^{} t}, \quad
///   \left| f(t)-\tilde{f}(t) \right| < \epsilon,
/// \f]
///
///
/// The signal \f$\boldsymbol{f}=(f_{k})_{k=0}^{\infty}\f$ sampled on
/// \f$k=0,1,\dots\f$ are expressed as
///
/// \f[
///   f_{k}:= f(k) = \sum_{j=1}^{N} c_{j} z_{j}^{k}
/// \f]
///
/// where \f$z_{j}=e^{-p_{j}}\in \mathbb{D}:=\{z \in \mathbb{C}:0<|z|<1\}.\f$
/// The problem to find truncated exponential sum can be recast to find a new
/// signal \f$\tilde{\boldsymbol{f}},\f$ a sparse approximation of signal
/// \f$\boldsymbol{f},\f$ such that
///
/// \f[
///   \tilde{f}_{k}=\sum_{j=1}^{M}\tilde{c}_{j}^{} \tilde{z}_{j}^{k}, \quad
///   \|\boldsymbol{f}-\tilde{\boldsymbol{f}}\|<\epsilon. \f$
/// \f]
///
/// Then, the exponents of reduced exponential sum
/// \f$\tilde{f}(t)\f$ are obtained as \f$\tilde{p}_{j}=\tilde{z}_{j}.\f$
///
///
/// #### References
///
/// 1. Gerlind Plonka and Vlada Pototskaia, "Application of the AAK theory for
///    sparse approximation of exponential sums", arXiv 1609.09603v1 (2016).
///    [URL: https://arxiv.org/abs/1609.09603v1]
/// 2. Gregory Beylkin and Lucas Monzón, "On approximation of functions by
///    exponential sums", Appl. Comput. Harmon. Anal. **19** (2005) 17-48.
///    [DOI: https://doi.org/10.1016/j.acha.2005.01.003]
/// 3. T. S. Haut and G. Beylkin, "FAST AND ACCURATE CON-EIGENVALUE ALGORITHM
///    FOR OPTIMAL RATIONAL APPROXIMATIONS", SIAM J. Matrix Anal. Appl. **33**
///    (2012) 1101-1125.
///    [DOI: https://doi.org/10.1137/110821901]
/// 4. Terry Haut, Gregory Beylkin, and Lucas Monzón, "Solving Burgers’
///    equation using optimal rational approximations", Appl. Comput. Harmon.
///    Anal. **34** (2013) 83-95.
///    [DOI: https://doi.org/10.1016/j.acha.2012.03.004]
///
template <typename T>
class AAKReduction
{
public:
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
    using ComplexScalar = std::complex<RealScalar>;
    using Index         = Eigen::Index;

    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ResultType = ExponentialSum<Scalar>;

    ///
    /// Compute truncated exponential sum \f$ \hat{f}(t) \f$
    ///
    /// \tparam DerivedF type of exponential sum inheriting ExponentialSumBase
    ///
    /// \param[in] orig original exponential sum function, \f$ f(t) \f$
    /// \param[in] threshold  prescribed accuracy \f$0 < \epsilon \ll 1\f$
    ///
    /// \return An instance of ExponentialSum represents \f$\hat{f}(t)\f$
    ///
    template <typename DerivedF>
    ResultType compute(const ExponentialSumBase<DerivedF>& orig,
                       RealScalar threshold);

private:
    using IndexVector        = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
    using ConeigenSolverType = SelfAdjointConeigenSolver<Scalar>;
    using RRDType =
        QuasiCauchyRRD<Scalar, QuasiCauchyRRDFunctorLogPole<Scalar>>;
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
    };
};

template <typename T>
template <typename DerivedF>
typename AAKReduction<T>::ResultType
AAKReduction<T>::compute(const ExponentialSumBase<DerivedF>& orig,
                         RealScalar threshold)
{
    static const auto pi  = RealScalar(4) * Eigen::numext::atan(RealScalar(1));
    static const auto eps = Eigen::NumTraits<RealScalar>::epsilon();
    using Eigen::numext::abs;
    using Eigen::numext::exp;
    using Eigen::numext::conj;
    using Eigen::numext::real;
    using Eigen::numext::sqrt;

    const Index n0 = orig.size();
    std::cout << "*** order of original function: " << n0 << std::endl;

    //-------------------------------------------------------------------------
    // Solve con-eigenvalue problem of the Cauchy-like matrix \f$C\f$ with
    // elements
    //
    // C(i,j)= sqrt(c[i]) * sqrt(conj(c[j])) / (1 - z[i] * conj(z[j]))
    //
    // where z[i] = exp(-p[i]).
    //-------------------------------------------------------------------------
    //
    // Rewrite the matrix element C(i,j) as
    //
    //                 a[i] * b[j]
    //   C(i,j) = ----------------------
    //             exp(x[i]) - exp(y[j])
    //
    // where,
    //
    //   a[i] = sqrt(c[i]) / exp(-p[i])
    //   b[j] = sqrt(conj(c[j]))
    //   x[i] = log(z[i]) = p[i]
    //   y[j] = log(1/conj(z[i])) = -conj(p[j])
    //

    VectorType a(n0), b(n0), x(n0), y(n0);
    {
        IndexVector index = IndexVector::LinSpaced(n0, 0, n0 - 1);
        if (IsComplex)
        {
            std::sort(index.data(), index.data() + n0, [&](Index i, Index j) {
                const auto re_i = real(orig.exponents()(i));
                const auto im_i = imag(orig.exponents()(i));
                const auto re_j = real(orig.exponents()(j));
                const auto im_j = imag(orig.exponents()(j));
                // return re_i > re_j || (!(re_j > re_i) && im_i < im_j);
                return std::tie(re_i, im_i) > std::tie(re_j, im_j);
            });
        }
        else
        {
            std::sort(index.data(), index.data() + n0, [&](Index i, Index j) {
                return real(orig.exponents()(i)) > real(orig.exponents()(j));
            });
        }

        for (Index i = 0; i < n0; ++i)
        {
            const auto tau_i    = orig.exponents()(index[i]);
            const auto sqrt_w_i = sqrt(orig.weights()(index[i]));
            a(i)                = sqrt_w_i * exp(tau_i);
            b(i)                = conj(sqrt_w_i);
            x(i)                = tau_i;
            y(i)                = -conj(tau_i);
        }
    }
    //
    // Compute partial Cholesky decomposition of matrix C, such that,
    //
    //   C = (P * L) * D^2 * (P * L)^H,
    //
    // where
    //
    //   - L: (n, m) matrix
    //   - D: (m, m) real diagonal matrix
    //   - P: (n, n) permutation matrix
    //
    //
    RRDType rrd;

    rrd.setThreshold(threshold * threshold * eps);
    rrd.compute(a, b, x, y);

    std::cout.precision(15);
    std::cout << "*** rank after quasi-Cauchy RRD: " << rrd.rank() << std::endl;

    //
    // Compute con-eigendecomposition
    //
    //   C = X D^2 X^H = U^C S U^T,
    //
    // where
    //
    //   - `X = P * L`: Cholesky factor (n, k)
    //   - `S`: (k, k) diagonal matrix. Diagonal elements `S(i,i)` are
    //          con-eigenvalues sorted in decreasing order.
    //   - `U`: (n, k) matrix. k-th column hold a con-eigenvector corresponding
    //          to k-th con-eigenvalue. The columns of `U` are orthogonal in the
    //          sense that `U^T * U = I`.
    //
    ConeigenSolverType ceig;
    ceig.compute(rrd.matrixPL(), rrd.vectorD());
    //
    // Determines the order of reduced system, \f$ M \f$ from the Hankel
    // singular values, which are corresponding to the con-eigenvalues of matrix
    // `C`.
    //
    const auto& sigma = ceig.coneigenvalues();
    auto pos          = std::upper_bound(
        sigma.data(), sigma.data() + sigma.size(), threshold,
        [&](RealScalar lhs, RealScalar rhs) { return lhs >= rhs; });
    const Index n1 = static_cast<Index>(pos - sigma.data());
    std::cout << "*** order after reduction: " << n1 << std::endl;

    if (n1 == Index())
    {
        return ResultType();
    }

    std::cout << "    sigma = " << sigma(n1 - 1) << std::endl;

    //-------------------------------------------------------------------------
    // Find the `n1` roots of rational function
    //
    //             n0  sqrt(alpha[i]) * conj(u[i])
    //   v(eta) = Sum  ---------------------------
    //            i=1     1 - conj(z[i]) * eta
    //
    // where eta is on the unit disk.
    //-------------------------------------------------------------------------

    auto vec_u = ceig.coneigenvectors().col(n1 - 1);
    // x          = -x;
    // y[i] = conj(u[i]) / sqrt(conj(w[i]))
    y.array() = vec_u.array().conjugate() / b.array();
    //
    // Approximation of v(eta) by continued fraction
    //
    detail::continued_fraction_interp_coeffs(x, y, a);

    VectorType tmp(n0);
    Index n_found = 0;
    for (Index i = 0; i < n0; ++i)
    {
        auto eta = detail::find_root_impl(x(i), eps, [&](const Scalar& z) {
            return detail::continued_fraction_interp_eval_with_derivative(z, x,
                                                                          a);
        });
        std::cout << "*** " << i << '\t' << eta << '\n';

        if (real(eta) > RealScalar())
        {
            if (n_found)
            {
                if (std::none_of(tmp.data(), tmp.data() + n_found,
                                 [&](const Scalar& lhs) {
                                     return detail::close_enough(lhs, eta,
                                                                 threshold);
                                 }))
                {
                    tmp(n_found) = eta;
                    ++n_found;
                    // std::cout << "root(" << n_found << "): " << eta << '\n';
                }
            }
            else
            {
                tmp(n_found) = eta;
                ++n_found;
            }
        }
    }

    //
    // Barycentric form of Lagrange interpolation
    //
    // VectorType& tau = x;
    // std::sort(tau.data(), tau.data() + tau.size(),
    //           [&](const Scalar& lhs, const Scalar& rhs) {
    //               return real(lhs) > real(rhs);
    //           });
    // VectorType& s = y;

    ResultType ret;

    return ret;
}

} // namespace: mxpfit

#endif /* MXPFIT_AAK_REDUCTION_HPP */

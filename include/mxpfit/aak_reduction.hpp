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
#include <numeric>
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
bool close_enough(T x, T y, T abstol, T reltol)
{
    using Eigen::numext::abs;
    const auto diff = abs(x - y);
    return diff <= abstol || diff <= reltol * abs(x) || diff <= reltol * abs(y);
}
//
// Compare two complex numbers
//
template <typename T>
bool close_enough(const std::complex<T>& x, const std::complex<T>& y, T abstol,
                  T reltol)
{
    using Eigen::numext::real;
    using Eigen::numext::imag;
    return close_enough(real(x), real(y), abstol, reltol) &&
           close_enough(imag(x), imag(y), abstol, reltol);
    // using Eigen::numext::abs;
    // const auto diff = abs(x - y);
    // return diff <= tol * abs(x) || diff <= tol * abs(y);
}

//
// Make argument of complex exponent tau in [-pi,pi]
//
// --- real number: nothing to do
template <typename T>
inline T regularize_arg(T x)
{
    return x;
}

template <typename T>
inline std::complex<T> regularize_arg(const std::complex<T>& x)
{
    constexpr const T pi = EIGEN_PI;
    return std::complex<T>(std::real(x), std::fmod(std::imag(x), pi));
}

//
// Newton method
//
template <typename T>
struct newton_method
{
    using Scalar = T;
    using Real   = typename Eigen::NumTraits<T>::Real;

    enum Status
    {
        SUCCESS,
        REACHED_MAX_ITER,
        DIVERGES,
    };

    newton_method()
        : m_max_iter(1000), m_threshold(10 * Eigen::NumTraits<Real>::epsilon())
    {
    }

    newton_method(int max_iter, Real threshold)
        : m_max_iter(max_iter), m_threshold(threshold)
    {
    }

    void set_threshold(Real new_thredshold)
    {
        m_threshold = new_thredshold;
    }

    Real threshold() const
    {
        return m_threshold;
    }

    void set_max_iteration(int max_iter)
    {
        m_max_iter = max_iter;
    }

    int max_iter() const
    {
        return m_max_iter;
    }

    template <typename Functor>
    std::pair<Scalar, Status> run(const T& guess, Functor fn) const
    {
        using Eigen::numext::abs;
        using Eigen::numext::sqrt;
        using Eigen::numext::real;
        static const Real eps = Real(16) * Eigen::NumTraits<Real>::epsilon();

        T f, df;
        int iter = m_max_iter;
        T x      = guess;

        while (--iter)
        {
            std::tie(f, df) = fn(x);
            auto delta = f / df;
            std::cout << "***** " << x << '\t' << delta << '\t' << f << '\t'
                      << df << std::endl;
            if (std::isnan(abs(df)) ||
                (abs(df) < m_threshold && abs(delta) > Real(1)))
            {
                return {x, DIVERGES};
            }
            if (abs(delta) < m_threshold * abs(x) || abs(f) < eps)
            {
                return {x, SUCCESS};
                break;
            }
            x -= delta;
        }
        return {x, REACHED_MAX_ITER};
    }

private:
    int m_max_iter;
    Real m_threshold;
};

//
// From a given finite sequence z[i]=exp(-t[i]) and function values y[i] =
// f(z[i]), formulate and evaluate continued fraction interpolation of the form
//
//                          a[0]
// g(t) = ------------------------------------------
//              a[1] * (exp(-t+t[0]) - 1)
//        1 + --------------------------------------
//                   a[2] * (exp(-t+t[1]) - 1)
//            1 + ----------------------------------
//                      a[3] * (exp(-t+t[2]) - 1)
//                1 + ------------------------------
//                     1 + ...
//
// where g(t[i]) = f(exp(-t[i])) = y[i].
//
// Let us define R[i](t) recursively as,
//
// R[1](t)   = 1 / (1 + a[1]   * expm1(-t+t[0])   * R[2](t))
// R[i](t)   = 1 / (1 + a[i]   * expm1(-t+t[i-1]) * R[i+1](t))  (i=1,2,...,n-2)
// R[n-1](t) = 1 / (1 + a[n-1] * expm1(-t+t[n-2]))
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
    using Eigen::numext::sqrt;

    constexpr static const Real one = Real(1);
    static const Real tiny          = sqrt(Eigen::NumTraits<Real>::lowest());

    const auto a0 = y[0];
    a[0]          = a0;

    for (Index i = 1; i < tau.size(); ++i)
    {
        auto v          = a0 / y[i];
        const auto taui = tau[i];
        for (Index j = 1; j < i; ++j)
        {
            v -= one;
            if (abs(v) < tiny)
            {
                v = tiny;
            }
            // v := 1 + a[j] * expm1(t[i-1]-t[j-1]) * R[j](t[i-1])
            //    = 1 / R[j](t[i-1])
            auto dt   = -taui + tau[j - 1];
            auto scal = expm1(-taui + tau[j - 1]);
            v         = a[j] * scal / v;
        }
        a[i] = (v - one) / expm1(-taui + tau[i - 1]);
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

    constexpr const Real one = Real(1);
    static const Real tiny   = Eigen::NumTraits<Real>::lowest();

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

private:
    using IndexVector        = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
    using ConeigenSolverType = SelfAdjointConeigenSolver<Scalar>;
    using RRDType =
        QuasiCauchyRRD<Scalar, QuasiCauchyRRDFunctorLogPole<Scalar>>;
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
    };

    RealScalar m_threshold;
    VectorType m_a;
    VectorType m_b;
    VectorType m_x;
    VectorType m_y;

    RRDType m_rrd;
    RRDType m_coneig_solver;

public:
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
    ResultType compute(const ExponentialSumBase<DerivedF>& orig);

    ///
    /// Compute truncated exponential sum \f$ \hat{f}(t) \f$
    ///
    /// This takes two arrays as input arguments: the first one is a array of
    /// adjacent sequence of exponents, and the second one is an array of
    /// coefficients. The function is useful in the case that some exponents are
    /// clustered but the difference between them can be computed accurately.
    ///
    /// \param[in] p_adjucent_difference adjacent difference of exponents \f$
    ///            p_{i}. \f$ The first element is \f$p_{0},\f$ and subsequent
    ///            elements are \f$p_{i}-p_{i-1}\, (i=1,2,\dots,n-1).\f$
    /// \param[in] c  array of coefficients \f$c_{i}.\f$
    ///
    /// \return An instance of ExponentialSum represents \f$\hat{f}(t)\f$
    ///
    template <typename ArrayDiffP, typename ArrayC>
    ResultType
    compute(const Eigen::ArrayBase<ArrayDiffP>& p_adjacent_difference,
            const Eigen::ArrayBase<ArrayC>& c);

    void setThreshold(RealScalar new_thredshold)
    {
        assert(new_thredshold > RealScalar());
        m_threshold = new_thredshold;
    }

    RealScalar threshold() const
    {
        return m_threshold;
    }

private:
    //
    // Resize working arrays
    //
    void resize(Index n)
    {
        m_a.resize(n);
        m_b.resize(n);
        m_x.resize(n);
        m_y.resize(n);
    }
    //
    // Solve con-eigenvalue problem of the Cauchy-like matrix \f$C\f$ with
    // elements
    //
    // C(i,j)= sqrt(c[i]) * sqrt(conj(c[j])) / (1 - z[i] * conj(z[j]))
    //
    // where z[i] = exp(-p[i]).
    //
    void solve_coneig();
};

template <typename T>
template <typename ArrayDiffP, typename ArrayC>
typename AAKReduction<T>::ResultType
AAKReduction<T>::compute(const Eigen::ArrayBase<ArrayDiffP>& p_diff,
                         const Eigen::ArrayBase<ArrayC>& c)
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArrayDiffP);
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(ArrayC);
    assert(p_diff.size() == c.size());

    const Index n0 = p_diff.size();
    resize(n0);

    // Set auxiliary arrays as,
    //
    //   a[i] = sqrt(c[i]) / exp(-p[i])
    //   b[j] = sqrt(conj(c[j]))
    //   x[i] = log(z[i]) = p[i]
    //   y[j] = log(1/conj(z[i]))
    //
    // so that the matrix element C(i,j) as
    //
    //                 a[i] * b[j]
    //   C(i,j) = ----------------------
    //             exp(x[i]) - exp(y[j])
    //
    {
        m_x(0) = p_diff(0);
        for (Index i = 1; i < n; ++i)
        {
            m_x(i) = m_x(i - 1) + p_diff(i);
        }
        m_y = -m_x.conjutage();

        m_b.array() = c.sqrt().cojugate();
        m_a.array() = m_b.conjutage() * m_x.array().exp();
    }

    //
    // Solve coneigenvalue problem of matrix C, and determines the order of
    // reduced system, \f$ M \f$ from the Hankel singular values, which are
    // corresponding to the con-eigenvalues.
    //
    solve_coneig();
    const auto& sigma = m_ceig.coneigenvalues();
    auto pos          = std::upper_bound(
        sigma.data(), sigma.data() + sigma.size(), threshold(),
        [&](RealScalar lhs, RealScalar rhs) { return lhs >= rhs; });
    const Index n1 = static_cast<Index>(std::distance(sigma.data(), pos));
    std::cout << "*** order after reduction: " << n1 << std::endl;

    if (n1 == Index())
    {
        return ResultType();
    }
    std::cout << "    sigma = " << sigma(n1 - 1) << std::endl;

    //-------------------------------------------------------------------------
    // Find the `n1` roots of rational function
    //
    //             n0     sqrt(c[i]) * conj(u[i])
    //   v(eta) = Sum  --------------------------------
    //            i=1  1 - conj(exp(-p[i])) * exp(-eta)
    //
    // where real(eta) > 0 is on the unit disk.
    //-------------------------------------------------------------------------
    auto vec_u = ceig.coneigenvectors().col(n1 - 1);
    // y[i] = conj(u[i]) / sqrt(conj(c[i]))
    m_y.array() = vec_u.array().conjugate() / m_b.array();
    //
    // Approximation of v(eta) by continued fraction
    //
    std::cout << x.transpose() << std::endl;
    detail::continued_fraction_interp_coeffs(x, y, a);
    std::cout << a.transpose() << std::endl;

    // std::cout << "*** Test continued fraction interpolation\n";
    // for (Index i = 0; i < n0; ++i)
    // {
    //     auto y_test =
    //         detail::continued_fraction_interp_eval_with_derivative(x(i), x,
    //         a);
    //     std::cout << x(i) << '\t' << y(i) << '\t' << std::get<0>(y_test)
    //               << '\n';
    // }
}

template <typename T>
void AAKReduction<T>::solve_coneig()
{
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
    m_rrd.setThreshold(threshold() * threshold() * eps);
    m_rrd.compute(m_a, m_b, m_x, m_y);

    std::cout.precision(15);
    std::cout << "*** rank after quasi-Cauchy RRD: " << m_rrd.rank()
              << std::endl;

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
    //
    //   - `U`: (n, k) matrix. k-th column hold a con-eigenvector corresponding
    //          to k-th con-eigenvalue. The columns of `U` are orthogonal in
    //          the sense that `U^T * U = I`.
    //
    m_ceig.compute(m_rrd.matrixPL(), m_rrd.vectorD());

    return;
}

// template <typename T>
// template <typename DerivedF>
// typename AAKReduction<T>::ResultType
// AAKReduction<T>::compute(const ExponentialSumBase<DerivedF>& orig,
//                          RealScalar threshold)
// {
//     // static const auto pi  = RealScalar(4) *
//     // Eigen::numext::atan(RealScalar(1));
//     static const auto eps = Eigen::NumTraits<RealScalar>::epsilon();
//     using Eigen::numext::abs;
//     using Eigen::numext::exp;
//     using Eigen::numext::conj;
//     using Eigen::numext::real;
//     using Eigen::numext::sqrt;

//     const Index n0 = orig.size();
//     std::cout << "*** order of original function: " << n0 << std::endl;

//     //-------------------------------------------------------------------------
//     // Solve con-eigenvalue problem of the Cauchy-like matrix \f$C\f$ with
//     // elements
//     //
//     // C(i,j)= sqrt(c[i]) * sqrt(conj(c[j])) / (1 - z[i] * conj(z[j]))
//     //
//     // where z[i] = exp(-p[i]).
//     //-------------------------------------------------------------------------
//     //
//     // Rewrite the matrix element C(i,j) as
//     //
//     //                 a[i] * b[j]
//     //   C(i,j) = ----------------------
//     //             exp(x[i]) - exp(y[j])
//     //
//     // where,
//     //
//     //   a[i] = sqrt(c[i]) / exp(-p[i])
//     //   b[j] = sqrt(conj(c[j]))
//     //   x[i] = log(z[i]) = p[i]
//     //   y[j] = log(1/conj(z[i])) = -conj(p[j])
//     //

//     VectorType a(n0), b(n0), x(n0), y(n0);
//     {
//         IndexVector index = IndexVector::LinSpaced(n0, 0, n0 - 1);
//         if (IsComplex)
//         {
//             std::sort(index.data(), index.data() + n0, [&](Index i, Index j)
//             {
//                 const auto re_i = real(orig.exponents()(i));
//                 const auto im_i = imag(orig.exponents()(i));
//                 const auto re_j = real(orig.exponents()(j));
//                 const auto im_j = imag(orig.exponents()(j));
//                 return std::tie(re_i, im_i) > std::tie(re_j, im_j);
//             });
//         }
//         else
//         {
//             std::sort(index.data(), index.data() + n0, [&](Index i, Index j)
//             {
//                 return real(orig.exponents()(i)) > real(orig.exponents()(j));
//             });
//         }

//         for (Index i = 0; i < n0; ++i)
//         {
//             const auto tau_i    = orig.exponents()(index[i]);
//             const auto sqrt_w_i = sqrt(orig.weights()(index[i]));
//             a(i)                = sqrt_w_i * exp(tau_i);
//             b(i)                = conj(sqrt_w_i);
//             x(i)                = tau_i;
//             y(i)                = -conj(tau_i);
//         }
//     }
//     //
//     // Compute partial Cholesky decomposition of matrix C, such that,
//     //
//     //   C = (P * L) * D^2 * (P * L)^H,
//     //
//     // where
//     //
//     //   - L: (n, m) matrix
//     //   - D: (m, m) real diagonal matrix
//     //   - P: (n, n) permutation matrix
//     //
//     //
//     RRDType rrd;

//     rrd.setThreshold(threshold * threshold * eps);
//     rrd.compute(a, b, x, y);

//     std::cout.precision(15);
//     std::cout << "*** rank after quasi-Cauchy RRD: " << rrd.rank() <<
//     std::endl;

//     //
//     // Compute con-eigendecomposition
//     //
//     //   C = X D^2 X^H = U^C S U^T,
//     //
//     // where
//     //
//     //   - `X = P * L`: Cholesky factor (n, k)
//     //   - `S`: (k, k) diagonal matrix. Diagonal elements `S(i,i)` are
//     //          con-eigenvalues sorted in decreasing order.
//     //   - `U`: (n, k) matrix. k-th column hold a con-eigenvector
//     corresponding
//     //          to k-th con-eigenvalue. The columns of `U` are orthogonal in
//     the
//     //          sense that `U^T * U = I`.
//     //
//     ConeigenSolverType ceig;
//     ceig.compute(rrd.matrixPL(), rrd.vectorD());
//     //
//     // Determines the order of reduced system, \f$ M \f$ from the Hankel
//     // singular values, which are corresponding to the con-eigenvalues of
//     matrix
//     // `C`.
//     //
//     const auto& sigma = ceig.coneigenvalues();
//     auto pos          = std::upper_bound(
//         sigma.data(), sigma.data() + sigma.size(), threshold,
//         [&](RealScalar lhs, RealScalar rhs) { return lhs >= rhs; });
//     const Index n1 = static_cast<Index>(std::distance(sigma.data(), pos));
//     std::cout << "*** order after reduction: " << n1 << std::endl;

//     if (n1 == Index())
//     {
//         return ResultType();
//     }

//     std::cout << "    sigma = " << sigma(n1 - 1) << std::endl;

//     //-------------------------------------------------------------------------
//     // Find the `n1` roots of rational function
//     //
//     //             n0  sqrt(alpha[i]) * conj(u[i])
//     //   v(eta) = Sum  ---------------------------
//     //            i=1     1 - conj(z[i]) * eta
//     //
//     // where eta is on the unit disk.
//     //-------------------------------------------------------------------------

//     auto vec_u = ceig.coneigenvectors().col(n1 - 1);
//     // x          = -x;
//     // y[i] = conj(u[i]) / sqrt(conj(w[i]))
//     y.array() = vec_u.array().conjugate() / b.array();
//     //
//     // Approximation of v(eta) by continued fraction
//     //
//     std::cout << x.transpose() << std::endl;
//     detail::continued_fraction_interp_coeffs(x, y, a);
//     std::cout << a.transpose() << std::endl;

//     // std::cout << "*** Test continued fraction interpolation\n";
//     // for (Index i = 0; i < n0; ++i)
//     // {
//     //     auto y_test =
//     //         detail::continued_fraction_interp_eval_with_derivative(x(i),
//     x,
//     //         a);
//     //     std::cout << x(i) << '\t' << y(i) << '\t' << std::get<0>(y_test)
//     //               << '\n';
//     // }

//     // VectorType tmp(n0);
//     // Index n_found = 0;
//     // detail::newton_method<Scalar> newton(1000, eps * RealScalar(100));
//     // Scalar eta;
//     // typename detail::newton_method<Scalar>::Status status;

//     // for (Index i = 0; i < n0; ++i)
//     // {
//     //     std::cout << "# " << i << " th iteration\n";
//     //     std::tie(eta, status) = newton.run(x(i), [&](const Scalar& z) {
//     //         return
//     detail::continued_fraction_interp_eval_with_derivative(z,
//     //         x,
//     // a);
//     //     });

//     //     if (!(status == detail::newton_method<Scalar>::SUCCESS &&
//     //           real(eta) > RealScalar()))
//     //     {
//     //         // Invalid solution
//     //         continue;
//     //     }

//     //     eta          = detail::regularize_arg(eta);
//     //     tmp(n_found) = eta;
//     //     ++n_found;
//     // }
//     // std::sort(tmp.data(), tmp.data() + n_found,
//     //           [&](const Scalar& lhs, const Scalar& rhs) {
//     //               return real(lhs) < real(rhs);
//     //           });

//     // {
//     //     auto pos = std::unique(tmp.data(), tmp.data() + n_found,
//     //                            [=](const Scalar& x, const Scalar& y) {
//     //                                return detail::close_enough(
//     //                                    x, y, threshold * 50, threshold *
//     50);
//     //                            });
//     //     n_found = static_cast<Index>(std::distance(tmp.data(), pos));
//     // }
//     // std::cout << "# " << n_found << " roots found\n";
//     // for (Index i = 0; i < n_found; ++i)
//     // {
//     //     std::cout << "root(" << i << "): " << tmp(i) << '\n';
//     // }

//     //
//     // Barycentric form of Lagrange interpolation
//     //
//     // VectorType& tau = x;
//     // std::sort(tau.data(), tau.data() + tau.size(),
//     //           [&](const Scalar& lhs, const Scalar& rhs) {
//     //               return real(lhs) > real(rhs);
//     //           });

//     ResultType ret;

//     return ret;
// }

} // namespace: mxpfit

#endif /* MXPFIT_AAK_REDUCTION_HPP */

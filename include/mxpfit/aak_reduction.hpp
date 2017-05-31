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
// g(x[i]), formulate and evaluate continued fraction interpolation of the form
//
//                          a[0]
// g(x) = ------------------------------------------
//              a[1] * (exp(-x+x[0]) - 1)
//        1 + --------------------------------------
//                   a[2] * (exp(-x+x[1]) - 1)
//            1 + ----------------------------------
//                      a[3] * (exp(-x+x[2]) - 1)
//                1 + ------------------------------
//                     1 + ...
//
// where g(x[i]) = f(exp(-x[i])) = y[i].
//

template <typename VecX, typename VecY, typename VecA>
void cont_frac_interp_coeffs(const VecX& x, const VecY& y, VecA& a)
{
    using ResultType = typename VecY::Scalar;
    using Real       = typename Eigen::NumTraits<ResultType>::Real;
    using Index      = Eigen::Index;

    constexpr static const Real one = Real(1);
    // static const Real eps           = Eigen::NumTraits<Real>::epsilon();
    // static const Real tiny          = eps * eps;
    static const Real tiny = Eigen::NumTraits<Real>::epsilon();

    const auto a0 = y[0];
    a[0]          = a0;

    for (Index i = 1; i < x.size(); ++i)
    {
        const auto fi = a0 / y[i];
        ResultType f  = one;
        ResultType C  = f;
        ResultType D  = ResultType();
        const auto xi = x[i];
        for (Index j = 1; j < i; ++j)
        {
            const auto coeff_aj = a[j] * expm1(x[j - 1] - xi);

            D = one + coeff_aj * D;
            if (D == ResultType())
            {
                D = tiny;
            }

            C = one + coeff_aj / C;
            if (D == ResultType())
            {
                D = tiny;
            }

            f *= C * D;
            std::cout << "xxxxx " << C << '\t' << D << '\t' << f << '\n';
        }

        const auto delta = fi / f;
        D                = (delta * D - one / C) * expm1(x[i - 1] - xi);
        if (D == ResultType())
        {
            D = tiny;
        }

        a[i] = (one - delta) / D;
        std::cout << "ooo  coeff(" << i << ")\t" << a[i] << '\n';
    }

    return;
}

//
// Compute value of the continued fraction interpolation g(t) and its
// derivative g'(x).
//
template <typename VecX, typename VecA>
// std::tuple<typename VecA::Scalar, typename VecA::Scalar>
typename VecA::Scalar eval_cont_frac_interp(typename VecX::Scalar x,
                                            const VecX& xi, const VecA& ai)
{
    using ResultType = typename VecA::Scalar;
    using Real       = typename Eigen::NumTraits<ResultType>::Real;
    using Index      = Eigen::Index;

    constexpr static const Real one = Real(1);
    static const Real eps           = Eigen::NumTraits<Real>::epsilon();
    static const Real tiny          = eps * eps;

    const auto n = ai.size();

    const auto a0 = ai[0];

    ResultType f = one;
    ResultType C = f;
    ResultType D = ResultType();

    for (Index j = 1; j < n; ++j)
    {
        const auto coeff_aj = ai[j] * expm1(xi[j] - x);

        D = one + coeff_aj * D;
        if (D == ResultType())
        {
            D = tiny;
        }

        C = one + coeff_aj / C;
        if (D == ResultType())
        {
            D = tiny;
        }

        f *= C * D;
    }

    return a0 / f;
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

    using ArrayType  = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
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
    ArrayType m_a;
    ArrayType m_b;
    ArrayType m_x;
    ArrayType m_y;

    RRDType m_rrd;
    ConeigenSolverType m_coneig_solver;

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
    // Set auxiliary arrays to form quasi-Cauchy matrix.
    //
    template <typename DerivedF>
    void set_aux_arrays(const ExponentialSumBase<DerivedF>& orig);
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
template <typename DerivedF>
typename AAKReduction<T>::ResultType
AAKReduction<T>::compute(const ExponentialSumBase<DerivedF>& orig)
{

    const Index n0 = orig.size();
    resize(n0);

    //
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
    set_aux_arrays(orig);

    //
    // Solve coneigenvalue problem of matrix C, and determines the order of
    // reduced system, \f$ M \f$ from the Hankel singular values, which are
    // corresponding to the con-eigenvalues.
    //
    solve_coneig();
    const auto& sigma = m_coneig_solver.coneigenvalues();
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
    auto vec_u = m_coneig_solver.coneigenvectors().col(n1 - 1);
    // y[i] = conj(u[i]) / sqrt(conj(c[i]))
    m_y.array() = vec_u.array().conjugate() / m_b.array();
    //
    // Approximation of v(eta) by continued fraction
    //
    // detail::continued_fraction_interp_coeffs(x, y, a);
    detail::cont_frac_interp_coeffs(m_x, m_y, m_a);
    std::cout << m_a.transpose() << std::endl;

    std::cout << "*** Test continued fraction interpolation\n";
    for (Index i = 0; i < n0; ++i)
    {
        auto y_test = detail::eval_cont_frac_interp(m_x(i), m_x, m_a);
        std::cout << m_x(i) << '\t' << m_y(i) << '\t' << y_test << '\n';
    }
    return ResultType();
}

template <typename T>
template <typename DerivedF>
void AAKReduction<T>::set_aux_arrays(const ExponentialSumBase<DerivedF>& orig)
{
    // TODO: sort the exponents in appropriate order
    m_x = orig.exponents();
    m_y = -m_x.conjugate();
    m_b = orig.weights().conjugate().sqrt();
    m_a = m_b.conjugate() * m_x.exp();
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
    static const auto eps = Eigen::NumTraits<RealScalar>::epsilon();
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
    m_coneig_solver.compute(m_rrd.matrixPL(), m_rrd.vectorD());

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

//     ArrayType a(n0), b(n0), x(n0), y(n0);
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

//     // ArrayType tmp(n0);
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
//     // ArrayType& tau = x;
//     // std::sort(tau.data(), tau.data() + tau.size(),
//     //           [&](const Scalar& lhs, const Scalar& rhs) {
//     //               return real(lhs) > real(rhs);
//     //           });

//     ResultType ret;

//     return ret;
// }

} // namespace: mxpfit

#endif /* MXPFIT_AAK_REDUCTION_HPP */

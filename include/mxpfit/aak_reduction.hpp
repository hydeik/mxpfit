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

#include <algorithm>

#include <mxpfit/exponential_sum.hpp>
#include <mxpfit/quasi_cauchy_rrd.hpp>
#include <mxpfit/self_adjoint_coneigensolver.hpp>

namespace mxpfit
{

namespace detail
{

///
/// \internal
///
/// Create a rational function \f$ f(x)\f$ in the continued fraction form from a
/// given finite set of points \f$ x_{i} \f$ and their function values
/// \f$y_{i}.\f$
///
/// \f[
///   f(x)=\cfrac{a_{0}}{1+a_{1}\cfrac{x-x_{0}}{1+a_{2}\cfrac{x-x_{1}}{1+\cdots}}}
/// \f]
///
/// \param[in] x  set of points
/// \param[in] y function values at given points `x`. All elements of `y` should
///              not be zero.
/// \param[out] a  the coefficients of continued fraction interpolation
///

template <typename VecX, typename VecY, typename VecA>
void continued_fraction_interpolation_coeffs(const VecX& x, const VecY& y,
                                             VecA& a)
{
    using ResultType = typename VecY::Scalar;
    using Real       = typename Eigen::NumTraits<ResultType>::Real;
    using Index      = Eigen::Index;

    static const Real tiny = Eigen::NumTraits<Real>::min();

    const auto a0 = y[0];
    a[0]          = a0;

    for (Index i = 1; i < x.size(); ++i)
    {
        auto v        = a0 / y[i];
        const auto xi = x[i];
        for (Index j = 0; j < i - 1; ++j)
        {
            v -= Real(1);
            if (v == ResultType())
            {
                v = tiny;
            }
            v = (a[j + 1] * (xi - x[j])) / v;
        }
        v -= Real(1);
        if (v == ResultType())
        {
            v = tiny;
        }
        a[i] = (xi - x[i - 1]) / v;
    }
}

//
// Compute value of the continued fraction interpolation f(x) and its derivative
// f'(x).
//
// Let define R_{i}(x) recursively as
//
//  R_{n-1}(x) = a_{n-2} / [1 + a_{n-1} (x - x_{n-1})]
//  R_{i}(x) = a_{i-1} / [1 +  (x - x_{i}) R_{i+1}(x)]  (i=n-2,n-3,...1)
//
// then
//
//  f(x) = a_{0} / [1 + (x - x_{0}) R_{1}(x)].
//
// The derivative of f(x) becomes
//
//   f'(x) = -a_{0} / [1 + (x - x_{0}) R_{1}(x)]^2
//         * [R_{1}(x) + (x - x_{0}) R_{1}'(x)]
//
// The derivative of R_{i}(x) can also evaluated recursively as
//
//   R_{i}'(x) = -a_{i-1} / [1 +  (x - x_{i}) R_{i+1}(x)]^2
//             * [R_{i+1}(x) + (x - x_{i}) R_{i+1}'(x)]
//
// with
//
//   R_{n-1}'(x) = -a_{n-2} a_{n-1} / [1 + a_{n-1}(x - x_{n-1})]^2
//
template <typename VecX, typename VecA>
void continued_fraction_interpolation_eval_with_derivative(
    typename VecX::Scalar x, const VecX& xi, const VecA& ai)
{
    using ResultType = typename VecA::Scalar;
    using Real       = typename Eigen::NumTraits<ResultType>::Real;
    using Index      = Eigen::Index;

    static const Real tiny   = Eigen::NumTraits<Real>::min();
    constexpr const Real one = Real(1);

    const auto n = ai.size();

    auto d  = one + ai[n - 1] * (x - xi[n - 1]);
    auto f  = ai[n - 2] / d;      // R_{n-1}(x)
    auto df = -ai[n - 1] * f / d; // R_{n-1}'(x)

    for (Index k = n - 2; k > 0; --k)
    {
        d = one + (x - xi[k]) * f;
        if (d == ResultType())
        {
            d = tiny;
        }
        const auto f_pre = f;

        f  = ai[k - 1] / d;
        df = -f * (f_pre + (x - xi[k]) * df) / d;
    }

    // now f = f(x), df = f'(x)
}

template <typename VecX, typename VecY, typename VecA>
struct continued_fraction_interpolation
{
    using ArgumentType = typename VecX::Scalar;
    using ResultType   = typename VecY::Scalar;
    using Index        = Eigen::Index;

    continued_fraction_interpolation(const VecX& x, VecY& y, VecA& a)
        : m_x(x), m_y(y), m_a(a)
    {
        assert(m_x.size() == m_y.size());
        assert(m_a.size() == m_x.size());
        compute_coeffs();
    }

    ResultType operator()(ArgumentType x) const;

    ResultType derivative(ArgumentType x) const;

private:
    void compute_coeffs();
    const VecX& m_x;
    VecY& m_y;
    VecA& m_a;
};

template <typename VecX, typename VecY, typename VecA>
void continued_fraction_interpolation<VecX, VecY, VecA>::compute_coeffs()
{
    using Real             = typename Eigen::NumTraits<ResultType>::Real;
    static const Real tiny = Eigen::NumTraits<Real>::min();

    m_a[0] = m_y[0];

    for (Index i = 1; i < m_x.size(); ++i)
    {
        const auto xi = m_x[i];
        ResultType v  = m_a[0] / m_y[i];
        for (Index j = 0; j < i - 1; ++j)
        {
            v -= Real(1);
            if (v == ResultType())
            {
                v = tiny;
            }
            v = (m_a[j + 1] * (xi - m_x[j])) / v;
        }
        v -= Real(1);
        if (v == ResultType())
        {
            v = tiny;
        }
        m_a[i] = (xi - m_x[i - 1]) / v;
    }
}

template <typename VecX, typename VecY, typename VecA>
typename VecY::Scalar continued_fraction_interpolation<VecX, VecY, VecA>::
operator()(ArgumentType x) const
{
    using Real = typename Eigen::NumTraits<ResultType>::Real;
    constexpr static const Real one = Real(1);

    ResultType u = one;

    for (Index k = m_a.size() - 1; k > 0; --k)
    {
        // TODO: Need zero check?
        u = one + m_a[k] * (x - m_x[k - 1]) / u;
    }

    return m_a[0] / u;
}

template <typename VecX, typename VecY, typename VecA>
typename VecY::Scalar
continued_fraction_interpolation<VecX, VecY, VecA>::derivative(
    ArgumentType x) const
{
    using Real = typename Eigen::NumTraits<ResultType>::Real;
    constexpr static const Real one = Real(1);

    auto& u                    = m_y; // overwrite m_y
    ResultType u(u.size() - 1) = one;

    for (Index k = m_a.size() - 1; k > 0; --k)
    {
        // TODO: Need zero check?
        u = one + m_a[k] * (x - m_x[k - 1]) / u;
    }

    return m_a[0] / u;
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
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
    };

    using ConeigenSolverType = SelfAdjointConeigenSolver<Scalar>;
    using RRDType =
        QuasiCauchyRRD<Scalar, QuasiCauchyRRDFunctorLogPole<Scalar>>;
};

template <typename T>
template <typename DerivedF>
typename AAKReduction<T>::ResultType
AAKReduction<T>::compute(const ExponentialSumBase<DerivedF>& orig,
                         RealScalar threshold)
{
    static const auto eps = Eigen::NumTraits<RealScalar>::epsilon();
    using Eigen::numext::abs;
    using Eigen::numext::conj;
    using Eigen::numext::real;

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
    VectorType b(orig.weights().conjugate().sqrt());
    VectorType a(b.array().conjugate() * orig.exponents().exp());
    VectorType x(orig.exponents());
    VectorType y(-orig.exponents().conjugate());
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
    //
    // Approximation of v(eta) by continued fraction as
    //
    // v(eta) = a1 / (1+a2(eta-z1)/(1+a3(eta-z2)/(1+...)))
    //

    a(0) = conj(vec_u(0)) / b(0);
    for (Index i = 1; i < n0; ++i)
    {

        auto t = a(0) * b(i) / conj(vec_u(i));
        for (Index j = 0; j < i; ++i)
        {
        }
        a(i) = t;
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

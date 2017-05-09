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
/// \file quasi_cauchy_rrd.hpp
///
#ifndef MXPFIT_QUASI_CAUCHY_RRD_HPP
#define MXPFIT_QUASI_CAUCHY_RRD_HPP

#include <cassert>

#include <Eigen/Core>

namespace mxpfit
{
namespace detail
{

//
// Compute exponential z minus one, exp(z) -1
//
// --- for real
template <typename T>
inline auto expm1(T x) -> decltype(std::expm1(x))
{
    return std::expm1(x);
}
// --- for complex
template <typename T>
std::complex<T> expm1(const std::complex<T>& z)
{
    constexpr const T inf = std::numeric_limits<T>::infinity();
    constexpr const T nan = std::numeric_limits<T>::quiet_NaN();

    const T x = std::real(z);
    const T y = std::imag(z);

    if (std::isnan(x))
    {
        return {x, y == T() ? y : x};
    }
    else if (!std::isfinite(y))
    {
        if (x == inf)
        {
            return {-x, nan};
        }
        else if (x == -inf)
        {
            return {-T(1), std::copysign(T(), y)};
        }
        else
        {
            return {nan, nan};
        }
    }
    else
    {
        const auto u = std::expm1(x);
        if (y == T())
        {
            return {u, y};
        }
        else
        {
            const auto v = u + T(1);
            const auto w = std::sin(y / 2);
            const auto re =
                std::isfinite(v) ? u - 2 * v * w * w : v * std::cos(y);
            return {re, v * std::sin(y)};
        }
    }
}

} // namespace: detail

///
/// ### DefaultQuasiCauchyRRDFunctor
///
/// Default Functor class for QuasiCauchyRRD.
///
/// \tparam T Scalar type of matrix elements
///
/// This functor is for a quasi-Cauchy matrix whose element is expressed as
/// \f$C_{ij}=\frac{a_{i}b_{j}}{x_{i}+y_{j}}\f$
///
template <typename T>
struct DefaultQuasiCauchyRRDFunctor
{
    constexpr static T matrix_element(T ai, T bj, T xi, T yj)
    {
        return ai * bj / (xi + yj);
    }

    constexpr static T update_coeff(T xi, T xj, T yj)
    {
        return (xi - xj) / (xi + yj);
    }
};

///
/// ### QuasiCauchyRRDFunctorLogPole
///
/// Functor class for the Cauchy-like matrix appearing in parameter reduction of
/// exponential sum.
///
/// \tparam T Scalar type of matrix elements
///
/// This functor is for a Cauchy-like matrix whose element is expressed as
/// \f[
///    C_{ij}=\frac{a_{i}b_{j}}{exp(p_{i}) - exp(q_{j})}
/// \f]
///
template <typename T>
struct QuasiCauchyRRDFunctorLogPole
{
    constexpr static T matrix_element(T ai, T bj, T pi, T qj)
    {
        using Eigen::numext::exp;
        return ai * bj / (exp(qj) * detail::expm1(pi - qj));
    }

    constexpr static T update_coeff(T pi, T pj, T qj)
    {
        using Eigen::numext::exp;
        // (1 - exp(pj - pi)) / (1 - exp(qj - pi));
        return detail::expm1(pj - pi) / detail::expm1(qj - pi);
    }
};

namespace detail
{

// Apply row permutation
template <typename IPiv, typename MatX, typename VecWork>
void apply_row_permutation(const Eigen::DenseBase<IPiv>& ipiv,
                           Eigen::DenseBase<MatX>& matX,
                           Eigen::DenseBase<VecWork>& work)
{
    using Index = Eigen::Index;
    for (Index j = 0; j < matX.cols(); ++j)
    {
        for (Index i = 0; i < matX.rows(); ++i)
        {
            work(ipiv(i)) = matX(i, j);
        }

        matX.col(j) = work;
    }
}

//
// Functor class for makeing dense matrix expression of a quasi-Cauchy matrix
//
template <typename VecA, typename VecB, typename VecX, typename VecY,
          typename FunctorBody>
struct quasi_cauchy_functor
{
    using Index = Eigen::Index;

    quasi_cauchy_functor(const VecA& a, const VecB& b, const VecX& x,
                         const VecY& y)
        : m_a(a), m_b(b), m_x(x), m_y(y)
    {
    }

    auto operator()(Index i, Index j) const
        -> decltype(FunctorBody::matrix_element(typename VecA::Scalar(),
                                                typename VecB::Scalar(),
                                                typename VecX::Scalar(),
                                                typename VecY::Scalar()))
    {
        // m_a(i) * m_b(j) / (m_x(i) + m_y(j));
        return FunctorBody::matrix_element(m_a(i), m_b(j), m_x(i), m_y(j));
    }

private:
    const VecA& m_a;
    const VecB& m_b;
    const VecX& m_x;
    const VecY& m_y;
};

} // namespace detail

///
/// ### QuasiCauchyRRD
///
/// \brief Compute the rank-revealing Cholesky decomposition of a self-adjoint
///   quasi-Cauchy matrix in high relative accuracy.
///
/// \tparam T the scalar type of matrix to be decomposed
/// \tparam Functor Functor class that provides static functions for calculating
///   matrix elements and Cholesky factors, such that,
/// ``` c++
///   /* Compute `ai * bj / (xi + yj)
///   static T matrix_element(T ai, T bj, T xi, T yj);
///   /* Compute `(xi - xj) / (xi + yj)
///   static T update_coeff(T xi, T xj, T yj);
/// ```
///
/// For given arrays of $a_{i},b_{i},x_{i},y_{i} \, (i=1,2,\dots,n)$ a
/// quasi-Cauchy matrix \f$ C \f$ is defined as
///
///   \f[
///     C_{ij} = \frac{a_{i}^{} b_{j}^{}}{x_{i}^{} + y_{j}^{}} \quad
///     (i,j=1,2,\dots,n).
///   \f]
///
/// We assume that the matrix \f$ C \f$ is self-adjoint and positive
/// definite. A quasi-Cauchy matrix is usually rank deficient. Let \f$ m =
/// \text{rank}(C) \f$ then, matrix `C` can have partial Cholesky decomposition
/// of the form
///
///   \f[ C = (PL)D^2(PL)^{\ast}, \f]
///
/// where \f$ L \f$ is \f$ n \times m \f$ unit triangular (trapezoidal) matrix,
/// \f$ D \f$ is \f$ m \times m \f$ diagonal matrix, and \f$ P \f$ is a \f$ m
/// \times n \f$ permutation matrix.
///
///
/// #### Algorithm
///
/// The factorization is made using the modified Gaussian elimination of
/// complete pivoting (GECP) proposed in Ref. [1], which is also described in
/// Algorithm 2 and 3 in Ref.[2]. Following the algorithms described in Ref.
/// [2], we stop the factorization when the diagonal element \f$ D_{mm} \f$
/// becomes smaller than the threshold value, \f$ D_{mm} \leq \delta^2 \epsilon
/// \f$, at certain \f$ m \f$ where \f$ \delta \f$ is mimimal target size of
/// singular values of the systems to be kept in the later step, and \f$
/// \epsilon \f$ is the machine epsilon. Note that the diagonal elements \f$
/// D_{ii} \f$ are sorted in non-increasing order by complete pivoting.
///
///
/// #### Complexity
///
/// The complexity of this algorithm is \f$
/// \mathcal{O}(n(\log(\delta\epsilon)^{-1})^2) \f$.
///
///
/// #### References
///
/// 1. J. Demmel, "ACCURATE SINGULAR VALUE DECOMPOSITIONS OF STRUCTURED
///    MATRICES", SIAM J. Matrix Anal. Appl. **21** (1999) 562-580.
///    [DOI: https://doi.org/10.1137/S0895479897328716]
/// 2. T. S. Haut and G. Beylkin, "FAST AND ACCURATE CON-EIGENVALUE ALGORITHM
///    FOR OPTIMAL RATIONAL APPROXIMATIONS", SIAM J. Matrix Anal. Appl. **33**
///    (2012) 1101-1125.
///    [DOI: https://doi.org/10.1137/110821901]
///

// --- Forward declaration
template <typename T, typename Functor = DefaultQuasiCauchyRRDFunctor<T>>
class QuasiCauchyRRD;
//
// --- Implementation
//
template <typename T, typename Functor>
class QuasiCauchyRRD
{
public:
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<T>::Real;
    using ComplexScalar = std::complex<RealScalar>;
    using StorageIndex  = Eigen::Index;
    using Index         = Eigen::Index;
    using MatrixType    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using IndicesType    = Eigen::Matrix<Index, Eigen::Dynamic, 1>;

    ///
    /// Default constructor
    ///
    QuasiCauchyRRD()
        : m_a(),
          m_b(),
          m_x(),
          m_y(),
          m_work(),
          m_ipiv(),
          m_matPL(),
          m_vecD(),
          m_threshold(Eigen::NumTraits<RealScalar>::epsilon()),
          m_is_initialized()
    {
    }

    ///
    /// Default destructor
    ///
    ~QuasiCauchyRRD() = default;

    ///
    /// Compute RRD of self-adjoint quasi-Cauchy matrix.
    ///
    /// \param[in] a  vector of length @f$ n @f$ defining matrix @f$ C. @f$.
    /// \param[in] b  vector of length @f$ n @f$ defining matrix @f$ C. @f$.
    /// \param[in] x  vector of length @f$ n @f$ defining matrix @f$ C. @f$
    /// \param[in] y  vector of length @f$ n @f$ defining matrix @f$ C. @f$
    ///
    template <typename VecA, typename VecB, typename VecX, typename VecY>
    void
    compute(const Eigen::EigenBase<VecA>& a, const Eigen::EigenBase<VecB>& b,
            const Eigen::EigenBase<VecX>& x, const Eigen::EigenBase<VecY>& y)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecA);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecB);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecX);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecY);

        m_a = a.derived();
        m_b = b.derived();
        m_x = x.derived();
        m_y = y.derived();

        m_work.resize(a.derived().size());
        m_ipiv.resize(a.derived().size());

        const Index rank = pivot_order();
        m_matPL.resize(m_a.derived().size(), rank);
        m_vecD.resize(rank);
        factorize();

        // apply row permutations
        detail::apply_row_permutation(m_ipiv, m_matPL, m_work);

        m_is_initialized = true;
    }

    ///
    /// Set threshold value for GECP termination.
    ///
    /// We stop GECP step as soon as the diagonal element of Cholesky factor
    /// \f$D_{mm}\f$ becomes smaller than this value.
    ///
    QuasiCauchyRRD& setThreshold(RealScalar threshold)
    {
        m_threshold = threshold;
        return *this;
    }

    ///
    /// \return the rank of matrix revealed.
    ///
    Index rank() const
    {
        return m_vecD.size();
    }

    ///
    /// \return const reference to the rank revealing factor \f$ X=PL \f$
    ///
    const MatrixType& matrixPL() const
    {
        assert(m_is_initialized && "QuasiCauchyRRD is not initialized");
        return m_matPL;
    }

    ///
    /// \return const reference to the rank revealing factor \f$ D \f$
    ///
    const RealVectorType& vectorD() const
    {
        assert(m_is_initialized && "QuasiCauchyRRD is not initialized");
        return m_vecD;
    }

    ///
    /// Create dense matrix expression of quasi-Cauchy matrix
    ///
    /// \param[in] a  vector of length @f$ n @f$ defining matrix @f$ C. @f$.
    /// \param[in] b  vector of length @f$ n @f$ defining matrix @f$ C. @f$.
    /// \param[in] x  vector of length @f$ n @f$ defining matrix @f$ C. @f$
    /// \param[in] y  vector of length @f$ n @f$ defining matrix @f$ C. @f$
    ///
    template <typename VecA, typename VecB, typename VecX, typename VecY>
    static Eigen::CwiseNullaryOp<
        detail::quasi_cauchy_functor<VecA, VecB, VecX, VecY, Functor>,
        MatrixType>
    makeDenseExpr(const Eigen::EigenBase<VecA>& a,
                  const Eigen::EigenBase<VecB>& b,
                  const Eigen::EigenBase<VecX>& x,
                  const Eigen::EigenBase<VecY>& y)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecA);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecB);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecX);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecY);

        assert(b.size() == a.size());
        assert(x.size() == a.size());
        assert(y.size() == a.size());

        using functor_t =
            detail::quasi_cauchy_functor<VecA, VecB, VecX, VecY, Functor>;

        return MatrixType::NullaryExpr(
            a.size(), a.size(),
            functor_t(a.derived(), b.derived(), x.derived(), y.derived()));
    }

protected:
    VectorType m_a;
    VectorType m_b;
    VectorType m_x;
    VectorType m_y;
    VectorType m_work;
    IndicesType m_ipiv;
    MatrixType m_matPL;
    RealVectorType m_vecD;

    RealScalar m_threshold;
    bool m_is_initialized;

    Index pivot_order();
    void factorize();
};

template <typename T, typename Functor>
typename QuasiCauchyRRD<T, Functor>::Index
QuasiCauchyRRD<T, Functor>::pivot_order()
{
    using Eigen::numext::abs;

    const Index n = m_a.size();

    assert(m_b.size() == n);
    assert(m_x.size() == n);
    assert(m_y.size() == n);
    assert(m_work.size() == n);
    assert(m_ipiv.size() == n);

    //
    // Form vector g(i) = a(i) * b(i) / (x(i) + y(i))
    //
    VectorType& g = m_work;

    for (Index i = 0; i < n; ++i)
    {
        g[i] = Functor::matrix_element(m_a[i], m_b[i], m_x[i], m_y[i]);
    }

    // Initialize rows transposition matrix
    for (Index i = 0; i < n; ++i)
    {
        m_ipiv[i] = i;
    }

    // GECP iteration
    Index m = 0;
    while (m < n)
    {
        //
        // Find m <= l < n such that |g(l)| = max_{m<=k<n}|g(k)|
        //
        Index l;
        auto max_diag = RealScalar();
        for (Index k = m; k < n; ++k)
        {
            const auto abs_gk = abs(g[k]);
            if (abs_gk > max_diag)
            {
                max_diag = abs_gk;
                l        = k;
            }
        }

        if (max_diag < m_threshold)
        {
            break;
        }

        if (l != m)
        {
            // Swap elements
            std::swap(m_ipiv[l], m_ipiv[m]);
            std::swap(g[l], g[m]);
            std::swap(m_a[l], m_a[m]);
            std::swap(m_b[l], m_b[m]);
            std::swap(m_x[l], m_x[m]);
            std::swap(m_y[l], m_y[m]);
        }

        // Update diagonal of Schur complement
        const auto xm = m_x[m];
        const auto ym = m_y[m];

        for (Index k = m + 1; k < n; ++k)
        {
            // g[k] *= (m_x[k] - xm) / (m_x[k] + ym);
            g[k] *= Functor::update_coeff(m_x[k], xm, ym);
            // g[k] *= (m_y[k] - ym) / (m_y[k] + xm);
            g[k] *= Functor::update_coeff(m_y[k], ym, xm);
        }
        ++m;
    }
    //
    // Returns the rank of input matrix
    //
    return m;
}

template <typename T, typename Functor>
void QuasiCauchyRRD<T, Functor>::factorize()
{
    using Eigen::numext::real;
    using Eigen::numext::sqrt;

    const auto n = m_matPL.rows();
    const auto m = m_matPL.cols();

    m_matPL.setZero();
    const auto b0 = m_b[0];
    const auto y0 = m_y[0];

    for (Index l = 0; l < n; ++l)
    {
        // m_matPL(l, 0) = m_a[l] * b0 / (m_x[l] + y0);
        m_matPL(l, 0) = Functor::matrix_element(m_a[l], b0, m_x[l], y0);
    }

    for (Index k = 1; k < m; ++k)
    {
        // Upgrade generators
        const auto xkm1 = m_x[k - 1];
        const auto ykm1 = m_y[k - 1];
        for (Index l = k; l < n; ++l)
        {
            // m_a[l] *= (m_x[l] - xkm1) / (m_x[l] + ykm1);
            m_a[l] *= Functor::update_coeff(m_x[l], xkm1, ykm1);
        }
        for (Index l = k; l < n; ++l)
        {
            // m_b[l] *= (m_y[l] - ykm1) / (m_y[l] + xkm1);
            m_b[l] *= Functor::update_coeff(m_y[l], ykm1, xkm1);
        }
        // Extract k-th column for Cholesky factors
        const auto bk = m_b[k];
        const auto yk = m_y[k];
        for (Index l = k; l < n; ++l)
        {
            m_matPL(l, k) = Functor::matrix_element(m_a[l], bk, m_x[l], yk);
        }
    }
    //
    // Scale strictly lower triangular part of G
    //   - diagonal part of G contains D**2
    //   - L = tril(G) * D^{-2} + I
    //
    for (Index j = 0; j < m; ++j)
    {
        const auto djj   = real(m_matPL(j, j));
        const auto scale = RealScalar(1) / djj;

        m_matPL(j, j) = RealScalar(1);
        m_vecD[j] = sqrt(djj);
        for (Index i = j + 1; i < n; ++i)
        {
            m_matPL(i, j) *= scale;
        }
    }

    return;
}

namespace detail
{

template <typename VecA, typename VecX>
struct self_adjoint_quasi_cauchy_helper
{
    using Scalar =
        typename Eigen::ScalarBinaryOpTraits<typename VecA::Scalar,
                                             typename VecX::Scalar>::ReturnType;
    using MatrixType =
        Eigen::Matrix<Scalar, VecA::SizeAtCompileTime, VecA::SizeAtCompileTime,
                      Eigen::ColMajor, VecA::MaxSizeAtCompileTime,
                      VecA::MaxSizeAtCompileTime>;
};

template <typename VecA, typename VecX>
struct self_adjoint_quasi_cauchy_functor
{
    using Scalar =
        typename Eigen::ScalarBinaryOpTraits<typename VecA::Scalar,
                                             typename VecX::Scalar>::ReturnType;
    using Index = Eigen::Index;
    using MatrixType =
        Eigen::Matrix<Scalar, VecA::SizeAtCompileTime, VecA::SizeAtCompileTime,
                      Eigen::ColMajor, VecA::MaxSizeAtCompileTime,
                      VecA::MaxSizeAtCompileTime>;

    self_adjoint_quasi_cauchy_functor(const VecA& a, const VecX& x)
        : m_a(a), m_x(x)
    {
    }

    Scalar operator()(Index i, Index j) const
    {
        using Eigen::numext::conj;
        return m_a(i) * conj(m_a(j)) / (m_x(i) + conj(m_x(j)));
    }

private:
    const VecA& m_a;
    const VecX& m_x;
};

} // namespace detail

///
/// ### SelfAdjointQuasiCauchyRRD
///
/// \brief  Compute the rank-revealing decomposition (RRD) of a self-adjoint
///   quasi-Cauchy matrix in high relative accuracy.
///
/// \tparam T  the scalar type of matrix to be decomposed
///
/// For given arrays of $a_{i},x_{i} \, (i=1,2,\dots,n)$ a self-adjoint
/// quasi-Cauchy matrix \f$ C \f$ is defined as
///
///   \f[
///     C_{ij} = \frac{a_{i}^{} a_{j}^{\ast}}{x_{i}^{} + x_{j}^{\ast}} \quad
///     (i,j=1,2,\dots,n).
///   \f]
///
/// We assume that the matrix \f$ C \f$ is also positive definite. A
/// quasi-Cauchy matrix is usually rank deficient. Let \f$ m = \text{rank}(C)
/// \f$ then, matrix `C` can have partial Cholesky decomposition of the form
///
///   \f[ C = (PL)D^2(PL)^{\ast}, \f]
///
/// where \f$ L \f$ is \f$ n \times m \f$ unit triangular (trapezoidal) matrix,
/// \f$ D \f$ is \f$ m \times m \f$ diagonal matrix, and \f$ P \f$ is a \f$ m
/// \times n \f$ permutation matrix.
///
/// #### Algorithm
///
/// The factorization is made using the modified Gaussian elimination of
/// complete pivoting (GECP) proposed in Ref. [1], which is also described in
/// Algorithm 2 and 3 in Ref.[2]. Following the algorithms described in Ref.
/// [2], we stop the factorization when the diagonal element \f$ D_{mm} \f$
/// becomes smaller than the threshold value, \f$ D_{mm} \leq \delta^2 \epsilon
/// \f$, at certain \f$ m \f$ where \f$ \delta \f$ is mimimal target size of
/// singular values of the systems to be kept in the later step, and \f$
/// \epsilon \f$ is the machine epsilon. Note that the diagonal elements \f$
/// D_{ii} \f$ are sorted in non-increasing order by complete pivoting.
///
/// #### Complexity
///
/// The complexity of this algorithm is \f$
/// \mathcal{O}(n(\log(\delta\epsilon)^{-1})^2) \f$.
///
///
/// #### References
///
/// 1. J. Demmel, "ACCURATE SINGULAR VALUE DECOMPOSITIONS OF STRUCTURED
///    MATRICES", SIAM J. Matrix Anal. Appl. **21** (1999) 562-580.
///    [DOI: https://doi.org/10.1137/S0895479897328716]
/// 2. T. S. Haut and G. Beylkin, "FAST AND ACCURATE CON-EIGENVALUE ALGORITHM
///    FOR OPTIMAL RATIONAL APPROXIMATIONS", SIAM J. Matrix Anal. Appl. **33**
///    (2012) 1101-1125.
///    [DOI: https://doi.org/10.1137/110821901]
///
template <typename T>
class SelfAdjointQuasiCauchyRRD
{
public:
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<T>::Real;
    using ComplexScalar = std::complex<RealScalar>;
    using StorageIndex  = Eigen::Index;
    using Index         = Eigen::Index;
    using MatrixType    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using IndicesType    = Eigen::Matrix<Index, Eigen::Dynamic, 1>;

    ///
    /// Default constructor
    ///
    SelfAdjointQuasiCauchyRRD()
        : m_a(),
          m_x(),
          m_work(),
          m_ipiv(),
          m_matPL(),
          m_vecD(),
          m_threshold(Eigen::NumTraits<RealScalar>::epsilon()),
          m_is_initialized(false)
    {
    }

    ///
    /// Default destructor
    ///
    ~SelfAdjointQuasiCauchyRRD() = default;

    ///
    /// Compute RRD of self-adjoint quasi-Cauchy matrix.
    ///
    /// @param[in] a  vector of length @f$ n @f$ defining matrix @f$ C. @f$.
    /// @param[in] x  vector of length @f$ n @f$ defining matrix @f$ C. @f$
    ///
    template <typename VecA, typename VecX>
    void compute(const Eigen::EigenBase<VecA>& a,
                 const Eigen::EigenBase<VecX>& x)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecA);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecX);

        m_a = a.derived();
        m_x = x.derived();
        m_work.resize(a.derived().size());
        m_ipiv.resize(a.derived().size());

        const Index rank = pivot_order();
        m_matPL.resize(m_a.derived().size(), rank);
        m_vecD.resize(rank);
        factorize();

        // apply row permutations
        detail::apply_row_permutation(m_ipiv, m_matPL, m_work);

        m_is_initialized = true;
    }

    ///
    /// Set threshold value for GECP termination.
    ///
    /// We stop GECP step as soon as the diagonal element of Cholesky factor
    /// \f$D_{mm}\f$ becomes smaller than this value.
    ///
    SelfAdjointQuasiCauchyRRD& setThreshold(RealScalar threshold)
    {
        m_threshold = threshold;
        return *this;
    }

    ///
    /// \return the rank of matrix revealed.
    ///
    Index rank() const
    {
        return m_vecD.size();
    }
    ///
    /// \return const reference to the rank revealing factor \f$ X=PL \f$
    ///
    const MatrixType& matrixPL() const
    {
        assert(m_is_initialized &&
               "SelfAdjointQuasiCauchyRRD is not initialized");
        return m_matPL;
    }

    ///
    /// \return const reference to the rank revealing factor \f$ D \f$
    ///
    const RealVectorType& vectorD() const
    {
        assert(m_is_initialized &&
               "SelfAdjointQuasiCauchyRRD is not initialized");
        return m_vecD;
    }

    ///
    /// Create dense matrix expression of quasi-Cauchy matrix
    ///
    /// \param[in] a  vector of length @f$ n @f$ defining matrix @f$ C. @f$.
    /// \param[in] x  vector of length @f$ n @f$ defining matrix @f$ C. @f$
    ///
    template <typename VecA, typename VecX>
    static Eigen::CwiseNullaryOp<
        detail::self_adjoint_quasi_cauchy_functor<VecA, VecX>, MatrixType>
    makeDenseExpr(const Eigen::DenseBase<VecA>& a,
                  const Eigen::DenseBase<VecX>& x)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecA);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecX);
        using functor_t = detail::self_adjoint_quasi_cauchy_functor<VecA, VecX>;

        assert(a.size() == x.size());

        return MatrixType::NullaryExpr(a.size(), a.size(),
                                       functor_t(a.derived(), x.derived()));
    }

protected:
    VectorType m_a;
    VectorType m_x;
    VectorType m_work;
    IndicesType m_ipiv;
    MatrixType m_matPL;
    RealVectorType m_vecD;

    RealScalar m_threshold;
    bool m_is_initialized;

    Index pivot_order();
    void factorize();
};

template <typename T>
Eigen::Index SelfAdjointQuasiCauchyRRD<T>::pivot_order()
{
    using Eigen::numext::abs;
    using Eigen::numext::abs2;
    using Eigen::numext::conj;

    const Index n = m_a.size();

    assert(m_x.size() == n);
    assert(m_work.size() == n);
    assert(m_ipiv.size() == n);

    //
    // Form vector g(i) = a(i) * b(i) / (x(i) + y(i))
    //
    VectorType& g = m_work;

    for (Index i = 0; i < n; ++i)
    {
        g[i] = m_a[i] * conj(m_a[i]) / (m_x[i] + conj(m_x[i]));
    }

    // Initialize rows transposition matrix
    for (Index i = 0; i < n; ++i)
    {
        m_ipiv(i) = i;
    }

    // GECP iteration
    Index m = 0;
    while (m < n)
    {
        //
        // Find m <= l < n such that |g(l)| = max_{m<=k<n}|g(k)|
        //
        Index l;
        auto max_diag = RealScalar();
        for (Index k = m; k < n; ++k)
        {
            const auto abs_gk = abs(g[k]);
            if (abs_gk > max_diag)
            {
                max_diag = abs_gk;
                l        = k;
            }
        }

        if (max_diag < m_threshold)
        {
            break;
        }

        if (l != m)
        {
            // Swap elements
            std::swap(m_ipiv[l], m_ipiv[m]);
            std::swap(g[l], g[m]);
            std::swap(m_a[l], m_a[m]);
            std::swap(m_x[l], m_x[m]);
        }

        // Update diagonal of Schur complement
        const auto xm = m_x[m];
        const auto ym = conj(m_x[m]);

        for (Index k = m + 1; k < n; ++k)
        {
            const auto s1 = (m_x[k] - xm) / (m_x[k] + ym);
            g[k] *= s1 * conj(s1);
        }
        ++m;
    }
    //
    // Returns the rank of input matrix
    //
    return m;
}

template <typename T>
void SelfAdjointQuasiCauchyRRD<T>::factorize()
{
    using Eigen::numext::conj;
    using Eigen::numext::real;
    using Eigen::numext::sqrt;

    const auto n = m_matPL.rows();
    const auto m = m_matPL.cols();

    m_matPL.setZero();
    const auto b0 = conj(m_a(0));
    const auto y0 = conj(m_x(0));

    for (Index l = 0; l < n; ++l)
    {
        m_matPL(l, 0) = m_a[l] * b0 / (m_x[l] + y0);
    }

    for (Index k = 1; k < m; ++k)
    {
        // Upgrade generators
        const auto xkm1 = m_x[k - 1];
        const auto ykm1 = conj(xkm1);
        for (Index l = k; l < n; ++l)
        {
            m_a[l] *= (m_x[l] - xkm1) / (m_x[l] + ykm1);
        }
        // Extract k-th column for Cholesky factors
        const auto bk = conj(m_a[k]);
        const auto yk = conj(m_x[k]);
        for (Index l = k; l < n; ++l)
        {
            m_matPL(l, k) = m_a[l] * bk / (m_x[l] + yk);
        }
    }
    //
    // Scale strictly lower triangular part of G
    //   - diagonal part of G contains D**2
    //   - L = tril(G) * D^{-2} + I
    //
    for (Index j = 0; j < m; ++j)
    {
        const auto djj   = real(m_matPL(j, j));
        const auto scale = RealScalar(1) / djj;

        m_matPL(j, j) = RealScalar(1);
        m_vecD[j] = sqrt(djj);
        for (Index i = j + 1; i < n; ++i)
        {
            m_matPL(i, j) *= scale;
        }
    }

    return;
}

} // namespace: mxpfit

#endif /* MXPFIT_QUASI_CAUCHY_RRD_HPP */

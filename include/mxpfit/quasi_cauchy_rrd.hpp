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
    const VecA m_a;
    const VecX m_x;
};

} // namespace detail

///
/// Matrix expression for self-adjoint quasi-Cauchy matrix
///

template <typename VecA, typename VecX>
Eigen::CwiseNullaryOp<
    detail::self_adjoint_quasi_cauchy_functor<VecA, VecX>,
    typename detail::self_adjoint_quasi_cauchy_helper<VecA, VecX>::MatrixType>
makeSelfAdjointQuasiCauchy(const Eigen::DenseBase<VecA>& a,
                           const Eigen::DenseBase<VecX>& x)
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecA);
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecX);
    using MatrixType =
        typename detail::self_adjoint_quasi_cauchy_helper<VecA,
                                                          VecX>::MatrixType;
    using Functor = detail::self_adjoint_quasi_cauchy_functor<VecA, VecX>;

    assert(a.size() == x.size());

    return MatrixType::NullaryExpr(a.size(), a.size(),
                                   Functor(a.derived(), x.derived()));
}

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
/// quasi-Cauchy matrix is usually rank deficient. Let \f$ m =
/// \text{rank}(C)
/// \f$ then, matrix `C` can have partial Cholesky decomposition of the form
///
///   \f[ C = (PL)D^2(PL)^{\ast}, \f]
///
/// where \f$ L \f$ is \f$ n \times m \f$ unit triangular (trapezoidal)
/// matrix,
/// \f$ D \f$ is \f$ m \times m \f$ diagonal matrix, and \f$ P \f$ is a \f$
/// m
/// \times n \f$ permutation matrix.
///
/// #### Algorithm
///
/// The factorization is made using the modified Gaussian elimination of
/// complete pivoting (GECP) proposed in Ref. [1], which is also described
/// in
/// Algorithm 2 and 3 in Ref.[2]. Following the algorithms described in Ref.
/// [2], we stop the factorization when the diagonal element \f$ D_{mm} \f$
/// becomes smaller than the threshold value, \f$ D_{mm} \leq \delta^2
/// \epsilon
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
/// 1. J. Demmel, "Accurate singular value decompositions of structured
///    matrices", SIAM J. Matrix Anal. Appl., **21** (1999), pp. 562–580.
/// 2. T. S. Haut and G. Beylkin, "Fast and accurate con-eigenvalue
/// algorithm
///    for optimal rational approximations", SIAM J. Matrix Anal. Appl.,
///    **33**
///    (2012), pp. 1101-1125.
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
        // m_matPL = m_rows_transposition * m_matPL;
        apply_row_permutation();

        m_is_initialized = true;
    }

    ///
    /// Set threshold value for GECP termination.
    ///
    /// We stop GECP step as soon as the diagonal element of Cholesky factor
    /// \f$D_{mm}\f$ becomes smaller than this value.
    ///
    void setThreshold(RealScalar threshold)
    {
        m_threshold = threshold;
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

private:
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
    void apply_row_permutation();
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

template <typename T>
void SelfAdjointQuasiCauchyRRD<T>::apply_row_permutation()
{
    for (Index j = 0; j < m_matPL.cols(); ++j)
    {
        for (Index i = 0; i < m_matPL.rows(); ++i)
        {
            m_work(m_ipiv(i)) = m_matPL(i, j);
        }

        m_matPL.col(j) = m_work;
    }
}

} // namespace: mxpfit

#endif /* MXPFIT_QUASI_CAUCHY_RRD_HPP */

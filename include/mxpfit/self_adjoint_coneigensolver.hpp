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
/// \file self_adjoint_coneigensolver.hpp
///
#ifndef MXPFIT_SELF_ADJOINT_CONEIGENSOLVER_HPP
#define MXPFIT_SELF_ADJOINT_CONEIGENSOLVER_HPP

#include <cassert>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/QR>

#include <mxpfit/jacobi_svd.hpp>

namespace mxpfit
{
///
/// ### SelfAdjointConeigenSolver
///
/// \brief Compute con-eigenvalue decomposition of self-adjoint matrix having a
///        rank-revealing decomposition
///
/// \tparam T  Scalar type of matrix to be decomposed.
///
/// Let \f$A\f$ be an \f$n \times n\f$ self-adjoint matrix having a rank
/// revealing decomposition of the form
///
/// \f[ A = X D^2 X^{\ast},\f]
///
/// where \f$X\f$ is a \f$n \times k \, (n \geq k)\f$ matrix and \f$D\f$ is a
/// \f$k \times k\f$ diagonal matrix with non-negative entries. This class
/// computes a con-eigenvalue decomposition of matrix \f$A\f$ defined as
///
/// \f[
///   A = \overline{U} \Sigma U^{\ast}
/// \f]
///
/// where \f$U\f$ is a \f$n \times k\f$ matrix satisfying \f$U^{-1}=U^{T},\f$
/// and overline denotes the element-wise complex conjugate of a matrix.
/// \f$\Sigma=\mathrm{diag}(\sigma_{1},\dots,\sigma_{k})\f$ is a \f$k \times
/// k\f$ diagonal matrix with \f$\sigma_{1}\geq\sigma_{2}\geq \cdots \geq
/// \sigma_{k} > 0.\f$ The con-eigenvalue decomposition defined above is a
/// special case of singular value decomposition (SVD) \f$A=W\Sigma V^{\ast}\f$,
/// where \f$ W=\overline{U} \f$ and \f$V=U.\f$
///
/// The con-eigenvalue decomposition is computed in high-relative accuracy using
/// the algorithm developed by Haut and Beylkin. The algorithm can also be
/// regarded as modification of Demmel's algorithm for high accuracy SVD of a
/// matrix with rank-revealing decomposition. See the references listed below
/// for the details.
///
///
/// #### References
///
/// 1. T. S. Haut and G. Beylkin, "FAST AND ACCURATE CON-EIGENVALUE ALGORITHM
///    FOR OPTIMAL RATIONAL APPROXIMATIONS", SIAM J. Matrix Anal. Appl. **33**
///    (2012) 1101-1125. [DOI: https://doi.org/10.1137/110821901]
/// 2. J. Demmel, "ACCURATE SINGULAR VALUE DECOMPOSITIONS OF STRUCTURED
///    MATRICES", SIAM J. Matrix Anal. Appl. **21** (1999) 562-580.
///    [DOI: https://doi.org/10.1137/S0895479897328716]
/// 3. J. Demmel, M. Gu, S. Eisenstat, I. Slapnicar, K. Veselic, and Z. Drmac,
///    "ACCURATE SINGULAR VALUE DECOMPOSITIONS OF STRUCTURED MATRICES", SIAM J.
///    Linear Algebra Appl. **299** (1999) 21-80. [DOI:
///    https://doi.org/10.1016/S0024-3795(99)00134-2]
///

enum DecompositionOption
{
    ConeigenvaluesOnly,
    ComputeConeigenvectors
};

template <typename T>
class SelfAdjointConeigenSolver
{
public:
    using Index         = Eigen::Index;
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
    using ComplexScalar = std::complex<RealScalar>;

    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    using ConeigenvalueType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

protected:
    enum
    {
        IsComplex  = Eigen::NumTraits<Scalar>::IsComplex,
        PacketSize = Eigen::internal::packet_traits<Scalar>::size,
        Alignment  = Eigen::internal::traits<MatrixType>::Alignment,
        // Alignment  = Eigen::internal::unpacket_traits<PacketType>::alignment
    };

    using MappedMatrix = Eigen::Map<MatrixType, Alignment>;
    using PacketType   = typename Eigen::internal::packet_traits<Scalar>::type;

    MatrixType m_ceigvecs;     // m x n (m >= n)
    RealVectorType m_ceigvals; // n
    MatrixType m_mat_work1;    // n x n
    MatrixType m_mat_work2;    // n x n

public:
    SelfAdjointConeigenSolver()                                 = default;
    SelfAdjointConeigenSolver(const SelfAdjointConeigenSolver&) = default;

    explicit SelfAdjointConeigenSolver(Index size, Index rank)
        : m_ceigvecs(size, rank),
          m_ceigvals(rank),
          m_mat_work1(rank, rank),
          m_mat_work2(rank, rank)
    {
        assert(size >= rank);
    }

    ~SelfAdjointConeigenSolver() = default;

    template <typename InputMatrix, typename InputVector>
    void compute(const Eigen::MatrixBase<InputMatrix>& matX,
                 const Eigen::MatrixBase<InputVector>& vecD,
                 DecompositionOption option = ComputeConeigenvectors);

    const MatrixType& coneigenvectors() const
    {
        return m_ceigvecs;
    }

    const RealVectorType& coneigenvalues() const
    {
        return m_ceigvals;
    }

protected:
    void resize(Index m, Index n)
    {
        m_ceigvecs.resize(m, n);
        m_ceigvals.resize(n);
        m_mat_work1.resize(n, n);
        m_mat_work2.resize(n, n);
    }
};

template <typename T>
template <typename InputMatrix, typename InputVector>
void SelfAdjointConeigenSolver<T>::compute(
    const Eigen::MatrixBase<InputMatrix>& matX,
    const Eigen::MatrixBase<InputVector>& vecD, DecompositionOption option)
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(InputVector);

    assert(matX.rows() >= matX.cols());
    assert(vecD.size() == matX.cols());

    const Index m = matX.rows();
    const Index n = matX.cols();
    resize(m, n);

    MappedMatrix matG(m_ceigvecs.data(), n, n);

    //
    // Form G = D * (X.st() * X) * D
    //
    matG.noalias() = matX.transpose() * matX;
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < n; ++i)
        {
            matG(i, j) *= vecD(i) * vecD(j);
        }
    }
    //
    // Compute G = Q * R by Householder QR factorization. G is overwritten by QR
    // factors.
    //
    Eigen::HouseholderQR<Eigen::Ref<MatrixType>> qr(matG);

    //
    // Compute SVD of `R = U S V^H` with high relative accuracy using the
    // one-sided Jacobi SVD algorithm. Only the singular values and left
    // singular vectors U are computed. The obtained singular values coincide
    // with coneigenvalues of input matrix
    //
    // Applying the one-sided Jacobi SVD to the matrix X = R^H is much faster
    // than applying the algorithm to R directly. This is because `R R^H` is
    // more diagonal than `R^H R`. Thus, `U` is computed as right singular
    // vectors of R^H.
    //
    MatrixType& matRt = m_mat_work1;
    matRt.setZero();
    matRt            = matG.template triangularView<Eigen::Upper>().adjoint();
    MatrixType& matU = m_mat_work2;
    const RealScalar tol_svd = Eigen::NumTraits<RealScalar>::epsilon() *
                               Eigen::numext::sqrt(RealScalar(n));
    one_sided_jacobi_svd(matRt, m_ceigvals, matU, tol_svd);

    //
    // Quick return if no con-eigenvector is required.
    //
    if (option == ConeigenvaluesOnly)
    {
        return;
    }

    //-------------------------------------------------------------------------
    //
    // The eigenvectors of A * conj(A) are given as
    //
    //   conj(X') = X * D * V * S^{-1/2}.
    //
    // However, direct evaluation of eigenvectors with this formula might be
    // inaccurate since D is ill-conditioned. The following formula are used
    // instead.
    //
    //   conj(X') = X * D * R.inv() * U * S^{1/2}
    //            = X * (D.inv() * R * D.inv()).inv() * (D.inv() * U *
    //            S^{1/2})
    //            = X * R1.inv() * X1
    //
    //-------------------------------------------------------------------------
    // Matrix R is stored on upper triangular part of G
    auto matR1 = matG.template triangularView<Eigen::Upper>();
    //
    // Compute R1 = D^(-1) * R * D^(-1). The upper triangular part of `matG` is
    // overwritten by R1.
    //
    for (Index j = 0; j < n; ++j)
    {
        const auto dj = vecD(j);
        for (Index i = 0; i <= j; ++i)
        {
            const auto di = vecD(i);
            matR1(i, j)   = matR1(i, j) / (di * dj);
        }
    }
    //
    // Compute X1 = D^{-1} * U * S^{1/2}, and then solve R1 * Y1 = X1 in-place.
    // Matrix `matU` is overwritten by X1 and then overwritten by Y1.
    //
    MatrixType& matY1 = matU;
    for (Index j = 0; j < n; ++j)
    {
        const auto sj = Eigen::numext::sqrt(m_ceigvals(j));
        for (Index i = 0; i < n; ++i)
        {
            matY1(i, j) *= sj / vecD(i);
        }
    }

    matR1.solveInPlace(matY1);
    //
    // Compute con-eigenvectors U = conj(X) * conj(Y).
    //
    m_ceigvecs.noalias() = matX.conjugate() * matY1.conjugate();

    if (IsComplex)
    {
        //
        // Adjust phase factor of each con-eigenvectors, so that U^{T} * U = I
        //
        for (Index j = 0; j < n; ++j)
        {
            auto xj          = m_ceigvecs.col(j);
            const auto t     = (xj.transpose() * xj).value();
            const auto phase = t / std::abs(t);
            const auto scale = std::sqrt(Eigen::numext::conj(phase));
            xj *= scale;
        }
    }
}

} // namespace: mxpfit

#endif /* MXPFIT_SELF_ADJOINT_CONEIGENSOLVER_HPP */

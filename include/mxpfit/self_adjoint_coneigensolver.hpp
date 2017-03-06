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
/// Compute con-eigenvalue decomposition of self-adjoint matrix having a
/// rank-revealing decomposition \f$ A = X D^2 X^{\ast} \f$
///
///
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
    RealVectorType m_vec_work;
    MatrixType m_mat_work1; // n x n
    MatrixType m_mat_work2; // n x n

public:
    SelfAdjointConeigenSolver()                                 = default;
    SelfAdjointConeigenSolver(const SelfAdjointConeigenSolver&) = default;

    explicit SelfAdjointConeigenSolver(Index size, Index rank)
        : m_ceigvecs(size, rank),
          m_ceigvals(rank),
          m_vec_work(rank),
          m_mat_work1(rank, rank),
          m_mat_work2(rank, rank)
    {
        assert(size >= rank);
    }

    ~SelfAdjointConeigenSolver() = default;

    template <typename InputMatrix, typename InputVector>
    void compute(const Eigen::MatrixBase<InputMatrix>& matX,
                 const Eigen::MatrixBase<InputVector>& vecD);

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
        m_vec_work.resize(n);
        m_mat_work1.resize(n, n);
        m_mat_work2.resize(n, n);
    }
};

template <typename T>
template <typename InputMatrix, typename InputVector>
void SelfAdjointConeigenSolver<T>::compute(
    const Eigen::MatrixBase<InputMatrix>& matX,
    const Eigen::MatrixBase<InputVector>& vecD)
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(InputVector);

    assert(matX.rows() >= matX.cols());
    assert(vecD.size() == matX.cols());

    const Index m = matX.rows();
    const Index n = matX.cols();
    resize(m, n);

    // Aliases
    // MatrixType& matG = m_mat_work1;              // D * X^T * X * D
    // MappedMatrix matR1(m_ceigvecs.data(), n, n); // D^{-1} * R * D^{-1}
    // MatrixType& matRt     = m_mat_work2;         // R^{T}
    // RealVectorType& sigma = m_ceigvals;
    // MatrixType& matU      = m_mat_work1; // left singular vectors of R
    // MatrixType& matY1     = m_mat_work2; // D^{-1} * U * S^{1/2}

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
            matR1(i, j) = matR1(i, j) / (di * dj);
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

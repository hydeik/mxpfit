#ifndef MXPFIT_SELF_ADJOINT_CONEIGENSOLVER_HPP
#define MXPFIT_SELF_ADJOINT_CONEIGENSOLVER_HPP

#include <cassert>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

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
    MatrixType m_matG;         // n x n

public:
    SelfAdjointConeigenSolver()                                 = default;
    SelfAdjointConeigenSolver(const SelfAdjointConeigenSolver&) = default;

    explicit SelfAdjointConeigenSolver(Index size, Index rank)
        : m_ceigvecs(size, rank), m_ceigvals(rank), m_matG(rank, rank)
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

    const Index n = matX.cols();
    m_ceigvecs.resize(matX.rows(), n);
    m_ceigvals.resize(n);
    m_matG.resize(n, n);

    //
    // Keep diagonal of D^(-1) for the latter use
    //
    RealVectorType diaginv(vecD.cwiseInverse());
    //
    // Form G = D * (X.st() * X) * D
    //
    m_matG.noalias() = matX.transpose() * matX;

    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < n; ++i)
        {
            m_matG(i, j) *= vecD(i) * vecD(j);
        }
    }
    //
    // Compute G = Q * R by Householder QR factorization. G is overwritten by QR
    // factors.
    //
    Eigen::HouseholderQR<Eigen::Ref<MatrixType>> qr(m_matG);
    //
    // Compute SVD of `R = U S V^H` with high relative accuracy using the
    // one-sided Jacobi SVD algorithm.
    //
    // Applying the one-sided Jacobi SVD to the matrix X = R^H is much faster
    // than applying the algorithm to R directly. This is because `R R^H` is
    // more diagonal than `R^H R`.
    //
    // Note: `d` is overwritten by the singular values
    //
    auto matR = m_matG.template triangularView<Eigen::Upper>();
    // MappedMatrix matRt(m_ceigvecs.data(), n, n);
    MatrixType matRt(n, n);
    matRt.setZero();
    matRt.template triangularView<Eigen::Lower>() = matR.adjoint();
    MatrixType matU(n, n);
    RealVectorType sigma(n);
    const RealScalar tol_svd = Eigen::NumTraits<RealScalar>::epsilon() *
                               Eigen::numext::sqrt(RealScalar(n));
    one_sided_jacobi_svd(matRt, sigma, matU, tol_svd);

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
    //
    // Compute R1 = D^(-1) * R * D^(-1)
    //
    MatrixType matR1(n, n);
    matR1.setZero();
    for (Index j = 0; j < n; ++j)
    {
        const auto dj = diaginv(j);
        for (Index i = 0; i <= j; ++i)
        {
            const auto di = diaginv(i);
            matR1(i, j) = di * dj * matR(i, j);
        }
    }
    //
    // Compute X1 = D^(-1) * U * S^{1/2} [in-place]
    //
    auto& matY1     = m_matG;
    m_ceigvals      = sigma;
    sigma           = sigma.cwiseSqrt();
    matY1.noalias() = diaginv.asDiagonal() * matU * sigma.asDiagonal();

    //
    // Solve R1 * Y1 = X1 [in-place]
    //
    // `ptr_mat1` points the first element of R1.
    auto R1 = matR1.template triangularView<Eigen::Upper>();
    R1.solveInPlace(matY1);
    //
    // Compute con-eigenvectors U = conj(X) * conj(Y).
    //
    // First `nvec` columns of `X` are overwritten by con-eigenvectors.
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

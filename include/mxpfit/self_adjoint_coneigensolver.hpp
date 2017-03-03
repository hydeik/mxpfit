#ifndef MXPFIT_SELF_ADJOINT_CONEIGENSOLVER_HPP
#define MXPFIT_SELF_ADJOINT_CONEIGENSOLVER_HPP

#include <cassert>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>

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
    using PacketType = typename Eigen::internal::packet_traits<Scalar>::type;

    enum
    {
        IsComplex  = Eigen::NumTraits<Scalar>::IsComplex,
        PacketSize = Eigen::internal::packet_traits<Scalar>::size,
        Alignment  = Eigen::internal::unpacket_traits<PacketType>::alignment
    };

    MatrixType m_ceigvecs;     // m x n (m >= n)
    RealVectorType m_ceigvals; // n
    MatrixType m_matG;         // n x n
    MatrixType m_matY;         // n x n
    MatrixType m_temp;         // n x 2

public:
    template <typename InputMatrix, typename InputVector>
    SelfAdjointConeigenSolver&
    compute(const Eigen::EigenBase<InputMatrix>& matX,
            const Eigen::EigenBase<InputVector>& vecD)
    {
        m_ceigvecs = matX;
        m_ceigvecs = vecD;
        compute_inplace();
        return *this;
    }

protected:
    void compute_inplace();
};

template <typename T>
void SelfAdjointConeigenSolver<T>::compute_inplace()
{
    auto& matX    = m_ceigvecs;
    auto& diag    = m_ceigvals;
    const Index n = matX.cols();

    assert(matX.rows() >= matX.cols());
    assert(diag.size() == n);
    //
    // Keep diagonal of D^(-1) for the latter use
    //
    RealVectorType diaginv(diag.cwiseInverse());
    //
    // Form G = D * (X.st() * X) * D
    //
    MatrixType matG(matX.transpose() * matX);

    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < n; ++i)
        {
            matG(i, j) *= diag(i) * diag(j);
        }
    }
    //
    // Compute G = Q * R by Householder QR factorization. G is overwritten by QR
    // factors.
    //
    Eigen::HouseholderQR<Eigen::Ref<MatrixType>> qr(matG);
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
    MatrixType matRt(matG.template triangularView<Eigen::Upper>().adjoint());
    Eigen::JacobiSVD<MatrixType> svd(matRt, Eigen::ComputeFullV);

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
    //            = X * (D.inv() * R * D.inv()).inv() * (D.inv() * U * S^{1/2})
    //            = X * R1.inv() * X1
    //
    //-------------------------------------------------------------------------
    //
    // Compute R1 = D^(-1) * R * D^(-1)
    //
    matRt.noalias() = diaginv.asDiagonal() *
                      matG.template triangularView<Eigen::Upper>() *
                      diaginv.asDiagonal();
    //
    // Compute X1 = D^(-1) * U * S^{1/2} [in-place]
    //
    auto& matY1     = matG;
    diag.noalias()  = svd.singularValues().cwiseSqrt();
    matY1.noalias() = diaginv.asDiagonal() * svd.matrixV() * diag.asDiagonal();

    //
    // Solve R1 * Y1 = X1 [in-place]
    //
    // `ptr_mat1` points the first element of R1.
    auto matR1 = matRt.template triangularView<Eigen::Upper>();
    matR1.solveInPlace(matY1);
    //
    // Compute con-eigenvectors U = conj(X) * conj(Y).
    //
    // First `nvec` columns of `X` are overwritten by con-eigenvectors.
    //
    matX = matX.conjugate() * matY1.conjugate();

    if (IsComplex)
    {
        //
        // Adjust phase factor of each con-eigenvectors, so that U^{T} * U = I
        //
        for (Index j = 0; j < n; ++j)
        {
            auto xj          = matX.col(j);
            const auto t     = (xj.transpose() * xj).value();
            const auto phase = t / std::abs(t);
            const auto scale = std::sqrt(Eigen::numext::conj(phase));
            xj *= scale;
        }
    }

    m_ceigvals = svd.singularValues();
}

namespace detail
{
///
/// Compute con-eigenvalue decomposition of real-symmetric or complex-Hermitian
/// matrix with rank-revealing decomposition `A = X * D^2 * X.adjoint()`
///
//

template <typename T>
struct SelfAdjointConeigenSolverImpl
{
    using Index         = Eigen::Index;
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
    using ComplexScalar = std::complex<RealScalar>;

    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor | Eigen::AutoAlign>;
    using VectorType =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::AutoAlign>;
    using RealVectorType =
        Eigen::Matrix<RealScalar, Eigen::Dynamic, 1, Eigen::AutoAlign>;

    using PacketType = typename Eigen::internal::packet_traits<Scalar>::type;

    enum
    {
        IsComplex  = Eigen::NumTraits<Scalar>::IsComplex,
        PacketSize = Eigen::internal::packet_traits<Scalar>::size,
        Alignment  = Eigen::internal::unpacket_traits<PacketType>::alignment
    };

    /// @return optimal size for `work`
    static Index optimal_work_size(Index nrows, Index rank)
    {
        using Eigen::internal::first_multiple;
        const auto n1 = first_multiple(nrows * rank, Index(PacketSize));
        const auto n2 = first_multiple(rank * rank, Index(PacketSize));
        const auto n3 =
            std::max(first_multiple(2 * rank, Index(PacketSize)), Index(6));
        return std::max(n1, 2 * n2) + n2 + n3;
    }
    /// @return optimal size for `rwork`
    static Index optimal_rwork_size(Index /*nrows*/, Index rank)
    {
        return 2 * rank + (IsComplex ? std::max(Index(6), rank) : Index());
    }

    ///
    /// Compute con-eigenpairs of @f$ A = X D^2 X^H @f$
    ///
    /// work size:
    ///    for matrix G : n * n
    ///    for matrix Y : n * n
    ///    for workspace: 2 * n
    ///    ------------------------------
    ///    Total        : 2 * n * (n + 1)
    ///
    /// rwork size:
    ///    if `T` is real type   : n
    ///    if `T` is complex type: 3 * n
    ///
    /// @param[in,out]  X  (m x n) matrix
    ///
    /// @param[in,out]  d  real vector of length `n`
    ///
    /// @param[in] threshold  threshold for the con-eigenvalue (=singular value)
    /// of
    ///   the matrix. Only the con-eigenvalues greater than this threshold value
    ///   and
    ///   corresponding con-eigenvectors are computed.
    ///
    /// @param work  `Scalar` array of size
    ///   `>= optimal_work_size(X.rows(),X.cols())` for working space
    ///
    /// @param rwork  `RealScalar` array of size
    ///   `>= optimal_rwork_size(X.rows,X.cols())` for working space
    ///
    template <typename MatX, typename VecD>
    static Index run(Eigen::MatrixBase<MatX>& X, Eigen::MatrixBase<VecD>& d,
                     RealScalar threshold, Scalar* work, RealScalar* rwork);

    template <typename MatX, typename VecD>
    static Index run(Eigen::MatrixBase<MatX>& X, Eigen::MatrixBase<VecD>& d,
                     RealScalar threshold)
    {
        VectorType work(X.rows(), X.cols());
        RealVectorType rwork(X.rows(), X.cols());

        run(X, d, threshold, work.data(), rwork.data());
    }

    template <typename MatA, typename MatX, typename VecD>
    static RealScalar residual(const Eigen::MatrixBase<MatA>& A,
                               const Eigen::MatrixBase<MatX>& X,
                               const Eigen::MatrixBase<VecD>& d)
    {
        const auto norm_A = A.norm();
        const auto resid =
            (A - X.conjugate() * d.asDiagonal() * X.transpose()).norm();

        return norm_A > RealScalar() ? resid / norm_A : resid;
    }

private:
    //
    // Compute QR factorization of matrix G = Q * R.
    //
    //  G: (N, N) matrix
    //  work size >= N * N
    //
    static void qr_inplace(Index N, Scalar* G, Scalar* work)
    {
        auto n          = static_cast<lapack_int>(N);
        auto lwork      = n * n;
        Scalar* tau     = work + n * n;
        lapack_int info = lapack::geqrf_work<Scalar>(LAPACK_COL_MAJOR, n, n, G,
                                                     n, tau, work, lwork);
        if (info)
        {
            std::ostringstream msg;
            msg << "(SelfAdjointConeigenSolverImpl) xGEQRF failed with "
                   "info "
                << info;
            throw std::runtime_error(msg.str());
        }
    }
    //
    // Solve R * Y = X, where R is upper triangular matrix.
    //
    //   R: upper triangular (N, N) matrix
    //   X: (N, NRHS) matrix
    //
    static void tri_solve(Index N, Index NRHS, Scalar* R, Scalar* X)
    {
        char uplo       = 'U';
        char trans      = 'N';
        char diag       = 'N';
        auto n          = static_cast<lapack_int>(N);
        auto nrhs       = static_cast<lapack_int>(NRHS);
        lapack_int info = lapack::trtrs_work<Scalar>(
            LAPACK_COL_MAJOR, uplo, trans, diag, n, nrhs, R, n, X, n);
        if (info)
        {
            std::ostringstream msg;
            msg << "(SelfAdjointConeigenSolverImpl) xTRTRS failed with info "
                << info;
            throw std::runtime_error(msg.str());
        }
    }
    //
    // Compute singular values and corresponding left singular vectors of upper
    // triangular matrix R using one-sided Jacobi method.
    //
    // --- for real matrix
    //    work size >= 2 * N
    //    rwork  not referred
    static void jacobi_svd(Index N, Scalar* R, RealScalar* sigma, Scalar* V,
                           RealScalar* work, RealScalar* /*rwork*/)
    {
        char joba = 'L'; // Input matrix R is upper triangular matrix
        char jobu = 'N'; // Compute left singular vectors
        char jobv = 'V'; // Do not compute right singular vectors
        auto n    = static_cast<lapack_int>(N);
        auto mv   = lapack_int();

        auto lwork = 2 * n;

        lapack_int info = lapack::gesvj_work<RealScalar>(
            LAPACK_COL_MAJOR, joba, jobu, jobv, n, n, R, n, sigma, mv, V, n,
            work, lwork);

        if (info)
        {
            std::ostringstream msg;
            msg << "(SelfAdjointConeigenSolverImpl) [s/d]GESVJ failed with "
                   "info = "
                << info;
            throw std::logic_error(msg.str());
        }
    }
    // --- for complex matrix
    //    work size  >= 2 * N
    //    rwork size >= max(N, 6)
    static void jacobi_svd(Index N, ComplexScalar* R, RealScalar* sigma,
                           ComplexScalar* V, ComplexScalar* work,
                           RealScalar* rwork)
    {
        char joba = 'L'; // Input matrix R is upper triangular matrix
        char jobu = 'N'; // Compute left singular vectors
        char jobv = 'V'; // Do not compute right singular vectors
        auto n    = static_cast<lapack_int>(N);
        auto mv   = lapack_int();

        auto lwork  = 2 * n;
        auto lrwork = std::max(lapack_int(6), n);

        lapack_int info = lapack::gesvj_work<ComplexScalar>(
            LAPACK_COL_MAJOR, joba, jobu, jobv, n, n, R, n, sigma, mv, V, n,
            work, lwork, rwork, lrwork);

        if (info)
        {
            std::ostringstream msg;
            msg << "(SelfAdjointConeigenSolverImpl) [s/d]GESVJ failed with "
                   "info = "
                << info;
            throw std::logic_error(msg.str());
        }
    }
};

template <typename T>
template <typename MatX, typename VecD>
Eigen::Index SelfAdjointConeigenSolverImpl<T>::run(Eigen::MatrixBase<MatX>& X_,
                                                   Eigen::MatrixBase<VecD>& d_,
                                                   RealScalar threshold,
                                                   Scalar* work,
                                                   RealScalar* rwork)
{
    static_assert(std::is_same<typename MatX::Scalar, Scalar>::value,
                  "MatX::Scalar must be the same as Scalar");
    static_assert(std::is_same<typename VecD::Scalar, RealScalar>::value,
                  "VecD::Scalar must be the same as RealScalar");
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecD);

    using MappedMatrix     = Eigen::Map<MatrixType, Alignment>;
    using MappedRealVector = Eigen::Map<RealVectorType, Alignment>;

    auto& X       = X_.derived();
    auto& d       = d_.derived();
    const Index n = X.cols();

    assert(X.rows() >= X.cols());
    assert(d.size() == n);

    const auto mat_size =
        Eigen::internal::first_multiple(n * n, Index(PacketSize));
    Scalar* ptr_mat1  = work;
    Scalar* ptr_mat2  = ptr_mat1 + mat_size;
    Scalar* ptr_mat3  = ptr_mat2 + mat_size;
    Scalar* ptr_extra = ptr_mat3 + mat_size;
    //
    // Keep diagonal of D^(-1) for the latter use
    //
    MappedRealVector dinv(rwork, n);
    dinv.noalias() = d.cwiseInverse();
    //
    // Form G = D * (X.st() * X) * D
    //
    MappedMatrix G(ptr_mat2, n, n);
    G.noalias() = X.transpose() * X;
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i < n; ++i)
        {
            G(i, j) *= d(i) * d(j);
        }
    }
    //
    // Compute G = Q * R by Householder QR factorization. G is overwritten by QR
    // factors.
    //
    qr_inplace(n, G.data(), ptr_mat3);

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
    MappedMatrix U(ptr_mat1, n, n);
    MappedMatrix RT(ptr_mat3, n, n);
    RT.noalias()          = G.adjoint(); // G = QR
    RealScalar* sigma     = rwork + n;
    RealScalar* ptr_rwork = sigma + n;

    jacobi_svd(n, RT.data(), sigma, U.data(), ptr_extra, ptr_rwork);

    //
    // Truncation
    //
    // Determines the order of reduced system, @f$ k @f$ from the error bound
    // computed from the singular values of Hankel system
    //
    // @f[
    //   \|\Sigma-\hat{\Sigma}\| \leq 2 \sum_{i=k+1}^{n} \sigma_{i}
    // @f]
    //
    auto sum_sigma = RealScalar();
    Index nvec     = n;
    while (nvec)
    {
        sum_sigma += sigma[nvec - 1];
        if (2 * sum_sigma > threshold)
        {
            break;
        }
        --nvec;
    }

    if (nvec == Index(0))
    {
        return nvec;
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
    //            = X * (D.inv() * R * D.inv()).inv() * (D.inv() * U * S^{1/2})
    //            = X * R1.inv() * X1
    //
    //-------------------------------------------------------------------------
    //
    // Compute X1 = D^(-1) * U * S^{1/2} [in-place]
    //
    MappedMatrix Y1(ptr_mat1, n, nvec); // Y1 refers first `nvec` columns of U
    for (Index j = 0; j < nvec; ++j)
    {
        const auto sj = std::sqrt(sigma[j]);
        for (Index i = 0; i < n; ++i)
        {
            Y1(i, j) *= sj * dinv(i);
        }
    }
    //
    // Compute R1 = D^(-1) * R * D^(-1) [in-place]
    //
    auto& R1 = G; // upper triangular part of G holds R
    for (Index j = 0; j < n; ++j)
    {
        for (Index i = 0; i <= j; ++i)
        {
            R1(i, j) *= dinv(i) * dinv(j);
        }
    }
    //
    // Solve R1 * Y1 = X1 [in-place]
    //
    // `ptr_mat1` points the first element of R1.
    tri_solve(n, nvec, R1.data(), Y1.data());
    //
    // Compute con-eigenvectors U = conj(X) * conj(Y).
    //
    // First `nvec` columns of `X` are overwritten by con-eigenvectors.
    //
    MappedMatrix Xc(ptr_mat2, X.rows(), nvec);
    Xc.noalias()               = X.conjugate() * Y1.conjugate();
    X.leftCols(nvec).noalias() = Xc;
    // X.leftCols(nvec) = X.conjugate() * Y1.conjugate();

    if (IsComplex)
    {
        //
        // Adjust phase factor of each con-eigenvectors, so that U^{T} * U = I
        //
        for (Index j = 0; j < nvec; ++j)
        {
            auto xj          = X.col(j);
            const auto t     = (xj.transpose() * xj).value();
            const auto phase = t / std::abs(t);
            const auto scale = std::sqrt(Eigen::numext::conj(phase));
            xj *= scale;
        }
    }

    for (Index k = 0; k < nvec; ++k)
    {
        d(k) = sigma[k];
    }
    // Return the number of con-eigenpairs computed.
    return nvec;
}

} // namespace: detail
} // namespace: mxpfit

#endif /* MXPFIT_SELF_ADJOINT_CONEIGENSOLVER_HPP */

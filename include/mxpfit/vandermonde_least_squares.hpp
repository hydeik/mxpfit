///
/// \file vandermonde.hpp
///
#ifndef MXPFIT_VANDERMONDE_LEAST_SQUARES_HPP
#define MXPFIT_VANDERMONDE_LEAST_SQUARES_HPP

#include <cassert>

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>

#include <mxpfit/matrix_free_gemv.hpp>
#include <mxpfit/vandermonde_matrix.hpp>

namespace mxpfit
{

namespace detail
{

///
/// \internal
///
/// Compute Cholesky decomposition of the gramian matrix of column Vandermonde
/// matrix.
///
/// This function compute the LDL decomposition of the gramian matrix,
///
/// \f[ G = V^{\ast} V = L^{} D L^{\ast}, \f]
///
/// where \f$ V=[v_{j}^{i}]_{i=1,\dots,m}^{n=1,\dots,n}\f$ is the column
/// Vandermonde matrix, \f$ L \f$ is a lower unit triangular matrix, and \f$ D
/// \f$ is a diagonal matrix.
///
/// \param[in] V  An \f$ m \times n \f$ column Vandermonde matrix to be
///   decomposed.
/// \param[out] ldlt  An \ f$ n \times n \f$ matrix to store the result of
///   decomposition. On exit, the diagonal elements of `ldlt` are those of
///   matrix \f$ D, \f$ and strict lower triangular part contains the off
///   diagonal elements of factor \f$ L \f$.
///
/// \param[out] work  An \f$ n \times 4 \f$ matrix used for workspace
///
template <typename T, typename MatrixT, typename MatrixWork>
void cholesky_vandermonde_gramian(const VandermondeMatrix<T>& V, MatrixT& ldlt,
                                  MatrixWork& work)
{
    using Scalar       = typename VandermondeMatrix<T>::Scalar;
    using CoeffsVector = typename VandermondeMatrix<T>::CoeffsVector;
    using RealScalar   = typename Eigen::NumTraits<Scalar>::Real;
    using Index        = Eigen::Index;

    using Eigen::numext::abs;
    using Eigen::numext::abs2;
    using Eigen::numext::conj;
    using Eigen::numext::real;
    using Eigen::numext::sqrt;

    static const auto tiny   = sqrt(std::numeric_limits<RealScalar>::min());
    constexpr const auto one = Scalar(1);

    auto z        = V.coeffs();
    const Index m = V.rows();
    const Index n = V.cols();

    assert(ldlt.rows() == n && ldlt.cols() == n);
    assert(work.rows() == n && work.cols() >= 4);

    // ----- Initialization
    auto y1 = work.col(0);
    auto y2 = work.col(1);
    auto x1 = work.col(2);
    auto x2 = work.col(3);

    auto gramian = [&](Index i, Index j) {
        const auto arg = conj(z(i)) * z(j);
        return arg == one ? Scalar(m) : (one - std::pow(arg, m)) / (one - arg);
    };

    auto sigma2 = gramian(0, 0);
    auto b0     = ldlt.col(0);
    b0(0)       = one;
    for (Index j = 1; j < n; ++j)
    {
        b0(j) = gramian(j, 0) / sigma2;
    }

    y1 = CoeffsVector::Ones(n) - b0;
    y2 = z.array().conjugate().pow(m);
    y2 -= y2(0) * b0;
    x1 = z.array().conjugate().inverse();
    x1 -= x1(0) * b0;
    x2 = -z.array().conjugate().pow(m - 1);
    x2 -= x2(0) * b0;

    b0(0) = sigma2;

    for (Index k = 1; k < n; ++k)
    {
        auto bk = ldlt.col(k);

        auto mu1 = x1(k);
        auto mu2 = x2(k);
        auto nu1 = conj(y1(k));
        auto nu2 = conj(y2(k));

        auto zk_inv = one / z(k);
        auto denom  = conj(zk_inv) - z(k);

        if (abs(denom) < tiny)
        {
            sigma2         = RealScalar(m);
            const auto bkk = real(ldlt(k, k));
            for (Index j = 0; j < k; ++j)
            {
                const auto bkj = ldlt(k, j);
                sigma2 -= abs2(bkj) * bkk;
            }
        }
        else
        {
            sigma2 = (mu1 * nu1 + mu2 * nu2) / denom;
        }

        bk(k) = sigma2;

        Index nt    = n - k - 1;
        bk.tail(nt) = (conj(mu1) / sigma2 * y1.tail(nt) +
                       conj(mu2) / sigma2 * y2.tail(nt));
        bk.array().tail(nt) /=
            (CoeffsVector::Constant(nt, zk_inv) - z.tail(nt).conjugate())
                .array();

        x1.tail(nt) -= mu1 * bk.tail(nt);
        x2.tail(nt) -= mu2 * bk.tail(nt);
        y1.tail(nt) -= conj(nu1) * bk.tail(nt);
        y2.tail(nt) -= conj(nu2) * bk.tail(nt);
    }

    return;
}

} // namespace detail

///
/// Preconditioner specialized for of Vandermonde matrix
///
template <typename T>
class VandermondePreconditioner
{
public:
    using Scalar     = T;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    using StorageIndex = typename Matrix::StorageIndex;
    using Index        = Eigen::Index;

    using VandermondeGEMV = MatrixFreeGEMV<VandermondeMatrix<Scalar>>;
    enum
    {
        ColsAtCompileTime    = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
    };

    VandermondePreconditioner() = default;

    ~VandermondePreconditioner() = default;

    Index rows() const
    {
        return m_ldlt.rows();
    }

    Index cols() const
    {
        return m_ldlt.cols();
    }

    VandermondePreconditioner& analyzePatter(const VandermondeGEMV&)
    {
        return *this;
    }

    VandermondePreconditioner& factorize(const VandermondeGEMV& mat)
    {
        const Index n = mat.cols();

        m_ldlt.resize(n, n);
        m_invdiag.resize(n);
        Matrix work(n, 4);
        //
        // Compute Cholesky decomposition of the Gramian matrix of the form
        // \f$ V^{\ast} V = L D L^{\ast} \f$
        //
        detail::cholesky_vandermonde_gramian(mat.nestedExpression(), m_ldlt,
                                             work);

        for (Index i = 0; i < n; ++i)
        {
            if (m_ldlt(i, i) == Scalar())
            {
                m_invdiag(i) = RealScalar(1);
            }
            else
            {
                m_invdiag(i) = RealScalar(1) / m_ldlt(i, i);
            }
        }
        m_is_initialized = true;
        return *this;
    }

    VandermondePreconditioner& compute(const VandermondeGEMV& mat)
    {
        return factorize(mat);
    }

    template <typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
        auto matL = m_ldlt.template triangularView<Eigen::UnitLower>();
        x         = matL.solve(b);
        x.array() *= m_invdiag.array();
        matL.adjoint().solveInPlace(x);
    }

    template <typename Rhs>
    inline const Eigen::Solve<VandermondePreconditioner, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const
    {
        eigen_assert(m_is_initialized &&
                     "VandermondePreconditioner is not initialized.");
        eigen_assert(m_ldlt.cols() == b.rows() &&
                     "VandermondePreconditioner::solve(): invalid "
                     "number of rows of the right hand side matrix b");
        return Eigen::Solve<VandermondePreconditioner, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info()
    {
        return Eigen::Success;
    }

private:
    Matrix m_ldlt;
    Vector m_invdiag;
    bool m_is_initialized;
};

template <typename T>
using VandermondeLeastSquaresSolver =
    Eigen::LeastSquaresConjugateGradient<MatrixFreeGEMV<VandermondeMatrix<T>>,
                                         VandermondePreconditioner<T>>;

} // namespace mxpfit

#endif /* MXPFIT_VANDERMONDE_LEAST_SQUARES_HPP */

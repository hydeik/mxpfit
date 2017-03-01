///
/// \file vandermonde.hpp
///
#ifndef MXPFIT_VANDERMONDE_HPP
#define MXPFIT_VANDERMONDE_HPP

#include <cassert>

#include <Eigen/Core>

#include <mxpfit/lsqr.hpp>
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
void cholesky_vandermonde_gramian(const VandermondeGEMV<T>& V, MatrixT& ldlt,
                                  MatrixWork& work)
{
    using Scalar       = typename VandermondeGEMV<T>::Scalar;
    using CoeffsVector = typename VandermondeGEMV<T>::CoeffsVector;
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

    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    using StorageIndex = typename MatrixType::StorageIndex;
    using Index        = Eigen::Index;
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

    template <typename MatT>
    VandermondePreconditioner& analyzePatter(const MatT&)
    {
        return *this;
    }

    template <typename MatT>
    VandermondePreconditioner& factorize(const MatT& mat)
    {
        const Index n = mat.cols();

        m_ldlt.resize(n, n);
        m_invdiag.resize(n);
        MatrixType work(n, 4);
        //
        // Compute Cholesky decomposition of the Gramian matrix of the form
        // \f$ V^{\ast} V = L D L^{\ast} \f$
        //
        detail::cholesky_vandermonde_gramian(mat, m_ldlt, work);

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

    template <typename MatT>
    VandermondePreconditioner& compute(const MatT& mat)
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
    MatrixType m_ldlt;
    VectorType m_invdiag;
    bool m_is_initialized;
};
// ///
// /// Preconditioner specialized for of Vandermonde matrix
// ///
// /// \tparam MatrixT  type of matrix
// ///
// template <typename MatrixT, typename VectorT>
// struct VandermondePreconditioner
// {
//     using Matrix       = MatrixT;
//     using Vector       = VectorT;
//     using StorageIndex = typename Matrix::StorageIndex;
//     using Index        = StorageIndex;
//     using Scalar       = typename Matrix::Scalar;
//     using RealScalar   = typename Matrix::RealScalar;

//     enum
//     {
//         ColsAtCompileTime    = Eigen::Dynamic,
//         MaxColsAtCompileTime = Eigen::Dynamic,
//     };

//     VandermondePreconditioner(const Matrix& mat_ldlt,
//                               const VectorT& vec_diaginv)
//         : ldlt(mat_ldlt), diaginv(vec_diaginv)
//     {
//     }

//     Index rows() const
//     {
//         return ldlt.rows();
//     }

//     Index cols() const
//     {
//         return ldlt.cols();
//     }

//     template <typename Rhs, typename Dest>
//     void _solve_impl(const Rhs& b, Dest& x) const
//     {
//         auto matL = ldlt.template triangularView<Eigen::UnitLower>();
//         x         = matL.solve(b);
//         x.array() *= diaginv.array();
//         matL.adjoint().solveInPlace(x);
//     }

//     template <typename Rhs>
//     inline const Eigen::Solve<VandermondePreconditioner, Rhs>
//     solve(const Eigen::MatrixBase<Rhs>& b) const
//     {
//         using ResultType = Eigen::Solve<VandermondePreconditioner, Rhs>;
//         assert(ldlt.cols() == b.rows());
//         return ResultType(*this, b.derived());
//     }

//     const Matrix& ldlt;
//     const Vector& diaginv;
// };

// ///
// /// Helper function for generating VandermondePreconditioner
// ///
// template <typename MatrixT, typename VectorT>
// inline VandermondePreconditioner<MatrixT, VectorT>
// makeVandermodePreconditioner(const MatrixT& ldlt, const VectorT& diaginv)
// {
//     return VandermondePreconditioner<MatrixT, VectorT>(ldlt, diaginv);
// }

///
/// Solver for Vandermonde least squares system
///
template <typename T>
class VandermondeLeastSquaresSolver
{
public:
    using Index        = Eigen::Index;
    using Scalar       = T;
    using RealScalar   = typename Eigen::NumTraits<Scalar>::Real;
    using PacketScalar = typename Eigen::internal::packet_traits<Scalar>::type;

    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                 (Eigen::ColMajor | Eigen::AutoAlign)>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1,
                                 (Eigen::ColMajor | Eigen::AutoAlign)>;

    using StorageIndex = typename Matrix::StorageIndex;

    using MappedMatrix =
        Eigen::Map<Matrix, Eigen::internal::traits<Matrix>::Alignment>;
    using MappedVector =
        Eigen::Map<Vector, Eigen::internal::traits<Vector>::Alignment>;

    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
        Alignment = Eigen::internal::unpacket_traits<PacketScalar>::alignment,
        SizePerPacket        = Eigen::internal::packet_traits<T>::size,
        ColsAtCompileTime    = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
    };

    /// Default constructor
    VandermondeLeastSquaresSolver()
        : m_max_iterations(-1),
          m_iterations(),
          m_tolerance(Eigen::NumTraits<Scalar>::epsilon()),
          m_error(),
          m_matV(nullptr),
          m_ldlt(),
          m_diaginv(),
          m_work()
    {
    }

    ///
    /// Create a solver with memory pre-allocation
    ///
    VandermondeLeastSquaresSolver(Index nrows, Index ncols)
        : m_max_iterations(-1),
          m_iterations(),
          m_tolerance(Eigen::NumTraits<Scalar>::epsilon()),
          m_error(),
          m_matV(nullptr),
          m_ldlt(ncols, ncols),
          m_diaginv(ncols),
          m_work(workspace_size(nrows, ncols))
    {
    }

    /// Default destructor
    ~VandermondeLeastSquaresSolver() = default;

    ///
    /// Perform matrix factorization required for solving the Vandermonde least
    /// squares problem.
    ///
    /// This function is part of Eigen sparse solver concept. Inside the
    /// function, \f$ LDL^T \f$ factorization of the Gramian matrix formed by
    /// the given Vandermonde matrix is performed, which is used to form an
    /// appropriate pre-conditioner of matrix V.
    ///
    /// \param[in] matV  A Vandermonde matrix
    ///
    void compute(const VandermondeGEMV<T>& matV)
    {
        const Index m = matV.rows();
        const Index n = matV.cols();
        m_matV        = &matV;

        resize(m, n);
        //
        // LDL^T decomposition of `V.adjoint() * V`.
        //
        Scalar* ptr_H = m_ldlt.data();
        Scalar* ptr_w = m_work.data();
        MappedMatrix matH(ptr_H, n, n);
        MappedMatrix vecs(ptr_w, n, 4);
        detail::cholesky_vandermonde_gramian(matV, matH, vecs);

        for (Index i = 0; i < n; ++i)
        {
            if (matH(i, i) == Scalar())
            {
                m_diaginv(i) = RealScalar(1);
            }
            else
            {
                m_diaginv(i) = RealScalar(1) / matH(i, i);
            }
        }
    }
    /// \return  the number of rows of Vandermonde matrix
    Index rows() const
    {
        return m_matV->rows();
    }

    /// \return  the number of columns of Vandermonde matrix
    Index cols() const
    {
        return m_matV->cols();
    }

    ///
    /// Computes the solution of the least squares problem for a given
    /// right-hand-side vector
    ///
    template <typename Rhs>
    inline const Eigen::Solve<VandermondeLeastSquaresSolver, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const
    {
        using ResultType = Eigen::Solve<VandermondeLeastSquaresSolver, Rhs>;
        assert(rows() == b.rows());
        return ResultType(*this, b.derived());
    }

    ///
    /// Solve `V * x = b` using preconditioned LSQR
    ///
    /// This function is called from `Eigen::Solve` class
    ///
    template <typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
        using MatVec = MatrixFreeGEMV<VandermondeGEMV<T>>;

        const Index m = m_matV->rows();
        const Index n = m_matV->cols();

        Scalar* ptr_H       = const_cast<Scalar*>(m_ldlt.data());
        Scalar* ptr_diaginv = const_cast<Scalar*>(m_diaginv.data());
        Scalar* ptr_u       = m_work.data();
        Scalar* ptr_v       = ptr_u + m;
        ptr_v = ptr_v + Eigen::internal::first_default_aligned(ptr_v, n * 3);

        assert(static_cast<Index>(ptr_v - ptr_u) <= m_work.size());

        MatVec opV(*const_cast<VandermondeGEMV<T>*>(m_matV));
        MappedMatrix matH(ptr_H, n, n);
        MappedVector diaginv(ptr_diaginv, n);
        MappedVector u(ptr_u, m);
        MappedMatrix tmp(ptr_v, n, 3);

        auto precond = makeVandermodePreconditioner(matH, diaginv);
        m_iterations = m_max_iterations < Index() ? 2 * n : m_max_iterations;
        m_error      = m_tolerance;
        u            = b.template cast<Scalar>();
        lsqr(opV, u, x, precond, tmp, m_iterations, m_error);
    }

    /// Check the convergence
    bool converged() const
    {
        return m_iterations <= m_max_iterations;
    }

    RealScalar tolerance() const
    {
        return m_tolerance;
    }

    void setTolerance(RealScalar tol)
    {
        m_tolerance = tol;
    }

    Index iterations() const
    {
        return m_iterations;
    }

    void setMaxIterations(Index max_iter)
    {
        m_max_iterations = max_iter;
    }

    /// \return An estimation of residual error
    RealScalar error() const
    {
        return m_error;
    }

    ///
    /// Resize work space
    ///
    void resize(Index nrows, Index ncols)
    {
        if (m_ldlt.rows() < ncols)
        {
            m_ldlt.resize(ncols, ncols);
            m_diaginv.resize(ncols);
        }

        const Index m_worksize = workspace_size(nrows, ncols);
        if (m_work.size() < m_worksize)
        {
            m_work.resize(m_worksize);
        }
    }

private:
    constexpr static Index workspace_size(Index nrows, Index ncols)
    {
        return ncols * 3 + std::max(ncols, nrows) + SizePerPacket;
    }

    Index m_max_iterations;
    mutable Index m_iterations;
    RealScalar m_tolerance;
    mutable RealScalar m_error;

    const VandermondeGEMV<T>* m_matV;
    Matrix m_ldlt;
    Vector m_diaginv;
    mutable Vector m_work;
};

} // namespace mxpfit

#endif /* MXPFIT_VANDERMONDE_HPP */

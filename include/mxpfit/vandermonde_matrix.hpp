///
/// \file vandermonde_matrix.hpp
///
#ifndef MXPFIT_VANDERMONDE_MATRIX_HPP
#define MXPFIT_VANDERMONDE_MATRIX_HPP

#include <cassert>

#include <Eigen/Core>

#include <mxpfit/lsqr.hpp>
#include <mxpfit/matrix_free_gemv.hpp>

namespace mxpfit
{

///
/// ### VandermondeWrapper
///
/// Expression of a rectangular column Vandermonde matrix.
///
/// \tparam T  Scalar type of matrix elements
///
/// This class represents a \f$ m \times n \f$ column Vandermonde matrix of
/// the form
///
/// \f[
///   \bm{V}(\bm{t}) = \left[ \begin{array}{cccc}
///     1         & 1         & \dots  & 1         \\
///     t_{1}^{}  & t_{2}^{}  & \dots  & t_{n}^{}  \\
///     \vdots    & \vdots    & \ddots & \vdots    \\
///     t_{1}^{m} & t_{2}^{m} & \dots  & t_{n}^{m} \\
///   \end{array} \right].
/// \f]
///
/// From the definition, a Vandermonde matrix \f$V\f$ is fully determined by the
/// coefficients of the second row. This class wraps the existing vector
/// expression of the coefficients of the first row.
///

template <typename T>
class VandermondeGEMV
{
public:
    using Scalar          = T;
    using RealScalar      = typename Eigen::NumTraits<T>::Real;
    using StorageIndex    = Eigen::Index;
    using Index           = Eigen::Index;
    using CoeffsVector    = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using CoeffsVectorRef = Eigen::Ref<const CoeffsVector>;
    using PlainObject = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

private:
    Index m_rows;
    mutable CoeffsVector m_workspace;
    CoeffsVectorRef m_coeffs;

public:
    VandermondeGEMV()
        : m_rows(),
          m_workspace(),
          m_coeffs(m_workspace) // need to initialize Eigen::Ref object
    {
    }

    ~VandermondeGEMV()
    {
    }

    VandermondeGEMV& operator=(const VandermondeGEMV&) = default;

    ///
    /// Create a Vandermonde matrix from number of rows and coefficients of the
    /// first row.
    ///
    template <typename InputType>
    VandermondeGEMV(Index nrows, const InputType& coeffs)
        : m_rows(nrows), m_workspace(coeffs.size()), m_coeffs(coeffs)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(InputType);
    }

    /// \return the number of rows
    Index rows() const
    {
        return m_rows;
    }

    /// \return the number of columns
    Index cols() const
    {
        return m_coeffs.size();
    }

    /// Resize matrix
    void resize(Index nrows, Index ncols)
    {
        m_rows = nrows;
        m_coeffs.resize(ncols);
        m_workspace.resize(ncols);
    }

    /// Set elements of Vandermonde matrix
    template <typename Derived>
    void setCoeffs(const Eigen::EigenBase<Derived>& coeffs)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        m_coeffs.~CoeffsVectorRef();
        ::new (&m_coeffs) CoeffsVectorRef(coeffs);
    }

    void setCoeffs(const CoeffsVector& coeffs)
    {
        if (&(coeffs.derived()) != &m_coeffs)
        {
            m_coeffs.~CoeffsVectorRef();
            ::new (&m_coeffs) CoeffsVectorRef(coeffs);
        }
    }

    const CoeffsVectorRef& coeffs() const
    {
        return m_coeffs;
    }

    ///
    /// \return The same Vandermonde matrix in dense form.
    ///
    PlainObject toDenseMatrix() const
    {
        PlainObject ret(rows(), cols());
        for (Index j = 0; j < cols(); ++j)
        {
            auto x = m_coeffs(j);
            auto v = Scalar(1);
            ret(0, j) = v;
            for (Index i = 1; i < rows(); ++i)
            {
                v *= x;
                ret(i, j) = v;
            }
        }
        return ret;
    }

    /// Compute matrix-vector product of the form `dst += alpha * A * rhs`
    template <typename Dest, typename RHS>
    void apply(Dest& dst, const Eigen::MatrixBase<RHS>& rhs, Scalar alpha) const
    {
        apply_impl(dst, rhs, alpha, coeffs());
    }

    /// Compute matrix-vector product of the form
    /// `dst += alpha * A.conjugate() * rhs`
    template <typename Dest, typename RHS>
    void applyConjugate(Dest& dst, const Eigen::MatrixBase<RHS>& rhs,
                        Scalar alpha) const
    {
        apply_impl(dst, rhs, alpha, coeffs().conjugate());
    }

    /// Compute matrix-vector product of the form
    /// `dst += alpha * A.transpose() * rhs`
    template <typename Dest, typename RHS>
    void applyTranspose(Dest& dst, const Eigen::MatrixBase<RHS>& rhs,
                        Scalar alpha) const
    {
        apply_transpose_impl(dst, rhs, alpha, coeffs());
    }

    /// Compute matrix-vector product of the form
    /// `dst += alpha * A.adjoint() * rhs`
    template <typename Dest, typename RHS>
    void applyAdjoint(Dest& dst, const Eigen::MatrixBase<RHS>& rhs,
                      Scalar alpha) const
    {
        apply_transpose_impl(dst, rhs, alpha, coeffs().conjugate());
    }

private:
    template <typename Dest, typename RHS, typename CoeffsV>
    void apply_impl(Dest& dst, const Eigen::MatrixBase<RHS>& rhs, Scalar alpha,
                    const Eigen::MatrixBase<CoeffsV>& coeffs) const
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Dest);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(RHS);

        assert(dst.size() == rows());
        assert(rhs.size() == cols());

        if (alpha == Scalar(/*zero*/))
        {
            return;
        }

        auto w = m_workspace.head(m_cols);
        w      = rhs.derived();

        dst(0) += alpha * w.sum();
        for (Index i = 1; i < rows(); ++i)
        {
            w.array() *= coeffs.array();
            dst(i) += alpha * w.sum();
        }
    }

    template <typename Dest, typename RHS, typename CoeffsV>
    void apply_transpose_impl(Dest& dst, const Eigen::MatrixBase<RHS>& rhs,
                              Scalar alpha,
                              const Eigen::MatrixBase<CoeffsV>& coeffs) const
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Dest);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(RHS);

        assert(dst.size() == cols());
        assert(rhs.size() == rows());

        for (Index i = 0; i < cols(); ++i)
        {
            auto x = coeffs(i);
            // Evaluate polynomial using Honer's method
            auto s = Scalar();
            for (Index j = 0; j < rows(); ++j)
            {
                s = s * x + rhs(rows() - j - 1);
            }

            dst(i) += alpha * s;
        }
    }
};

} // namespace mxpfit

#endif /* MXPFIT_VANDERMONDE_MATRIX_HPP */

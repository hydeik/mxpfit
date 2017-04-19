///
/// \file vandermonde_matrix.hpp
///
#ifndef MXPFIT_VANDERMONDE_MATRIX_HPP
#define MXPFIT_VANDERMONDE_MATRIX_HPP

#include <cassert>

#include <Eigen/Core>

#include <mxpfit/matrix_free_gemv.hpp>

namespace mxpfit
{

///
/// ### VandermondeMatrix
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
/// This class represents a Vandermonde matrix expression from the given number
/// of rows, \f$ m, \f$ and a vector expression for the coefficients of the
/// second row, \f$(t_1,t_2,\dots,t_n).\f$ If the given vector expression is
/// l-value, this class wraps the existing vector expression, otherwise storage
/// for coefficients are allocated.
///
/// This class also provides the interface for matrix-vector multiplication
/// compatible to `MatrixFreeGEMV`. For this purpose, the class also allocate
/// internal a vector of size \f$n\f$ as working space.
///

template <typename T>
class VandermondeMatrix
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
    /// Default constructor: create an empty matrix
    VandermondeMatrix()
        : m_rows(),
          m_workspace(),
          m_coeffs(m_workspace) // need to initialize Eigen::Ref object
    {
    }

    ///
    /// Create a Vandermonde matrix from number of rows and coefficients of the
    /// first row.
    ///
    template <typename InputType>
    VandermondeMatrix(Index nrows, const InputType& coeffs)
        : m_rows(nrows), m_workspace(coeffs.size()), m_coeffs(coeffs)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(InputType);
    }

    /// Copy constructor
    VandermondeMatrix(const VandermondeMatrix& other)
        : m_rows(other.m_rows),
          m_workspace(other.m_workspace),
          m_coeffs(other.m_coeffs)
    {
    }

    /// Default destructor
    ~VandermondeMatrix()
    {
    }

    /// Delete assignment operator as a consequence that Eigen::Ref is
    /// non-assignable
    VandermondeMatrix& operator=(const VandermondeMatrix& other) = delete;

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

    /// Set elements of Vandermonde matrix
    template <typename Derived>
    void setMatrix(Index nrows, const Eigen::EigenBase<Derived>& coeffs)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        m_rows = nrows;
        if (m_workspace.size() != coeffs.size())
        {
            m_workspace.resize(coeffs.size());
        }
        m_coeffs.~CoeffsVectorRef();
        ::new (&m_coeffs) CoeffsVectorRef(coeffs.derived());
    }

    void setMatrix(Index nrows, const CoeffsVectorRef& coeffs)
    {
        m_rows = nrows;
        if (&(coeffs.derived()) != &m_coeffs)
        {
            if (m_workspace.size() != coeffs.size())
            {
                m_workspace.resize(coeffs.size());
            }
            m_coeffs.~CoeffsVectorRef();
            ::new (&m_coeffs) CoeffsVectorRef(coeffs);
        }
    }

    ///
    /// \return A const reference to the coefficients of the second row
    ///
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

        m_workspace = rhs.derived();

        dst(0) += alpha * m_workspace.sum();
        for (Index i = 1; i < rows(); ++i)
        {
            m_workspace.array() *= coeffs.array();
            dst(i) += alpha * m_workspace.sum();
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

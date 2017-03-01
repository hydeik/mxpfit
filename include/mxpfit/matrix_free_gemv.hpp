///
/// \file matrix_free_gemv.hpp
///
#ifndef MXPFIT_FITTING_MATRIX_FREE_GEMV_HPP
#define MXPFIT_FITTING_MATRIX_FREE_GEMV_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore> // for sparse product interface

namespace mxpfit
{
///
/// ### OperationType
///
/// Flags describing the form of matrix operator
///
///  | Variable     | Form of operator                       |
///  |:------------ |:-------------------------------------- |
///  | `NoTranspose`| \f$ A \f$                              |
///  | `Transposed` | transpose \f$ A^{T} \f$                |
///  | `Conjugate`  | complex conjugate \f$ \overline{A} \f$ |
///  | `Adjoint`    | Hermitian conjugate \f$ A^{\ast} \f$   |
///
enum OperationType : std::uint32_t
{
    NoTranspose = 0,
    Transposed  = 1,
    Conjugate   = 1 << 1u,
    Adjoint     = Transposed | Conjugate
};

template <typename MatrixType, std::uint32_t Trans = NoTranspose>
class MatrixFreeGEMV;

} // namespace mxpfit

namespace Eigen
{
namespace internal
{

///
/// Specialization of `Eigen::internal::traits` for `mxpfit::MatrixFreeGEMV`
///
template <typename MatrixType, std::uint32_t Trans>
struct traits<mxpfit::MatrixFreeGEMV<MatrixType, Trans>>
{
    using Scalar       = typename MatrixType::Scalar;
    using StorageIndex = Index;
    using StorageKind  = Sparse;
    using XprKind      = MatrixXpr;
    enum
    {
        RowsAtCompileTime    = Dynamic,
        ColsAtCompileTime    = Dynamic,
        MaxRowsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic,
        Flags                = 0u // unused(?)
    };
};

} // namespace internal
} // namespace Eigen

namespace mxpfit
{
///
/// Generic wrapper of matrix-vector product for Einge matrix-free solver
///
/// \tparam MatrixType  A type of matrix operator which have special member
///     functions with the following signature:
/// \tparam Trans  A parameter of type `mxpfit::OperationType` specifying
///     the operator form
///
/// ``` c++
///   /* Compute `dst += alpha * A * rhs` */
///   template <typename Dest, typename RHS>
///   void apply(Dest& dst, const Eigen::MatrixBase<RHS>& rhs, Scalar alpha);
///
///   /* Compute `dst += alpha * A.conjugate() * rhs */
///   template <typename Dest, typename RHS>
///   void applyConjugate(Dest& dst, const Eigen::MatrixBase<RHS>& rhs,
///                       Scalar alpha);
///
///   /* Compute `dst += alpha * A.transpose() * rhs` */
///   template <typename Dest, typename RHS>
///   void applyTranspose(Dest& dst, const Eigen::MatrixBase<RHS>& rhs,
///                       Scalar alpha);
///
///   /* Compute `dst += alpha * A.transpose() * rhs` */
///   template <typename Dest, typename RHS>
///   void applyAdjoint(Dest& dst, const Eigen::MatrixBase<RHS>& rhs,
///                     Scalar alpha);
/// ```
///
/// where `Dest` and `RHS` are the expressions for target and destination
/// vectors.
///
///
template <typename MatrixType, std::uint32_t Trans>
class MatrixFreeGEMV
    : public Eigen::EigenBase<MatrixFreeGEMV<MatrixType, Trans>>
{
public:
    using Scalar       = typename MatrixType::Scalar;
    using RealScalar   = typename MatrixType::RealScalar;
    using StorageIndex = Eigen::Index;
    using Index        = StorageIndex;

    using TransposeReturnType =
        MatrixFreeGEMV<MatrixType, (Trans ^ Transposed)>;

    using ConjugateReturnType = MatrixFreeGEMV<MatrixType, (Trans ^ Conjugate)>;

    using AdjointReturnType = MatrixFreeGEMV<MatrixType, (Trans ^ Adjoint)>;

    enum
    {
        ColsAtCompileTime    = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor           = false
    };

    MatrixFreeGEMV() : m_matrix(nullptr)
    {
    }

    MatrixFreeGEMV(const MatrixType& mat) : m_matrix(&mat)
    {
    }

    MatrixFreeGEMV(const MatrixFreeGEMV& other) : m_matrix(other.m_matrix)
    {
    }

    ~MatrixFreeGEMV()
    {
    }

    /// \return  The number of rows of matrix `A`
    Index rows() const
    {
        return (Trans & Transposed) ? m_matrix->cols() : m_matrix->rows();
    }

    /// \return  The number of columns of matrix `A`
    Index cols() const
    {
        return (Trans & Transposed) ? m_matrix->rows() : m_matrix->cols();
    }

    ///
    /// Overlaoded `operator*` with Eigen dense column vector
    ///
    template <typename Rhs>
    Eigen::Product<MatrixFreeGEMV, Rhs, Eigen::AliasFreeProduct>
    operator*(const Eigen::MatrixBase<Rhs>& x) const
    {
        using ResultType =
            Eigen::Product<MatrixFreeGEMV, Rhs, Eigen::AliasFreeProduct>;
        return ResultType(*this, x.derived());
    }
    ///
    /// Set new matrix-like operator
    ///
    void attachMatrix(MatrixType& mat)
    {
        m_matrix = &mat;
    }
    ///
    /// Get the reference to the attached matrix. The return type is a reference
    /// to `MatrixType`, rather than const reference, as your matrix operator
    /// might have internal state modified on multiplying to vectors.
    ///
    const MatrixType& nestedExpression() const
    {
        return *m_matrix;
    }
    ///
    /// Retrun expression to the transposed matrix operator.
    ///
    TransposeReturnType transpose() const
    {
        return TransposeReturnType(*m_matrix);
    }
    ///
    /// Retrun expression to the complex conjugate of matrix operator.
    ///
    ConjugateReturnType conjugate() const
    {
        return ConjugateReturnType(*m_matrix);
    }
    ///
    /// Retrun expression to the adjoint of matrix operator.
    ///
    AdjointReturnType adjoint() const
    {
        return AdjointReturnType(*m_matrix);
    }

private:
    const MatrixType* m_matrix;
};

} // namespace mxpfit

namespace Eigen
{
namespace internal
{

template <typename MatrixType, typename Rhs>
struct generic_product_impl<
    mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::NoTranspose>, Rhs, SparseShape,
    DenseShape, GemvProduct>
    : generic_product_impl_base<
          mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::NoTranspose>, Rhs,
          generic_product_impl<
              mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::NoTranspose>, Rhs>>
{
    using Lhs    = mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::NoTranspose>;
    using Scalar = typename Product<Lhs, Rhs>::Scalar;
    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs,
                              const Scalar& alpha)
    {
        lhs.nestedExpression().apply(dst, rhs, alpha);
    }
};

template <typename MatrixType, typename Rhs>
struct generic_product_impl<
    mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Transposed>, Rhs, SparseShape,
    DenseShape, GemvProduct>
    : generic_product_impl_base<
          mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Transposed>, Rhs,
          generic_product_impl<
              mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Transposed>, Rhs>>
{
    using Lhs    = mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Transposed>;
    using Scalar = typename Product<Lhs, Rhs>::Scalar;
    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs,
                              const Scalar& alpha)
    {
        lhs.nestedExpression().applyTranspose(dst, rhs, alpha);
    }
};

template <typename MatrixType, typename Rhs>
struct generic_product_impl<
    mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Conjugate>, Rhs, SparseShape,
    DenseShape, GemvProduct>
    : generic_product_impl_base<
          mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Conjugate>, Rhs,
          generic_product_impl<
              mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Conjugate>, Rhs>>
{
    using Lhs    = mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Conjugate>;
    using Scalar = typename Product<Lhs, Rhs>::Scalar;
    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs,
                              const Scalar& alpha)
    {
        lhs.nestedExpression().applyConjugate(dst, rhs, alpha);
    }
};

template <typename MatrixType, typename Rhs>
struct generic_product_impl<mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Adjoint>,
                            Rhs, SparseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<
          mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Adjoint>, Rhs,
          generic_product_impl<
              mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Adjoint>, Rhs>>
{
    using Lhs    = mxpfit::MatrixFreeGEMV<MatrixType, mxpfit::Adjoint>;
    using Scalar = typename Product<Lhs, Rhs>::Scalar;
    template <typename Dest>
    static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs,
                              const Scalar& alpha)
    {
        lhs.nestedExpression().applyAdjoint(dst, rhs, alpha);
    }
};

} // namespace internal
} // namespace Eigen

#endif /* MXPFIT_FITTING_MATRIX_FREE_GEMV_HPP */

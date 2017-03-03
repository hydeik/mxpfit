///
/// \file hankel_gemv.hpp
///
#ifndef MXPFIT_HANKEL_MATRIX_HPP
#define MXPFIT_HANKEL_MATRIX_HPP

#include <Eigen/Core>

#include <fftw3/shared_plan.hpp>

namespace mxpfit
{

///
/// ### HankelMatrix
///
/// \brief Expression of a generalized Hankel matrix.
///
/// \tparam T  Scalar type of matrix elements
///
/// For the given vector \f$ \boldsymbol{h} = [h_0,h_1,...,h_{n+m-1}]^{T},\f$ a
/// \f$ m \times n \f$ Hankel matrix, \f$ A, \f$ is defined as
///
/// \f[
///   A = \left[ \begin{array}{cccccc}
///     h_0    & h_1    & h_2    & h_3   & \cdots & h_{n-1} \\
///     h_1    & h_2    & h_3    & h_4   & \cdots & h_{n}   \\
///     h_2    & h_3    & h_4    & h_5   & \cdots & h_{n+1} \\
///     h_3    & h_4    & h_5    & h_6   & \cdots & h_{n+2} \\
///     \vdots & \vdots & \vdots &\vdots & \ddots & \vdots \\
///     h_{m-1} & h_{m} & h_{m+1} & h_{m+2} & \cdots & h_{m+n-1}
///   \end{array} \right]
/// \f]
///
/// This class represents a Hankel matrix expression from the given number of
/// rows, \f$ m, \f$ the number of columns \f$n,\f$ and a vector expression for
/// the coefficients, \f$ \boldsymbol{h}. \f$ If the given vector expression is
/// l-value, this class wraps the existing vector expression, otherwise storage
/// for coefficients are allocated and stored.
///
/// This class also provides the interface for matrix-vector multiplication
/// compatible to `MatrixFreeGEMV`. This operation can be performed efficiently
/// using the fast Fourier transform (FFT) in \f$O((m+n)\log(m+n).\f$
///
/// For this purpose, the class also allocate internal vectors as working space.
/// The FFT plans are automatically generated whenever matrix size is updated.
/// This operation is done in thread-safe manner, using mutex-lock.
///
template <typename T>
class HankelMatrix
{
public:
    using Scalar              = T;
    using RealScalar          = typename Eigen::NumTraits<T>::Real;
    using ComplexScalar       = std::complex<RealScalar>;
    using StorageIndex        = Eigen::Index;
    using Index               = Eigen::Index;
    using CoeffsVector        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using CoeffsVectorRef     = Eigen::Ref<const CoeffsVector>;
    using ComplexCoeffsVector = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1>;
    using PlainObject = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

private:
    using FFT  = fftw3::FFT<RealScalar>;
    using IFFT = fftw3::IFFT<RealScalar>;

    typename FFT::PlanPointer m_fft_plan;
    typename IFFT::PlanPointer m_ifft_plan;

    Index m_rows;
    Index m_cols;

    mutable CoeffsVector m_work;
    CoeffsVectorRef m_coeffs;
    ComplexCoeffsVector m_caux;
    mutable ComplexCoeffsVector m_xaux;

public:
    enum
    {
        /// `Scalar` is complex number?
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex
    };

    /// Default constructor: create an empty matrix
    HankelMatrix()
        : m_fft_plan(),
          m_ifft_plan(),
          m_rows(),
          m_cols(),
          m_work(),
          m_coeffs(m_work), // need to initialize Eigen::Ref object
          m_caux(),
          m_xaux()
    {
    }

    /// Copy constructor
    HankelMatrix(const HankelMatrix &other)
        : m_fft_plan(other.m_fft_plan),
          m_ifft_plan(other.m_ifft_plan),
          m_rows(other.m_rows),
          m_cols(other.m_cols),
          m_work(other.m_work),
          m_coeffs(other.m_coeffs),
          m_caux(other.m_caux),
          m_xaux(other.m_xaux)
    {
    }

    ///
    /// Construct a Hankel matrix with memory allocation
    ///
    /// \param[in] nrows the number of rows
    /// \param[in] ncols the number of columns
    /// \param[in] prescribed_fft_size the FFT length (optional)
    ///
    /// \pre  `nrows >= 0 && ncols >= 0` is required
    ///
    HankelMatrix(Index nrows, Index ncols, Index prescribed_fft_size = 0)
        : m_fft_plan(),
          m_ifft_plan(),
          m_rows(nrows),
          m_cols(ncols),
          m_work(get_fft_length(nrows, ncols, prescribed_fft_size)),
          m_coeffs(m_work), // need to initialize Eigen::Ref
          m_caux(IsComplex ? m_work.size() : m_work.size() / 2 + 1),
          m_xaux(m_caux.size())
    {
        assert(nrows >= Index() && "nrows must be a non-negative integer");
        assert(ncols >= Index() && "ncols must be a non-negative integer");
        set_fft_plans();
    }

    /// Default destructor
    ~HankelMatrix()
    {
    }

    /// Assignment operator is deleted as a consequence that Eigen::Ref is
    /// non-assignable
    HankelMatrix &operator=(const HankelMatrix &other) = delete;

    /// \return the number of rows
    Index rows() const
    {
        return m_rows;
    }

    /// \return the number of columns
    Index cols() const
    {
        return m_cols;
    }

    /// \return size of vector that defines current Hankel matrix
    Index size() const
    {
        // Return 0 if the matrix is empty
        return std::max(Index(), rows() + cols() - 1);
    }

    /// \return  A const reference of the coefficient vector
    const CoeffsVectorRef &coeffs() const
    {
        return m_coeffs;
    }

    ///
    /// Resize internal Hankel matrix and update FFT plan for matrix-vector
    /// multiplication if necessary.
    ///
    void resize(Index nrows, Index ncols, Index prescribed_fft_size = 0)
    {
        assert(nrows >= Index() && "nrows must be a non-negative integer");
        assert(ncols >= Index() && "ncols must be a non-negative integer");

        m_rows = nrows;
        m_cols = ncols;

        const Index n1 = m_work.size();
        const Index n2 = get_fft_length(nrows, ncols, prescribed_fft_size);

        if (n1 != n2)
        {
            m_work.resize(n2);
            m_caux.resize(IsComplex ? m_work.size() : m_work.size() / 2 + 1);
            m_xaux.resize(m_caux.size());
            set_fft_plans();

            if (coeffs().size() != size())
            {
                // Eigen::Ref to dummy object
                m_coeffs.~CoeffsVectorRef();
                ::new (&m_coeffs) CoeffsVectorRef(m_work);
            }
        }
    }

    ///
    /// Set a Hankel matrix from the number of rows, columns and vector
    /// expression of matrix coefficients.
    ///
    /// \param[in] coeffs   the vector expression of matrix coefficients
    ///
    /// \pre `coeffs.size() ==  size()` is required
    ///
    template <typename Derived>
    void setCoeffs(const Eigen::EigenBase<Derived> &coeffs)
    {
        assert(coeffs.size() == size() &&
               "Invalid size for the vector of coefficients");
        if (rows() == Index() || cols() == Index())
        {
            // Matrix is empty. Nothing to do.
            return;
        }

        m_coeffs.~CoeffsVectorRef();
        ::new (&m_coeffs) CoeffsVectorRef(coeffs.derived());

        compute_aux_vector();
    }

    void setCoeffs(const CoeffsVectorRef &coeffs)
    {
        assert(coeffs.size() == size() &&
               "Invalid size for the vector of coefficients");
        if (rows() == Index() || cols() == Index())
        {
            // Matrix is empty. Nothing to do.
            return;
        }

        if (&(coeffs.derived()) != &m_coeffs)
        {
            m_coeffs.~CoeffsVectorRef();
            ::new (&m_coeffs) CoeffsVectorRef(coeffs);
            compute_aux_vector();
        }
    }

    /// \return The same Hankel matrix in dense form.
    PlainObject toDenseMatrix() const
    {
        PlainObject ret(rows(), cols());
        for (Index j = 0; j < cols(); ++j)
        {
            for (Index i = 0; i < rows(); ++i)
            {
                ret(i, j) = coeffs()(i + j);
            }
        }
        return ret;
    }

    /// Compute `dst += alpha * A * rhs`
    template <typename Dest, typename RHS>
    void apply(Dest &dst, const Eigen::MatrixBase<RHS> &rhs, Scalar alpha) const
    {
        apply_impl<false>(dst, rhs, alpha);
    }

    /// Compute `dst += alpha * A.conjugate() * rhs`
    template <typename Dest, typename RHS>
    void applyConjugate(Dest &dst, const Eigen::MatrixBase<RHS> &rhs,
                        Scalar alpha) const
    {
        apply_impl<(IsComplex != 0)>(dst, rhs, alpha);
    }

    /// Compute `dst += alpha * A.transpose() * rhs`
    template <typename Dest, typename RHS>
    void applyTranspose(Dest &dst, const Eigen::MatrixBase<RHS> &rhs,
                        Scalar alpha) const
    {
        apply_transpose_impl<false>(dst, rhs, alpha);
    }

    /// Compute `dst += alpha * A.transpose() * rhs`
    template <typename Dest, typename RHS>
    void applyAdjoint(Dest &dst, const Eigen::MatrixBase<RHS> &rhs,
                      Scalar alpha) const
    {
        apply_transpose_impl<(IsComplex != 0)>(dst, rhs, alpha);
    }

private:
    static Index get_fft_length(Index nrows, Index ncols,
                                Index prescribed_fft_size)
    {
        // Enforce fft_size becomes even
        const Index n = (nrows + ncols) / 2;
        return std::max({prescribed_fft_size, 2 * n});
    }

    void set_fft_plans()
    {
        const int n       = static_cast<int>(m_work.size());
        const int howmany = 1;
        m_fft_plan  = FFT::make_plan(n, howmany, m_work.data(), m_xaux.data());
        m_ifft_plan = IFFT::make_plan(n, howmany, m_xaux.data(), m_work.data());
    }

    // Pre-compute auxiliary vector for matrix-vector multiplication operation
    void compute_aux_vector()
    {
        // Set first column of circulant matrix C. Then compute the discrete
        // Fourier transform this vector and store the result into \c caux.
        const auto nhead    = rows();
        const auto ntail    = cols() - 1;
        const auto npadding = m_work.size() - nhead - ntail;

        m_work.head(nhead) = coeffs().tail(nhead);
        if (npadding > Index())
        {
            m_work.segment(nhead, npadding).setZero();
        }
        m_work.tail(ntail) = coeffs().head(ntail);

        // m_caux <-- FFT[m_work]
        FFT::run(m_fft_plan, m_work.data(), m_caux.data());
        m_caux *= RealScalar(1) / m_work.size();
    }

    template <bool ComplexConjugate, typename Dest, typename RHS>
    void apply_impl(Dest &dst, const Eigen::MatrixBase<RHS> &rhs,
                    Scalar alpha) const
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Dest);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(RHS);

        assert(dst.size() == rows());
        assert(rhs.size() == cols());

        if (alpha == Scalar(/*zero*/))
        {
            return;
        }
        //
        // Form new vector x' = [x(n-1),x(n-2),...,x(0),0....0] of
        // length n + m - 1, and compute FFT.
        //
        if (ComplexConjugate)
        {
            m_work.head(cols()) = rhs.reverse().conjugate();
        }
        else
        {
            m_work.head(cols()) = rhs.reverse();
        }

        m_work.tail(m_work.size() - cols()).setZero();

        // m_xaux <-- FFT[m_work]
        FFT::run(m_fft_plan, m_work.data(), m_xaux.data());
        //
        // y[0:nrows] = IFFT(FFT(c') * FFT(x'))[0:nrows]
        //
        m_xaux = m_xaux.cwiseProduct(m_caux);
        IFFT::run(m_ifft_plan, m_xaux.data(), m_work.data());

        if (ComplexConjugate)
        {
            dst += alpha * m_work.head(rows()).conjugate();
        }
        else
        {
            dst += alpha * m_work.head(rows());
        }
    }

    // Compute `dst += alpha * A^T * rhs`
    template <bool ComplexConjugate, typename Dest, typename RHS>
    void apply_transpose_impl(Dest &dst, const Eigen::MatrixBase<RHS> &rhs,
                              Scalar alpha) const
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Dest);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(RHS);

        assert(dst.size() == cols());
        assert(rhs.size() == rows());

        if (alpha == Scalar(/*zero*/))
        {
            return;
        }
        //
        // Form new vector x' = [0,0,...,0,x(m-1),x(m-2),...,x(0)] of
        // length n + m - 1, and compute FFT.
        //
        m_work.head(m_work.size() - rows()).setZero();
        if (ComplexConjugate)
        {
            m_work.tail(rows()) = rhs.reverse().conjugate();
        }
        else
        {
            m_work.tail(rows()) = rhs.reverse();
        }

        // m_xaux <-- FFT[m_work]
        FFT::run(m_fft_plan, m_work.data(), m_xaux.data());
        //
        // y[0:ncols-1] = IFFT(FFT(c') * FFT(x'))[nrows:size - 1]
        //
        m_xaux = m_xaux.cwiseProduct(m_caux);
        IFFT::run(m_ifft_plan, m_xaux.data(), m_work.data());

        if (ComplexConjugate)
        {
            dst += alpha * m_work.tail(cols()).conjugate();
        }
        else
        {
            dst += alpha * m_work.tail(cols());
        }
    }
};

} // namespace mxpfit

#endif /* MXPFIT_HANKEL_MATRIX_HPP */

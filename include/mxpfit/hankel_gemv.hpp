///
/// \file hankel_gemv.hpp
///
#ifndef MXPFIT_HANKEL_GEMV_HPP
#define MXPFIT_HANKEL_GEMV_HPP

#include <Eigen/Core>

#include <fftw3/shared_plan.hpp>

namespace mxpfit
{

///
/// ### HankelGEMV
///
/// Fast Hankel matrix-vector multiplication using FFT
///
/// \tparam T  scalar type of matrix elements (either real or complex number)
///
template <typename T>
class HankelGEMV
{
public:
    using Scalar              = T;
    using RealScalar          = typename Eigen::NumTraits<T>::Real;
    using ComplexScalar       = std::complex<RealScalar>;
    using StorageIndex        = Eigen::Index;
    using Index               = Eigen::Index;
    using CoeffsVector        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ComplexCoeffsVector = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1>;

private:
    using FFT  = fftw3::FFT<RealScalar>;
    using IFFT = fftw3::IFFT<RealScalar>;

    typename FFT::PlanPointer m_fft_plan;
    typename IFFT::PlanPointer m_ifft_plan;

    Index m_rows;
    Index m_cols;

    mutable CoeffsVector m_work;
    ComplexCoeffsVector m_caux;
    mutable ComplexCoeffsVector m_xaux;

public:
    enum
    {
        /// `Scalar` is complex number?
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex
    };

    HankelGEMV() = default;

    HankelGEMV(Index nrows, Index ncols, Index fft_size = -1)
        : m_fft_plan(),
          m_ifft_plan(),
          m_rows(nrows),
          m_cols(ncols),
          m_work(fft_size > 0 ? fft_size : nrows + ncols - 1),
          m_caux(IsComplex ? m_work.size() : m_work.size() / 2 + 1),
          m_xaux(m_caux.size())
    {
        assert(m_work.size() >= nrows + ncols - 1);
        set_fft_plans();
    }

    template <typename Derived>
    HankelGEMV(Index nrows, Index ncols,
               const Eigen::MatrixBase<Derived> &coeffs, Index fft_size = -1)
        : HankelGEMV(nrows, ncols, fft_size)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
        assert(coeffs.size() == nrows + ncols - 1);

        setCoeffs(coeffs);
    }

    HankelGEMV(const HankelGEMV &) = default;
    HankelGEMV(HankelGEMV &&)      = default;
    ~HankelGEMV()                  = default;

    HankelGEMV &operator=(const HankelGEMV &) = default;
    HankelGEMV &operator=(HankelGEMV &&) = default;

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
        return rows() + cols() - 1;
    }

    /// Resize internal Hankel matrix
    void resize(Index nrows, Index ncols, Index fft_size = -1)
    {
        if (fft_size <= 0)
        {
            fft_size = nrows + ncols - 1;
        }
        assert(fft_size >= nrows + ncols - 1);
        m_rows = nrows;
        m_cols = ncols;
        m_work.resize(fft_size);
        m_caux.resize(IsComplex ? m_work.size() : m_work.size() / 2 + 1);
        m_xaux.resize(m_caux.size());

        set_fft_plans();
    }

    /// Set coefficients of Hankel matrix
    template <typename Derived>
    void setCoeffs(const Eigen::MatrixBase<Derived> &coeffs)
    {
        assert(coeffs.size() == size() &&
               "Invalid size for coefficients array");
        //
        // Set first column of circulant matrix C. Then compute the discrete
        // Fourier transform this vector and store the result into \c caux.
        //
        const auto nhead    = rows();
        const auto ntail    = cols() - 1;
        const auto npadding = m_work.size() - nhead - ntail;

        m_work.head(nhead) = coeffs.tail(nhead);
        if (npadding > Index())
        {
            m_work.segment(nhead, npadding).setZero();
        }
        m_work.tail(ntail) = coeffs.head(ntail);

        // m_caux <-- FFT[m_work]
        FFT::run(m_fft_plan, m_work.data(), m_caux.data());
        m_caux *= RealScalar(1) / m_work.size();
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
    void set_fft_plans()
    {
        const int n       = static_cast<int>(m_work.size());
        const int howmany = 1;
        m_fft_plan  = FFT::make_plan(n, howmany, m_work.data(), m_xaux.data());
        m_ifft_plan = IFFT::make_plan(n, howmany, m_xaux.data(), m_work.data());
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

///
/// Create a Hankel matrix in dense form from the sequence of elements.
///
/// This function creates a \f$ m \times n \f$ Hankel matrix \f$ A \f$ in the
/// dense form from a given vector of matrix element \f$ h =
/// [h_0,h_1,...,h_{n+m-1}]^{T}\f$ such that,
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
/// \param[in] nrows  number of rows, \f$ m \f$
/// \param[in] ncols  number of columns, \f$ n \f$
/// \param[in] h      vector of elments of Hankel matrix with length
///                   \f$ m + n - 1 \f$:w
///
/// \return \f$ m \times n \f$ matrix with same scalar type of input vector type
/// `T1`.
///
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic,
              Eigen::ColMajor>
make_hankel_matrix(Eigen::Index nrows, Eigen::Index ncols,
                   const Eigen::DenseBase<Derived> &h)
{
    assert(h.size() == nrows + ncols - 1);
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::ColMajor>
        A(nrows, ncols);
    for (Eigen::Index col = 0; col < ncols; ++col)
    {
        for (Eigen::Index row = 0; row < nrows; ++row)
        {
            A(row, col) = h(row + col);
        }
    }

    return A;
}

} // namespace mxpfit

#endif /* MXPFIT_HANKEL_GEMV_HPP */

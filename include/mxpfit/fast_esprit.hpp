/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2017 Hidekazu Ikeno
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

///
/// \file fast_esprit.hpp
///
#ifndef MXPFIT_FAST_ESPRIT_HPP
#define MXPFIT_FAST_ESPRIT_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <mxpfit/hankel_gemv.hpp>
#include <mxpfit/matrix_free_gemv.hpp>
#include <mxpfit/partial_lanczos_bidiagonalization.hpp>
#include <mxpfit/vandermonde.hpp>

namespace mxpfit
{
///
/// Fast ESPRIT method
///
template <typename T>
class FastESPRIT
{
public:
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
    using ComplexScalar = std::complex<RealScalar>;
    using Index         = Eigen::Index;

    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::AutoAlign>;
    using RealVector =
        Eigen::Matrix<RealScalar, Eigen::Dynamic, 1, Eigen::AutoAlign>;
    using ComplexVector =
        Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1, Eigen::AutoAlign>;

    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                 (Eigen::ColMajor | Eigen::AutoAlign)>;
    using RealMatrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic,
                                     (Eigen::ColMajor | Eigen::AutoAlign)>;
    using ComplexMatrix =
        Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic,
                      (Eigen::ColMajor | Eigen::AutoAlign)>;

private:
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
        Alignment = Eigen::internal::traits<Matrix>::Alignment
    };

    using MappedMatrix = Eigen::Map<Matrix, Alignment>;

    Index nrows_;
    Index ncols_;
    Index nterms_;

    ComplexVector exponent_;
    ComplexVector weight_;
    HankelGEMV<T> matH_;
    VandermondeGEMV<ComplexScalar> matV_;
    VandermondeLeastSquaresSolver<ComplexScalar> vandermonde_solver_;

    Matrix work_;
    RealMatrix rwork_;

public:
    ///
    /// Default constructor
    ///
    FastESPRIT() = default;

    ///
    /// Constructor with memory preallocation
    ///
    /// \param[in] N  Number of sampling points
    /// \param[in] L  Window size. This is equals to the number of rows of
    ///               generalized Hankel matrix.
    /// \param[in] M  Maxumum number of terms used for the exponential sum.
    /// \pre  `N >= M >= 1` and `N - L + 1 >= M >= 1`.
    ///
    FastESPRIT(Index N, Index L, Index M)
        : nrows_(L),
          ncols_(N - L + 1),
          nterms_(),
          exponent_(M),
          weight_(M),
          matH_(nrows_, ncols_),
          matV_(N, M),
          vandermonde_solver_(N, M)
    {
        assert(nrows_ >= M && ncols_ >= M && M >= 1);
    }

    ///
    /// Destructor
    ///
    ~FastESPRIT() = default;

    ///
    /// Memory reallocation
    ///
    /// \param[in] N  Number of sampling points
    /// \param[in] L  Window size. This is equals to the number of rows of
    ///               generalized Hankel matrix.
    /// \param[in] M  Maxumum number of terms used for the exponential sum.
    /// \pre  `N >= L >= N / 2 >= 1` and `N - L + 1 >= M >= 1`.
    ///
    void resize(Index N, Index L, Index M)
    {
        nrows_  = L;
        ncols_  = N - L + 1;
        nterms_ = 0;

        assert(nrows_ >= M && ncols_ >= M && M >= 1);

        exponent_.resize(M);
        weight_.resize(M);
        matH_.resize(nrows_, ncols_);
        matV_.resize(N, M);
        vandermonde_solver_.resize(N, M);
    }

    ///
    /// \return Number of sampling points.
    ///
    Index size() const
    {
        return nrows_ + ncols_ - 1;
    }

    ///
    /// \return Number of rows of trajectory matrix.
    ///
    Index rows() const
    {
        return nrows_;
    }

    ///
    /// \return Number of columns of trajectory matrix. This should be a upper
    ///         bound of the number of exponential functions.
    ///
    Index cols() const
    {
        return ncols_;
    }

    ///
    /// Fit signals by a multi-exponential sum
    ///
    /// \param[in] f The array of signals sampled on the equispaced grid. The
    ///    first `size()` elemnets of `f` are used as a sampled data. In case
    ///    `f.size() < size()` then, last `size() - f.size()` elements are
    ///    padded by zeros.
    /// \param[in] eps  Small positive number `(0 < eps < 1)` that
    ///    controlls the accuracy of the fit.
    /// \param[in] x0  Argument of first sampling point
    /// \param[in] delta Spacing between neighboring sample points.
    ///
    template <typename VectorT>
    void compute(const Eigen::MatrixBase<VectorT>& h, RealScalar x0,
                 RealScalar delta, RealScalar eps);

    ///
    /// \return Vector view to the exponents.
    ///
    auto exponents() const -> decltype(exponent_.head(nterms_))
    {
        return exponent_.head(nterms_);
    }

    ///
    /// \return Vector view to the weights.
    ///
    auto weights() const -> decltype(weight_.head(nterms_))
    {
        return weight_.head(nterms_);
    }

    ///
    /// Evaluate exponential sum at a point
    ///
    ComplexScalar evalAt(RealScalar x) const
    {
        return ((x * exponents().array()).exp() * weights().array()).sum();
    }
};

template <typename T>
template <typename VectorT>
void FastESPRIT<T>::compute(const Eigen::MatrixBase<VectorT>& h, RealScalar x0,
                            RealScalar delta, RealScalar eps)
{
    //
    // Form rectangular Hankel matrix and pre-compute for fast multiplication to
    // the vector.
    //
    matH_.setCoeffs(h);
    const Index nr = matH_.rows();
    const Index nc = matH_.cols();

    //-------------------------------------------------------------------------
    // Compute Mxpfit roots
    //-------------------------------------------------------------------------
    //
    // Partial Lanczos bidiagonalization of Hankel matrix H = P B Q^H
    //
    const Index max_rank = exponent_.size();
    RealScalar tol_error = eps;
    MatrixFreeGEMV<HankelGEMV<Scalar>> opH(matH_);
    // Local matrices and vectors
    Matrix P(nr, max_rank);
    Matrix Q(nc, max_rank);
    Vector tmp(max_rank);
    RealVector alpha(max_rank);
    RealVector beta(max_rank);

    nterms_ =
        partialLanczosBidiagonalization(opH, alpha, beta, P, Q, tol_error, tmp);
    //
    // Form the views of matrix Q
    //
    auto Q0 = Q.block(0, 0, nc - 1, nterms_);
    auto Q1 = Q.block(1, 0, nc - 1, nterms_);
    auto nu = tmp.head(nterms_);
    nu      = Q.block(nc - 1, 0, 1, nterms_).adjoint();
    //
    // Compute the spectral matrix G = pinv(Q0) * Q1, where pinv indicate the
    // Moore-Penrose pseudo-inverse.
    //
    MappedMatrix G(P.data(), nterms_, nterms_);
    auto scal = RealScalar(1) / (RealScalar(1) - nu.squaredNorm());
    G         = Q0.adjoint() * Q1;
    auto phi  = Q.block(0, 0, nterms_, 1);
    phi       = G.adjoint() * nu;
    G += scal * nu * phi.adjoint();

    exponent_.head(nterms_) = G.eigenvalues();

    if (nterms_ > Index())
    {
        //----------------------------------------------------------------------
        // Solve overdetermined Vandermonde system
        //----------------------------------------------------------------------
        matV_.setCoeffs(exponents());
        vandermonde_solver_.setTolerance(eps);
        vandermonde_solver_.setMaxIterations(nterms_);
        vandermonde_solver_.compute(matV_);

        auto dst = weight_.head(nterms_);
        dst      = vandermonde_solver_.solve(h.template cast<ComplexScalar>());
        //
        // adjust computed parameters
        //
        auto xi_ = exponent_.head(nterms_).array();
        auto w_  = weight_.head(nterms_).array();
        xi_      = xi_.log() / delta;
        w_ *= (xi_ * x0).exp();
    }
}

} // namespace: mxpfit

#endif /* MXPFIT_FAST_ESPRIT_HPP */

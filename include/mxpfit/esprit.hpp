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
#ifndef MXPFIT_ESPRIT_HPP
#define MXPFIT_ESPRIT_HPP

#include <algorithm>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <mxpfit/exponential_sum.hpp>
#include <mxpfit/prony_like_method_common.hpp>

namespace mxpfit
{
///
/// ### ESPRIT
///
/// \brief ESPRIT method for finding parameters of exponential sum
/// approximation from sampled data on uniform grid.
///
/// \tparam T  Scalar type of function values.
///
/// #### Description
///
/// This class implements the Estimation of Signal Parameters via Rotational
/// Invariance Techniques (ESPRIT) method for parameter estimation of decaying
/// exponential sum funcitons.
///
/// This class is implemented only for the benchmarking purpose to compare with
/// the fast ESPRIT algorithm. DO NOT use the class for practical applications,
/// use `FastESPRIT` class instead, which is much faster than ESPRIT.
///
///
/// #### References
///
/// 1. D. Potts and M. Tasche,"Parameter estimation for nonincreasing
///    exponential sums by Prony-like methods", Linear Algebra Appl. **439**
///    (2013) 1024-1039.
///    [DOI: https://doi.org/10.1016/j.laa.2012.10.036]
///
template <typename T>
class ESPRIT
{
public:
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
    using ComplexScalar = std::complex<RealScalar>;
    using Index         = Eigen::Index;

    using Vector        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RealVector    = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using ComplexVector = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1>;

    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealMatrix =
        Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ComplexMatrix =
        Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>;

    using ResultType =
        typename detail::gen_prony_like_method_result<T>::ResultType;

private:
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
        Alignment = Eigen::internal::traits<Matrix>::Alignment
    };

    Index m_rows;
    Index m_cols;
    Index m_max_terms;
    Matrix m_matH;

public:
    ///
    /// Default constructor
    ///
    ESPRIT() = default;

    ///
    /// Constructor with memory preallocation
    ///
    /// \param[in] N  Number of sampling points
    /// \param[in] L  Window size. This is equals to the number of rows of
    ///               generalized Hankel matrix.
    /// \param[in] M  Maxumum number of terms used for the exponential sum.
    /// \pre  `N >= M >= 1` and `N - L + 1 >= M >= 1`.
    ///
    ESPRIT(Index N, Index L, Index M)
        : m_rows(L), m_cols(N - L + 1), m_max_terms(M), m_matH(m_rows, m_cols)
    {
        assert(m_rows >= M && m_cols >= M && M >= 1);
    }

    ///
    /// Destructor
    ///
    ~ESPRIT()
    {
    }

    ///
    /// Memory reallocation
    ///
    /// \param[in] N  Number of sampling points
    /// \param[in] L  Window size. This is equals to the number of rows of
    ///               generalized Hankel matrix.
    /// \param[in] M  Maxumum number of terms used for the exponential sum.
    /// \pre  `N >= M >= 1` and `N - L + 1 >= M >= 1`.
    ///
    void resize(Index N, Index L, Index M)
    {
        m_rows = L;
        m_cols = N - L + 1;
        assert(m_rows >= M && m_cols >= M && M >= 1);
        m_max_terms = M;
        m_matH.resize(m_rows, m_cols);
    }

    ///
    /// \return Number of sampling points.
    ///
    Index size() const
    {
        return m_rows + m_cols - 1;
    }

    ///
    /// Fit signals by a exponential sum
    ///
    /// \param[in] f The array of signals sampled on the equispaced grid. The
    ///    first `size()` elemnets of `nterms` are used as a sampled data. In
    ///    case `f.size() < size()` then, last `size() - f.size()` elements are
    ///    padded by zeros.
    /// \param[in] eps  Small positive number `(0 < eps < 1)` that
    ///    controlls the accuracy of the fit.
    /// \param[in] x0  Argument of first sampling point
    /// \param[in] delta Spacing between neighboring sample points.
    ///
    template <typename VectorT>
    ResultType compute(const Eigen::MatrixBase<VectorT>& h, RealScalar x0,
                       RealScalar delta, RealScalar eps);
};

template <typename T>
template <typename VectorT>
typename ESPRIT<T>::ResultType
ESPRIT<T>::compute(const Eigen::MatrixBase<VectorT>& h, RealScalar x0,
                   RealScalar delta, RealScalar eps)
{
    assert(h.size() == size() && "Number of data points mismatch.");
    //
    // Form rectangular Hankel matrix from sequance h.
    //
    for (Index j = 0; j < m_cols; ++j)
    {
        for (Index i = 0; i < m_rows; ++i)
        {
            m_matH.coeffRef(i, j) = h(i + j);
        }
    }
    // const Index nr = m_matH.rows();
    const Index nc = m_matH.cols();

    //-------------------------------------------------------------------------
    // Compute roots of Prony polynomials H = U * S * W^H
    //-------------------------------------------------------------------------
    Eigen::BDCSVD<Matrix> svd(m_matH, Eigen::ComputeThinV);

    const auto& sigma = svd.singularValues();
    Index nterms      = 0;
    // extract rank of matrix H from singular values
    while (nterms < m_max_terms)
    {
        if (sigma(nterms) < eps)
        {
            break;
        }
        ++nterms;
    }

    if (nterms == 0)
    {
        return ResultType();
    }

    ComplexVector roots(nterms);
    ComplexVector weights(nterms);

    {
        // --- Form the views of matrix W
        // Matrix W excluding the last row
        auto W0 = svd.matrixV().block(0, 0, nc - 1, nterms);
        // Matrix W excluding the first row
        auto W1 = svd.matrixV().block(1, 0, nc - 1, nterms);
        // adjoint of the last row of matrix W
        auto nu = svd.matrixV().block(nc - 1, 0, 1, nterms).adjoint();
        //
        // Compute the spectral matrix G = pinv(W0) * W1, where pinv indicate
        // the Moore-Penrose pseudo-inverse. The computation of the
        // pseudo-inverse of W0 can be avoided.
        //
        Matrix G(W0.adjoint() * W1);
        Vector phi(G.adjoint() * nu);
        auto scal = RealScalar(1) / (RealScalar(1) - nu.squaredNorm());
        G += scal * nu * phi.adjoint();
        //
        // Prony roots \f$\{z_i\{\}\f$ are the eigenvalues of matrix G. The
        // exponents for approximation are obtained as \f$ \log z_i \f$
        //
        roots = G.eigenvalues();
    }

    //----------------------------------------------------------------------
    // Solve overdetermined Vandermonde system to obtain the weights
    //----------------------------------------------------------------------
    {
        // Create Vandermonde matrix from prony roots
        ComplexMatrix matV(h.size(), nterms);
        for (Index j = 0; j < matV.cols(); ++j)
        {
            auto x     = roots(j);
            auto v     = Scalar(1);
            matV(0, j) = v;
            for (Index i = 1; i < matV.rows(); ++i)
            {
                v *= x;
                matV(i, j) = v;
            }
        }
        // Solve least-squares problem V x = h
        weights = matV.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                      .solve(h.template cast<ComplexScalar>());
    }

    // Rescale the exponents and weights, and return the result
    return detail::gen_prony_like_method_result<T>::create(
        roots.array(), weights.array(), x0, delta);
}

} // namespace mxpfit

#endif /* MXPFIT_ESPRIT_HPP */

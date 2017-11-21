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

#include <algorithm>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <mxpfit/exponential_sum.hpp>
#include <mxpfit/hankel_matrix.hpp>
#include <mxpfit/matrix_free_gemv.hpp>
#include <mxpfit/partial_lanczos_bidiagonalization.hpp>
#include <mxpfit/vandermonde_least_squares.hpp>

namespace mxpfit
{
///
/// ### FastESPRIT
///
/// \brief Fast ESPRIT method for finding parameters of exponential sum
/// approximation from sampled data on uniform grid.
///
/// \tparam T  Scalar type of function values.
///
/// #### Description
///
/// For a given sequence \f$ f_{k}=f(t_{k}) \f$ sampled on a uniform grid
/// \f$t_{k}=t_{0}+hk\, (k=0,1,\dots,N-1)\f$ and prescribed accuracy
/// \f$\epsilon\f$, this class finds exponential sum approximation of function
/// \f$ f(t) \f$ such that,
///
/// \f[
///   \left| f_{k}-\sum_{j=1}^{M}c_{j}e^{-a_{j} t} \right| < \epsilon
/// \f]
///
/// with \f$\mathrm{Re}(a_{j}) > 0.\f$ The problem is solved using the fast
/// ESPRIT algorithm via partial Lanczos bidiagonalization which has been
/// developed by Potts and Tasche (2015). The present algorithm was slightly
/// modified from the original one.

/// #### References
///
/// 1. D. Potts and M. Tasche, "Fast ESPRIT algorithms based on partial singular
///    value decompositions", Appl. Numer. Math. **88** (2015) 31-45.
///    [DOI: https://doi.org/10.1016/j.apnum.2014.10.003]
/// 2. D. Potts and M. Tasche,"Parameter estimation for nonincreasing
///    exponential sums by Prony-like methods", Linear Algebra Appl. **439**
///    (2013) 1024-1039.
///    [DOI: https://doi.org/10.1016/j.laa.2012.10.036]
/// 3. K. Browne, S. Qiao, and Y. Wei, "A Lanczos bidiagonalization algorithm
///    for Hankel matrices", Linear Algebra Appl. **430** (2009) 1531-1543.
///    [DOI: https://doi.org/10.1016/j.laa.2008.01.012]
///
template <typename T>
class FastESPRIT
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

    using ResultType = ExponentialSum<ComplexScalar, ComplexScalar>;

private:
    using HankelGEMV      = MatrixFreeGEMV<HankelMatrix<Scalar>>;
    using VandermondeGEMV = MatrixFreeGEMV<VandermondeMatrix<ComplexScalar>>;
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
        Alignment = Eigen::internal::traits<Matrix>::Alignment
    };

    using MappedMatrix = Eigen::Map<Matrix, Alignment>;

    Index m_rows;
    Index m_cols;
    Index m_max_terms;
    HankelMatrix<Scalar> m_matH;
    PartialLanczosBidiagonalization<HankelGEMV> m_plbd;

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
        : m_rows(L),
          m_cols(N - L + 1),
          m_max_terms(M),
          m_matH(m_rows, m_cols),
          m_plbd(m_rows, m_cols, m_max_terms)
    {
        assert(m_rows >= M && m_cols >= M && M >= 1);
    }

    /// Destructor
    ~FastESPRIT()
    {
    }

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
        m_rows = L;
        m_cols = N - L + 1;
        assert(m_rows >= M && m_cols >= M && M >= 1);
        m_max_terms = M;
        m_matH.resize(m_rows, m_cols);
        // Internal matrices of `m_plbd` are resized automatically during
        // computation
    }

    ///
    /// \return Number of sampling points.
    ///
    Index size() const
    {
        return m_matH.size();
    }

    ///
    /// Fit signals by a exponential sum
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
    ResultType compute(const Eigen::MatrixBase<VectorT>& h, RealScalar x0,
                       RealScalar delta, RealScalar eps);

private:
    template <typename VectorZ, typename VectorW>
    ResultType make_results_from_prony_roots_and_weights(
        const Eigen::MatrixBase<VectorZ>& z,
        const Eigen::MatrixBase<VectorW>& w, RealScalar x0, RealScalar delta,
        std::true_type /* Scalar is complex */);

    template <typename VectorZ, typename VectorW>
    ResultType make_results_from_prony_roots_and_weights(
        const Eigen::MatrixBase<VectorZ>& z,
        const Eigen::MatrixBase<VectorW>& w, RealScalar x0, RealScalar delta,
        std::false_type /* Scalar is real*/);
};

template <typename T>
template <typename VectorT>
typename FastESPRIT<T>::ResultType
FastESPRIT<T>::compute(const Eigen::MatrixBase<VectorT>& h, RealScalar x0,
                       RealScalar delta, RealScalar eps)
{
    assert(h.size() == size() && "Number of data points mismatch.");
    //
    // Form rectangular Hankel matrix and pre-compute for fast multiplication to
    // the vector.
    //
    m_matH.setCoeffs(h);
    // const Index nr = m_matH.rows();
    const Index nc = m_matH.cols();

    //-------------------------------------------------------------------------
    // Compute roots of Prony polynomials
    //-------------------------------------------------------------------------
    //
    // Partial Lanczos bidiagonalization of Hankel matrix H = P B Q^H
    //
    HankelGEMV opH(m_matH);
    m_plbd.setTolerance(eps);
    m_plbd.compute(m_matH, m_max_terms);
    const Index nterms = m_plbd.rank();

    if (nterms == 0)
    {
        return ResultType();
    }
    // --- Form the views of matrix Q
    // Matrix Q excluding the last row
    auto Q0 = m_plbd.matrixQ().block(0, 0, nc - 1, nterms);
    // Matrix Q excluding the first row
    auto Q1 = m_plbd.matrixQ().block(1, 0, nc - 1, nterms);
    // adjoint of the last row of matrix Q
    auto nu = m_plbd.matrixQ().block(nc - 1, 0, 1, nterms).adjoint();
    //
    // Compute the spectral matrix G = pinv(Q0) * Q1, where pinv indicate the
    // Moore-Penrose pseudo-inverse. The computation of the pseudo-inverse of Q0
    // can be avoided.
    //
    Matrix G(Q0.adjoint() * Q1);
    Vector phi(G.adjoint() * nu);
    auto scal = RealScalar(1) / (RealScalar(1) - nu.squaredNorm());
    G += scal * nu * phi.adjoint();
    //
    // Prony roots \f$\{z_i\{\}\f$ are the eigenvalues of matrix G.
    // The exponents for approximation are obtained as \f$ \log z_i \f$
    //
    ComplexVector roots(G.eigenvalues());

    //----------------------------------------------------------------------
    // Solve overdetermined Vandermonde system to obtain the weights
    //----------------------------------------------------------------------
    VandermondeMatrix<ComplexScalar> matV(size(), roots);
    VandermondeGEMV opV(matV);
    VandermondeLeastSquaresSolver<ComplexScalar> solver(opV);
    solver.setTolerance(eps);

    ComplexVector weights(nterms);
    weights = solver.solve(h.template cast<ComplexScalar>());

    // //
    // // adjust computed parameters
    // //
    // roots.array() = -roots.array().log() / delta;
    // if (x0 != RealScalar())
    // {
    //     weights.array() = (-x0 * ret.exponents()).exp();
    // }

    // return ResultType(roots, weights);

    return make_results_from_prony_roots_and_weights(
        roots, weights, x0, delta, std::integral_constant<bool, IsComplex>());
}

template <typename T>
template <typename VectorZ, typename VectorW>
typename FastESPRIT<T>::ResultType
FastESPRIT<T>::make_results_from_prony_roots_and_weights(
    const Eigen::MatrixBase<VectorZ>& z, const Eigen::MatrixBase<VectorW>& w,
    RealScalar x0, RealScalar delta, std::true_type /*is complex*/)
{
    ResultType ret(z.size());
    ret.exponents() = -z.array().log() / delta;
    if (x0 != RealScalar())
    {
        ret.weights() = w.array() * (-x0 * ret.exponents()).exp();
    }

    return ret;
}

template <typename T>
template <typename VectorZ, typename VectorW>
typename FastESPRIT<T>::ResultType
FastESPRIT<T>::make_results_from_prony_roots_and_weights(
    const Eigen::MatrixBase<VectorZ>& z, const Eigen::MatrixBase<VectorW>& w,
    RealScalar x0, RealScalar delta, std::false_type /*is complex*/)
{
    using Eigen::numext::abs;
    using Eigen::numext::conj;
    using Eigen::numext::exp;
    using Eigen::numext::log;
    using Eigen::numext::imag;
    using Eigen::numext::real;

    static const auto eps     = Eigen::NumTraits<RealScalar>::epsilon();
    static const auto pi      = 4 * Eigen::numext::atan(RealScalar(1));
    constexpr const auto zero = RealScalar();
    constexpr const auto half = RealScalar(0.5);

    //-------------------------------------------------------------------------
    // The exponents are obtained as a_i = -log(z_i), where {z_i} are the roots
    // of the Prony polynomial.
    //
    // Some z_i might be real and negative: in this case, the corresponding
    // parameter a_i becomes a complex, i.e, a_i = -ln|z_i|+i \pi.
    // However, its complex conjugate a_i^* is not included in the final
    // exponential sum approximation which makes the approximated function
    // non-real. Thus, we introduce additional parameters so as to include
    // a exponential term whose exponents is a_i^*.
    //-------------------------------------------------------------------------

    // Count negative, real-valued Prony roots
    Index count = 0;
    for (Index i = 0; i < z.size(); ++i)
    {
        const auto xi = real(z(i));
        const auto yi = imag(z(i));
        if (xi < zero && abs(yi) < eps)
        {
            ++count;
        }
    }

    if (count)
    {
        // Found negative real roots.
        ResultType ret(z.size() + count);

        Index n = 0;
        for (Index i = 0; i < z.size(); ++i)
        {
            const auto xi = real(z(i));
            const auto yi = imag(z(i));
            if (xi < zero && abs(yi) < eps)
            {
                // Log(z) = -log(xi) + i pi
                const auto an       = ComplexScalar(-log(-xi), -pi) / delta;
                ret.exponent(n)     = an;
                ret.exponent(n + 1) = conj(an);

                const auto wn     = half * w(i) * exp(-x0 * an);
                ret.weight(n)     = wn;
                ret.weight(n + 1) = conj(wn);

                n += 2;
            }
            else
            {
                const auto an   = -log(z(i)) / delta;
                ret.exponent(n) = an;
                ret.weight(n)   = w(i) * exp(-x0 * an);

                ++n;
            }
        }

        return ret;
    }
    else
    {
        ResultType ret(z.size());
        ret.exponents() = -z.array().log() / delta;
        if (x0 != RealScalar())
        {
            ret.weights() = w.array() * (-x0 * ret.exponents()).exp();
        }
        return ret;
    }
}

} // namespace: mxpfit

#endif /* MXPFIT_FAST_ESPRIT_HPP */

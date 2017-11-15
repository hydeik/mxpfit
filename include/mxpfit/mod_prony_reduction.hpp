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
/// \file mod_prony_reduction.hpp
///
/// Sparse approximation of exponential sum by the modified Prony method.
///

#ifndef MXPFIT_MOD_PRONY_REDUCTION_HPP
#define MXPFIT_MOD_PRONY_REDUCTION_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <mxpfit/exponential_sum.hpp>

namespace mxpfit
{

///
/// ### ModPronyReduction
///
/// \brief Find a truncated exponential sum function with smaller number of
///        terms by the modified Prony's method
///
/// \tparam T  Scalar type of exponential sum function.
///
/// For a given exponential sum function with real exponents,
///
/// \f[
///   f(t)=\sum_{j=1}^{n} c_{j}^{} e^{-a_{j}^{} t}, \quad
///   (a_{j} > 0),
/// \f]
///
/// and prescribed accuracy \f$\epsilon > 0,\f$ this class calculates truncated
/// exponential \f$\hat{f}(t)\f$ sum such that
///
/// \f[
///   \hat{f}(t)=\sum_{j=1}^{k} \hat{c}_{j}^{}e^{-\hat{a}_{j}^{} t}, \quad
///   \left| f(t)-\hat{f}(t) \right| < \epsilon,
/// \f]
///
/// where \f$k \leq n.\f$ Exponents are assumed to be real and sorted in
/// ascending order.
///
template <typename T>
class ModPronyReduction
{
public:
    using Scalar     = T;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Index      = Eigen::Index;

    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ResultType = ExponentialSum<Scalar>;

    ///
    /// Compute truncated exponential sum \f$ \hat{f}(t) \f$
    ///
    /// \tparam DerivedF type of exponential sum inheriting ExponentialSumBase
    ///
    /// \param[in] orig original exponential sum function, \f$ f(t) \f$
    /// \param[in] threshold  prescribed accuracy \f$0 < \epsilon \ll 1\f$
    ///
    /// \return An instance of ExponentialSum represents \f$\hat{f}(t)\f$
    ///
    template <typename DerivedF>
    static ResultType compute(const ExponentialSumBase<DerivedF>& orig,
                              RealScalar threshold);
};

template <typename T>
template <typename DerivedF>
typename ModPronyReduction<T>::ResultType
ModPronyReduction<T>::compute(const ExponentialSumBase<DerivedF>& fn,
                              RealScalar threshold)
{
    using Eigen::numext::real;
    //
    // Express the exponential sum whose exponents are less than a threshold
    // (set to 1 here) with the linear combinations of fewer exponential
    // functions.
    //
    // Note that exponents are all positive, and sorted in ascending order.
    //
    Index m0 = 0;
    for (; m0 < fm.size(); ++m0)
    {
        if (fn.exponent(m0) >= T(1))
        {
            break;
        }
    }

    VectorType h(2 * m0);
    const auto w_small = fn.weights().head(m0);   // View
    const auto p_small = fn.exponents().head(m0); // View
    h(0)               = w_small.sum();
    h(1)               = -(w_small * p_small).sum();
    Index m            = 1;
    auto factorial     = T(1);
    for (; m < m0; ++m)
    {
        h(2 * m + 0) = (w_small * p_small.pow(2 * m + 0)).sum();
        h(2 * m + 1) = -(w_small * p_small.pow(2 * m + 1)).sum();
        factorial *= T((2 * m) * (2 * m + 1));
        if (-h(2 * m + 1) / factorial < eps)
        {
            // Taylor expansion converges with the tolerance eps.
            ++m;
            break;
        }
    }

    //
    // Construct a Hankel matrix from the sequence h, and solve the linear
    // equation, H q = b, with b = -h(m:2m-1).
    //
    MatrixType H(m, m);
    for (Index i = 0; i < m; ++i)
    {
        H.col(i) = h.segment(i, m);
    }
    VectorType b(-h.segment(m, m));
    VectorType q(H.colPivHouseholderQr().solve(b));

    //
    // Find the roots of the Prony polynomial,
    //
    // q(z) = \sum_{k=0}^{m-1} q_k z^{k}.
    //
    // The roots of q(z) can be obtained as the eigenvalues of the companion
    // matrix,
    //
    //     (0  0  ...  0 -p[0]  )
    //     (1  0  ...  0 -p[1]  )
    // C = (0  1  ...  0 -p[2]  )
    //     (.. .. ...  .. ..    )
    //     (0  0  ...  1 -p[m-1])
    //
    MatrixType& companion = H;
    companion.setZeros();
    companion.diagonal(-1).setOnes();
    companion.col(m - 1) = -q;
    Eigen::EigenSolver<MatrixType> es(companion,
                                      /*compute eigenvectors*/ false);
    // Vector gamma(arma::real(arma::eig_gen(companion)));

    // --- Update exponents & weights
    const Index keep = fn.size() - m0;
    ResultType ret(keep + m);
    ret.exponents().head(m)    = -es.eigenvalues().real();
    ret.exponents().tail(keep) = fn.exponents().tail(keep);
    //
    // Construct Vandermonde matrix from gamma
    //
    MatrixType V(2 * m, m);
    for (Index i = 0; i < m; ++i)
    {
        const auto z = real(gamma(i));
        V(0, i)      = T(1);
        for (Index j = 1; j < V.rows(); ++j)
        {
            V(j, i) = V(j - 1, i) * z; // z[i]**j
        }
    }
    //
    // Solve overdetermined Vandermonde system,
    //
    // V(0:2m-1,0:m-1) w(0:m-1) = h(0:2m-1)
    //
    // by the least square method.
    //
    ret.weights().head(m)    = V.colPivHouseholderQr().solve(h.head(2 * m));
    ret.weights().tail(keep) = fn.weights().tail(keep);

    return ret;
}
/// \internal
///
/// ### reduce_terms_with_small_exponents
///
/// Find shorter exponential sum approximation by removing terms with small
/// exponents using the modified Prony's method.
///
template <typename T>
GaussianSum<T> reduce_terms_with_small_exponents(const GaussianSum<T>& gs_orig,
                                                 T eps)
{
    using UIndex  = arma::uword;
    using SUIndex = arma::sword;
    using Vector  = arma::Col<T>;
    using Matrix  = arma::Mat<T>;
    //
    // Express the sum of Gaussians whose exponents are less than a threshold
    // (set to 1 here) with the linear combinations of fewer Gaussians.
    //
    // Note that exponents are all positive, and sorted in ascending order.
    //
    UIndex m0 = 0;
    for (; m0 < gs_orig.size(); ++m0)
    {
        if (gs_orig.exponent(m0) >= T(1))
        {
            break;
        }
    }

    Vector h(2 * m0);
    const auto w_small = gs_orig.weights().head(m0);   // View
    const auto p_small = gs_orig.exponents().head(m0); // View
    h(0)               = arma::sum(w_small);
    h(1)               = -arma::sum(w_small % p_small);
    UIndex m           = 1;
    auto factorial     = T(1);
    for (; m < m0; ++m)
    {
        h(2 * m + 0) = arma::sum(w_small % arma::pow(p_small, 2 * m + 0));
        h(2 * m + 1) = -arma::sum(w_small % arma::pow(p_small, 2 * m + 1));
        factorial *= T((2 * m) * (2 * m + 1));
        if (-h(2 * m + 1) / factorial < eps)
        {
            // Taylor expansion converges with the tolerance eps.
            ++m;
            break;
        }
    }

    //
    // Construct a Hankel matrix from the sequence h, and solve the linear
    // equation, H q = b, with b = -h(m:2m-1).
    //
    Matrix H(m, m);
    for (UIndex i = 0; i < m; ++i)
    {
        H.col(i) = h.subvec(i, i + m - 1);
    }
    Vector b(-h.subvec(m, 2 * m - 1));
    Vector q(arma::solve(H, b));
    //
    // Find the roots of the Prony polynomial,
    //
    // q(z) = \sum_{k=0}^{m-1} q_k z^{k}.
    //
    // The roots of q(z) can be obtained as the eigenvalues of the companion
    // matrix,
    //
    //     (0  0  ...  0 -p[0]  )
    //     (1  0  ...  0 -p[1]  )
    // C = (0  1  ...  0 -p[2]  )
    //     (.. .. ...  .. ..    )
    //     (0  0  ...  1 -p[m-1])
    //
    Matrix companion(m, m, arma::fill::zeros);
    companion.diag(-1).ones();
    companion.col(m - 1) = -q;
    Vector gamma(arma::real(arma::eig_gen(companion)));

    // --- Update exponents & weights
    auto keep = gs_orig.size() - m0;
    GaussianSum<T> gs(keep + m);
    gs.exponents_unsafe().head(m)    = -gamma;
    gs.exponents_unsafe().tail(keep) = gs_orig.exponents().tail(keep);
    //
    // Construct Vandermonde matrix from gamma
    //
    Matrix V(2 * m, m);
    for (UIndex i = 0; i < m; ++i)
    {
        const auto z = gamma(i);
        V(0, i)      = T(1);
        for (UIndex j = 1; j < V.n_rows; ++j)
        {
            V(j, i) = V(j - 1, i) * z; // z[i]**j
        }
    }
    //
    // Solve overdetermined Vandermonde system,
    //
    // V(0:2m-1,0:m-1) w(0:m-1) = h(0:2m-1)
    //
    // by the least square method.
    //
    gs.weights_unsafe().head(m)    = arma::solve(V, h.head(2 * m));
    gs.weights_unsafe().tail(keep) = gs_orig.weights().tail(keep);

    return gs;
}

} // namespace mxpfit

#endif /* MXPFIT_MOD_PRONY_REDUCTION_HPP */

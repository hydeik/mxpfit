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
/// \file aak_reduction.hpp
///
/// Sparse approximation of exponential sum using modified Prony method.
///
#ifndef MXPFIT_MODIFIED_PRONY_REDUCTION_HPP
#define MXPFIT_MODIFIED_PRONY_REDUCTION_HPP

#include <Eigen/Core>
#include <Eigen/QR>

#include <unsupported/Eigen/Polynomials>

///
/// ### ModifiedPronyReduction
///
/// \brief Find a truncated exponential sum function with smaller number of
///        terms by the modified balanced truncation method.
///
/// \tparam T  Scalar type of exponential sum function.
///
/// Let us consider an exponential sum function, in which the weights and
/// exponetns are strictly positive, i.e.,
///
/// \f[
///   f(t)=\sum_{j=1}^{n} w_{j}^{} e^{-a_{j}^{} t}, \quad
///   (a_{j} > 0, \, w_{j} > 0).
/// \f]
///
/// This class calculates truncated exponential \f$\hat{f}(t)\f$ sum such that
///
/// \f[
///   \hat{f}(t)=\sum_{j=1}^{k} \hat{w}_{j}^{}e^{-\hat{a}_{j}^{} t}, \quad
///   \left| f(t)-\hat{f}(t) \right| < \epsilon, \, (k < n)
/// \f]
///
/// where \f$\epsilon > 0\f$ is the prescribed accuracy. The weights
/// \f$\hat{w}_{j}\f$ and exponents \f$\hat{w}_{j}\f$ in the trucated sum are
/// all positive.
///
/// The modified Prony method proposed by Beylkin and Monzon are adopted. We
/// refer to the literature listed below for the detial about the method.
///
/// #### References
///
/// 1. G. Beylkin and L. Monz\'{o}n, "Approximation by exponential sums
///    revisited", Appl. Comput. Harmon. Anal. 28 (2010) 131-149.
///    [DOI: https://doi.org/10.1016/j.acha.2009.08.011]
/// 2. W. McLean, "Exponential sum approximations for \f$t^{-\beta}\f$",
///    arXiv:1606.00123 [math]
///
template <typename T>
class ModifiedPronyReduction
{
public:
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
    using ComplexScalar = std::complex<RealScalar>;
    using Index         = Eigen::Index;

    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ResultType = ExponentialSum<Scalar>;

    ///
    /// Compute truncated exponential sum \f$ \hat{f}(t) \f$
    ///
    /// \tparam DerivedF type of exponential sum inheriting ExponentialSumBase
    ///
    /// \param[in] orig original exponential sum function, \f$ f(t) \f$
    /// \param[in] n_target Index smaller than `orig.size()`, which denotes
    ///            number of terms to be target for reduction.
    /// \param[in] threshold  prescribed accuracy \f$0 < \epsilon \ll 1\f$
    ///
    /// \pre The weights \f$w_{j}\f$ and exponents \f${a_{j}}\f$ are strictly
    /// positive and \f$a_{j}\f$ must be sorted in ascending order.
    ///
    /// \remark The exponents \f$\hat{a}_{j}\f$ of truncated sum are obtained as
    /// a root of Prony polynomials, which might not be real and positive.
    /// Internaly, obtaind roots of polynomial is casted to be a real number,
    /// which might introduce significant errors in the approximation.
    ///
    /// \return An instance of ExponentialSum represents \f$\hat{f}(t)\f$
    ///
    template <typename DerivedF>
    ResultType compute(const ExponentialSumBase<DerivedF>& orig, Index n_target,
                       RealScalar threshold);
};

template <typename T>
typename ModifiedPronyReduction<T>::ResultType template <typename DerivedF>
ModifiedPronyReduction<T>::compute(const ExponentialSumBase<DerivedF>& orig,
                                   Index n_target, RealScalar eps)
{
    using Eigen::numext::abs;
    using Eigen::numext::real;
    assert(Index(0) <= n_target && n_target <= orig.size());
    assert(threshold > Real());

    if (n_target == Index())
    {
        ResultType ret(orig);
        return ret; // quick return
    }

    //
    // Compute a sequence
    //
    // \f[
    //   h_{j} = \sum_{k=1}^{n} w_{k} a_{k}^{j}
    // \f]
    //
    VectorType h(2 * n_target);
    const auto w_target = orig.weights().head(n_target);
    const auto a_target = orig.exponents().head(n_target);
    VectorType a_pow(a_target);

    h(0) = w_target.sum();
    h(1) = -(w_target * a_pow).sum();

    Index m        = 1;
    auto factorial = Real(1);
    for (; m < n_target; ++m)
    {
        a_pow *= a_target;
        h(2 * m + 0) = (w_target * a_pow).sum();
        a_pow *= a_target;
        h(2 * m + 1) = -(w_target * a_pow).sum();
        factorial *= Real(2 * m * (2 * m + 1));

        if (abs(h(2 * m + 1)) / factorial < eps)
        {
            // Taylor expansion converges with the tolerance eps.
            ++m;
            break;
        }
    }

    if (m == n_target)
    {
        // no further reduction
        ResultType ret(orig);
        return ret;
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
    MatrixType q(H.colPivHouseholderQr().solve(-h.segment(m, m)));

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
    Eigen::PolynomialSolver<Real, Eigen::Dynamic> poly_solver(q);

    // --- Update exponents & weights
    const Index keep = orig.size() - n_target;
    ResultType ret(keep + m);
    ret.exponents().head(m)    = poly_solver.roots().real();
    ret.exponents().tail(keep) = orig.exponents().tail(keep);

    //
    // Construct Vandermonde matrix from Prony roots
    //
    Matrix V(2 * m, m);
    for (UIndex i = 0; i < m; ++i)
    {
        const auto z = real(poly_solver.roots()(i));
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
    ret.weights().head(m)    = V.colPivHouseholderQr().solve(h.head(2 * m));
    ret.weights().tail(keep) = orig.weights().tail(keep);

    return ret;
}

#endif /* MXPFIT_MODIFIED_PRONY_REDUCTION_HPP */

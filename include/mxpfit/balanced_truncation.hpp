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
/// \file balanced_truncation.hpp
///
/// Balanced truncation method specialized for multi-exponential sum
///
#ifndef MXPFIT_BALANCED_TRUNCATION_HPP
#define MXPFIT_BALANCED_TRUNCATION_HPP
#include <cassert>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <mxpfit/exponential_sum.hpp>
#include <mxpfit/quasi_cauchy_rrd.hpp>
#include <mxpfit/self_adjoint_coneigensolver.hpp>

namespace mxpfit
{

///
/// ### BalancedTruncation
///
/// \brief Find a truncated exponential sum function with smaller number of
///        terms by the modified balanced truncation method.
///
/// \tparam T  Scalar type of exponential sum function.
///
/// For a given exponential sum function,
///
/// \f[
///   f(t)=\sum_{j=1}^{n} c_{j}^{} e^{-a_{j}^{} t}, \quad
///   (\mathrm{Re}(a_{j}) > 0),
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
/// where \f$k \leq n.\f$ Let \f$F(s)\f$ and \f$\hat{F}(s)\f$ be the Laplace
/// transform of \f$f(t)\f$ and \f$\hat{f}(t),\f$ respectively. \f$F(s)\f$ can
/// be evaluated analytically as
///
/// \f[
///  F(s)=\sum_{j=1}^{n}\frac{c_{j}^{}}{s+a_{j}^{}}
/// \f]
///
/// and similar to \f$\hat{F}(s).\f$ Now, the problem can be rewritten as
/// finding optimal rational sum approximation \f$\hat{F}(s)\f$ such that \f$
/// \left|F(s)-\hat{F}(s)\right| < \epsilon.\f$
///
/// This class computes the truncated rational sum approximation
/// \f$\hat{F}(s)\f$ by the modified balanced truncation method combined with
/// the first and accurate con-eigensolver of a quasi-Cauchy matrix.
///
///
/// #### References
///
/// 1. K. Xu and S. Jiang, "A Bootstrap Method for Sum-of-Poles Approximations",
///    J. Sci. Comput. **55** (2013) 16-39.
///    [DOI: https://doi.org/10.1007/s10915-012-9620-9]
/// 2. T. S. Haut and G. Beylkin, "FAST AND ACCURATE CON-EIGENVALUE ALGORITHM
///    FOR OPTIMAL RATIONAL APPROXIMATIONS", SIAM J. Matrix Anal. Appl. **33**
///    (2012) 1101-1125.
///    [DOI: https://doi.org/10.1137/110821901]
/// 3. W. H. A. Schilders, H. A. van der Vorst, and J. Rommes, "Model Order
///    Reduction: Theory, Research Aspects and Applications", Springer (2008).
///    [DOI: https://doi.org/10.1007/978-3-540-78841-6]
///

template <typename T>
class BalancedTruncation
{
public:
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
    using ComplexScalar = std::complex<RealScalar>;
    using Index         = Eigen::Index;

    using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using ResultType = ExponentialSum<Scalar>;

    template <typename DerivedF>
    ResultType compute(const ExponentialSumBase<DerivedF>& orig);

    RealScalar threshold() const
    {
        return m_threshold;
    }

    BalancedTruncation& setThreshold(const RealScalar& new_threshold)
    {
        m_threshold = new_threshold;
        return *this;
    }

private:
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
    };

    using EigenSolverType = typename Eigen::internal::conditional<
        IsComplex, Eigen::ComplexEigenSolver<MatrixType>,
        Eigen::SelfAdjointEigenSolver<MatrixType>>::type;
    using ConeigenSolverType = SelfAdjointConeigenSolver<Scalar>;

    RealScalar m_threshold;
};

template <typename T>
template <typename DerivedF>
typename BalancedTruncation<T>::ResultType
BalancedTruncation<T>::compute(const ExponentialSumBase<DerivedF>& fn)
{
    //--------------------------------------------------------------------------
    //
    // The controllability Gramian matrix of the system is defined as
    //
    //   C(i, j) = sqrt(w[i] * conj(w[j])) / (p[i] + p[j]).
    //
    // C is a quasi-Cauchy matrix. Then compute partial Cholesky factorization
    // of matrix `C`
    //
    //   C = (P * L) * D^2 * (P * L)^H,
    //
    // where
    //
    //   - L: (n, m) matrix
    //   - D: (m, m) real diagonal matrix
    //   - P: (n, n) permutation matrix
    //
    // and `m = rank(C)`
    //
    //--------------------------------------------------------------------------
    static const RealScalar eps = Eigen::NumTraits<RealScalar>::epsilon();

    const Index n0 = fn.size();
    VectorType b(n0);
    b.array() = fn.weights().sqrt();

    const RealScalar rrd_threshold = threshold() * eps * eps;
    SelfAdjointQuasiCauchyRRD<T> rrd;
    rrd.setThreshold(rrd_threshold);
    rrd.compute(b, fn.exponents());

    if (rrd.rank() == Index(0))
    {
        return ResultType();
    }

    //--------------------------------------------------------------------------
    //
    // Compute con-eigendecomposition of the controllability Gramian matrix
    //
    //   C = X D^2 X^H = U^C S U^T,
    //
    // where
    //
    //   - `X = P * L`: Cholesky factor (n, k)
    //   - `S`: (k, k) diagonal matrix. Diagonal elements `S(i,i)` are
    //          con-eigenvalues sorted in decreasing order.
    //   - `U`: (n, k) matrix. k-th column hold a con-eigenvector corresponding
    //          to k-th con-eigenvalue. The columns of `U` are orthogonal in the
    //          sense that `U^T * U = I`.
    //
    //---------------------------------------------------------------------------
    //
    // `diag` is overwritten by con-eigenvalues, and first k column of `matX` is
    // overwritten by con-eigenvectors `U`
    //
    ConeigenSolverType ceig;
    ceig.compute(rrd.matrixPL(), rrd.vectorD());
    //--------------------------------------------------------------------------
    //
    // Truncation
    //
    // Determines the order of reduced system, \f$ k \f$ from the error bound
    // computed from the Hankel singular values system. The Hankel singular
    // values are coincide with con-eigenvalues of the Gramian matrix.
    //
    // \f[
    //   \|\Sigma-\hat{\Sigma}\| \leq 2 \sum_{i=k+1}^{n} \sigma_{i}
    // \f]
    //
    //--------------------------------------------------------------------------
    auto sum_sigma       = RealScalar();
    const auto& sigma    = ceig.coneigenvalues();
    const auto sigma_tol = threshold() * sigma(0);
    Index n1             = sigma.size();
    while (n1)
    {
        sum_sigma += sigma(n1 - 1);
        if (2 * sum_sigma > sigma_tol)
        {
            break;
        }
        --n1;
    }

    if (n1 == Index())
    {
        return ResultType();
    }

    //--------------------------------------------------------------------------
    //
    // Apply transformation matrix
    //
    //  A1 = U.adjoint() * S * U.conjugate()
    //  b1 = U.adjoint() * b
    //
    //--------------------------------------------------------------------------

    MatrixType A1(n1, n1);
    VectorType b1(n1);
    auto U = ceig.coneigenvectors().leftCols(n1);
    A1.noalias() =
        U.adjoint() * fn.exponents().matrix().asDiagonal() * U.conjugate();
    b1.noalias() = U.adjoint() * b;

    //--------------------------------------------------------------------------
    //
    // Compute eigenvalue decomposition of the (k x k) matrix, A1. Since A1
    // real/complex symmetric matrix, the eigen decomposition has the form
    //
    //   A1 = X2 * D * X2.transpose(), (X2.transpose() * X2 = I).
    //
    //--------------------------------------------------------------------------
    EigenSolverType eig(A1, Eigen::ComputeEigenvectors);

    if (IsComplex)
    {
        //
        // Enforce X2.transpose() * X2 = I
        //
        using EigenVectorsType = typename Eigen::internal::remove_all<decltype(
            eig.eigenvectors())>::type;
        auto& X2 = *const_cast<EigenVectorsType*>(&eig.eigenvectors());
        for (Index j = 0; j < X2.cols(); ++j)
        {
            auto xj          = X2.col(j);
            const auto t     = (xj.transpose() * xj).value();
            const auto scale = RealScalar(1) / std::sqrt(t);
            xj *= scale;
        }
    }

    //
    // Apply the state space transformation by X2,
    //
    // A2 = X2.transpose() * A1 * X2 = D
    // b2 = X2.transpose() * b1
    // c2 = b1 * X2 = b2.transpose()
    //
    // Finally parameters for truncated exponential sum can be obtained as
    //
    // p' = D.diagonal()
    // w' = c2 * b2 = square(b2)
    //
    auto b2      = b.head(n1);
    b2.noalias() = eig.eigenvectors().transpose() * b1;
    return ResultType(eig.eigenvalues(), b2.array().square());
}

} // namespace: mxpfit

#endif /* MXPFIT_BALANCED_TRUNCATION_HPP */

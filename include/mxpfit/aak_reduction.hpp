/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2017  Hidekazu Ikeno
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
/// Sparse approximation of exponential sum using AAK theory.
///

#ifndef MXPFIT_AAK_REDUCTION_HPP
#define MXPFIT_AAK_REDUCTION_HPP

#include <algorithm>

#include <mxpfit/exponential_sum.hpp>
#include <mxpfit/quasi_cauchy_rrd.hpp>
#include <mxpfit/self_adjoint_coneigensolver.hpp>

namespace mxpfit
{

namespace detail
{
} // namespace: detail

///
/// ### AAKReduction
///
/// \brief Find a truncated exponential sum function with smaller number of
///        terms using AAK theory.
///
/// \tparam T  Scalar type of exponential sum function.
///
/// For a given exponential sum function
/// \f[
///   f(t)=\sum_{j=1}^{N} c_{j}^{} e^{-p_{j}^{} t}, \quad
///   (\mathrm{Re}(a_{j}) > 0).
/// \f]
///
/// and prescribed accuracy \f$\epsilon > 0,\f$ this class calculates truncated
/// exponential \f$\tilde{f}(t)\f$ sum such that
///
/// \f[
///   \tilde{f}(t)=\sum_{j=1}^{M} \tilde{c}_{j}^{}e^{-\tilde{p}_{j}^{} t}, \quad
///   \left| f(t)-\tilde{f}(t) \right| < \epsilon,
/// \f]
///
///
/// The signal \f$\boldsymbol{f}=(f_{k})_{k=0}^{\infty}\f$ sampled on
/// \f$k=0,1,\dots\f$ are expressed as
///
/// \f[
///   f_{k}:= f(k) = \sum_{j=1}^{N} c_{j} z_{j}^{k}
/// \f]
///
/// where \f$z_{j}=e^{-p_{j}}\in \mathbb{D}:=\{z \in \mathbb{C}:0<|z|<1\}.\f$
/// The problem to find truncated exponential sum can be recast to find a new
/// signal \f$\tilde{\boldsymbol{f}},\f$ a sparse approximation of signal
/// \f$\boldsymbol{f},\f$ such that
///
/// \f[
///   \tilde{f}_{k}=\sum_{j=1}^{M}\tilde{c}_{j}^{} \tilde{z}_{j}^{k}, \quad
///   \|\boldsymbol{f}-\tilde{\boldsymbol{f}}\|<\epsilon. \f$
/// \f]
///
/// Then, the exponents of reduced exponential sum
/// \f$\tilde{f}(t)\f$ are obtained as \f$\tilde{p}_{j}=\tilde{z}_{j}.\f$
///
///
/// #### References
///
/// 1. Gerlind Plonka and Vlada Pototskaia, "Application of the AAK theory for
///    sparse approximation of exponential sums", arXiv 1609.09603v1 (2016).
///    [URL: https://arxiv.org/abs/1609.09603v1]
/// 2. Gregory Beylkin and Lucas Monzón, "On approximation of functions by
///    exponential sums", Appl. Comput. Harmon. Anal. **19** (2005) 17-48.
///    [DOI: https://doi.org/10.1016/j.acha.2005.01.003]
/// 3. T. S. Haut and G. Beylkin, "FAST AND ACCURATE CON-EIGENVALUE ALGORITHM
///    FOR OPTIMAL RATIONAL APPROXIMATIONS", SIAM J. Matrix Anal. Appl. **33**
///    (2012) 1101-1125.
///    [DOI: https://doi.org/10.1137/110821901]
/// 4. Terry Haut, Gregory Beylkin, and Lucas Monzón, "Solving Burgers’
///    equation using optimal rational approximations", Appl. Comput. Harmon.
///    Anal. **34** (2013) 83-95.
///    [DOI: https://doi.org/10.1016/j.acha.2012.03.004]
///
template <typename T>
class AAKReduction
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
    /// \param[in] threshold  prescribed accuracy \f$0 < \epsilon \ll 1\f$
    ///
    /// \return An instance of ExponentialSum represents \f$\hat{f}(t)\f$
    ///
    template <typename DerivedF>
    ResultType compute(const ExponentialSumBase<DerivedF>& orig,
                       RealScalar threshold);

private:
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
    };

    using ConeigenSolverType = SelfAdjointConeigenSolver<Scalar>;
    using RRDType =
        QuasiCauchyRRD<Scalar, QuasiCauchyRRDFunctorLogPole<Scalar>>;
};

template <typename T>
template <typename DerivedF>
typename AAKReduction<T>::ResultType
AAKReduction<T>::compute(const ExponentialSumBase<DerivedF>& orig,
                         RealScalar threshold)
{
    static const auto eps = Eigen::NumTraits<RealScalar>::epsilon();
    using Eigen::numext::abs;
    using Eigen::numext::real;

    const Index n0 = orig.size();
    std::cout << "*** order of original function: " << n0 << std::endl;

    //-------------------------------------------------------------------------
    // Solve con-eigenvalue problem of the Cauchy-like matrix \f$C\f$ with
    // elements
    //
    // C(i,j)= sqrt(c[i]) * sqrt(conj(c[j])) / (1 - z[i] * conj(z[j]))
    //
    // where z[i] = exp(-p[i]).
    //-------------------------------------------------------------------------
    //
    // Rewrite the matrix element C(i,j) as
    //
    //                 a[i] * b[j]
    //   C(i,j) = ----------------------
    //             exp(x[i]) - exp(y[j])
    //
    // where,
    //
    //   a[i] = sqrt(c[i]) / exp(-p[i])
    //   b[j] = sqrt(conj(c[j]))
    //   x[i] = p[i]
    //   y[j] = -conj(p[j])
    //
    VectorType b(orig.weights().conjugate().sqrt());
    VectorType a(b.array().conjugate() * orig.exponents().exp());
    VectorType x(orig.exponents());
    VectorType y(-orig.exponents().conjugate());
    //
    // Compute partial Cholesky decomposition of matrix C, such that,
    //
    //   C = (P * L) * D^2 * (P * L)^H,
    //
    // where
    //
    //   - L: (n, m) matrix
    //   - D: (m, m) real diagonal matrix
    //   - P: (n, n) permutation matrix
    //
    //
    RRDType rrd;

    rrd.setThreshold(threshold * threshold * eps);
    rrd.compute(a, b, x, y);

    std::cout << "*** rank after quasi-Cauchy RRD: " << rrd.rank() << std::endl;

    //
    // Compute con-eigendecomposition
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
    ConeigenSolverType ceig;
    ceig.compute(rrd.matrixPL(), rrd.vectorD());
    //
    // Determines the order of reduced system, \f$ M \f$ from the Hankel
    // singular values, which are corresponding to the con-eigenvalues of matrix
    // `C`.
    //
    const auto& sigma = ceig.coneigenvalues();
    auto pos          = std::upper_bound(
        sigma.data(), sigma.data() + sigma.size(), threshold,
        [&](RealScalar lhs, RealScalar rhs) { return lhs >= rhs; });
    const Index n1 = static_cast<Index>(pos - sigma.data());
    std::cout << "*** order after reduction: " << n1 << std::endl;

    if (n1 == Index())
    {
        return ResultType();
    }

    std::cout << "    sigma = " << sigma(n1 - 1) << std::endl;

    //-------------------------------------------------------------------------
    // Find the `n1` roots of rational function
    //
    //             n1  sqrt(alpha[i]) * conj(u[i])
    //   v(eta) = Sum  ---------------------------
    //            i=1     1 - conj(z[i]) * eta
    //
    // where eta is on the unit disk.
    //-------------------------------------------------------------------------

    //
    // Barycentric form of Lagrange interpolation
    //
    // VectorType& tau = x;
    // std::sort(tau.data(), tau.data() + tau.size(),
    //           [&](const Scalar& lhs, const Scalar& rhs) {
    //               return real(lhs) > real(rhs);
    //           });
    // VectorType& s = y;

    ResultType ret;

    return ret;
}

} // namespace: mxpfit

#endif /* MXPFIT_AAK_REDUCTION_HPP */

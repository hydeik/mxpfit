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
/// Find an optimal exponential-sum function by the balanced truncation method.
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

    // Clear memory used for internal working space
    void clear()
    {
    }

private:
    enum
    {
        IsComplex      = Eigen::NumTraits<Scalar>::IsComplex,
        Alignment      = Eigen::internal::traits<VectorType>::Alignment,
        PacketSize     = Eigen::internal::packet_traits<Scalar>::size,
        RealPacketSize = Eigen::internal::packet_traits<RealScalar>::size
    };

    // using MappedMatrix     = Eigen::Map<Matrix, Alignment>;
    // using MappedVector     = Eigen::Map<Vector, Alignment>;
    // using MappedRealVector = Eigen::Map<RealVector, Alignment>;

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
    // Define a self-adjoint quasi-Cauchy matrix
    //
    //   C(i, j) = sqrt(w[i] * conj(w[j])) / (p[i] + p[j]),
    //
    // then compute partial Cholesky factorization of matrix `C`
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
    // computed from the singular values of Hankel system
    //
    // \f[
    //   \|\Sigma-\hat{\Sigma}\| \leq 2 \sum_{i=k+1}^{n} \sigma_{i}
    // \f]
    //
    //--------------------------------------------------------------------------
    auto sum_sigma    = RealScalar();
    const auto& sigma = ceig.coneigenvalues();
    Index n1          = sigma.size();
    while (n1)
    {
        sum_sigma += sigma(n1 - 1);
        if (2 * sum_sigma > threshold())
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
        using EigenVectorsType = typename Eigen::internal::remove_all<decltype(
            eig.eigenvectors())>::type;
        // Enforce X2.transpose() * X2 = I
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

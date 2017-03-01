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
/// \file partial_lanczos_bidiagonalization.hpp
///
#ifndef MXPFIT_PARTIAL_LANCZOS_BIDIAGONALIZATION_HPP
#define MXPFIT_PARTIAL_LANCZOS_BIDIAGONALIZATION_HPP

#include <Eigen/Core>

namespace mxpfit
{

///
/// Partial Lanczos bidiagonalization with full reorthogonalization
///
template <typename MatrixA, typename VectorAlpha, typename VectorBeta,
          typename MatrixP, typename MatrixQ, typename VectorWork>
Eigen::Index partialLanczosBidiagonalization(
    const MatrixA& A, VectorAlpha& alpha, VectorBeta& beta, MatrixP& P,
    MatrixQ& Q, typename VectorAlpha::RealScalar& tol_error, VectorWork& work)
{
    using RealScalar = typename VectorAlpha::RealScalar;
    using Index      = Eigen::Index;

    const Index m        = P.rows();
    const Index k        = P.cols();
    const Index n        = Q.rows();
    const Index max_rank = std::min({m, n, k});

    assert(Q.cols() == k);
    assert(alpha.size() >= max_rank && beta.size() >= max_rank);
    assert(work.size() >= max_rank);
    assert(RealScalar() < tol_error && tol_error < RealScalar(1));

    auto q0 = Q.col(0);
    q0.setRandom().normalize();
    auto p0 = P.col(0);
    p0      = A * q0;
    auto a1 = p0.norm();
    if (a1 > RealScalar())
    {
        p0 *= RealScalar(1) / a1;
    }
    alpha(0) = a1;

    const RealScalar tol2 = tol_error * tol_error;
    RealScalar fnorm_A    = a1 * a1; // Estimation of |A|_F
    RealScalar rel_err    = RealScalar();
    Index irank           = 0;

    while (++irank < max_rank)
    {
        auto p1 = P.col(irank - 1);
        auto p2 = P.col(irank);
        auto q1 = Q.col(irank - 1);
        auto q2 = Q.col(irank);
        //
        // --- Recursion for right Lanczos vector
        //
        q2 = A.adjoint() * p1 - alpha(irank - 1) * q1;
        // Reorthogonalization
        auto tmp   = work.head(irank);
        auto viewQ = Q.leftCols(irank);
        tmp        = viewQ.adjoint() * q2;
        q2 -= viewQ * tmp;
        auto b1 = q2.norm();
        if (b1 > RealScalar())
        {
            q2 *= RealScalar(1) / b1;
        }
        beta(irank - 1) = b1;
        //
        // --- Recursion for left Lanczos vector
        //
        // p2 <-- A * q2 - beta(i) * p1
        p2 = A * q2 - beta(irank - 1) * p1;
        // Reorthogonalization
        auto viewP = P.leftCols(irank);
        tmp        = viewP.adjoint() * p2;
        p2 -= viewP * tmp;

        auto a2 = p2.norm();
        if (a2 > RealScalar())
        {
            p2 *= RealScalar(1) / a2;
            alpha(irank) = a2;
        }
        //
        // Update frobenius norm of matrix A via
        //
        // ||A||_{F}^{2} = \sum_{K=1}^{rank(A)-1}
        //       (\alpha_{K}^{2} + \beta_{K}^{2}) + \alpha_{rank(A)}}
        //
        auto t = a2 * a2 + b1 * b1;
        fnorm_A += t;
        if (t <= tol2 * fnorm_A)
        {
            rel_err = t / fnorm_A;
            break; // converged
        }
    }

    tol_error = rel_err;
    return irank;
}

} // namespace: mxpfit

#endif /* MXPFIT_PARTIAL_LANCZOS_BIDIAGONALIZATION_HPP*/

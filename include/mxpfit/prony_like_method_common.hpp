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
/// \file prony_like_method_common.hpp
///
#ifndef MXPFIT_PRONY_LIKE_METHOD_COMMON_HPP
#define MXPFIT_PRONY_LIKE_METHOD_COMMON_HPP

#include <cassert>

#include <Eigen/Core>
#include <Eigen/QR>

#include <mxpfit/matrix_free_gemv.hpp>
#include <mxpfit/vandermonde_least_squares.hpp>

namespace mxpfit
{

namespace detail
{

/// \internal
///
/// Solve overdetermined linear system
///
/// \f[
///    V\boldsymbol{x} = \boldsymbol{b}
/// \f]
///
/// to compute weights of the exponential sum approximation, where \f$V\f$ is a
/// column Vandermonde matrix constructed from roots of a Prony polynomial.
///
/// \param[in] prony_roots  roots of a Prony polynomial
/// \param[in] rhs   a sequence of function/data values on uniform grid
/// \param[out] dst  the solution of overdetermined Vandermonde system
/// \param[in] eps      a prescribed accuracy required for the least-squares
/// \param[in] max_iter maximum number of iterations for CGLS linear solver
///
template <typename VecNodes, typename VecRHS, typename VecDest, typename RealT>
void solve_overdetermined_vandermonde(
    const Eigen::DenseBase<VecNodes>& prony_roots,
    const Eigen::MatrixBase<VecRHS>& rhs, Eigen::DenseBase<VecDest>& dst,
    RealT eps, Eigen::Index max_iter)
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecNodes);
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecRHS);
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecDest);

    assert(prony_roots.size() == dst.size());

    using Scalar          = typename VecNodes::Scalar; // usually complex type
    using VandermondeGEMV = MatrixFreeGEMV<VandermondeMatrix<Scalar>>;

    // --- Solve with CGLS method.
    VandermondeMatrix<Scalar> matV(rhs.size(), prony_roots);
    VandermondeGEMV opV(matV);
    VandermondeLeastSquaresSolver<Scalar> solver(opV);
    solver.setTolerance(eps);
    solver.setMaxIterations(max_iter);

    //
    // We need cast the scalar type of RHS vector as the VecRHS::Scalar might be
    // different from VecNodes::Scalar. We should be consider the following
    // case:
    //
    // VecNodes::Scalar -> complex type
    // VecRHS::Scalar   -> real type
    //
    dst = solver.solve(rhs.template cast<Scalar>());

    if (solver.info() == Eigen::NoConvergence)
    {
        //
        // CGLS did not converge.
        // Fall back to least-squares with dense QR factorization.
        //
        auto denseV = matV.toDenseMatrix();
        dst = denseV.colPivHouseholderQr().solve(rhs.template cast<Scalar>());
    }

    return;
}

} // namespace detail
} // namespace mxpfit

#endif /* MXPFIT_PRONY_LIKE_METHOD_COMMON_HPP */

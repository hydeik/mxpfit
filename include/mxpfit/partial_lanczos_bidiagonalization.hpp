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
/// ### PartialLanczosBidiagonalization
///
/// \brief Low-rank approximation of a matrix by the Lanczos bidiagonalization
/// with full reorthogonalization
///
/// \tparam MatrixT Matrix type to be decomposed. We expect that `MatrixT`
/// inherits Eigen::EigenBase class.
///
/// For a given \f$m \times n\f$ matrix \f$A\f$ and prescribed accuracy
/// \f$\epsilon,\f$ this class computes the approximate decomposition such that
///
/// \f[
///   \|A - P_{k}^{} B_{k} Q_{k}^{\ast} \|_{F} < \epsilon.
/// \f]
///
/// where \f$P_{k}\f$ is a \f$ m \times k \f$ matrix with orthonormal columns,
/// and \f$ Q_{k} \f$ is a \f$m \times k\f$ matrix with orthonormal columns.
/// \f$B_{k}\f$ is a real bidiagonal matrix
///
/// #### References
///
/// 1. D. Potts and M. Tasche, "Fast ESPRIT algorithms based on partial singular
///    value decompositions", Appl. Numer. Math. **88** (2015) 31-45.
///    [DOI: http://doi.org/10.1016/j.apnum.2014.10.003]
///
template <typename MatrixT>
class PartialLanczosBidiagonalization
{
public:
    using MatrixType    = MatrixT;
    using Scalar        = typename MatrixType::Scalar;
    using RealScalar    = typename MatrixType::RealScalar;
    using Index         = Eigen::Index;
    using Vector        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RealVector    = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using Matrix        = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealVectorRef = Eigen::Ref<RealVector>;
    using MatrixRef     = Eigen::Ref<Matrix>;

    PartialLanczosBidiagonalization()
        : m_matP(),
          m_matQ(),
          m_alpha(),
          m_beta(),
          m_rank(),
          m_tolerance(Eigen::NumTraits<RealScalar>::epsilon()),
          m_error(),
          m_is_initialized()

    {
    }

    PartialLanczosBidiagonalization(Index nrows, Index ncols, Index nsteps)
        : m_matP(nrows, nsteps),
          m_matQ(ncols, nsteps),
          m_alpha(nsteps),
          m_beta(nsteps),
          m_rank(),
          m_tolerance(Eigen::NumTraits<RealScalar>::epsilon()),
          m_error(),
          m_is_initialized()
    {
    }

    PartialLanczosBidiagonalization(const MatrixType& mat, Index nsteps)
        : m_matP(mat.rows(), nsteps),
          m_matQ(mat.cols(), nsteps),
          m_alpha(nsteps),
          m_beta(nsteps),
          m_rank(),
          m_tolerance(Eigen::NumTraits<RealScalar>::epsilon()),
          m_error(),
          m_is_initialized()
    {
        compute(mat, nsteps);
    }

    void compute(const MatrixType& matA, Index nsteps);

    void setTolerance(RealScalar tolerance) noexcept
    {
        m_tolerance = tolerance;
    }

    RealScalar tolerance() const noexcept
    {
        return m_tolerance;
    }

    Index rank() const noexcept
    {
        assert(m_is_initialized &&
               "PartialLanczosBidiagonalization is not initialized.");
        return m_rank;
    }

    RealScalar error() const noexcept
    {
        assert(m_is_initialized &&
               "PartialLanczosBidiagonalization is not initialized.");
        return m_error;
    }

    const Matrix& matrixP() const noexcept
    {
        assert(m_is_initialized &&
               "PartialLanczosBidiagonalization is not initialized.");
        return m_matP;
    }

    const Matrix& matrixQ() const noexcept
    {
        assert(m_is_initialized &&
               "PartialLanczosBidiagonalization is not initialized.");
        return m_matQ;
    }

    const RealVector& diagonalAlpha() const noexcept
    {
        assert(m_is_initialized &&
               "PartialLanczosBidiagonalization is not initialized.");
        return m_alpha;
    }

    const RealVector& superdiagonalBeta() const noexcept
    {
        assert(m_is_initialized &&
               "PartialLanczosBidiagonalization is not initialized.");
        return m_beta;
    }

    Matrix reconstructedMatrix() const
    {
        assert(m_is_initialized &&
               "PartialLanczosBidiagonalization is not initialized.");

        if (m_rank == Index())
        {
            return Matrix();
        }

        auto viewP = m_matP.leftCols(m_rank);
        auto viewQ = m_matQ.leftCols(m_rank);
        Matrix matB(Matrix::Zero(m_rank, m_rank));
        matB.diagonal()  = m_alpha.head(m_rank);
        matB.diagonal(1) = m_beta.head(m_rank - 1);

        return viewP * matB * viewQ.adjoint();
    }

protected:
    Matrix m_matP;
    Matrix m_matQ;
    RealVector m_alpha;
    RealVector m_beta;

    Index m_rank;
    RealScalar m_tolerance;
    RealScalar m_error;
    bool m_is_initialized;
};

template <typename MatrixT>
void PartialLanczosBidiagonalization<MatrixT>::compute(const MatrixType& matA,
                                                       Index nsteps)
{
    assert(Index() <= nsteps && nsteps <= matA.rows() && nsteps <= matA.cols());

    m_matP.resize(matA.rows(), nsteps);
    m_matQ.resize(matA.cols(), nsteps);
    m_alpha.resize(nsteps);
    m_beta.resize(nsteps - 1);

    Vector workspace(nsteps);

    auto q0 = m_matQ.col(0);
    // Set q0 as unit vector (1,0,0,....)^T plus small perturbation
    q0.setRandom();
    q0(0) = RealScalar(1) /
            Eigen::numext::sqrt(Eigen::NumTraits<RealScalar>::epsilon());
    q0.normalize(); // make q0 normalized
    auto p0       = m_matP.col(0);
    p0            = matA * q0;
    const auto a1 = p0.norm();
    if (a1 > RealScalar())
    {
        p0 *= RealScalar(1) / a1;
    }
    m_alpha(0) = a1;

    const RealScalar tol2 = m_tolerance * m_tolerance;
    RealScalar fnorm_A    = a1 * a1; // Estimation of |A|_F
    m_error               = RealScalar();
    Index irank           = 0;

    while (++irank < nsteps)
    {
        auto p1 = m_matP.col(irank - 1);
        auto p2 = m_matP.col(irank);
        auto q1 = m_matQ.col(irank - 1);
        auto q2 = m_matQ.col(irank);
        //
        // --- Recursion for right Lanczos vector
        //
        q2 = matA.adjoint() * p1 - m_alpha(irank - 1) * q1;
        // Reorthogonalization
        auto tmp   = workspace.head(irank);
        auto viewQ = m_matQ.leftCols(irank);
        tmp        = viewQ.adjoint() * q2;
        q2 -= viewQ * tmp;
        auto b1 = q2.norm();
        if (b1 > RealScalar())
        {
            q2 *= RealScalar(1) / b1;
        }
        m_beta(irank - 1) = b1;
        //
        // --- Recursion for left Lanczos vector
        //
        // p2 <-- A * q2 - beta(i) * p1
        p2 = matA * q2 - m_beta(irank - 1) * p1;
        // Reorthogonalization
        auto viewP = m_matP.leftCols(irank);
        tmp        = viewP.adjoint() * p2;
        p2 -= viewP * tmp;

        auto a2 = p2.norm();
        if (a2 > RealScalar())
        {
            p2 *= RealScalar(1) / a2;
        }
        m_alpha(irank) = a2;
        //
        // Update frobenius norm of matrix A via
        //
        // ||A||_{F}^{2} = \sum_{K=1}^{rank(A)-1}
        //       (\alpha_{K}^{2} + \beta_{K}^{2}) + \alpha_{rank(A)}}
        //
        auto t = a2 * a2 + b1 * b1;
        fnorm_A += t;
        m_error = t / fnorm_A;
        if (m_error <= tol2)
        {
            break; // converged
        }
    }

    m_rank           = irank;
    m_is_initialized = true;
}

} // namespace mxpfit

#endif /* MXPFIT_PARTIAL_LANCZOS_BIDIAGONALIZATION_HPP*/

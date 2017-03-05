// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-
#ifndef MXPFIT_JACOBI_SVD_HPP
#define MXPFIT_JACOBI_SVD_HPP

#include <cassert>
#ifdef DEBUG
#include <iomanip>
#include <iostream>
#endif /* DEBUG */

#include <Eigen/Core>

namespace mxpfit
{
namespace detail
{
//
// Make Jacobi rotation matrix \c J of the form
//
//  J = [ cs conj(sn)]
//      [-sn conj(cs)]
//
// that diagonalize the 2x2 matirix
//
// B = [     a  c]
//     [conj(c) b]
//
// as D = J.t() * B * J
//
// J.t() = [ conj(cs)    -conj(sn)] = J.inv()
//         [      sn           cs ]
//
// J * [a11 a12] = [ cs*a11 + conj(sn)*a21   cs*a12 + conj(sn)*a22]
//     [a21 a22]   [-sn*a11 + conj(cs)*a21  -sn*a12 + conj(cs)*a22]
//
// [a11 a12] * J = [     cs *a11 -      sn *a12        cs *a21 -      sn *a22]
// [a21 a22]       [conj(sn)*a11 +E conj(cs)*a12  -conj(sn)*a21 + conj(cs)*a22]
//
template <typename RealT, typename ScalarT>
bool make_jacobi_rotation(RealT a, RealT b, ScalarT c, RealT& cs, ScalarT& sn)
{
    using Eigen::numext::abs;
    using Eigen::numext::abs2;
    using Eigen::numext::conj;
    using Eigen::numext::sqrt;

    constexpr const RealT one = RealT(1);

    if (c == ScalarT())
    {
        cs = one;
        sn = ScalarT();
        return false;
    }

    auto zeta = (a - b) / (RealT(2) * abs(c));
    auto w    = sqrt(abs2(zeta) + one);
    auto t    = (zeta > RealT() ? one / (zeta + w) : one / (zeta - w));

    cs = one / sqrt(abs2(t) + one);
    sn = -t * cs * (conj(c) / abs(c));

    return true;
}

template <typename T>
struct one_sided_jacobi_helper
{
    using Index      = Eigen::Index;
    using Scalar     = T;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    //
    // Sum of the square of the absolute value of the elments in the i-th column
    // of the matrix G, i.e. \f$ \sum_{k=0}^{n-1} |G_{ki}|^{2}. \f$
    //
    template <typename MatG>
    static RealScalar submat_diag(const Eigen::MatrixBase<MatG>& G, Index i)
    {
        return G.col(i).squaredNorm();
    }

    template <typename MatG>
    static Scalar submat_offdiag(const Eigen::MatrixBase<MatG>& G, Index i,
                                 Index j)
    {
        return G.col(i).dot(G.col(j));
    }

    template <typename MatU>
    static void update_vecs(Eigen::MatrixBase<MatU>& U, Index i, Index j,
                            RealScalar cs, const Scalar& sn)
    {
        using Eigen::numext::conj;
        for (Index k = 0; k < U.rows(); ++k)
        {
            const auto t1 = U(k, i);
            const auto t2 = U(k, j);
            // Apply Jacobi rotation matrix on the right
            U(k, i) = cs * t1 - sn * t2;
            U(k, j) = conj(sn) * t1 + cs * t2;
        }
    }
};

} // namespace: detail

//
// Compute the singular value decomposition (SVD) by one-sided Jacobi algorithm.
//
// This function compute the singular value decomposition (SVD) using modified
// one-sided Jacobi algorithm.
//
// #### References
//  - James Demmel and Kresimir Veselic, "Jacobi's method is more accurate than
//    QR", LAPACK working note 15 (lawn15) (1989), Algorithm 4.1.
//
// @A     On entry, an ``$m \times n$`` matrix to be decomposed.
//        On exit, columns of  `A` holds left singular vectors.
// @sigma singular values of matrix `A`.
// @V     right singular vectors.
// @tol   small positive real number that determines stopping criterion of
//        Jacobi algorithm.
//
template <typename MatA, typename VecS, typename MatV, typename RealT>
void one_sided_jacobi_svd(Eigen::MatrixBase<MatA>& A,
                          Eigen::MatrixBase<VecS>& sigma,
                          Eigen::MatrixBase<MatV>& V, RealT tol)
{
    using Index      = Eigen::Index;
    using Scalar     = typename MatA::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using jacobi_aux = detail::one_sided_jacobi_helper<Scalar>;

    using Eigen::numext::abs;
    using Eigen::numext::sqrt;
    // constexpr const Index max_sweep = 50;

    auto n                = A.cols();
    const Index max_sweep = n * (n - 1) / 2;
    assert(V.rows() == A.cols());
    //
    // Initialize right singular vectors
    //
    V.setIdentity();

    Scalar sn;
    RealScalar cs;
    RealScalar max_resid = tol + RealScalar(1);
    Index nsweep         = max_sweep + 1;
    while (--nsweep && max_resid > tol)
    {
        max_resid = RealScalar();
        for (Index j = 1; j < n; ++j)
        {
            auto b = jacobi_aux::submat_diag(A, j);
            for (Index i = 0; i < j; ++i)
            {
                //
                // For all pairs i < j, compute the 2x2 submatrix of A.t() * A
                // constructed from i-th and j-th columns, such as
                //
                //   M = [     a   c]
                //       [conj(c)  b]
                //
                // where
                //
                //   a = \sum_{k=0}^{n-1} |A(k,i)|^{2} (computed in outer loop)
                //   b = \sum_{k=0}^{n-1} |A(k,j)|^{2}
                //   c = \sum_{k=0}^{n-1} conj(A(k,i)) * A(k,j)
                //
                auto a    = jacobi_aux::submat_diag(A, i);
                auto c    = jacobi_aux::submat_offdiag(A, i, j);
                max_resid = std::max(max_resid, abs(c) / sqrt(a * b));
                //
                // Compute the Jacobi rotation matrix which diagonalize M.
                //
                if (detail::make_jacobi_rotation(a, b, c, cs, sn))
                {
                    // If the return vlaue of make_jacobi_rotation is false, no
                    // rotation is made.
                    //
                    // Update columns i and j of A
                    //
                    jacobi_aux::update_vecs(A, i, j, cs, sn);
                    //
                    // Update right singular vector V
                    //
                    jacobi_aux::update_vecs(V, i, j, cs, sn);
                }
            }
        }
#ifdef DEBUG
        std::cout << "(one_sided_jacobi_svd) iter " << std::setw(8)
                  << max_sweep - nsweep << ": max residual = " << max_resid
                  << '\n';
#endif /* DEBUG */
    }
    //
    // Set singular values and left singular vectors.
    //
    for (Index j = 0; j < A.cols(); ++j)
    {
        // Singular values are the norms of the columns of the final A
        sigma(j) = A.col(j).norm();
        // Left singular vectors are the normalized columns of the final A
        A.col(j) /= sigma(j);
    }
    //
    // Sort singular values in descending order. The corresponding singular
    // vectors are also rearranged.
    //
    for (Index i = 0; i < n - 1; ++i)
    {
        //
        // Find the index of the maximum value of a sub-array, sigma(i:n-1).
        //
        Index imax = i;
        for (Index k = i + 1; k < n; ++k)
        {
            if (sigma(k) > sigma(imax))
            {
                imax = k;
            }
        }

        if (imax != i)
        {
            // Move the largest singular value to the beggining of the
            // sub-array, and corresponsing singular vectors by swapping columns
            // of A and V.
            std::swap(sigma(i), sigma(imax));
            A.col(i).swap(A.col(imax));
            V.col(i).swap(V.col(imax));
        }
    }
}

} // namespace: mxpfit

#endif /* MXPFIT_JACOBI_SVD_HPP */

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
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <mxpfit/exponential_sum.hpp>
#include <mxpfit/matrix_free_gemv.hpp>
#include <mxpfit/vandermonde_least_squares.hpp>

namespace mxpfit
{

namespace detail
{

/// \internal
/// Select roots of Prony polynomials on the unit disk
template <typename T>
struct prony_roots_on_unit_disk
{
    using Scalar        = T;
    using RealScalar    = typename Eigen::NumTraits<Scalar>::Real;
    using ComplexScalar = std::complex<RealScalar>;
    using Index         = Eigen::Index;

    using Vector        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RealVector    = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
    using ComplexVector = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1>;

    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex,
    };

    /// \param z  array of roots to be filtered
    /// \param tolerance  small real number to acceptable tolerance for the
    //                    mangnitude of roots.
    static ComplexVector compute(const Eigen::Ref<const ComplexVector>& z,
                                 RealScalar tolerance)
    {
        using Eigen::numext::abs;
        using Eigen::numext::imag;
        static const auto eps     = Eigen::NumTraits<RealScalar>::epsilon();
        constexpr const auto zero = RealScalar();
        constexpr const auto one  = RealScalar(1);

        // Count the number of roots insize unit disk
        Index count = 0;
        for (Index i = 0; i < z.size(); ++i)
        {
            const auto abs_zi = abs(z(i));
            if (abs_zi <= one + tolerance)
            {
                if (IsComplex)
                {
                    ++count;
                }
                else
                {
                    // Discard negative real z(i).
                    const auto xi = real(z(i));
                    const auto yi = imag(z(i));
                    if (!(abs(yi) < eps && xi < zero))
                    {
                        ++count;
                    }
                }
            }
        }

        ComplexVector ret(count);
        Index n = 0;
        for (Index i = 0; i < z.size(); ++i)
        {
            const auto abs_zi = abs(z(i));
            if (abs_zi <= one + tolerance)
            {
                if (IsComplex)
                {
                    ret(n) = z(i);
                    ++n;
                }
                else
                {
                    // Discard negative real z(i).
                    const auto xi = real(z(i));
                    const auto yi = imag(z(i));
                    if (!(abs(yi) < eps && xi < zero))
                    {
                        ret(n) = z(i);
                        ++n;
                    }
                }
            }
        }

        return ret;
    }
};

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

    // if (solver.info() == Eigen::NoConvergence)
    // {
    //     //
    //     // CGLS did not converge.
    //     // Fall back to least-squares with dense QR factorization.
    //     //
    //     auto denseV = matV.toDenseMatrix();
    //     dst = denseV.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
    //               .solve(rhs.template cast<Scalar>());
    // }

    return;
}

/// \internal
///
/// ### gen_prony_like_method_result
///
/// Generate and manipulate a result (an exponential sum) of Prony-like method.
///

template <typename T>
struct gen_prony_like_method_result;

template <typename T>
struct gen_prony_like_method_result
{
    using Real       = T;
    using Complex    = std::complex<Real>;
    using ResultType = ExponentialSum<Complex, Complex>;

    template <typename ArrayZ, typename ArrayW>
    static ResultType
    create(const Eigen::ArrayBase<ArrayZ>& z, // exp(-exponents)
           const Eigen::ArrayBase<ArrayW>& w, // weights
           Real x0, Real delta)
    {
        using Index = Eigen::Index;

        using Eigen::numext::abs;
        using Eigen::numext::conj;
        using Eigen::numext::exp;
        using Eigen::numext::imag;
        using Eigen::numext::log;
        using Eigen::numext::real;

        static const auto eps     = Eigen::NumTraits<Real>::epsilon();
        static const auto pi      = 4 * Eigen::numext::atan(Real(1));
        constexpr const auto zero = Real();
        constexpr const auto half = Real(0.5);

        //-------------------------------------------------------------------------
        // The exponents are obtained as a_i = -log(z_i), where {z_i} are the
        // roots of the Prony polynomial.
        //
        // Some z_i might be real and negative: in this case, the corresponding
        // parameter a_i becomes a complex, i.e, a_i = -ln|z_i|+i \pi.
        // However, its complex conjugate a_i^* is not included in the final
        // exponential sum approximation which makes the approximated function
        // non-real. Thus, we disregards those terms.
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
                    const auto an       = Complex(-log(-xi), -pi) / delta;
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
            ret.exponents() = -z.log() / delta;
            if (x0 == Real())
            {
                ret.weights() = w; // Don't forget to copy weights
            }
            else
            {
                ret.weights() = w * (-x0 * ret.exponents()).exp();
            }
            return ret;
        }
    }
};

template <typename T>
struct gen_prony_like_method_result<std::complex<T>>
{
    using Real       = T;
    using Complex    = std::complex<Real>;
    using ResultType = ExponentialSum<Complex, Complex>;

    template <typename ArrayZ, typename ArrayW>
    static ResultType
    create(const Eigen::ArrayBase<ArrayZ>& z, // exp(-exponents)
           const Eigen::ArrayBase<ArrayW>& w, // weights
           Real x0, Real delta)
    {
        ResultType ret(z.size());
        ret.exponents() = -z.log() / delta;
        if (x0 == Real())
        {
            ret.weights() = w; // Don't forget to copy weights
        }
        else
        {
            ret.weights() = w * (-x0 * ret.exponents()).exp();
        }

        return ret;
    }
};

} // namespace detail
} // namespace mxpfit

#endif /* MXPFIT_PRONY_LIKE_METHOD_COMMON_HPP */

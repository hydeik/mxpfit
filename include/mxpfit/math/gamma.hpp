/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2014 Hidekazu Ikeno
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
/// \file gamma.hpp
///
/// Implementations of Gamma and related functions.
///
#ifndef MATH_IGAMMA_HPP
#define MATH_IGAMMA_HPP

#include <cmath>
#include <limits>

namespace math
{
//==============================================================================
// Gamma and log
//==============================================================================

//==============================================================================
// Regularized incomplete gamma functions
//==============================================================================
namespace detail
{
//
// Calculate regularized lower incomplete gamma function by the series
// expansion.
//
template <typename T>
T igamma_series(T a, T x, T epsilon, unsigned maxIter)
{
    T sum  = T(1);
    T term = T(1);
    for (unsigned k = 1; k < maxIter; ++k)
    {
        term *= x / (a + static_cast<T>(k));
        sum += term;
        if (term < sum * epsilon)
        {
            break;
        }
    }
    return std::exp(a * std::log(x) - x - std::lgamma(a + T(1))) * sum;
}

//
// Calculate regularized upper incomplete gamma function by the continued
// fraction.
//
template <typename T>
T igammac_cont_frac(T a, T x, T epsilon, unsigned maxIter)
{
    static const T tiny = std::numeric_limits<T>::min();
    T f, C, D, delta;
    f = T(1) - a + x;
    if (f == T())
    {
        f = tiny;
    }
    C = f;
    D = T();

    for (unsigned i = 1; i < maxIter; ++i)
    {
        auto A = i * (a - i);
        auto B = T(2 * i + 1) - a + x;
        D      = B + A * D;
        if (D == T())
        {
            D = tiny;
        }
        C = B + A / C;
        if (C == T())
        {
            C = tiny;
        }
        D     = T(1) / D;
        delta = C * D;
        f *= delta;
        if (std::abs(delta - T(1)) < epsilon)
        {
            break;
        }
    }

    return std::exp(a * std::log(x) - x - std::lgamma(a)) / f;
}

} // namespace: detail

/**
 * Calculate regularized lower incomplete gamma function,
 * \f$P(a,x)=\frac{\gamma(a,x)}{\Gamma(a)}\f$.
 *
 * \param[in] a  the argument \f$ a \f$
 * \param[in] x  the argument \f$ x \f$
 * \param[in] epsilon Terminate summation when the absolute value of the term in
 * the series is less than this value.
 * \param[in] maxIter maximum number of terms in the series expansion.
 * \return   the value of \f$P(a,x)\f$
 */
template <typename T>
T igamma(T a, T x, T epsilon, unsigned maxIter)
{
    if (x == T())
    {
        return T();
    }
    return x < a + 1 ? detail::igamma_series(a, x, epsilon, maxIter)
                     : T(1) - detail::igammac_cont_frac(a, x, epsilon, maxIter);
}

/**
 * Calculate regularized lower incomplete gamma function,
 * \f$P(a,x)=\frac{\gamma(a,x)}{\Gamma(a)}\f$.
 *
 * \param[in] a  the argument \f$ a \f$
 * \param[in] x  the argument \f$ x \f$
 * \return   the value of \f$P(a,x)\f$
 */
template <typename T>
T igamma(T a, T x)
{
    static_assert(std::is_floating_point<T>::value,
                  "invalid type for template "
                  "argument T: a floating "
                  "point type expected.");
    static const T epsilon        = 4 * std::numeric_limits<T>::epsilon();
    static const unsigned maxIter = std::numeric_limits<unsigned>::max();
    // static const unsigned maxIter = 10000;
    return igamma(a, x, epsilon, maxIter);
}

/**
 * Calculate regularized upper incomplete gamma function,
 * \f$Q(a,x)=\frac{\Gamma(a,x)}{\Gamma(a)}\f$.
 *
 * \param[in] a  the argument \f$ a \f$
 * \param[in] x  the argument \f$ x \f$
 * \param[in] epsilon Terminate summation when the absolute value of the term in
 * the series is less than this value.
 * \param[in] maxIter maximum number of terms in the series expansion.
 * \return   the value of \f$Q(a,x)\f$
 */
template <typename T>
T igammac(T a, T x, T epsilon, unsigned maxIter)
{
    if (x == T())
    {
        return T(1);
    }
    return x >= a + 1 ? detail::igammac_cont_frac(a, x, epsilon, maxIter)
                      : T(1) - detail::igamma_series(a, x, epsilon, maxIter);
}

/**
 * Calculate regularized lower incomplete gamma function,
 * \f$Q(a,x)=\frac{\Gamma(a,x)}{\Gamma(a)}\f$.
 *
 * \param[in] a  the argument \f$ a \f$
 * \param[in] x  the argument \f$ x \f$
 * \return   the value of \f$Q(a,x)\f$
 */
template <typename T>
T igammac(T a, T x)
{
    static_assert(std::is_floating_point<T>::value,
                  "invalid type for template "
                  "argument T: a floating "
                  "point type expected.");
    static const T epsilon        = 4 * std::numeric_limits<T>::epsilon();
    static const unsigned maxIter = std::numeric_limits<unsigned>::max();
    // static const unsigned maxIter = 1000;
    return igammac(a, x, epsilon, maxIter);
}

} // namespace: math

#endif /* MATH_IGAMMA_HPP */

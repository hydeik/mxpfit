// -*- mode: c++; fill-column: 80; indent-tabs-mode: nil; -*-

#ifndef MATH_LEGENDRE_HPP
#define MATH_LEGENDRE_HPP

#include <cassert>
#include <cmath>
#include <iterator>

namespace math
{

/** Compute value of next Legendre polynomials
 *
 *  Compute value of next\ Legendre polynomials, \f$ P_{l+1}(x) \f$ from lower
 *  order values, \f$ P_{l}(x), \f$ P_{l-1}(x) \f$ and \f$ x \f$ using the
 *  reccurence realtion:
 *
 *  \f[
 *    (l+1)P_{l+1}(x) = (2l+1) x P_{l}(x) - l P_{l-1}(x)
 *  \f]
 *
 * \param l  order of legendre polynomial
 * \param x  cooddinate that function values is computed.
 * \param pl   value of \f$ P_{l}(x) \f$
 * \param plm1 value of \f$ P_{l-1}(x) \f$
 * \return   value of \f$ P_{l+1}(x) \f$
 */
template <typename T>
inline T next_legendre(int l, T x, T pl, T plm1)
{
    return (T(2 * l + 1) * x * pl - T(l) * plm1) / T(l + 1);
}

/** Legendre polynomials of first kind.
 *
 *  Calculate a value of Legendre polynomial of first kind, \f$ P_{l}(x) \f$ at
 *  given point. The value of \f$ P_{l}(x) \f$ is computed using the recurrence
 *  relation (\see `next_legendre()`).
 *
 * \remark functions with \f$ l < 0 \f$ are not considered, because we do not
 *         need them.
 */
template <typename T>
T legendre_p(int l, T x)
{
    assert(l >= 0);

    if (l == 0)
        return T(1);

    auto plm1 = T(1); // P(0,x) = 1
    auto pl   = x;    // P(1,x) = x
    T plp1;
    for (int k = 1; k < l; ++k)
    {
        plp1 = next_legendre(k, x, pl, plm1);
        plm1 = pl;
        pl   = plp1;
    }
    return pl;
}

/** Legendre polynomials of first kind.
 *
 *  Calculate list of Legendre polynomials of first kind
 *  \f$\{P_0(x),P_1(x),\dots,P_l(x)\}\f$ at given point \f$ x \f$.
 *
 * \param  lmax maximum order of Legendre polynomials
 * \param  x  coordinate that the polynomials to be calculated
 * \param  Px random access iterator where polynomial values are stored
 *
 * \pre `lmax >= 0`
 * \pre `-1 <= x && x <= 1` because legendre polynomial P(x) is defined for
 *      this domain.
 * \pre `Px[0],Px[1],...,Px[lmax]` must be valid.
 */

template <typename T, typename RandomAccessIterator>
void list_legendre_p(int lmax, T x, RandomAccessIterator Plx)
{
    assert(lmax >= 0);
    assert(std::abs(x) <= T(1));

    if (lmax == 0)
    {
        Plx[0] = T(1);
        return;
    }

    Plx[0] = T(1);
    Plx[1] = x;

    for (std::size_t l = 1; l < static_cast<std::size_t>(lmax); ++l)

    {
        // Plx[l + 1] = ((2 * l + 1) * x * Plx[l] - l * Plx[l-1]) / (l + 1);
        Plx[l + 1] = next_legendre(l, x, Plx[l], Plx[l - 1]);
    }
}

} // namespace: math

#endif /* MATH_LEGENDRE_HPP */

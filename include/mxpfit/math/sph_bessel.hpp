///
/// \file sph_bessel.hpp
///
/// Spherical Bessel functions of integer order with complex arguments
///
#ifndef MATH_SPH_BESSEL_HPP
#define MATH_SPH_BESSEL_HPP

#include <cassert>

#include <limits>

#include <math/complex_math.hpp>

namespace math
{

namespace detail
{
//------------------------------------------------------------------------------
// Auxiliary functions to compute spherical Bessel function j_n(z)
//------------------------------------------------------------------------------
template <typename T>
struct max_factorial_arg;

template <>
struct max_factorial_arg<float>
{
    constexpr static const int value = 34;
};

template <>
struct max_factorial_arg<double>
{
    constexpr static const int value = 170;
};

template <>
struct max_factorial_arg<long double>
{
    constexpr static const int value = 170;
};

/// \internal
///
/// Compute spherical Bessel of the first kind \f$j_{n}(z)\f$ with complex
/// argument \f$z\f$ with small magnitude.
///
/// \f$j_{n}(z)\f$ is evaluated using the power series expansion,
///
/// \f[
///   j_{n}(z) = z^{n} \sum_{k=0}^{\infty}
///      \frac{1}{k!(2n+2k+1)!!} \left(-\frac{z^{2}}{2}\right)^{2}.
/// \f]
///
template <typename T>
T sph_bessel_j_small_z_power_series(int n, const T& z)
{
    using real_type = decltype(std::real(z));

    constexpr static const auto tol =
        std::numeric_limits<real_type>::epsilon() / 4;
    // sqrt(pi) / 2
    static const auto sqrt_pi_half =
        std::sqrt(std::atan2(real_type(1), real_type(1)));
    const auto max_iter = std::numeric_limits<int>::max();
    const auto nphalf   = real_type(n + 0.5);

    T mult = z / real_type(2);
    T term;
    //
    // Initial term: z^n / (2n+1)!! = (z/2)^n / Gamma(n+3/2)
    //
    if (n + 3 > max_factorial_arg<real_type>::value)
    {
        term =
            real_type(n) * std::log(mult) - std::lgamma(nphalf + real_type(1));
        term = std::exp(term);
    }
    else
    {
        term = std::pow(mult, n) / std::tgamma(nphalf + real_type(1));
    }
    mult *= -mult; // mult = -z^2/4

    auto result = term;
    for (int k = 1; k < max_iter; ++k)
    {
        if (std::abs(term) < tol * std::abs(result))
        {
            break;
        }
        term *= mult / (real_type(k) * (real_type(k) + nphalf));
        result += term;
    }

    return sqrt_pi_half * result;
}

//
// Compute spherical Bessel of the first kind of order 0,
// \f$j_{0}(z) = \frac{\sin  z}{z}.\f$
//
template <typename T>
T sph_bessel_j0(const T& z)
{
    using Real = typename RealTypeOf<T>::type;
    using std::abs;
    using std::sqrt;
    using std::sin;

    constexpr static const Real eps  = std::numeric_limits<Real>::epsilon();
    static const Real sqrt_eps       = sqrt(eps);
    static const Real forth_root_eps = sqrt(sqrt_eps);

    const auto abs_z = abs(z);
    if (abs_z >= forth_root_eps)
    {
        return sin(z) / z;
    }
    else
    {
        T result = T(1);
        if (abs_z >= eps)
        {
            const T z2 = z * z;
            result -= z2 / Real(6);
            if (abs_z >= sqrt_eps)
            {
                result += z2 * z2 / Real(120);
            }
        }
        return result;
    }
}

//
// Compute spherical Bessel function of the first kind of order 1,
//   \f$j_{1}(z)=\frac{\sin z}{z^2}-\frac{\cos z}{z}.f$
//
template <typename T>
T sph_bessel_j1(const T& z)
{
    using std::cos;
    using std::sin;
    return (sin(z) / z - cos(z)) / z;
}

//
// Compute spherical Bessel function of the first kind \f$j_{n}(z)\f$ using the
// backward recurrence algorithm proposed by Liang-Wu Cai.
//
template <typename T>
int sph_bessel_rec_cai_start_order(int n, const T& z)
{
    const auto abs_z = std::abs(z);
    const auto m     = (1.83 + 4.1) * std::pow(abs_z, 0.91) + 9.0;
    const auto m_max = 235.0 + std::floor(50.0 * std::sqrt(abs_z));

    return std::min(std::max(static_cast<int>(m), n + 1),
                    static_cast<int>(m_max));
}

template <typename T>
int sph_bessel_rec_cai_start_order(int n, const std::complex<T>& z)
{
    const int N      = std::max(50, n);
    const auto sn    = std::sin(std::abs(std::arg(z)));
    const auto abs_z = std::abs(z);
    const auto m     = (1.83 + 4.1 * std::pow(sn, 0.36)) *
                       std::pow(abs_z, 0.91 - 0.43 * std::pow(sn, 0.33)) +
                   9.0 * (1.0 - std::sqrt(sn));
    const auto m_max = 235.0 + std::floor(50.0 * std::sqrt(abs_z));

    return std::min(std::max(static_cast<int>(m), N + 1),
                    static_cast<int>(m_max));
}

template <typename T>
T sph_bessel_j_rec_cai(int n, const T& z)
{
    using real_type          = decltype(std::real(z));
    constexpr const auto eps = std::numeric_limits<real_type>::epsilon();
    constexpr const auto tiny =
        std::numeric_limits<real_type>::min() / (eps * eps);

    const T z_inv  = real_type(1) / z;
    const T j1_val = (std::sin(z) * z_inv - std::cos(z)) * z_inv;

    if (n == 1)
    {
        return j1_val;
    }
    //
    // let f_{M1} = 0, f_{M} = tiny and apply backward recursion to n = 0
    //

    T fmp1 = T();
    T fm   = tiny;
    int m  = sph_bessel_rec_cai_start_order(n, z);

    // std::cout << "*** m = " << m << " for z = " << z << std::endl;

    while (--m > n)
    {
        const T fmm1 = real_type(2 * m + 1) * z_inv * fm - fmp1;
        fmp1         = fm;
        fm           = fmm1;
    }

    ++m;
    T result = fm; // fm = S * j_{n}(z) where S is scaling factor

    while (--m)
    {
        const T fmm1 = real_type(2 * m + 1) * z_inv * fm - fmp1;
        fmp1         = fm;
        fm           = fmm1;
    }
    //
    // Re-scaling: determine scaling factor either from j_0(z) or j_1(z) value.
    //
    const T j0_val = sph_bessel_j0(z);
    if (std::abs(j0_val) >= std::abs(j1_val))
    {
        result *= j0_val / fm;
    }
    else
    {
        result *= j1_val / fmp1;
    }

    return result;
}

//------------------------------------------------------------------------------
// Auxiliary functions to compute spherical Hankel function h_n^{(1)}(z) and
// h_n^{(2)}(z).
//------------------------------------------------------------------------------

template <typename T>
struct sph_hankel_resutl_type
{
    using type = std::complex<T>;
};

template <typename T>
struct sph_hankel_resutl_type<std::complex<T>>
{
    using type = std::complex<T>;
};

template <int kind, typename T>
typename sph_hankel_resutl_type<T>::type sph_hankel_impl(int n, const T& z)
{
    static_assert(kind == 1 || kind == 2,
                  "invalid value for kind of spherical Hankel function");
    using result_type                = typename sph_hankel_resutl_type<T>::type;
    using real_type                  = decltype(std::real(z));
    constexpr static const auto zero = real_type();
    constexpr static const auto one  = real_type(1);

    constexpr static const auto ci =
        (kind == 1) ? result_type(zero, one) : result_type(zero, -one);

    const auto z_inv = one / z;
    //
    // Compute spherical Hankel function of order 0
    //
    // h_{0}^{(1)}(z)= -i exp(iz) / z
    // h_{0}^{(2)}(z)= i exp(-iz) / z
    auto hlm1 = -ci * std::exp(ci * z) * z_inv;

    if (n == 0)
    {
        return hlm1;
    }
    //
    // Apply forward recursion
    //
    //
    // h_{1}^{(1)}(z) = (1/z - i) * h_{0}^{(1)}(z)
    // h_{1}^{(2)}(z) = (1/z + i) * h_{0}^{(2)}(z)
    //
    auto hl = (z_inv - ci) * hlm1;
    for (int l = 1; l < n; ++l)
    {
        const auto hlp1 = T(2 * l + 1) * z_inv * hl - hlm1;
        hlm1            = hl;
        hl              = hlp1;
    }

    return hl;
}

} // namespace: detail

///
/// Compute spherical Bessel functions of the first kind \f$j_{n}(z)\f$ for
/// integer order \f$n\f$ with a complex argument \f$z.\f$
///
/// \tparam T  scalar type of argument `z`, either real or complex number.
///
/// \param[in]  n  order of spherical Bessel function
/// \param[in]  z  input argument
///
/// \return Value of \f$j_{n}(z)\f$
///
template <typename T>
T sph_bessel_j(int n, const T& z)
{
    using real_type = decltype(std::abs(z));
    //
    // Special case at n = 0: j_{0}(z) = sinc(z) = sin(z)/z
    //
    if (n == 0)
    {
        return detail::sph_bessel_j0(z);
    }
    //
    // Special case at z = 0: j_{n}(0) = 0 for n >= 1
    //
    if (z == T())
    {
        return T();
    }
    //
    // Case |z| < 1: Evaluate power series
    //
    if (std::abs(z) < real_type(1))
    {
        return detail::sph_bessel_j_small_z_power_series(n, z);
    }
    //
    // Default case: Evaluate by Cai's backward recursion algorithm
    //
    return detail::sph_bessel_j_rec_cai(n, z);
}

///
/// Compute spherical Hankel functions of the first kind \f$h_{n}^{(1)}(z)\f$
/// for integer order \f$n\f$ with a complex argument \f$z.\f$
///
/// \tparam T  scalar type of argument `z`, either real or complex number.
///
/// \param[in]  n  order of spherical Bessel function
/// \param[in]  z  input argument
///
/// \return Value of \f$h_{n}^{(1)}(z)\f$
///
template <typename T>
typename detail::sph_hankel_resutl_type<T>::type sph_hankel_1(int n, const T& z)
{
    using real_type = decltype(std::real(z));

    if (std::imag(z) >= real_type())
    {
        return detail::sph_hankel_impl<1>(n, z);
    }
    //
    // In case imag(z) < 0, evaluate as
    //   h_{n}^{(1)}(z) = 2 j_{n}(z) - h_{n}^{(2)}(z)
    //
    const auto jn  = sph_bessel_j(n, z);
    const auto hn2 = detail::sph_hankel_impl<2>(n, z);

    return real_type(2) * jn - hn2;
}

///
/// Compute spherical Hankel functions of the second kind \f$h_{n}^{(2)}(z)\f$
/// for integer order \f$n\f$ with a complex argument \f$z.\f$
///
/// \tparam T  scalar type of argument `z`, either real or complex number.
///
/// \param[in]  n  order of spherical Bessel function
/// \param[in]  z  input argument
///
/// \return Value of \f$h_{n}^{(2)}(z)\f$
///
template <typename T>
typename detail::sph_hankel_resutl_type<T>::type sph_hankel_2(int n, const T& z)
{
    using real_type = decltype(std::real(z));

    if (std::imag(z) <= real_type())
    {
        return detail::sph_hankel_impl<2>(n, z);
    }
    //
    // In case imag(z) < 0, evaluate as
    //   h_{n}^{(2)}(z) = 2 j_{n}(z) + h_{n}^{(1)}(z)
    //
    const auto jn  = sph_bessel_j(n, z);
    const auto hn1 = detail::sph_hankel_impl<1>(n, z);

    return real_type(2) * jn - hn1;
}
///
/// Compute spherical Bessel functions of the second kind \f$y_{n}(z)\f$ for an
/// integer order \f$n\f$ with a complex argument \f$z.\f$
///
/// \tparam T  scalar type of argument `z`, either real or complex number.
///
/// \param[in]  n  order of spherical Bessel function
/// \param[in]  z  input argument
///
/// \return Value of \f$j_{n}(z)\f$
///
// --- for real argument
template <typename T>
T sph_bessel_y(int n, const T& z)
{
    const auto hn1 = detail::sph_hankel_impl<1>(n, z);
    return std::imag(hn1);
}

// --- for complex argument
template <typename T>
std::complex<T> sph_bessel_y(int n, const std::complex<T>& z)
{
    using real_type   = T;
    using result_type = std::complex<T>;

    result_type ret = sph_bessel_j(n, z);
    if (std::imag(z) >= real_type())
    {
        ret -= detail::sph_hankel_impl<1>(n, z);
    }
    else
    {
        ret = detail::sph_hankel_impl<2>(n, z) - ret;
    }

    return result_type(std::imag(ret), std::real(ret));
}

} // namespace: math

#endif /* MATH_SPH_BESSEL_HPP */

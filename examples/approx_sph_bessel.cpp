#include <iomanip>
#include <iostream>

#include <mxpfit/approx_sph_bessel.hpp>

// #include <mxpfit/math/sph_bessel.hpp>

#include <boost/math/special_functions/bessel.hpp>

//==============================================================================
// Main
//==============================================================================

using Index              = Eigen::Index;
using Real               = double;
using Complex            = std::complex<Real>;
using RealArray          = Eigen::Array<Real, Eigen::Dynamic, 1>;
using ComplexArray       = Eigen::Array<Complex, Eigen::Dynamic, 1>;
using ExponentialSumType = mxpfit::ExponentialSum<Complex, Complex>;

void sph_bessel_kernel_error(int l, const RealArray& x,
                             const ExponentialSumType& ret,
                             bool verbose_print = false)
{
    RealArray exact(x.size());
    RealArray approx(x.size());

    for (Index i = 0; i < x.size(); ++i)
    {
        // exact(i)  = math::sph_bessel_j(l, x(i));
        exact(i)  = boost::math::sph_bessel(l, x(i));
        approx(i) = std::real(ret(x(i)));
    }

    RealArray abserr(Eigen::abs(exact - approx));

    if (verbose_print)
    {
        for (Index i = 0; i < x.size(); ++i)
        {
            std::cout << std::setw(24) << x(i) << ' '      // point
                      << std::setw(24) << exact(i) << ' '  // exact value
                      << std::setw(24) << approx(i) << ' ' // approximation
                      << std::setw(24) << abserr(i) << '\n';
        }
    }

    Index imax;
    abserr.maxCoeff(&imax);

    std::cout << "\n  abs. error in interval [" << x(0) << ","
              << x(x.size() - 1) << "]\n"
              << "    maximum : " << abserr(imax) << '\n'
              << "    averaged: " << abserr.sum() / x.size() << std::endl;
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const Real threshold = 1.0e-12;
    const Real eps       = Eigen::NumTraits<Real>::epsilon();
    const Index lmax     = 20;
    const Index N        = 1000000; // # of sampling points

    std::cout
        << "# Approximation of spherical Bessel function by exponential sum\n";

    RealArray x = Eigen::pow(10.0, RealArray::LinSpaced(N, -5.0, 7.0));
    ExponentialSumType ret;
    for (Index l = 0; l <= lmax; ++l)
    {
        std::cout << "\n# --- order " << l;
        ret = mxpfit::approxSphBessel(l, threshold);
        const auto thresh_weight =
            std::max(eps, threshold) / std::sqrt(Real(ret.size()));
        ret = mxpfit::removeIf(
            ret, [=](const Complex& /*exponent*/, const Complex& wi) {
                return std::abs(std::real(wi)) < thresh_weight &&
                       std::abs(std::imag(wi)) < thresh_weight;
            });

        const bool verbose = false;
        sph_bessel_kernel_error(l, x, ret, verbose);
        std::cout << " (" << ret.size() << " terms approximation)\n";
        std::cout << "# real(exponent), imag(exponent), real(weight), "
                     "imag(weight)\n";
        for (Index i = 0; i < ret.size(); ++i)
        {
            std::cout << std::setw(24) << std::real(ret.exponent(i)) << '\t'
                      << std::setw(24) << std::imag(ret.exponent(i)) << '\t'
                      << std::setw(24) << std::real(ret.weight(i)) << '\t'
                      << std::setw(24) << std::imag(ret.weight(i)) << '\n';
        }
        std::cout << '\n' << std::endl;

        // std::cout << "# no. of terms and (exponents, weights)\n" << ret <<
        // '\n'; std::cout << "# sum of weights: " << ret.weights().sum() <<
        // '\n';
    }

    return 0;
}

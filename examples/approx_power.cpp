#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <mxpfit/approx_power.hpp>

//==============================================================================
// Main
//==============================================================================

using Index              = Eigen::Index;
using Real               = double;
using RealArray          = Eigen::Array<Real, Eigen::Dynamic, 1>;
using ExponentialSumType = mxpfit::ExponentialSum<Real>;

void approx_error_power(const mxpfit::ApproxPowerFunction<Real>& approx_pow,
                        const ExponentialSumType& ret)
{
    const Index N = 1001;
    RealArray x =
        Eigen::pow(10.0, RealArray::LinSpaced(N, std::log10(approx_pow.rmin()),
                                              std::log10(approx_pow.rmax())));
    RealArray exact(x.size());
    RealArray approx(x.size());

    const Real beta = approx_pow.power_factor();

    for (Index i = 0; i < x.size(); ++i)
    {
        exact(i)  = std::pow(x(i), -beta);
        approx(i) = ret(x(i));
    }

    RealArray abserr(Eigen::abs(exact - approx));
    RealArray relerr(
        Eigen::abs(RealArray::Ones(exact.size()) - approx / exact));

    Index imax;
    const Real max_relerr = relerr.maxCoeff(&imax);
    std::cout << "#  relative error in interval [" << approx_pow.rmin() << ","
              << approx_pow.rmax() << "]\n"
              << "#    maximum : " << max_relerr << " at r = " << x(imax)
              << '\n'
              << "#    averaged: " << relerr.sum() / relerr.size() << "\n\n";

    for (Index i = 0; i < x.size(); ++i)
    {
        std::cout << std::setw(24) << x(i) << ' '        // point
                  << std::setw(24) << exact(i) << ' '    // exact value
                  << std::setw(24) << approx(i) << ' '   // approximation
                  << std::setw(24) << abserr(i) << ' '   // absolute error
                  << std::setw(24) << relerr(i) << '\n'; // relative error
    }
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const Real rmin        = 1.0e-9;
    const Real rmax        = 1.0e1;
    const Real beta_list[] = {1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0,
                              1.0,        2.0,       4.0,       8.0};

    const Real eps = 1.0e-12;

    std::cout << "# Approximation of power function, r^{-beta}, by an "
                 "exponential sum.\n";

    mxpfit::ApproxPowerFunction<Real> pow_approx;
    for (auto beta : beta_list)
    {
        ExponentialSumType ret = pow_approx.compute(beta, eps, rmin, rmax);

        std::cout << pow_approx << '\n';
        std::cout << ret << '\n';
        approx_error_power(pow_approx, ret);
        std::cout << '\n' << std::endl;
    }

    return 0;
}

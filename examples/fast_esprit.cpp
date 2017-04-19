#include <cassert>
#include <cmath>

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <mxpfit/fast_esprit.hpp>

#include "timer.hpp"

using Index         = Eigen::Index;
using Real          = double;
using Complex       = std::complex<double>;
using RealVector    = Eigen::VectorXd;
using ComplexVector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using RealMatrix    = Eigen::MatrixXd;
using ComplexMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

//
// f(x) = sinc(x) = sin(x) / x
//
Real sinc(Real x)
{
    static const Real eps            = std::numeric_limits<Real>::epsilon();
    static const Real sqrt_eps       = std::sqrt(eps);
    static const Real forth_root_eps = std::sqrt(sqrt_eps);

    const Real abs_x = std::abs(x);
    if (abs_x >= forth_root_eps)
    {
        return std::sin(x) / x;
    }
    else
    {
        Real result = 1.0;
        if (abs_x >= eps)
        {
            Real x2 = x * x;
            result -= x2 / 6.0;

            if (abs_x >= sqrt_eps)
            {
                result += x2 * x2 / 120.0;
            }
        }
        return result;
    }
}

///
/// Fit sinc(x) by exponential sum in the interval [0, N*h) via fast ESPRIT
/// method
///
/// \param[in] N  number of grid points
/// \param[in] h  interval between neighboring points
/// \pr arm[in] eps  prescribed accuracy \f$ \epsilon > 0\f$
///
void fit_sinc(Index N, double h, double eps)
{
    using ESPRIT = mxpfit::FastESPRIT<Real>;
    using ExpSum = typename ESPRIT::ResultType;

    // sample function on uniform grid in interval [0, (N-1)h]
    RealVector fvals =
        RealVector::NullaryExpr(N, [&](Index i) { return sinc(i * h); });
    const Index L = N / 2;                   // window length
    const Index M = std::min(L, Index(500)); // upper bound of # of terms

    std::cout << "# --- N = " << N << ", L = " << L << ", M_upper = " << M
              << std::endl;
    Timer time;
    ESPRIT esprit(N, L, M);
    std::cout << "# --- preparation done (elapsed time: "
              << time.elapsed().count() << " us)\n";
    time.restart();
    ExpSum ret = esprit.compute(fvals, 0.0, h, eps);
    std::cout << "# --- Fitting done (elapsed time: " << time.elapsed().count()
              << " us)\n";

    std::cout
        << "# Parameters: real(a[i]), imag(a[i]), real(w[i]), imag(w[i]) [n = "
        << ret.size() << "]\n";

    for (Index i = 0; i < ret.size(); ++i)
    {
        const auto ai = ret.exponent(i);
        const auto wi = ret.weight(i);
        std::cout << std::setw(24) << std::real(ai) << '\t'  // real part
                  << std::setw(24) << std::imag(ai) << '\t'  // imaginary part
                  << std::setw(24) << std::real(wi) << '\t'  // real part
                  << std::setw(24) << std::imag(wi) << '\n'; // imaginary part
    }

    std::cout << "\n# errors on sample points\n"
                 "# x, f(x), f(x) approx., abs. error, rel. error\n";
    Real max_abserr = Real();
    Real max_relerr = Real();
    for (Index i = 0; i < N; ++i)
    {
        const Real xi     = i * h;
        const Real v1     = fvals(i);
        const Real v2     = std::real(ret(xi));
        const Real abserr = std::abs(v1 - v2);
        const Real relerr = abserr / std::abs(v1);
        max_abserr        = std::max(max_abserr, abserr);
        max_relerr        = std::max(max_relerr, relerr);

        std::cout << std::setw(24) << xi << '\t'      // grid
                  << std::setw(24) << v1 << '\t'      // exact value
                  << std::setw(24) << v2 << '\t'      // approx. value
                  << std::setw(24) << abserr << '\t'  // abs. error
                  << std::setw(24) << relerr << '\n'; // rel. error
    }

    std::cout << "# --- max abs. error = " << std::setw(24) << max_abserr
              << "\n# --- max rel. error = " << std::setw(24) << max_relerr
              << '\n'
              << std::endl;
}

//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------
int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    std::cout << "# Fitting f(x) = sinc(x) in [0, N / 16)\n" << std::endl;

    const Index nsamples[] = {1 << 8,  1 << 9,  1 << 10, 1 << 11,
                              1 << 12, 1 << 13, 1 << 14, 1 << 15,
                              1 << 16, 1 << 17, 1 << 18};
    const Real h   = 1.0 / 16.0;
    const Real eps = 1.0e-12;

    for (auto N : nsamples)
    {
        fit_sinc(N, h, eps);
    }

    return 0;
}

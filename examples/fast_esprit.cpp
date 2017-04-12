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

//------------------------------------------------------------------------------
// Test functors
//------------------------------------------------------------------------------

//
// f(x) = sinc(x) = sin(x) / x
//
struct SincFn
{
    double operator()(double x) const
    {
        static const double eps      = std::numeric_limits<double>::epsilon();
        static const double sqrt_eps = std::sqrt(eps);
        static const double forth_root_eps = std::sqrt(sqrt_eps);

        const double abs_x = std::abs(x);
        if (abs_x >= forth_root_eps)
        {
            return std::sin(x) / x;
        }
        else
        {
            double result = 1.0;
            if (abs_x >= eps)
            {
                double x2 = x * x;
                result -= x2 / 6.0;

                if (abs_x >= sqrt_eps)
                {
                    result += x2 * x2 / 120.0;
                }
            }
            return result;
        }
    }
};

//
// f(x) = 1 / x
//
struct rinv
{
    double operator()(double x) const
    {
        return 1.0 / x;
    }
};

//
// Compute function values on a uniform grid
//
// \param[in]  f  user defined function \f$ f(x) \f$
// \param[in]  xmin  lower bound of \f$x\f$
// \param[in]  xmax  upper bound of \f$x\f$
// \param[out] result  a vector to store values of \f$f(x_{k})\f$
//
template <typename F, typename Vec>
void make_sample(F f, double xmin, double xmax, Vec& result)
{
    auto np = result.size();
    auto h  = (xmax - xmin) / (np - 1);
    for (Index n = 0; n < np; ++n)
    {
        result(n) = f(xmin + n * h);
    }

    return;
}

//
// Fit function via fast ESPRIT method
//
template <typename F>
void test_fast_esprit(F fn, Index N, Index L, Index M, double xmin, double xmax,
                      double eps)
{
    using Scalar = decltype(fn(xmin));
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ESPRIT = mxpfit::FastESPRIT<Scalar>;
    using ExpSum = typename ESPRIT::ResultType;

    Vector exact(N);
    make_sample(fn, xmin, xmax, exact);
    auto delta = (xmax - xmin) / (N - 1);

    std::cout << "# N = " << N << ", L = " << L << ", M_upper = " << M
              << std::endl;
    Timer time;
    ESPRIT esprit(N, L, M);
    std::cout << "# --- preparation done (elapsed time: "
              << time.elapsed().count() << " us)\n";
    time.restart();
    ExpSum ret = esprit.compute(exact, xmin, delta, eps);
    std::cout << "# --- Fitting done (elapsed time: " << time.elapsed().count()
              << " us)\n";

    std::cout << "# Parameters of exponential sum approximation\n"
              << ret << "\n\n";

    std::cout << "# x, approx, exact, abserr, relerr\n";

    for (Index i = 0; i < N; ++i)
    {
        auto x      = xmin + i * delta;
        auto approx = ret(x);
        auto abserr = std::abs(approx - exact(i));
        auto relerr = (abserr == Real()) ? Real() : abserr / std::abs(exact(i));

        std::cout << x << '\t' << approx << '\t' << exact(i) << '\t' << abserr
                  << '\t' << relerr << '\n';
    }

    std::cout << "\n" << std::endl;
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    std::cout << "# --- Fitting f(x) = sinc(x)\n" << std::endl;

    const double xmin = 0.0;
    const double eps  = 1.0e-12;
    // const double eps       = std::numeric_limits<double>::epsilon() * 10;
    const double h         = 0.125;
    const Index nsamples[] = {100, 500, 1000, 5000, 10000, 50000, 100000};

    for (auto N : nsamples)
    {
        const Index L     = N / 2;                   // window length
        const Index M     = std::min(L, Index(500)); // max # of terms
        const double xmax = xmin + h * (N - 1);
        test_fast_esprit(SincFn(), N, L, M, xmin, xmax, eps);
    }

    return 0;
}

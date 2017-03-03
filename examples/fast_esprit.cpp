#include <cassert>
#include <cmath>

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <mxpfit/fast_esprit.hpp>

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

struct rinv
{
    double operator()(double x) const
    {
        return 1.0 / x;
    }
};

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

template <typename F>
void test_fast_esprit(F fn, Index N, Index L, Index M, double xmin, double xmax,
                      double eps)
{
    using Scalar     = decltype(fn(xmin));
    using Vector     = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ESPRIT     = mxpfit::FastESPRIT<Scalar>;
    using RealScalar = typename ESPRIT::RealScalar;
    using ExpSum     = typename ESPRIT::ResultType;

    Vector exact(N);
    make_sample(fn, xmin, xmax, exact);
    auto delta = (xmax - xmin) / (N - 1);

    ESPRIT esprit(N, L, M);

    ExpSum ret = esprit.compute(exact, xmin, delta, eps);

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
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    std::srand(static_cast<unsigned int>(std::time(0)));

    std::cout << "# sinc(x):  x in [0, 1000]" << std::endl;
    Index N     = 2000;  // # of sampling points
    Index L     = N / 2; // window length
    Index M     = 100;   // max # of terms
    double xmin = 0.0;
    double xmax = 500.0;
    double eps  = 1.0e-10;
    test_fast_esprit(SincFn(), N, L, M, xmin, xmax, eps);

    // std::cout << "# 1/r: r in [1, 10^{6}]" << std::endl;
    // N    = (1 << 12);
    // L    = N / 2;
    // M    = 100;
    // xmin = 1.0;
    // xmax = 1.0e+6;
    // eps  = 1.0e-8;
    // test_fast_esprit(rinv(), N, L, M, xmin, xmax, eps);

    return 0;
}

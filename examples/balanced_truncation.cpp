#include <iomanip>
#include <iostream>
#include <random>

#include <mxpfit/balanced_truncation.hpp>
#include <mxpfit/exponential_sum.hpp>

#include "timer.hpp"

using Index = Eigen::Index;

//
// Test balanced truncation for exponential sum
//

template <typename Real, typename F1, typename F2>
void print_funcs(const Real& xmin, const Real& xmax,
                 const mxpfit::ExponentialSumBase<F1>& orig,
                 const mxpfit::ExponentialSumBase<F2>& truncated)
{
    using Scalar    = decltype(orig(Real()));
    using Array     = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
    using RealArray = Eigen::Array<Real, Eigen::Dynamic, 1>;

    const Index n = 100001;

    RealArray grid(RealArray::LinSpaced(n, xmin, xmax));
    Array f1(n);
    Array f2(n);
    RealArray abserr(n);
    RealArray relerr(n);

    for (Index i = 0; i < n; ++i)
    {
        const auto x  = grid(i);
        const auto v1 = orig(x);
        const auto v2 = truncated(x);
        f1(i)         = v1;
        f2(i)         = v2;
        abserr(i)     = std::abs(v1 - v2);
        relerr(i)     = (v1 != Scalar()) ? abserr(i) / std::abs(v1) : abserr(i);
    }

    std::cout << "# parameters of original exponential sum\n" << orig << '\n';
    std::cout << "# parameters of truncated exponential sum\n"
              << truncated << '\n';

    Index imax;
    abserr.maxCoeff(&imax);
    std::cout << "# Error estimation:\n"
              << "#    max abs. error in [" << xmin << ", " << xmax
              << "]: " << abserr(imax) << " (rel. error: " << relerr(imax)
              << ")\n";

    relerr.maxCoeff(&imax);
    std::cout << "#    max rel. error in [" << xmin << ", " << xmax
              << "]: " << relerr(imax) << " (abs. error: " << abserr(imax)
              << ")\n";
    std::cout << "\n#    x, f(x), f(x)[truncated], abs. error, rel. error\n";
    for (Index i = 0; i < n; ++i)
    {
        std::cout << grid(i) << '\t'    // x
                  << f1(i) << '\t'      // f(x) original
                  << f2(i) << '\t'      // f(x) truncated
                  << abserr(i) << '\t'  // abs. error
                  << relerr(i) << '\n'; // rel. error
    }

    std::cout << std::endl;
}

template <typename T>
struct RandomScalar
{
    RandomScalar() : engine_(std::random_device()()), distr_(T(), T(1)){};

    T operator()()
    {
        return distr_(engine_);
    }

    std::mt19937 engine_;
    std::uniform_real_distribution<T> distr_;
};

template <typename T>
struct RandomScalar<std::complex<T>>
{
    RandomScalar() : engine_(std::random_device()()), distr_(T(), T(1)){};

    std::complex<T> operator()()
    {
        static const T pi_half = T(2) * std::atan2(T(1), T(1));
        return {distr_(engine_), pi_half * distr_(engine_)};
    }

    std::mt19937 engine_;
    std::uniform_real_distribution<T> distr_;
};

template <typename T>
void test_balanced_truncation(Index n)
{
    using TruncationBody = mxpfit::BalancedTruncation<T>;
    using RealScalar     = typename TruncationBody::RealScalar;
    using Array          = Eigen::Array<T, Eigen::Dynamic, 1>;

    const auto xmin = RealScalar();
    const auto xmax = RealScalar(50);

    const auto delta = std::sqrt(n) * Eigen::NumTraits<RealScalar>::epsilon();
    RandomScalar<T> rng;
    Array a = Array::NullaryExpr(n, [&](Eigen::Index) { return rng(); });
    Array w = Array::NullaryExpr(n, [&](Eigen::Index) { return rng(); });
    w *= T(5);
    auto f_orig = mxpfit::makeExponentialSum(a, w);

    Timer time;
    TruncationBody truncation;
    auto f_truncated = truncation.compute(f_orig, delta);
    std::cout << "    elapsed time: " << time.elapsed().count() << " us\n";

    print_funcs(xmin, xmax, f_orig, f_truncated);
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const Index n       = 1000;
    const Index n_trial = 2;

    std::cout << "# real exponential sum of length " << n << '\n';

    for (Index i = 0; i < n_trial; ++i)
    {
        std::cout << "--- trial " << i + 1 << '\n';
        test_balanced_truncation<double>(n);
    }

    std::cout << "# complex exponential sum of length " << n << '\n';
    for (Index i = 0; i < n_trial; ++i)
    {
        std::cout << "--- trial " << i + 1 << '\n';
        test_balanced_truncation<std::complex<double>>(n);
    }

    return 0;
}

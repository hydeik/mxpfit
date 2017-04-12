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

template <typename T, typename F1, typename F2>
void print_funcs(const T& xmin, const T& xmax,
                 const mxpfit::ExponentialSumBase<F1>& orig,
                 const mxpfit::ExponentialSumBase<F2>& truncated)
{
    using RealArray = Eigen::Array<T, Eigen::Dynamic, 1>;

    const Index n = 100001;

    RealArray grid(RealArray::LinSpaced(n, xmin, xmax));
    RealArray abserr(n);
    RealArray relerr(n);

    for (Index i = 0; i < n; ++i)
    {
        const auto x  = grid(i);
        const auto f1 = orig(x);
        const auto f2 = truncated(x);
        abserr(i)     = std::abs(f1 - f2);
        relerr(i)     = (f1 != T()) ? abserr(i) / std::abs(f1) : abserr(i);
    }

    std::cout << "    size before truncation = " << orig.size() << '\n'
              << "    size after truncation  = " << truncated.size() << '\n';
    Index imax;
    abserr.maxCoeff(&imax);
    std::cout << "    max abs. error in [" << xmin << ", " << xmax
              << "]: " << abserr(imax) << " (rel. error: " << relerr(imax)
              << ")\n";

    relerr.maxCoeff(&imax);
    std::cout << "    max rel. error in [" << xmin << ", " << xmax
              << "]: " << relerr(imax) << " (abs. error: " << abserr(imax)
              << ")\n";
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
    const Index n_trial = 10;

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

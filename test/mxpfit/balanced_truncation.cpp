#include <iomanip>
#include <iostream>
#include <random>

#include <mxpfit/balanced_truncation.hpp>
#include <mxpfit/exponential_sum.hpp>

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
        return {distr_(engine_), distr_(engine_)};
    }

    std::mt19937 engine_;
    std::uniform_real_distribution<T> distr_;
};

template <typename T>
void test_balanced_truncation(Index n, Index n_trial)
{
    using ExponentialSum = mxpfit::ExponentialSum<T>;
    using TruncationBody = mxpfit::BalancedTruncation<T>;
    using Real           = typename Eigen::NumTraits<T>::Real;

    const auto xmin = Real();
    const auto xmax = Real(10);

    const auto delta = std::sqrt(n) * Eigen::NumTraits<Real>::epsilon();

    ExponentialSum orig(n);
    ExponentialSum trunc;
    TruncationBody truncation;
    truncation.setThreshold(delta);
    RandomScalar<T> rng;

    for (Index i_trial = 1; i_trial <= n_trial; ++i_trial)
    {
        std::cout << "--- trial " << i_trial << '\n';
        auto& x = orig.exponents();
        auto& a = orig.weights();

        x = ExponentialSum::ExponentsArray::NullaryExpr(
            n, [&](Eigen::Index) { return rng(); });
        a = ExponentialSum::WeightsArray::NullaryExpr(
            n, [&](Eigen::Index) { return rng(); });
        a *= Real(5);

        trunc = truncation.compute(orig);
        print_funcs(xmin, xmax, orig, trunc);
    }
}

int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    const Index n       = 500;
    const Index n_trial = 5;

    std::cout << "# Reduction for real exponential sum\n";
    test_balanced_truncation<double>(n, n_trial);

    std::cout << "# Reduction for complex exponential sum\n";
    test_balanced_truncation<std::complex<double>>(n, n_trial);

    return 0;
}

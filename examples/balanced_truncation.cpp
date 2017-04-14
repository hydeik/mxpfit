#include <iomanip>
#include <iostream>
#include <random>

#include <mxpfit/balanced_truncation.hpp>
#include <mxpfit/exponential_sum.hpp>

#include "timer.hpp"

using Index = Eigen::Index;

//
// Generate random real number between [0,1]
//
template <typename T>
struct RandomReal
{
    RandomReal() : engine_(std::random_device()()), distr_(T(), T(1)){};

    T operator()()
    {
        return distr_(engine_);
    }

    std::mt19937 engine_;
    std::uniform_real_distribution<T> distr_;
};

//
//
template <typename T>
struct RandomExponentialSum
{
    using ResultType = mxpfit::ExponentialSum<T>;

    static ResultType create(Index n)
    {
        ResultType ret(n);
        RandomReal<T> rng;
        //
        // 0 < exponent[i] < 1
        // 0 < weight[i]   < 1
        //
        for (Index i = 0; i < n; ++i)
        {
            ret.exponents()(i) = rng();
            ret.weights()(i)   = rng();
        }

        return ret;
    }
};

template <typename T>
struct RandomExponentialSum<std::complex<T>>
{
    using Complex    = std::complex<T>;
    using ResultType = mxpfit::ExponentialSum<std::complex<T>>;

    static ResultType create(Index n)
    {
        static const T pi2 = T(8) * std::atan2(T(1), T(1));
        ResultType ret(n);
        RandomReal<T> rng;

        for (Index i = 0; i < n; ++i)
        {
            // random complex number on unit dist
            const auto zi      = std::polar(rng(), pi2 * rng());
            ret.exponents()(i) = -std::log(zi);
            ret.weights()(i) =
                Complex(T(2) * (rng() - T(0.5)), T(2) * (rng() - T(0.5)));
        }

        return ret;
    }
};

template <typename T>
inline void print_scalar(const T& x, const char* sep)
{
    std::cout << x << sep;
}

template <typename T>
inline void print_scalar(const std::complex<T>& x, const char* sep)
{
    std::cout << std::real(x) << '\t' << std::imag(x) << sep;
}
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
    enum
    {
        IsComplex = Eigen::NumTraits<Scalar>::IsComplex
    };

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

    std::cout << "# parameters of original exponential sum (order = "
              << orig.size() << ")\n";
    for (Index i = 0; i < orig.size(); ++i)
    {
        // "exponent(i)[TAB]weight(i)\n"
        print_scalar(orig.exponents()(i), "\t");
        print_scalar(orig.weights()(i), "\n");
    }

    std::cout << "\n# parameters of truncated exponential sum (order = "
              << truncated.size() << ")\n";
    for (Index i = 0; i < truncated.size(); ++i)
    {
        // "exponent(i)[TAB]weight(i)\n"
        print_scalar(truncated.exponents()(i), "\t");
        print_scalar(truncated.weights()(i), "\n");
    }

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
        std::cout << grid(i) << '\t';   // x
        print_scalar(f1(i), "\t");      // f(x) original
        print_scalar(f2(i), "\t");      // f(x) truncated
        std::cout << abserr(i) << '\t'  // abs. error
                  << relerr(i) << '\n'; // rel. error
    }

    std::cout << std::endl;
}

template <typename T>
void test_balanced_truncation(Index n)
{
    using TruncationBody = mxpfit::BalancedTruncation<T>;
    using RealScalar     = typename TruncationBody::RealScalar;

    const auto xmin = RealScalar();
    const auto xmax = RealScalar(100);

    // const auto delta = std::sqrt(n) *
    // Eigen::NumTraits<RealScalar>::epsilon();
    const auto delta = RealScalar(1.0e-12);

    auto f_orig = RandomExponentialSum<T>::create(n);

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

    const Index n = 500;

    std::cout << "# real exponential sum of length " << n << '\n';
    test_balanced_truncation<double>(n);

    std::cout << "# complex exponential sum of length " << n << '\n';
    test_balanced_truncation<std::complex<double>>(n);

    return 0;
}

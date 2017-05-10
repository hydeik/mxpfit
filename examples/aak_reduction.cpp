#include <iomanip>
#include <iostream>
#include <random>

#include <mxpfit/aak_reduction.hpp>
#include <mxpfit/exponential_sum.hpp>

#include "timer.hpp"

using Index = Eigen::Index;

template <typename T>
void test_aak_reduction(Index n)
{
    using Body       = mxpfit::AAKReduction<T>;
    using RealScalar = typename Body::RealScalar;

    // const auto xmin = RealScalar();
    // const auto xmax = RealScalar(100);

    const auto delta = RealScalar(1.0e-12);

    // auto f_orig = RandomExponentialSum<T>::create(n);
    mxpfit::ExponentialSum<T> f_orig(n);
    f_orig.setRandom();

    Timer time;
    Body truncation;
    auto f_truncated = truncation.compute(f_orig, delta);
    std::cout << "    elapsed time: " << time.elapsed().count() << " us\n";

    // print_funcs(xmin, xmax, f_orig, f_truncated);
}

int main()
{
    test_aak_reduction<std::complex<double>>(200);
    return 0;
}

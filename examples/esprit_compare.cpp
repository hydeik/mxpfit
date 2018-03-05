#include <iomanip>
#include <iostream>
#include <random>

#include <mxpfit/esprit.hpp>
#include <mxpfit/fast_esprit.hpp>

#include "constants.hpp"
#include "timer.hpp"

using Index   = Eigen::Index;
using Real    = double;
using Complex = std::complex<Real>;

using RealArray      = Eigen::Array<Real, Eigen::Dynamic, 1>;
using RealArrayXX    = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>;
using ComplexArray   = Eigen::Array<Complex, Eigen::Dynamic, 1>;
using ExponentialSum = mxpfit::ExponentialSum<Complex, Complex>;
using ESPRIT         = mxpfit::ESPRIT<Real>;
using FastESPRIT     = mxpfit::FastESPRIT<Real>;

constexpr static const Real pi = math::constant<Real>::pi;

ExponentialSum sort_exponential_sum(const ExponentialSum& orig)
{
    const Index n = orig.size();
    Eigen::ArrayXi index(n);
    for (Index i = 0; i < n; ++i)
    {
        index(i) = i;
    }

    std::sort(index.data(), index.data() + n, [&](Index i, Index j) {
        return std::imag(orig.exponent(i)) < std::imag(orig.exponent(j));
    });

    ExponentialSum ret(n);
    for (Index i = 0; i < n; ++i)
    {
        ret.exponent(i) = orig.exponent(index(i));
        ret.weight(i)   = orig.weight(index(i));
    }

    return ret;
}

void print_exponential_sum(const ExponentialSum& orig,
                           const ExponentialSum& ret)
{
    std::cout
        << "# Parameters: a[i], w[i], abs. error a[i], abs. error c[i], [n = "
        << ret.size() << "]\n";
    for (Index i = 0; i < ret.size(); ++i)
    {
        const auto ai = ret.exponent(i);
        const auto wi = ret.weight(i);

        const auto abserr_ai = std::abs(ai - orig.exponent(i));
        const auto abserr_wi = std::abs(wi - orig.weight(i));

        std::cout << std::setw(48) << ai << '\t'        // a[i]
                  << std::setw(48) << wi << '\t'        // c[i]
                  << std::setw(24) << abserr_ai << '\t' //
                  << std::setw(24) << abserr_wi << '\n';
    }

    std::cout << std::endl;
}

//
// Make sequence of sampling data h[k] = fn(k) + e(k) for k = 0,1,2...,N
// where e(k) is random noise uniformly distributed in
// -noise_magnitude <= e(k) <= noise_magnitude
//
template <typename Functor>
void make_sampling_data(Functor fn, Real noise_magnitude,
                        Eigen::Ref<RealArray> h_exact,
                        Eigen::Ref<RealArrayXX> h_noise)
{
    assert(h_exact.size() == h_noise.rows());

    const Index n_samples = h_noise.rows();
    const Index n_trial   = h_noise.cols();

    std::random_device seed_gen;
    std::mt19937 rnd(seed_gen());
    std::uniform_real_distribution<Real> noise(-noise_magnitude,
                                               noise_magnitude);

    // Sampling data without noise
    for (Index i = 0; i < n_samples; ++i)
    {
        h_exact(i) = fn(Real(i));
    }

    // Sampling data with noise
    for (Index j = 0; j < n_trial; ++j)
    {
        for (Index i = 0; i < n_samples; ++i)
        {
            h_noise(i, j) = h_exact(i) + noise(rnd);
        }
    }
}

//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------
int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    ExponentialSum orig(5);
    orig.exponent(0) = Complex(Real(), -pi / 2);
    orig.exponent(1) = Complex(Real(), -pi / 4);
    orig.exponent(2) = Complex();
    orig.exponent(3) = Complex(Real(), pi / 4);
    orig.exponent(4) = Complex(Real(), pi / 2);

    orig.weight(0) = Complex(1);
    orig.weight(1) = Complex(300);
    orig.weight(2) = Complex(34);
    orig.weight(3) = Complex(300);
    orig.weight(4) = Complex(1);

    const Index nsamples[] = {1 << 8,  1 << 9,  1 << 10, 1 << 11,
                              1 << 12, 1 << 13, 1 << 14, 1 << 15,
                              1 << 16, 1 << 17, 1 << 18};

    const Index n_terms        = orig.size();
    const Index n_trial        = 10;
    const auto delta           = Real(1);
    const auto t0              = Real();
    const Real eps             = 1.0e-5;
    const Real noise_magnitude = 3.0;

    std::cout << "# Exact exponential sum\n" << orig << std::endl;


    for (auto N : nsamples)
    {
        const Index L = N / 2;
        const Index M = n_terms;

        RealArray h_exact(N);
        RealArrayXX h_noise(N, n_trial);
        ExponentialSum ret;

        Timer timer;
        auto total_time_orig = std::chrono::microseconds();
        auto total_time_fast = std::chrono::microseconds();
        auto elapsed         = std::chrono::microseconds();

        make_sampling_data([&](Real x) { return std::real(orig(x)); },
                           noise_magnitude, h_exact, h_noise);

        // Fitting by ESPRIT
        if (N < 10000)
        {
            std::cout << "# Original ESPRIT: (N = " << N << ", L = " << L
                      << ", M = " << M << ", eps = " << eps << ")" << std::endl;
            ESPRIT esprit(N, L, M);
            for (Index iter = 0; iter < n_trial; ++iter)
            {
                std::cout << "# --- Trial " << iter + 1 << std::flush;
                timer.restart();
                ExponentialSum tmp =
                    esprit.compute(h_noise.matrix().col(iter), t0, delta, eps);
                elapsed = timer.elapsed();
                ret     = sort_exponential_sum(tmp);
                std::cout << " (elapsed time: " << elapsed.count()
                          << " microseconds)\n";
                total_time_orig += elapsed;
                print_exponential_sum(orig, ret);
                std::cout << std::endl;
            }
            std::cout << "#$ [original] Averaged running time: "
                      << total_time_orig.count() / n_trial << " microseconds"
                      << std::endl;
        }

        // Fitting by FastESPRIT
        FastESPRIT fast_esprit(N, L, M);

        std::cout << "# Fast ESPRIT: (N = " << N << ", L = " << L
                  << ", M = " << M << ", eps = " << eps << ")" << std::endl;
        for (Index iter = 0; iter < n_trial; ++iter)
        {
            std::cout << "# --- Trial " << iter + 1 << std::flush;

            timer.restart();
            auto tmp =
                fast_esprit.compute(h_noise.matrix().col(iter), t0, delta, eps);
            elapsed = timer.elapsed();
            ret     = sort_exponential_sum(tmp);
            std::cout << " (elapsed time: " << elapsed.count()
                      << " microseconds)\n";
            print_exponential_sum(orig, ret);
            std::cout << std::endl;
            total_time_fast += elapsed;
        }
        std::cout << "#$ [fast] Averaged running time: "
                  << total_time_fast.count() / n_trial << " microseconds"
                  << std::endl;
    }

    return 0;
}

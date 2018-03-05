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
using ComplexArray   = Eigen::Array<Complex, Eigen::Dynamic, 1>;
using ExponentialSum = mxpfit::ExponentialSum<Complex, Complex>;

constexpr static const Real pi = math::constant<Real>::pi;

void print_exponential_sum(const ExponentialSum& ret)
{
    std::cout << "# Parameters: real(a[i]), imag(a[i]), real(w[i]), "
                 "imag(w[i]), [n = "
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
                        Eigen::Ref<RealArray> h_noise)
{
    assert(h_exact.size() == h_noise.size());

    const Index n_samples = h_noise.size();
    std::random_device seed_gen;
    std::mt19937 rnd(seed_gen());
    std::uniform_real_distribution<Real> noise(-noise_magnitude,
                                               noise_magnitude);

    for (Index i = 0; i < n_samples; ++i)
    {
        const auto val = fn(Real(i));
        h_exact(i)     = val;
        h_noise(i)     = val + noise(rnd);
    }
}

//
// Fit sampling data by original ESPRIT algorithm
//
std::pair<ExponentialSum, std::chrono::microseconds>
fit_by_original_esprit(const Eigen::Ref<const RealArray>& h, Index n_terms,
                       Real eps)
{
    const Index N    = h.size(); // number of sampling data
    const Index L    = N / 2;    // window size of Hankel matrix
    const auto delta = Real(1);  // step width is unity

    Timer time;
    mxpfit::ESPRIT<Complex> esprit(N, L, n_terms);
    ExponentialSum ret = esprit.compute(h.matrix(), Real(), delta, eps);
    return {ret, time.elapsed()};
}

//
// Fit sampling data by fast ESPRIT algorithm
//
std::pair<ExponentialSum, std::chrono::microseconds>
fit_by_fast_esprit(const Eigen::Ref<const RealArray>& h, Index n_terms,
                   Real eps)
{
    const Index N    = h.size(); // number of sampling data
    const Index L    = N / 2;    // window size of Hankel matrix
    const auto delta = Real(1);  // step width is unity

    Timer time;
    mxpfit::FastESPRIT<Real> esprit(N, L, n_terms);
    ExponentialSum ret = esprit.compute(h.matrix(), Real(), delta, eps);

    return {ret, time.elapsed()};
}

//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------
int main()
{
    std::cout.precision(15);
    std::cout.setf(std::ios::scientific);

    ExponentialSum orig(5);
    orig.exponent(0) = Complex();
    orig.exponent(1) = Complex(Real(), pi / 4);
    orig.exponent(2) = Complex(Real(), -pi / 4);
    orig.exponent(3) = Complex(Real(), pi / 2);
    orig.exponent(4) = Complex(Real(), -pi / 2);

    orig.weight(0) = Complex(34);
    orig.weight(1) = Complex(300);
    orig.weight(2) = Complex(300);
    orig.weight(3) = Complex(1);
    orig.weight(4) = Complex(1);

    const Index nsamples[] = {128, 256, 512, 1024, 2048};

    const Index n_terms        = orig.size();
    const Index n_trial        = orig.size();
    const Real eps             = 1.0e-5;
    const Real noise_magnitude = 3.0;

    std::cout << "# Exact exponential sum\n";
    print_exponential_sum(orig);

    for (auto N : nsamples)
    {
        RealArray h_exact(N);
        RealArray h_noise(N);
        ExponentialSum ret;
        auto total_time_orig = std::chrono::microseconds();
        auto total_time_fast = std::chrono::microseconds();
        auto elapsed         = std::chrono::microseconds();

        std::cout << "# Fitting by ESPRIT N = " << N << ", L = " << N / 2
                  << ", M = " << n_terms << ", eps = " << eps << ")"
                  << std::endl;

        for (Index iter = 0; iter < n_trial; ++iter)
        {
            make_sampling_data([&](Real x) { return std::real(orig(x)); },
                               noise_magnitude, h_exact, h_noise);

            std::cout << "# --- Trial " << iter + 1 << '\n';
            std::tie(ret, elapsed) =
                fit_by_original_esprit(h_noise, n_terms, eps);
            total_time_orig += elapsed;
            std::cout << "#     Original ESPRIT (elapsed time: " << elapsed.count()
                      << " microseconds)\n";
            print_exponential_sum(ret);

            std::tie(ret, elapsed) = fit_by_fast_esprit(h_noise, n_terms, eps);
            total_time_fast += elapsed;
            std::cout << "#     Fast ESPRIT     (elapsed time: " << elapsed.count()
                      << " microseconds)\n";
            print_exponential_sum(ret);
            std::cout << std::endl;
        }

        std::cout << "#$ Averaged running time:\n"
                  << "   [original]: " << total_time_orig.count() / n_trial
                  << " microseconds\n"
                  << "   [fast]:     " << total_time_fast.count() / n_trial
                  << " microseconds\n"
                  << std::endl;
    }

    return 0;
}


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

void print_fitting_results(const ExponentialSum& ret,
                           const ExponentialSum& orig)
{
    if (ret.size() < orig.size())
    {
        std::cout << "*** failed to extract all parameters" << std::endl;
        return;
    }

    std::cout << "# Parameters: real(a[i]), imag(a[i]), real(w[i]), "
                 "imag(w[i]), abserr a[i], abserr w[i], [n = "
              << ret.size() << "]\n";
    for (Index i = 0; i < ret.size(); ++i)
    {
        const auto ai        = ret.exponent(i);
        const auto wi        = ret.weight(i);
        const auto abserr_ai = std::abs(ai - orig.exponent(i));
        const auto abserr_wi = std::abs(wi - orig.weight(i));

        std::cout << std::setw(24) << std::real(ai) << '\t' // real part
                  << std::setw(24) << std::imag(ai) << '\t' // imaginary part
                  << std::setw(24) << std::real(wi) << '\t' // real part
                  << std::setw(24) << std::imag(wi) << '\t' // imaginary part
                  << std::setw(24) << abserr_ai << '\t'     // abs. err. a[i]
                  << std::setw(24) << abserr_wi << '\n';    // abs. err. w[i]
    }

    std::cout << std::endl;
}

void fit(Index N, Real eps, const ExponentialSum& orig, Real noise_magnitude)
{
    const Index L    = N / 2;
    const Index M    = orig.size(); // upper bound of # of terms
    const Real delta = Real(1);     // step width is unity

    std::random_device seed_gen;
    std::mt19937 rnd(seed_gen());
    std::uniform_real_distribution<Real> noise(-noise_magnitude,
                                               noise_magnitude);
    // Prepare sampling data
    ComplexArray h(N);
    for (Index i = 0; i < N; ++i)
    {
        h(i) = orig(Real(i)) + noise(rnd);
    }

    {
        std::cout << "# --- Fit by ESPRIT (N = " << N << ", L = " << L
                  << ", M = " << M << ", eps = " << eps << ")\n";
        Timer time;
        mxpfit::ESPRIT<Complex> esprit(N, L, M);
        ExponentialSum ret = esprit.compute(h.matrix(), Real(), delta, eps);

        std::cout << "# --- done (elapsed time: " << time.elapsed().count()
                  << " us)\n";

        print_fitting_results(ret, orig);
    }
    {
        std::cout << "# --- Fit by FastESPRIT (N = " << N << ", L = " << L
                  << ", M = " << M << ", eps = " << eps << ")\n";
        Timer time;
        mxpfit::FastESPRIT<Complex> esprit(N, L, M);
        ExponentialSum ret = esprit.compute(h.matrix(), Real(), delta, eps);

        std::cout << "# --- done (elapsed time: " << time.elapsed().count()
                  << " us)\n";

        print_fitting_results(ret, orig);
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

    std::cout << "# Exact exponential sum\n" << orig << std::endl;

    // const Index nsamples[] = {1 << 8,  1 << 9,  1 << 10, 1 << 11,
    //                           1 << 12, 1 << 13, 1 << 14, 1 << 15,
    //                           1 << 16, 1 << 17, 1 << 18};
    const Index N              = 500;
    const Real eps             = 1.0e-12;
    const Real noise_magnitude = 3.0;

    fit(N, eps, orig, noise_magnitude);

    return 0;
}

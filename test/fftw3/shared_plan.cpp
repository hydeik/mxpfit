#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <ctime>

#include <complex>
#include <iostream>

#include <Eigen/Core>

#include <fftw3/shared_plan.hpp>

using Index         = Eigen::Index;
using Real          = double;
using Complex       = std::complex<Real>;
using RealVector    = Eigen::VectorXd;
using ComplexVector = Eigen::VectorXcd;
using RealMatrix    = Eigen::MatrixXd;
using ComplexMatrix = Eigen::MatrixXcd;

using FFT  = fftw3::FFT<Real>;
using IFFT = fftw3::IFFT<Real>;

static const double eps = 1.0e-14;

void test_complex_fft_many_1d(Index n, Index howmany)
{
    INFO("1D complex-to-complex FFT (n = " << n << ", howmany = " << howmany
                                           << ')');

    ComplexMatrix f1(n, howmany);
    ComplexMatrix re_f1(n, howmany);
    ComplexMatrix g1(n, howmany);

    RealMatrix f2(n, howmany);
    RealMatrix re_f2(n, howmany);
    ComplexMatrix g2(n / 2 + 1, howmany);

    int n_       = static_cast<int>(n);
    int howmany_ = static_cast<int>(howmany);

    // Complex-to-complex
    auto plan_fft  = FFT::make_plan(n_, howmany_, f1.data(), g1.data());
    auto plan_ifft = IFFT::make_plan(n_, howmany_, f1.data(), g1.data());

    f1.setRandom();
    FFT::run(plan_fft, f1.data(), g1.data());
    IFFT::run(plan_ifft, g1.data(), re_f1.data());
    re_f1 /= Real(n);

    for (Index i = 0; i < howmany; ++i)
    {
        const Real residual = (f1.col(i) - re_f1.col(i)).norm();
        CHECK(residual == Approx(0.0).margin(n * eps));
    }
}

void test_real_fft_many_1d(Index n, Index howmany)
{
    INFO("1D real-to-complex FFT (n = " << n << ", howmany = " << howmany
                                        << ')');

    RealMatrix f2(n, howmany);
    RealMatrix re_f2(n, howmany);
    ComplexMatrix g2(n / 2 + 1, howmany);

    RealVector residuals(howmany);

    int n_       = static_cast<int>(n);
    int howmany_ = static_cast<int>(howmany);

    // Real-to-complex transform
    auto plan_fft_r2c  = FFT::make_plan(n_, howmany_, f2.data(), g2.data());
    auto plan_ifft_r2c = IFFT::make_plan(n_, howmany_, g2.data(), re_f2.data());

    f2.setRandom();
    FFT::run(plan_fft_r2c, f2.data(), g2.data());
    IFFT::run(plan_ifft_r2c, g2.data(), re_f2.data());
    re_f2 /= Real(n);

    for (Index i = 0; i < howmany; ++i)
    {
        const Real residual = (f2.col(i) - re_f2.col(i)).norm();
        CHECK(residual == Approx(0.0).margin(n * eps));
    }
}

TEST_CASE("Test thread-safe FFTW3 wrapper for multiple 1D FFT")
{
    std::srand(static_cast<unsigned>(std::time(0)));
    SECTION("Complex-to-complex FFT")
    {
        test_complex_fft_many_1d(100, 20);
        test_complex_fft_many_1d(1000, 10);
        test_complex_fft_many_1d(10000, 5);
        test_complex_fft_many_1d(100000, 2);
        test_complex_fft_many_1d(1000000, 1);
    }

    SECTION("Real-to-complex FFT")
    {
        test_real_fft_many_1d(100, 20);
        test_real_fft_many_1d(1000, 10);
        test_real_fft_many_1d(10000, 5);
        test_real_fft_many_1d(100000, 2);
        test_real_fft_many_1d(1000000, 1);
    }
}

// #define CATCH_CONFIG_MAIN
// #include <catch/catch.hpp>

#include <iostream>

#include <mxpfit/quasi_cauchy_rrd.hpp>

template <typename T>
void test_quasi_cauchy_rrd_impl(Eigen::Index n, Eigen::Index n_trial)
{
    using Real = typename Eigen::NumTraits<T>::Real;

    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    std::cout << "self-adjoint quasi-Cauchy RRD (size " << n << ")"
              << std::endl;

    VectorType a(n);
    VectorType x(n);
    MatrixType C(n, n);
    MatrixType C_reconstructed(n, n);

    const Real delta     = 1.0e-10;
    const Real threshold = Eigen::NumTraits<Real>::epsilon() * delta * delta;

    for (Eigen::Index i_trial = 1; i_trial <= n_trial; ++i_trial)
    {
        a.setRandom();
        a *= Real(5);
        x.setRandom();
        x = (x.array().real() < Real()).select(-x, x);
        x = x.cwiseAbs();

        C = mxpfit::makeSelfAdjointQuasiCauchy(a, x);

        mxpfit::SelfAdjointQuasiCauchyRRD<T> rrd;
        rrd.setThreshold(threshold);

        rrd.compute(a, x);

        C_reconstructed = rrd.matrixPL() *
                          rrd.vectorD().cwiseAbs2().asDiagonal() *
                          rrd.matrixPL().adjoint();

        const auto norm_C = C.norm();
        const auto resid  = (C - C_reconstructed).norm();
        const auto error  = (norm_C == Real()) ? resid : resid / norm_C;

        std::cout << " --- Trial " << i_trial << ":\n"
                  << "     rank = " << rrd.rank() << '\n'
                  << "     ||C - X D^2 X^H|| / ||C|| = " << error << std::endl;
    }
}

int main()
{
    test_quasi_cauchy_rrd_impl<double>(100, 10);
    test_quasi_cauchy_rrd_impl<std::complex<double>>(100, 10);
    test_quasi_cauchy_rrd_impl<double>(500, 10);
    test_quasi_cauchy_rrd_impl<std::complex<double>>(500, 10);

    return 0;
}

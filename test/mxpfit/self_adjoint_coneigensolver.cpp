#include <iostream>

#include <mxpfit/quasi_cauchy_rrd.hpp>
#include <mxpfit/self_adjoint_coneigensolver.hpp>

template <typename T>
void test_self_adjoint_coneigensolver(Eigen::Index n, Eigen::Index n_trial)
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

    const Real delta     = 1.0e-15;
    const Real threshold = Eigen::NumTraits<Real>::epsilon() * delta * delta;

    mxpfit::SelfAdjointQuasiCauchyRRD<T> rrd;
    rrd.setThreshold(threshold);
    mxpfit::SelfAdjointConeigenSolver<T> ceig;

    for (Eigen::Index i_trial = 1; i_trial <= n_trial; ++i_trial)
    {
        a.setRandom();
        a *= Real(5);
        x.setRandom();
        x = (x.array().real() < Real()).select(-x, x);
        x = x.cwiseAbs();

        C = mxpfit::makeSelfAdjointQuasiCauchy(a, x);

        // --- quasi-Cauchy RRD
        rrd.compute(a, x);

        C_reconstructed = rrd.matrixPL() *
                          rrd.vectorD().cwiseAbs2().asDiagonal() *
                          rrd.matrixPL().adjoint();

        const auto norm_C    = C.norm();
        const auto resid_rrd = (C - C_reconstructed).norm();
        const auto error = (norm_C == Real()) ? resid_rrd : resid_rrd / norm_C;

        std::cout << " --- Trial " << i_trial << ":\n"
                  << "   Quasi-Cauchy RRD: rank = " << rrd.rank() << '\n'
                  << "     ||C - X * D^2 * X^H|| / ||C|| = " << error
                  << std::endl;

        // --- con-eigendecomposition

        std::cout << "   Con-eigendecomposition:\n";
        ceig.compute(rrd.matrixPL(), rrd.vectorD());

        std::cout << "     sigma(i), |C * u_i - sigma(i) * u_i.conjugate()|, "
                     "1 - u_i.transpose() * u_i:\n";

        for (Eigen::Index i = 0; i < ceig.coneigenvalues().size(); ++i)
        {
            const auto sigma    = ceig.coneigenvalues()(i);
            const auto ui       = ceig.coneigenvectors().col(i);
            const auto resid_ui = (C * ui - sigma * ui.conjugate()).norm();
            const auto resid_norm =
                std::abs(T(1) - (ui.transpose() * ui).value());
            std::cout << sigma << '\t' << resid_ui << '\t' << resid_norm
                      << '\n';
        }

        std::cout << std::endl;
        // const auto resid_ceig = (C -
        //                          ceig.coneigenvectors().conjugate() *
        //                              ceig.coneigenvalues().asDiagonal() *
        //                              ceig.coneigenvectors().transpose())
        //                             .norm();
        // std::cout << "     |C - U.conjugate() * D * U.transpose()| = "
        //           << resid_ceig << '\n';

        // std::cout
        //     << " |C - U.conjugate() * S * U.transpose()| / |C| = " <<
        //     resid_ceig
        //     << ", |I - U.transpose() * U| = "
        //     << (Matrix::Identity(size_, size_) - Uk.transpose() * Uk).norm()
        //     << std::endl;
    }
}

int main()
{
    std::cout.precision(15);
    std::cout << "# Con-eigenvalue decompositoin of RRD matrix (Real)"
              << std::endl;
    test_self_adjoint_coneigensolver<double>(100, 5);
    std::cout << "# Con-eigenvalue decompositoin of RRD matrix (Complex)"
              << std::endl;
    test_self_adjoint_coneigensolver<std::complex<double>>(100, 5);

    return 0;
}

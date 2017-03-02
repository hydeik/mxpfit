#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <ctime>

#include <mxpfit/matrix_free_gemv.hpp>
#include <mxpfit/vandermonde_matrix.hpp>

using Index = Eigen::Index;

TEST_CASE("Test Vandermonde matrix expression", "[vandermonde_matrix]")
{
    mxpfit::VandermondeMatrix<int> v0;

    mxpfit::VandermondeMatrix<int>::CoeffsVector coeff(5);
    coeff << 1, 2, 3, 4, 5;

    SECTION("Create an empty matrix")
    {
        REQUIRE(v0.rows() == 0);
        REQUIRE(v0.cols() == 0);
    }

    v0.setMatrix(4, coeff);

    SECTION("Set matrix coefficients")
    {
        REQUIRE(v0.rows() == 4);
        REQUIRE(v0.cols() == 5);

        CHECK(&v0.coeffs().coeffRef(0) == &coeff.coeffRef(0));
        REQUIRE(v0.coeffs()(0) == 1);
        REQUIRE(v0.coeffs()(1) == 2);
        REQUIRE(v0.coeffs()(2) == 3);
        REQUIRE(v0.coeffs()(3) == 4);
        REQUIRE(v0.coeffs()(4) == 5);

        SECTION("Make dense matrix")
        {
            mxpfit::VandermondeMatrix<int>::PlainObject m0 = v0.toDenseMatrix();
            REQUIRE(m0(0, 0) == 1);
            REQUIRE(m0(1, 0) == 1);
            REQUIRE(m0(2, 0) == 1);
            REQUIRE(m0(3, 0) == 1);

            REQUIRE(m0(0, 1) == 1);
            REQUIRE(m0(1, 1) == 2);
            REQUIRE(m0(2, 1) == 4);
            REQUIRE(m0(3, 1) == 8);

            REQUIRE(m0(0, 2) == 1);
            REQUIRE(m0(1, 2) == 3);
            REQUIRE(m0(2, 2) == 9);
            REQUIRE(m0(3, 2) == 27);

            REQUIRE(m0(0, 3) == 1);
            REQUIRE(m0(1, 3) == 4);
            REQUIRE(m0(2, 3) == 16);
            REQUIRE(m0(3, 3) == 64);

            REQUIRE(m0(0, 4) == 1);
            REQUIRE(m0(1, 4) == 5);
            REQUIRE(m0(2, 4) == 25);
            REQUIRE(m0(3, 4) == 125);
        }
    }

    v0.setMatrix(5, coeff.tail(3));

    SECTION("Reset matrix from other vector expression (l-value)")
    {
        REQUIRE(v0.rows() == 5);
        REQUIRE(v0.cols() == 3);

        CHECK(&v0.coeffs().coeffRef(0) == &coeff.coeffRef(2));
        REQUIRE(v0.coeffs()(0) == 3);
        REQUIRE(v0.coeffs()(1) == 4);
        REQUIRE(v0.coeffs()(2) == 5);

        SECTION("Make dense matrix")
        {
            mxpfit::VandermondeMatrix<int>::PlainObject m0 = v0.toDenseMatrix();
            REQUIRE(m0(0, 0) == 1);
            REQUIRE(m0(1, 0) == 3);
            REQUIRE(m0(2, 0) == 9);
            REQUIRE(m0(3, 0) == 27);
            REQUIRE(m0(4, 0) == 81);

            REQUIRE(m0(0, 1) == 1);
            REQUIRE(m0(1, 1) == 4);
            REQUIRE(m0(2, 1) == 16);
            REQUIRE(m0(3, 1) == 64);
            REQUIRE(m0(4, 1) == 256);

            REQUIRE(m0(0, 2) == 1);
            REQUIRE(m0(1, 2) == 5);
            REQUIRE(m0(2, 2) == 25);
            REQUIRE(m0(3, 2) == 125);
            REQUIRE(m0(4, 2) == 625);
        }
    }

    v0.setMatrix(3, 2 * coeff);

    SECTION("Reset matrix from other vector expression (r-value)")
    {
        REQUIRE(v0.rows() == 3);
        REQUIRE(v0.cols() == 5);

        CHECK(&v0.coeffs().coeffRef(0) != &coeff.coeffRef(0));

        REQUIRE(v0.coeffs()(0) == 2);
        REQUIRE(v0.coeffs()(1) == 4);
        REQUIRE(v0.coeffs()(2) == 6);
        REQUIRE(v0.coeffs()(3) == 8);
        REQUIRE(v0.coeffs()(4) == 10);

        SECTION("Make dense matrix")
        {
            mxpfit::VandermondeMatrix<int>::PlainObject m0 = v0.toDenseMatrix();
            REQUIRE(m0(0, 0) == 1);
            REQUIRE(m0(1, 0) == 2);
            REQUIRE(m0(2, 0) == 4);

            REQUIRE(m0(0, 1) == 1);
            REQUIRE(m0(1, 1) == 4);
            REQUIRE(m0(2, 1) == 16);

            REQUIRE(m0(0, 2) == 1);
            REQUIRE(m0(1, 2) == 6);
            REQUIRE(m0(2, 2) == 36);

            REQUIRE(m0(0, 3) == 1);
            REQUIRE(m0(1, 3) == 8);
            REQUIRE(m0(2, 3) == 64);

            REQUIRE(m0(0, 4) == 1);
            REQUIRE(m0(1, 4) == 10);
            REQUIRE(m0(2, 4) == 100);
        }
    }

    mxpfit::VandermondeMatrix<int> v1(v0);

    SECTION("Copy Vandermonde matrix")
    {
        REQUIRE(v1.rows() == 3);
        REQUIRE(v1.cols() == 5);

        CHECK(&v1.coeffs().coeffRef(0) == &v0.coeffs().coeffRef(0));

        REQUIRE(v1.coeffs()(0) == 2);
        REQUIRE(v1.coeffs()(1) == 4);
        REQUIRE(v1.coeffs()(2) == 6);
        REQUIRE(v1.coeffs()(3) == 8);
        REQUIRE(v1.coeffs()(4) == 10);
    }
}

template <typename T>
void test_vandermonde_matvec(Index nrows, Index ncols)
{
    using VandermondeMatrix = mxpfit::VandermondeMatrix<T>;
    using CoeffsVector      = typename VandermondeMatrix::CoeffsVector;
    using RealScalar        = typename VandermondeMatrix::RealScalar;
    using RealVector        = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    static const RealScalar zero = RealScalar();
    const RealScalar eps         = Eigen::NumTraits<RealScalar>::epsilon();
    const Index niter            = 100;

    std::srand(static_cast<unsigned>(std::time(0)));

    CoeffsVector h(CoeffsVector::Random(ncols));
    VandermondeMatrix matV(nrows, h);
    typename VandermondeMatrix::PlainObject V = matV.toDenseMatrix();

    mxpfit::MatrixFreeGEMV<VandermondeMatrix> opV(matV);

    CoeffsVector x1(ncols);
    CoeffsVector x2(nrows);
    CoeffsVector y1(nrows);
    CoeffsVector y2(nrows);
    CoeffsVector y3(ncols);
    CoeffsVector y4(ncols);
    RealVector r1(nrows);
    RealVector r2(ncols);

    const RealScalar V_norm = V.norm();

    Approx tol = Approx(zero).margin(eps);

    for (Index i = 0; i < niter; ++i)
    {
        INFO("Trial " << i << ": y = V * x");
        x1.setRandom();
        y1                = opV * x1;
        y2                = V * x1;
        r1                = (y1 - y2).cwiseAbs();
        RealScalar maxerr = r1.maxCoeff();
        RealScalar rnorm  = r1.norm();
        CHECK((maxerr / V_norm) == Approx(zero).margin(eps));
        CHECK((rnorm / V_norm) == Approx(zero).margin(eps));

        y1 = opV.conjugate() * x1;
        y2 = V.conjugate() * x1;
        r1 = (y1 - y2).cwiseAbs();
        INFO("Trial " << i << "y = V.conjugate() * x");
        maxerr = r1.maxCoeff();
        rnorm  = r1.norm();
        CHECK((maxerr / V_norm) == Approx(zero).margin(eps));
        CHECK((rnorm / V_norm) == Approx(zero).margin(eps));

        x2.setRandom();
        y3 = opV.transpose() * x2;
        y4 = V.transpose() * x2;
        r2 = (y3 - y4).cwiseAbs();
        INFO("Trial " << i << "y = V.transpose() * x");
        maxerr = r2.maxCoeff();
        rnorm  = r2.norm();
        CHECK((maxerr / V_norm) == Approx(zero).margin(eps));
        CHECK((rnorm / V_norm) == Approx(zero).margin(eps));

        y3 = opV.adjoint() * x2;
        y4 = V.adjoint() * x2;
        r2 = (y3 - y4).cwiseAbs();
        INFO("Trial " << i << "y = V.adjoint() * x");
        maxerr = r2.maxCoeff();
        rnorm  = r2.norm();
        CHECK((maxerr / V_norm) == Approx(zero).margin(eps));
        CHECK((rnorm / V_norm) == Approx(zero).margin(eps));
    }

    return;
}

TEST_CASE("Test Vandermonde matrix-vector multiplication",
          "[vandermonde_matrix]")
{
    SECTION("# (50 x 50) real matrix")
    {
        test_vandermonde_matvec<double>(50, 50);
    }
    SECTION("# (50 x 71) real matrix")
    {
        test_vandermonde_matvec<double>(50, 71);
    }
    SECTION("# (71 x 50) real matrix")
    {
        test_vandermonde_matvec<double>(71, 50);
    }

    SECTION("# (50 x 50) complex matrix")
    {
        test_vandermonde_matvec<std::complex<double>>(50, 50);
    }
    SECTION("# (50 x 71) complex matrix")
    {
        test_vandermonde_matvec<std::complex<double>>(50, 71);
    }
    SECTION("# (71 x 50) complex matrix")
    {
        test_vandermonde_matvec<std::complex<double>>(71, 50);
    }
}

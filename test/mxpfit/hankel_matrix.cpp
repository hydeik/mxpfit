#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <ctime>

#include <mxpfit/hankel_matrix.hpp>
#include <mxpfit/matrix_free_gemv.hpp>

TEST_CASE("Test Hankel matrix expression", "[hankel_matrix]")
{
    using HankelMatrix = mxpfit::HankelMatrix<double>;

    constexpr const double eps = std::numeric_limits<double>::epsilon();

    HankelMatrix h1;
    HankelMatrix h2(4, 3);

    HankelMatrix::CoeffsVector c1(7);
    c1 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;

    SECTION("Create empty Hankel matrix")
    {
        REQUIRE(h1.rows() == 0);
        REQUIRE(h1.cols() == 0);
        REQUIRE(h1.size() == 0);
    }

    h1.resize(3, 5);

    SECTION("Resize Hankel matrix")
    {

        REQUIRE(h1.rows() == 3);
        REQUIRE(h1.cols() == 5);
        // Number of coeffs should be equals to rows() + cols() - 1
        REQUIRE(h1.size() == 7);
    }

    h2.setCoeffs(c1.tail(6));

    SECTION("Set matrix coefficients (l-value)")
    {
        REQUIRE(h2.rows() == 4);
        REQUIRE(h2.cols() == 3);
        // Number of coeffs should be equals to rows() + cols() - 1
        REQUIRE(h2.coeffs().size() == 6);

        CHECK(&h2.coeffs().coeffRef(0) == &c1.coeffRef(1));

        REQUIRE(h2.coeffs()(0) == Approx(2.0).epsilon(eps));
        REQUIRE(h2.coeffs()(1) == Approx(3.0).epsilon(eps));
        REQUIRE(h2.coeffs()(2) == Approx(4.0).epsilon(eps));
        REQUIRE(h2.coeffs()(3) == Approx(5.0).epsilon(eps));
        REQUIRE(h2.coeffs()(4) == Approx(6.0).epsilon(eps));
        REQUIRE(h2.coeffs()(5) == Approx(7.0).epsilon(eps));

        HankelMatrix::PlainObject m1 = h2.toDenseMatrix();
        SECTION("Make dense matrix")
        {
            REQUIRE(m1.rows() == 4);
            REQUIRE(m1.cols() == 3);

            REQUIRE(m1(0, 0) == Approx(2.0).epsilon(eps));
            REQUIRE(m1(1, 0) == Approx(3.0).epsilon(eps));
            REQUIRE(m1(2, 0) == Approx(4.0).epsilon(eps));
            REQUIRE(m1(3, 0) == Approx(5.0).epsilon(eps));

            REQUIRE(m1(0, 1) == Approx(3.0).epsilon(eps));
            REQUIRE(m1(1, 1) == Approx(4.0).epsilon(eps));
            REQUIRE(m1(2, 1) == Approx(5.0).epsilon(eps));
            REQUIRE(m1(3, 1) == Approx(6.0).epsilon(eps));

            REQUIRE(m1(0, 2) == Approx(4.0).epsilon(eps));
            REQUIRE(m1(1, 2) == Approx(5.0).epsilon(eps));
            REQUIRE(m1(2, 2) == Approx(6.0).epsilon(eps));
            REQUIRE(m1(3, 2) == Approx(7.0).epsilon(eps));
        }
    }

    h2.setCoeffs(2 * c1.head(6));

    SECTION("Set matrix coefficients (r-value)")
    {
        REQUIRE(h2.rows() == 4);
        REQUIRE(h2.cols() == 3);
        // Number of coeffs should be equals to rows() + cols() - 1
        REQUIRE(h2.coeffs().size() == 6);

        CHECK(&h2.coeffs().coeffRef(0) != &c1.coeffRef(0));

        REQUIRE(h2.coeffs()(0) == Approx(2.0).epsilon(eps));
        REQUIRE(h2.coeffs()(1) == Approx(4.0).epsilon(eps));
        REQUIRE(h2.coeffs()(2) == Approx(6.0).epsilon(eps));
        REQUIRE(h2.coeffs()(3) == Approx(8.0).epsilon(eps));
        REQUIRE(h2.coeffs()(4) == Approx(10.0).epsilon(eps));
        REQUIRE(h2.coeffs()(5) == Approx(12.0).epsilon(eps));
    }

    HankelMatrix h3(h2);

    SECTION("Copy Hankel matrix")
    {
        REQUIRE(h3.rows() == 4);
        REQUIRE(h3.cols() == 3);
        // Number of coeffs should be equals to rows() + cols() - 1
        REQUIRE(h3.coeffs().size() == 6);

        CHECK(&h2.coeffs().coeffRef(0) == &h3.coeffs().coeffRef(0));

        REQUIRE(h3.coeffs()(0) == Approx(2.0).epsilon(eps));
        REQUIRE(h3.coeffs()(1) == Approx(4.0).epsilon(eps));
        REQUIRE(h3.coeffs()(2) == Approx(6.0).epsilon(eps));
        REQUIRE(h3.coeffs()(3) == Approx(8.0).epsilon(eps));
        REQUIRE(h3.coeffs()(4) == Approx(10.0).epsilon(eps));
        REQUIRE(h3.coeffs()(5) == Approx(12.0).epsilon(eps));
    }

    h2.resize(0, 0);

    SECTION("Resize to empty matrix")
    {
        REQUIRE(h2.rows() == 0);
        REQUIRE(h2.cols() == 0);
        REQUIRE(h2.size() == 0);
    }
}

template <typename T>
void test_hankel_gemv(Eigen::Index nrows, Eigen::Index ncols)
{
    using HankelMatrix = mxpfit::HankelMatrix<T>;
    using RealScalar   = typename HankelMatrix::RealScalar;

    using VectorType = typename HankelMatrix::CoeffsVector;
    using RealVectorType =
        Eigen::Matrix<typename HankelMatrix::RealScalar, Eigen::Dynamic, 1>;

    constexpr const auto zero = RealScalar();
    constexpr const auto eps = std::numeric_limits<RealScalar>::epsilon() * 100;
    const Eigen::Index niter = 100;

    std::srand(static_cast<unsigned>(std::time(0)));

    VectorType h(VectorType::Random(nrows + ncols - 1));
    HankelMatrix matA(nrows, ncols);
    matA.setCoeffs(h);

    typename HankelMatrix::PlainObject A = matA.toDenseMatrix();
    const RealScalar A_norm              = A.norm();
    mxpfit::MatrixFreeGEMV<HankelMatrix> opA(matA);

    REQUIRE(matA.rows() == A.rows());
    REQUIRE(matA.cols() == A.cols());
    REQUIRE(matA.rows() == opA.rows());
    REQUIRE(matA.cols() == opA.cols());

    VectorType x1(ncols);
    VectorType x2(nrows);

    VectorType y1(nrows);
    VectorType y2(nrows);
    VectorType y3(ncols);
    VectorType y4(ncols);

    RealVectorType r1(nrows);
    RealVectorType r2(ncols);

    for (Eigen::Index i = 0; i < niter; ++i)
    {
        INFO("Trial " << i << ": y = A * x");
        x1.setRandom();
        y1          = opA * x1;
        y2          = A * x1;
        r1          = (y1 - y2).cwiseAbs();
        auto maxerr = r1.maxCoeff();
        auto rnorm  = r1.norm();
        CHECK((maxerr / A_norm) == Approx(zero).margin(eps));
        CHECK((rnorm / A_norm) == Approx(zero).margin(eps));

        INFO("Trial " << i << ": y = A.conjugate() * x");
        y1     = opA.conjugate() * x1;
        y2     = A.conjugate() * x1;
        r1     = (y1 - y2).cwiseAbs();
        maxerr = r1.maxCoeff();
        rnorm  = r1.norm();
        CHECK((maxerr / A_norm) == Approx(zero).margin(eps));
        CHECK((rnorm / A_norm) == Approx(zero).margin(eps));

        INFO("Trial " << i << ": y = A.transpose() * x");
        x2.setRandom();
        y3     = opA.transpose() * x2;
        y4     = A.transpose() * x2;
        r2     = (y3 - y4).cwiseAbs();
        maxerr = r2.maxCoeff();
        rnorm  = r2.norm();
        CHECK((maxerr / A_norm) == Approx(zero).margin(eps));
        CHECK((rnorm / A_norm) == Approx(zero).margin(eps));

        INFO("Trial " << i << ": y = A.transpose() * x");
        y3     = opA.adjoint() * x2;
        y4     = A.adjoint() * x2;
        r2     = (y3 - y4).cwiseAbs();
        maxerr = r2.maxCoeff();
        rnorm  = r2.norm();
        CHECK((maxerr / A_norm) == Approx(zero).margin(eps));
        CHECK((rnorm / A_norm) == Approx(zero).margin(eps));
    }

    return;
}

TEST_CASE("Test Hankel matrix-vector multiplication", "[hankel_matrix]")
{
    SECTION("# (50 x 50) real matrix\n")
    {
        test_hankel_gemv<double>(50, 50);
    }
    SECTION("# (50 x 71) real matrix\n")
    {
        test_hankel_gemv<double>(50, 71);
    }
    SECTION("# (71 x 50) real matrix\n")
    {
        test_hankel_gemv<double>(71, 50);
    }

    SECTION("# (50 x 50) complex matrix\n")
    {
        test_hankel_gemv<std::complex<double>>(50, 50);
    }
    SECTION("# (50 x 71) complex matrix\n")
    {
        test_hankel_gemv<std::complex<double>>(50, 71);
    }
    SECTION("# (71 x 50) complex matrix\n")
    {
        test_hankel_gemv<std::complex<double>>(71, 50);
    }
}

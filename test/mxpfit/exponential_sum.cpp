#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

#include <mxpfit/exponential_sum.hpp>

TEST_CASE("Test ExponentialSum class", "[exponential_sum]")
{
    using Eigen::numext::exp;

    constexpr const double eps = std::numeric_limits<double>::epsilon() * 10;

    mxpfit::ExponentialSum<double> expsum1;
    mxpfit::ExponentialSum<double> expsum2(3);
    expsum2.exponent(0) = 1.0;
    expsum2.exponent(1) = 1.1;
    expsum2.exponent(2) = 1.2;

    expsum2.weight(0) = 1.0;
    expsum2.weight(1) = 2.0;
    expsum2.weight(2) = 3.0;

    Eigen::ArrayXd xi(4);
    Eigen::ArrayXd w(4);
    xi << 0.1, 0.2, 0.3, 0.4;
    w << 1.0, 2.0, 3.0, 4.0;

    SECTION("Create empty exponential sum")
    {
        REQUIRE(expsum1.size() == 0);
    }
    SECTION("Resize exponential sum")
    {
        expsum1.resize(5);
        REQUIRE(expsum1.size() == 5);
    }
    SECTION("Create exponential sum of size n and set parameters")
    {
        REQUIRE(expsum2.exponent(0) == Approx(1.0).epsilon(eps));
        REQUIRE(expsum2.exponent(1) == Approx(1.1).epsilon(eps));
        REQUIRE(expsum2.exponent(2) == Approx(1.2).epsilon(eps));

        REQUIRE(expsum2.weight(0) == Approx(1.0).epsilon(eps));
        REQUIRE(expsum2.weight(1) == Approx(2.0).epsilon(eps));
        REQUIRE(expsum2.weight(2) == Approx(3.0).epsilon(eps));

        REQUIRE(expsum2(0.0) == Approx(6.0).epsilon(eps));
        REQUIRE(expsum2(1.0) ==
                Approx(1.0 * exp(-1.0) + 2.0 * exp(-1.1) + 3.0 * exp(-1.2))
                    .epsilon(eps));
    }
    SECTION("Create exponential sum from arrays of exponents and weights")
    {
        mxpfit::ExponentialSum<double> expsum3(xi, w);

        REQUIRE(expsum3.exponent(0) == Approx(0.1).epsilon(eps));
        REQUIRE(expsum3.exponent(1) == Approx(0.2).epsilon(eps));
        REQUIRE(expsum3.exponent(2) == Approx(0.3).epsilon(eps));
        REQUIRE(expsum3.exponent(3) == Approx(0.4).epsilon(eps));

        REQUIRE(expsum3.weight(0) == Approx(1.0).epsilon(eps));
        REQUIRE(expsum3.weight(1) == Approx(2.0).epsilon(eps));
        REQUIRE(expsum3.weight(2) == Approx(3.0).epsilon(eps));
        REQUIRE(expsum3.weight(3) == Approx(4.0).epsilon(eps));

        REQUIRE(expsum3(0.0) == Approx(10.0).epsilon(eps));
        REQUIRE(expsum3(1.0) ==
                Approx(1.0 * exp(-0.1) + 2.0 * exp(-0.2) + 3.0 * exp(-0.3) +
                       4.0 * exp(-0.4))
                    .epsilon(eps));
    }
}

TEST_CASE("Test ExponentialSumWrapper class", "[exponential_sum]")
{
    using Eigen::numext::exp;

    constexpr const double eps = std::numeric_limits<double>::epsilon() * 10;

    Eigen::ArrayXXd a(2, 4);
    a << 0.1, 0.2, 0.3, 0.4, // exponents
        1.0, 2.0, 3.0, 4.0;  // weights

    SECTION("Exponential sum from existing array expressions")
    {
        auto expsum = mxpfit::makeExponentialSum(a.row(0), a.row(1));

        REQUIRE(expsum.exponent(0) == Approx(0.1).epsilon(eps));
        REQUIRE(expsum.exponent(1) == Approx(0.2).epsilon(eps));
        REQUIRE(expsum.exponent(2) == Approx(0.3).epsilon(eps));
        REQUIRE(expsum.exponent(3) == Approx(0.4).epsilon(eps));

        REQUIRE(expsum.weight(0) == Approx(1.0).epsilon(eps));
        REQUIRE(expsum.weight(1) == Approx(2.0).epsilon(eps));
        REQUIRE(expsum.weight(2) == Approx(3.0).epsilon(eps));
        REQUIRE(expsum.weight(3) == Approx(4.0).epsilon(eps));

        REQUIRE(expsum(0.0) == Approx(10.0).epsilon(eps));
        REQUIRE(expsum(1.0) ==
                Approx(1.0 * exp(-0.1) + 2.0 * exp(-0.2) + 3.0 * exp(-0.3) +
                       4.0 * exp(-0.4))
                    .epsilon(eps));
    }
    SECTION("Exponential sum from r-value array expression")
    {
        auto expsum = mxpfit::makeExponentialSum(a.row(0), 2.0 * a.row(1));

        REQUIRE(expsum.exponent(0) == Approx(0.1).epsilon(eps));
        REQUIRE(expsum.exponent(1) == Approx(0.2).epsilon(eps));
        REQUIRE(expsum.exponent(2) == Approx(0.3).epsilon(eps));
        REQUIRE(expsum.exponent(3) == Approx(0.4).epsilon(eps));

        REQUIRE(expsum.weight(0) == Approx(2.0).epsilon(eps));
        REQUIRE(expsum.weight(1) == Approx(4.0).epsilon(eps));
        REQUIRE(expsum.weight(2) == Approx(6.0).epsilon(eps));
        REQUIRE(expsum.weight(3) == Approx(8.0).epsilon(eps));

        REQUIRE(expsum(0.0) == Approx(20.0).epsilon(eps));
        REQUIRE(expsum(1.0) ==
                Approx(2.0 * exp(-0.1) + 4.0 * exp(-0.2) + 6.0 * exp(-0.3) +
                       8.0 * exp(-0.4))
                    .epsilon(eps));
    }
}

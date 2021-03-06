/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2017 Hidekazu Ikeno
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/// \file exponential_sum.hpp

#ifndef MXPFIT_EXPONENTIAL_SUM_HPP
#define MXPFIT_EXPONENTIAL_SUM_HPP

#include <algorithm>
#include <iosfwd>
#include <random>

#include <Eigen/Core>

namespace mxpfit
{
// --- Forward declarations
template <typename ExpScalarT, typename WScalarT = ExpScalarT>
class ExponentialSum;

template <typename ExponentsArrayT, typename WeightsArrayT>
class ExponentialSumWrapper;

namespace detail
{

template <typename T>
struct ExponentialSumTraits;

//
// Generate random real number between [0,1]
//
template <typename T>
struct random_real_scalar
{
    random_real_scalar() : distr_(T(), T(1))
    {
        std::random_device rd;
        std::seed_seq seeds({rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()});
        engine_.seed(seeds);
    }

    T operator()()
    {
        return distr_(engine_);
    }

    std::mt19937 engine_;
    std::uniform_real_distribution<T> distr_;
};

template <typename T>
struct random_exponent_impl
{
    T operator()()
    {
        return m_rng(); // distributed on [0, 1]
    }

private:
    random_real_scalar<T> m_rng;
};

template <typename T>
struct random_exponent_impl<std::complex<T>>
{
    std::complex<T> operator()()
    {
        static const T pi = EIGEN_PI;
        // zi distributed on unit disk
        const auto zi = std::polar(m_rng(), pi * (m_rng() - T(0.5)));
        return -Eigen::numext::log(zi);
    }

private:
    random_real_scalar<T> m_rng;
};

template <typename T>
struct random_weight_impl
{
    T operator()()
    {
        return m_rng(); // distributed on [0, 1]
    }

private:
    random_real_scalar<T> m_rng;
};

template <typename T>
struct random_weight_impl<std::complex<T>>
{
    std::complex<T> operator()()
    {
        // distributed on [-1, 1] + i[-1,1]
        return std::complex<T>(T(2) * m_rng() - 1, T(2) * m_rng() - T(1));
    }

private:
    random_real_scalar<T> m_rng;
};

} // namespace detail

///
/// ### ExponentialSumBase
///
/// Base class for expressions of exponential sum function
///
template <typename Derived>
class ExponentialSumBase
{
public:
    using Index  = Eigen::Index;
    using Traits = detail::ExponentialSumTraits<Derived>;

    using ExponentsArray = typename Traits::ExponentsArray;
    using WeightsArray   = typename Traits::WeightsArray;

    using ExponentScalar = typename ExponentsArray::Scalar;
    using WeightScalar   = typename WeightsArray::Scalar;

    using ExponentsArrayNested =
        typename Eigen::internal::ref_selector<ExponentsArray>::type;
    using WeightsArrayNested =
        typename Eigen::internal::ref_selector<WeightsArray>::type;

    using PlainExponentsArray =
        Eigen::Array<typename ExponentsArray::Scalar, Eigen::Dynamic, 1>;
    using PlainWeightsArray =
        Eigen::Array<typename WeightsArray::Scalar, Eigen::Dynamic, 1>;

    /// \return reference to the derived object
    Derived& derived()
    {
        return *static_cast<Derived*>(this);
    }
    /// \return const reference to the derived object
    const Derived& derived() const
    {
        return *static_cast<const Derived*>(this);
    }

    /// \return the number of exponential terms
    Index size() const
    {
        assert(exponents().size() == weights().size());
        return exponents().size();
    }

    /// \return value of the multi-exponential function at given argument x
    template <typename ArgT>
    auto operator()(const ArgT& x) const
        -> decltype(ExponentScalar() * WeightScalar())
    {
        // return derived().evalAt(x);
        return ((-x * exponents()).exp() * weights()).sum();
    }

    /// \return const reference to the array of exponents
    ExponentsArrayNested exponents() const
    {
        return derived().exponents();
    }

    /// \return reference to the array of weights
    WeightsArrayNested weights() const
    {
        return derived().weights();
    }

    /// \return value or reference of `i`-th exponents
    typename ExponentsArray::CoeffReturnType exponent(Index i) const
    {
        assert(Index() <= i && i < size());
        return exponents().coeff(i);
    }

    /// \return value or reference of `i`-th weight
    typename WeightsArray::CoeffReturnType weight(Index i) const
    {
        assert(Index() <= i && i < size());
        return weights().coeff(i);
    }
};

/// Output stream operator for `ExponentialSumBase`
template <typename Ch, typename Tr, typename Derived>
std::basic_ostream<Ch, Tr>&
operator<<(std::basic_ostream<Ch, Tr>& os,
           const ExponentialSumBase<Derived>& expsum)
{
    auto n = expsum.size();
    os << n << '\n';
    for (decltype(n) i = 0; i < n; ++i)
    {
        os << expsum.exponent(i) << '\t' << expsum.weight(i) << '\n';
    }

    return os;
}

///
/// Remove terms in exponential sum satisfying the specific criteria
///
/// \param[in] esum original exponential sum,
///   \f$f(t)=\sum_{j=1}^{n}c_{j}e^{-a_{j}t}.\f$
/// \param[in] pred unary predicate which returns ​`true` if the term should
///   be removed. The signature of the predicate function should be equivalent
///   to the following:
///   ``` c++
///     bool pred(const Scalar& aj, const Scalar& cj);
///   ```
///   where `aj` and `cj` are j-th exponent, \f$a_{j},\$ and coefficient
///   \f$c_{j},\f$ respectively, and `Scalar` is the scalar type of original
///   exponential sum, `esum`. The signature does not need to have `const &`,
///   but the function must not modify the objects passed to it.
///
/// \return an object of ExponentialSum with same scalar types for `esum`.
///
template <typename Derived, typename Predicate>
ExponentialSum<typename ExponentialSumBase<Derived>::ExponentScalar,
               typename ExponentialSumBase<Derived>::WeightScalar>
removeIf(const ExponentialSumBase<Derived>& esum, Predicate pred)
{
    using Index      = Eigen::Index;
    using IndexArray = Eigen::Array<Index, Eigen::Dynamic, 1>;
    using ResultType =
        ExponentialSum<typename ExponentialSumBase<Derived>::ExponentScalar,
                       typename ExponentialSumBase<Derived>::WeightScalar>;
    using Eigen::numext::abs;
    using Eigen::numext::real;

    if (esum.size() == Index())
    {
        return ResultType();
    }

    IndexArray index(IndexArray::LinSpaced(esum.size(), 0, esum.size() - 1));

    auto* last =
        std::remove_if(index.data(), index.data() + index.size(), [&](Index x) {
            return pred(esum.exponent(x), esum.weight(x));
        });

    Index n = static_cast<Index>(last - index.data());
    ResultType ret(n);
    for (Index i = 0; i < n; ++i)
    {
        ret.exponent(i) = esum.exponent(index(i));
        ret.weight(i)   = esum.weight(index(i));
    }

    return ret;
}

//==============================================================================
// ExponentialSum class
//==============================================================================
namespace detail
{

template <typename ExpScalarT, typename WScalarT>
struct ExponentialSumTraits<ExponentialSum<ExpScalarT, WScalarT>>
{
    using ExponentsArray = Eigen::Array<ExpScalarT, Eigen::Dynamic, 1>;
    using WeightsArray   = Eigen::Array<WScalarT, Eigen::Dynamic, 1>;
};

} // namespace detail

///
/// ### ExponentialSum
///
/// Representation of an exponential sum function with its storage.
///
/// \tparam T  the scalar type for parameters
/// \tparam Size_ size of internal arrays to store exponents and weights. Set
///     Eigen::Dynamic for changing the size dynamically. Default is
///     Eigen::Dynamic.
/// \tparam MaxSize_ maximum size of internal arrays. Default is `Size_`.
///
template <typename ExpScalarT, typename WScalarT>
class ExponentialSum
    : public ExponentialSumBase<ExponentialSum<ExpScalarT, WScalarT>>
{
    using Base = ExponentialSumBase<ExponentialSum<ExpScalarT, WScalarT>>;

public:
    using Index = Eigen::Index;

    using ExponentsArray = typename Base::ExponentsArray;
    using WeightsArray   = typename Base::WeightsArray;
    using ExponentScalar = typename ExponentsArray::Scalar;
    using WeightScalar   = typename WeightsArray::Scalar;

protected:
    using IndexArray = Eigen::Array<Index, Eigen::Dynamic, 1>;

    ExponentsArray m_exponents;
    WeightsArray m_weights;

public:
    /// Default constructor
    ExponentialSum() = default;

    /// Create an exponential sum function with number of terms
    explicit ExponentialSum(Index n) : m_exponents(n), m_weights(n)
    {
    }

    /// Create an exponential sum function from other expression
    template <typename Derived>
    explicit ExponentialSum(const ExponentialSumBase<Derived>& other)
        : m_exponents(other.exponents()), m_weights(other.weights())
    {
    }

    /// Copy constructor
    ExponentialSum(const ExponentialSum&) = default;

    /// Move constructor
    ExponentialSum(ExponentialSum&&) = default;

    /// Destuctor
    ~ExponentialSum() = default;

    /// Copy assignment operator
    ExponentialSum& operator=(const ExponentialSum&) = default;

    /// Move assignment operator
    ExponentialSum& operator=(ExponentialSum&&) = default;

    /// Create an exponential sum function from expressions of arrays
    template <typename Derived1, typename Derived2>
    explicit ExponentialSum(const Eigen::DenseBase<Derived1>& exponents_,
                            const Eigen::DenseBase<Derived2>& weights_)
        : m_exponents(exponents_), m_weights(weights_)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived1);
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived2);
        eigen_assert(exponents_.size() == weights_.size() &&
                     "Size of exponents and weights array must be the same");
    }

    /// \return const reference to the array of exponents
    const ExponentsArray& exponents() const
    {
        return m_exponents;
    }

    /// \return  reference to the array of exponents
    ExponentsArray& exponents()
    {
        return m_exponents;
    }

    /// \return const reference to the array of exponents
    const WeightsArray& weights() const
    {
        return m_weights;
    }

    /// \return  reference to the array of exponents
    WeightsArray& weights()
    {
        return m_weights;
    }

    /// Set the number of exponential terms
    void resize(Index n)
    {
        m_exponents.resize(n);
        m_weights.resize(n);
    }

    void swap(ExponentialSum& other)
    {
        m_exponents.swap(other.m_exponents);
        m_weights.swap(other.m_weights);
    }

    using Base::size;
    using Base::exponent;
    using Base::weight;
    using Base::operator();

    ExponentScalar& exponent(Index i)
    {
        return m_exponents(i);
    }

    WeightScalar& weight(Index i)
    {
        return m_weights(i);
    }

    /// sort exponents \f$ \xi_i \f$ and weights \f$ w_i \f$ by the ratio
    /// \f$|w_i| / |\mathrm{Re}(\xi)|\f$
    void sortByDominanceRatio()
    {
        using Eigen::numext::abs;
        using Eigen::numext::real;
        using std::swap;

        if (size() <= Index(1))
        {
            return;
        }

        IndexArray order(IndexArray::LinSpaced(size(), 0, size() - 1));

        std::sort(order.data(), order.data() + order.size(),
                  [this](Index x, Index y) {
                      return abs(m_weights(x)) / abs(real(m_exponents(x))) >
                             abs(m_weights(y)) / abs(real(m_exponents(y)));
                  });

        ExponentsArray e_tmp(m_exponents);
        WeightsArray w_tmp(m_weights);
        for (Index i = 0; i < size(); ++i)
        {
            m_exponents(i) = e_tmp(order(i));
            m_weights(i)   = w_tmp(order(i));
        }
    }

    ///
    /// Set exponents and weights to a random number.
    ///
    /// Exponents \f$a_{i}\f$ are distributed on the right half-plane, i.e.,
    /// \f$Re(a_{i})>0,\f$ while weights are located at the region \f$w_{i}\in
    /// [0,1] + i[-1,1].\f$
    ///
    void setRandom()
    {
        static detail::random_exponent_impl<ExponentScalar> rnd_c;
        static detail::random_weight_impl<WeightScalar> rnd_w;

        for (Index i = 0; i < size(); ++i)
        {
            m_exponents(i) = rnd_c();
            m_weights(i)   = rnd_w();
        }
    }

    ///
    /// Merge the terms with same exponents on an exponential sum.
    ///
    /// \param[in] pred binary predicate which returns ​`true` if two
    ///   exponents can be regarded as the same. The signature of the predicate
    ///   function should be equivalent to the following:
    ///
    ///   ``` c++
    ///     bool pred(const Scalar& AI, const Scalar& aj);
    ///   ```
    ///
    ///   where `ai` and `aj` are the i-th and j-th exponentand `Scalar` is the
    ///   scalar type of original exponential sum, `esum`. The signature does
    ///   not need to have `const &`, but the function must not modify the
    ///   objects passed to it.
    ///
    void
    uniqueExponents(typename Eigen::NumTraits<ExponentScalar>::Real tolerance)
    {
        if (size() <= Index(1))
        {
            return;
        }
        IndexArray order(IndexArray::LinSpaced(size(), 0, size() - 1));
        std::sort(order.data(), order.data() + order.size(),
                  [this](Index x, Index y) {
                      return std::make_tuple(std::abs(m_exponents(x)),
                                             std::arg(m_exponents(x))) >
                             std::make_tuple(std::abs(m_exponents(y)),
                                             std::arg(m_exponents(y)));
                  });

        Index first = 0;
        Index ret   = first;
        Index last  = size();
        while (++first != last)
        {
            const auto lhs = m_exponents(order(ret));
            const auto rhs = m_exponents(order(first));
            // if (pred(m_exponents(order(ret)), m_exponents(order(first))))
            if (std::abs(rhs - lhs) <= tolerance * std::abs(rhs) ||
                std::abs(rhs - lhs) <= tolerance * std::abs(lhs))
            {
                m_weights(order(first)) += m_weights(order(ret));
            }
            else if (++ret != first)
            {
                m_exponents(order(ret)) = m_exponents(order(first));
                m_weights(order(ret))   = m_weights(order(first));
            }
        }
        ++ret;

        m_exponents.conservativeResize(ret);
        m_weights.conservativeResize(ret);

        return;
    }
};

//==============================================================================
// ExponentialSumWrapper class
//==============================================================================
namespace detail
{

template <typename ExpArrayT, typename WArrayT>
struct ExponentialSumTraits<ExponentialSumWrapper<ExpArrayT, WArrayT>>
{
    using ExponentsArray = ExpArrayT;
    using WeightsArray   = WArrayT;
};

} // namespace: detail

///
/// ### ExponentialSumWrapper
///
/// Expression of an exponential sum function formed by wrapping existing array
/// expressions.
///
template <typename ExpArrayT, typename WArrayT>
class ExponentialSumWrapper
    : public ExponentialSumBase<ExponentialSumWrapper<ExpArrayT, WArrayT>>
{
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(ExpArrayT);
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(WArrayT);
    using Base = ExponentialSumBase<ExponentialSumWrapper<ExpArrayT, WArrayT>>;

public:
    using Index = typename Base::Index;

    using ExponentsArray = typename Base::ExponentsArray;
    using WeightsArray   = typename Base::WeightsArray;
    using ExponentsArrayNested =
        typename Eigen::internal::ref_selector<ExponentsArray>::type;
    using WeightsArrayNested =
        typename Eigen::internal::ref_selector<WeightsArray>::type;

protected:
    ExponentsArrayNested m_exponents;
    WeightsArrayNested m_weights;

public:
    /// Create an exponential sum function from expressions of arrays
    explicit ExponentialSumWrapper(const ExponentsArray& exponents_,
                                   const WeightsArray& weights_)
        : m_exponents(exponents_), m_weights(weights_)
    {
        eigen_assert(exponents_.size() == weights_.size() &&
                     "Size of exponents and weights array must be the same");
    }

    /// \return const reference to the array of exponents
    const ExponentsArray& exponents() const
    {
        return m_exponents;
    }

    /// \return const reference to the array of exponents
    const WeightsArray& weights() const
    {
        return m_weights;
    }

    using Base::size;
    using Base::exponent;
    using Base::weight;
    using Base::operator();
};

///
/// Create an instance of `ExponentialSumWrapper` from given arrays.
///
template <typename ExpArrayT, typename WArrayT>
ExponentialSumWrapper<ExpArrayT, WArrayT>
makeExponentialSum(const Eigen::ArrayBase<ExpArrayT>& exponents,
                   const Eigen::ArrayBase<WArrayT>& weights)
{
    return ExponentialSumWrapper<ExpArrayT, WArrayT>(exponents.derived(),
                                                     weights.derived());
}

} // namespace: mxpfit

#endif /* MXPFIT_EXPONENTIAL_SUM_HPP */

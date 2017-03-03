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

#include <iosfwd>

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
        m_exponents.swap(other.exponent_);
        m_weights.swap(other.weight_);
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

/// Create an instance of `ExponentialSumWrapper` from given arrays.
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

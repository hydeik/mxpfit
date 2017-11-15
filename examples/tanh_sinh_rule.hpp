#ifndef QUAD_TANH_SINH_RULE_HPP
#define QUAD_TANH_SINH_RULE_HPP

#include <tuple>

#include <Eigen/Core>

#include "constants.hpp"

namespace quad
{

//==============================================================================
// Double exponential formula for numerical integration
//==============================================================================

///
/// ### DESinhTanhRule
///
/// \brief Double exponential quadrature for finite interval [-1, 1]
/// \tparam T the scalar type for quadrature nodes and weights
///
/// This class computes nodes and weights of the sinh-tanh quadrature rule which
/// can efficiently compute the integral over the finite interval. ///
///
/// \f[
///   I = \int_{-1}^{1}f(x) dx \approx \sum_{k=N_{-}}^{N_{+}} w_{k} f(x_k)
/// \f]
///
/// This sinh-tanh rule is obtained by applying the change of variable in the
/// form
///
/// \f[
///   x = \phi(t)
///     = \tanh\left(\frac{\pi}{2}\sinh(t)\right) ,\quad t \in (-\infty,\infty).
/// \f]
///
/// and discretizing the integral over \f$ t \f$ by the trapezoidal rule of the
/// step width \f$h\f$. The integration nodes and weighs are given as
///
/// \f[
///   x_k = \tanh\left(\frac{\pi}{2}\sinh(kh)\right), \\
///   w_k =
///   \frac{\frac{1}{2}h\pi\cosh(kh)}{\cosh^\left(\frac{1}{2}\pi\sinh(kh)\right)}.
/// \f]
///
template <typename T>
class DESinhTanhRule
{
public:
    constexpr static const T alpha = math::constant<T>::pi / 2;

    using Scalar    = T;
    using ArrayType = Eigen::Array<T, Eigen::Dynamic, 1>;
    using Index     = Eigen::Index;

private:
    ArrayType m_node;            // nodes
    ArrayType m_weight;          // weights
    ArrayType m_dist_from_lower; // distance of nodes from lower bound
    ArrayType m_dist_from_upper; // distance of nodes from upper bound
    ArrayType m_adj_diff;        // adjacent difference of nodes
public:
    /// Default constructor
    DESinhTanhRule() = default;
    /// Copy constructor
    DESinhTanhRule(const DESinhTanhRule&) = default;
    /// Move constructor
    DESinhTanhRule(DESinhTanhRule&&) = default;
    /// Destructor
    ~DESinhTanhRule() = default;

    /// Copy assignment operator
    DESinhTanhRule& operator=(const DESinhTanhRule&) = default;
    /// Move assignment operator
    DESinhTanhRule& operator=(DESinhTanhRule&&) = default;

    ///
    /// Construst `n`-points quadrature rule
    ///
    /// \param[in] n  number of integration points
    /// \param[in] tiny A threshold value that the integration weights can be
    ///    negligibly small. If the integrand is a regular function at
    ///    boundaries,
    ///
    /// \pre `n > 2` is required
    ///
    explicit DESinhTanhRule(Index n, T tiny)
    {
        compute(n, tiny);
    }

    ///
    /// \return number of integration points.
    ///
    Index size() const
    {
        return m_node.size();
    }
    ///
    /// \return const reference to the vector of integration points.
    ///
    const ArrayType& x() const
    {
        return m_node;
    }

    ///
    /// Get a node
    ///
    /// \param[in] i  index of integration point. `i < size()` required.
    /// \return `i`-th integration point.
    ///
    Scalar x(Index i) const
    {
        assert(i < size());
        return m_node[i];
    }
    ///
    /// \return const reference to the vector of integration weights.
    ///
    const ArrayType& w() const
    {
        return m_weight;
    }

    ///
    /// Get a weight
    ///
    /// \parma[in] i  index of integration point. `i < size()` required.
    /// \return `i`-th integration weight.
    ///
    Scalar w(Index i) const
    {
        assert(i < size());
        return m_weight[i];
    }
    ///
    /// Get the distance of node from lower bound, \f$ x + 1, \f$ where \f$ x
    /// \in [-1,1].\f$
    ///
    Scalar distanceFromLower(Index i) const
    {
        assert(i < size());
        return m_dist_from_lower[i];
    }
    ///
    /// Get the distance of node from upper bound, \f$ 1 - x, \f $where \f$ x
    /// \in [-1,1].\f$
    ///
    Scalar distanceFromUpper(Index i) const
    {
        assert(i < size());
        return m_dist_from_upper[i];
    }
    ///
    /// Get the adjacent difference of nodes.
    ///
    /// \param[in] i index of nodes
    ///
    /// \return `x(0)` for `i=0`, `x(i)-x(i-1)` for `i=1,2,...,size()-1.`
    ///
    Scalar adjacentDifference(Index i) const
    {
        assert(i < size());
        return m_adj_diff[i];
    }

    ///
    /// Compute nodes and weights of the \c n point Gauss-Legendre
    /// quadrature rule. Nodes are located in the interval [-1,1].
    ///
    /// \param[in] n  number of integration points
    /// \param[in] tiny A threshold value that the integration weights can be
    ///    negligibly small. If the integrand is a regular function at
    ///    boundaries,
    ///
    /// \pre `n > 2` is required
    ///
    void compute(Index n, T tiny)
    {
        using Eigen::numext::log;
        assert(T() < tiny && tiny < T(1));

        resize(n);

        const auto t_bound =
            log(log(T(2) / (alpha * tiny) * log(T(2) / tiny)) / alpha);
        const auto h     = T(2) * t_bound / (n - 1);
        const auto w_pre = h * alpha;

        const auto sh_half = std::sinh(h / T(2));
        const auto ch_half = std::cosh(h / T(2));

        T di_pre;

        {
            const auto u0        = alpha * std::sinh(-t_bound);
            const auto d0        = std::cosh(u0);
            m_node[0]            = std::tanh(u0);
            m_weight[0]          = w_pre * std::cosh(-t_bound) / (d0 * d0);
            m_dist_from_lower[0] = std::exp(u0) / d0;
            m_dist_from_upper[0] = std::exp(-u0) / d0;
            m_adj_diff[0]        = m_node[0];

            di_pre = d0;
        }

        for (Index i = 1; i < n; ++i)
        {
            const auto ti = -t_bound + T(i) * h;
            const auto si = alpha * std::sinh(ti);
            const auto ci = alpha * std::cosh(ti);
            const auto di = std::cosh(si);

            m_node[i]            = std::tanh(si);
            m_weight[i]          = w_pre * std::cosh(ti) / (di * di);
            m_dist_from_lower[i] = std::exp(si) / di;
            m_dist_from_upper[i] = std::exp(-si) / di;
            // alpha * (sinh(t[i]) - sinh(t[i-1]))
            const auto ui_diff = T(2) * sh_half * (ci * ch_half - si * sh_half);
            m_adj_diff[i]      = std::sinh(ui_diff) / (di * di_pre);
            di_pre             = di;
        }

        return;
    }

    ///
    /// Inverse of variable transformation
    ///
    /// \return \f$ t = \phi^{-1}(x) \f$
    ///
    static T inverse(T x)
    {
        return std::asinh(std::atanh(x) / alpha);
    }

private:
    void resize(Index n)
    {
        m_node.resize(n);
        m_weight.resize(n);
        m_dist_from_lower.resize(n);
        m_dist_from_upper.resize(n);
        m_adj_diff.resize(n);
    }
};

} // namespace: quad

#endif /* QUAD_TANH_SINH_RULE_HPP */

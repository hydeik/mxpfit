#ifndef QUAD_DOUBLE_EXPONENTIAL_HPP
#define QUAD_DOUBLE_EXPONENTIAL_HPP

#include <tuple>

#include <quad/quadrature_rule.hpp>

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
///   w_k = \frac{\frac{1}{2}h\pi\cosh(kh)}{\cosh^\left(\frac{1}{2}\pi\sinh(kh)\right)}.
/// \f]
///
template <typename T>
class DESinhTanhRule
{
public:
    constexpr static const T alpha = constant::pi<T>() / 2;

    using Scalar    = T;
    using ArrayType = Eigen::Array<T, Eigen::Dynamic, 1>;
    using Index     = Eigen::Index;

private:
    ArrayType m_node;            // nodes
    ArrayType m_weight;          // weights
    ArrayType m_dist_from_lower; // distance of nodes from lower bound
    ArrayType m_dist_from_upper; // distance of nodes from upper bound
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
        //
        // Compute nodes on x < 0 and corresponding weights
        //
        const auto h     = T(2) * t_bound / (n - 1);
        const auto w_pre = h * alpha;
        for (Index i = 0; i < n; ++i)
        {
            const auto ti = -t_bound + T(i) * h;
            const auto ui = alpha * std::sinh(ti);
            const auto di = std::cosh(ui);

            m_node[i]   = std::tanh(ui);
            m_weight[i] = w_pre * std::cosh(ti) / (di * di);

            m_dist_from_lower[i] = std::exp(ui) / di;
            m_dist_from_upper[i] = -std::exp(-ui) / di;
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
    }
};

namespace detail
{

//
// Generator for double exponential rule on half-open interval [0, inf]
//
template <typename T>
struct gen_de_exp_sinh_rule
{
    constexpr static const T alpha = constant::pi<T>() / 2;

    using Index = Eigen::Index;

    template <typename ArrayX, typename ArrayW>
    static void run(Eigen::DenseBase<ArrayX>& node,
                    Eigen::DenseBase<ArrayW>& weight, T t_lower, T t_upper)
    {
        const Index n = node.size(); // number of nodes

        assert(t_lower < t_upper);
        if (n == 0)
        {
            return;
        }

        if (n == 1)
        {
            std::tie(node[0], weight[0]) = m_nodeand_weight(T());
            return;
        }
        //
        // Compute nodes on x < 0 and corresponding weights
        //
        const auto h = (t_upper - t_lower) / (n - 1);
        for (Index i = 0; i < n; ++i)
        {
            std::tie(node[i], weight[i]) = m_nodeand_weight(t_lower + i * h);
            weight[i] *= h;
        }
    }

    static std::tuple<T, T> m_nodeand_weight(T t)
    {
        const auto u = alpha * std::sinh(t);
        const auto x = std::exp(u);      // node
        const auto w = x * std::cosh(t); // weight

        return std::make_tuple(x, w);
    }

    // Inverse of variable transformation
    static T inverse(T x)
    {
        return std::asinh(std::log(x) / alpha);
    }
};

//
// Generator for double exponential rule on half-open interval [0, inf] which is
// suitable for exponentially decaying function
//
template <typename T>
struct gen_de_exp_exp_rule
{
    using Index = Eigen::Index;

    template <typename ArrayX, typename ArrayW>
    static void run(Eigen::DenseBase<ArrayX>& node,
                    Eigen::DenseBase<ArrayW>& weight, T t_lower, T t_upper)
    {
        const Index n = node.size(); // number of nodes

        assert(t_lower < t_upper);
        if (n == 0)
        {
            return;
        }

        if (n == 1)
        {
            std::tie(node[0], weight[0]) = m_nodeand_weight(T());
            return;
        }

        const auto h = (t_upper - t_lower) / (n - 1);
        for (Index i = 0; i < n; ++i)
        {
            std::tie(node[i], weight[i]) = m_nodeand_weight(t_lower + i * h);
            weight[i] *= h;
        }
    }

    static std::tuple<T, T> m_nodeand_weight(T t)
    {
        const auto em = std::exp(-t);
        const auto x  = std::exp(t - em); // node
        const auto w  = x * (T(1) + em);

        return std::make_tuple(x, w);
    }
};

} // namespace: detail

//------------------------------------------------------------------------------
// Aliases that defines quadrature rule
//------------------------------------------------------------------------------

template <typename T>
using DEExpSinhRule = QuadratureRule<T, detail::gen_de_exp_sinh_rule<T>>;

template <typename T>
using DEExpExpRule = QuadratureRule<T, detail::gen_de_exp_exp_rule<T>>;

} // namespace: quadrature

#endif /* QUAD_DOUBLE_EXPONENTIAL_HPP */

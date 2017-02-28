///
/// \file shared_plan.hpp
///
/// Templated wrapper classes for FFTW3
///
#ifndef FFTW3_SHARED_PLAN_HPP
#define FFTW3_SHARED_PLAN_HPP

#include <cassert>
#include <complex>
#include <memory>
#include <type_traits>

#include <fftw3.h>

namespace fftw3 {
namespace detail {

///
/// @internal
///
/// Cast pointer of the C++ arithmetic types to corresponding FFTW type.
///
inline float* fftw_cast(float* ptr) { return ptr; }

inline double* fftw_cast(double* ptr) { return ptr; }

inline long double* fftw_cast(long double* ptr) { return ptr; }

inline fftwf_complex* fftw_cast(std::complex<float>* ptr) {
    return reinterpret_cast<fftwf_complex*>(ptr);
}

inline fftw_complex* fftw_cast(std::complex<double>* ptr) {
    return reinterpret_cast<fftw_complex*>(ptr);
}

inline fftwl_complex* fftw_cast(std::complex<long double>* ptr) {
    return reinterpret_cast<fftwl_complex*>(ptr);
}
///
/// @internal
///
/// Cast const pointer of the C++ arithmetic types to corresponding FFTW type.
///
/// @note  Ugly `const_cast` is mandatory, as FFTW3 uses non-const pointer.
/// pointer.
///
inline float* fftw_cast(const float* ptr) { return const_cast<float*>(ptr); }

inline double* fftw_cast(const double* ptr) { return const_cast<double*>(ptr); }

inline long double* fftw_cast(const long double* ptr) {
    return const_cast<long double*>(ptr);
}

inline fftwf_complex* fftw_cast(const std::complex<float>* ptr) {
    return reinterpret_cast<fftwf_complex*>(
        const_cast<std::complex<float>*>(ptr));
}

inline fftw_complex* fftw_cast(const std::complex<double>* ptr) {
    return reinterpret_cast<fftw_complex*>(
        const_cast<std::complex<double>*>(ptr));
}

inline fftwl_complex* fftw_cast(const std::complex<long double>* ptr) {
    return reinterpret_cast<fftwl_complex*>(
        const_cast<std::complex<long double>*>(ptr));
}

///
/// @internal
///
/// Mutex for FFTW plan
///
template <typename T>
struct fftw_plan_mutex {
    static std::mutex m;
};

template <typename T>
std::mutex fftw_plan_mutex<T>::m;

/// @internal
///
/// 1D FFT plans
///
/// \@tparam  T a floating point type compatible to the FFTW internal type
///
template <typename T>
struct fftw_impl;

#define FFTW_IMPL_SPECIALIZATION_MACRO(PRECISION, PREFIX)                      \
    template <>                                                                \
    struct fftw_impl<PRECISION> {                                              \
        using real_type    = PRECISION;                                        \
        using complex_type = std::complex<real_type>;                          \
        using plan_type    = PREFIX##_plan;                                    \
        using plan_s_type  = typename std::remove_pointer<plan_type>::type;    \
        using plan_deleter = decltype(&PREFIX##_destroy_plan);                 \
        using r2r_kind     = PREFIX##_r2r_kind;                                \
                                                                               \
        static plan_type make_plan_fft(int n, int howmany,                     \
                                       const complex_type* in,                 \
                                       complex_type* out, unsigned flags) {    \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_dft(1, &n, howmany, fftw_cast(in),       \
                                          nullptr, 1, n, fftw_cast(out),       \
                                          nullptr, 1, n, FFTW_FORWARD, flags); \
        }                                                                      \
                                                                               \
        static plan_type make_plan_ifft(int n, int howmany,                    \
                                        const complex_type* in,                \
                                        complex_type* out, unsigned flags) {   \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_dft(                                     \
                1, &n, howmany, fftw_cast(in), nullptr, 1, n, fftw_cast(out),  \
                nullptr, 1, n, FFTW_BACKWARD, flags);                          \
        }                                                                      \
                                                                               \
        static plan_type make_plan_fft(int n, int howmany,                     \
                                       const real_type* in, complex_type* out, \
                                       unsigned flags) {                       \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_dft_r2c(1, &n, howmany, fftw_cast(in),   \
                                              nullptr, 1, n, fftw_cast(out),   \
                                              nullptr, 1, n / 2 + 1, flags);   \
        }                                                                      \
                                                                               \
        static plan_type make_plan_ifft(int n, int howmany, complex_type* in,  \
                                        real_type* out, unsigned flags) {      \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_dft_c2r(                                 \
                1, &n, howmany, fftw_cast(in), nullptr, 1, n / 2 + 1,          \
                fftw_cast(out), nullptr, 1, n, flags);                         \
        }                                                                      \
                                                                               \
        static plan_type make_plan_r2r(int n, int howmany,                     \
                                       const real_type* in, real_type* out,    \
                                       PREFIX##_r2r_kind kind,                 \
                                       unsigned flags) {                       \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            return PREFIX##_plan_many_r2r(1, &n, howmany, fftw_cast(in),       \
                                          nullptr, 1, n, fftw_cast(out),       \
                                          nullptr, 1, n, &kind, flags);        \
        }                                                                      \
                                                                               \
        static void destroy_plan(plan_type p) {                                \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            PREFIX##_destroy_plan(p);                                          \
        }                                                                      \
                                                                               \
        static void execute(plan_type p, const complex_type* in,               \
                            complex_type* out) {                               \
            PREFIX##_execute_dft(p, fftw_cast(in), fftw_cast(out));            \
        }                                                                      \
                                                                               \
        static void execute(plan_type p, const real_type* in,                  \
                            complex_type* out) {                               \
            PREFIX##_execute_dft_r2c(p, fftw_cast(in), fftw_cast(out));        \
        }                                                                      \
                                                                               \
        static void execute(plan_type p, complex_type* in, real_type* out) {   \
            PREFIX##_execute_dft_c2r(p, fftw_cast(in), fftw_cast(out));        \
        }                                                                      \
                                                                               \
        static void execute(plan_type p, const real_type* in,                  \
                            real_type* out) {                                  \
            PREFIX##_execute_r2r(p, fftw_cast(in), fftw_cast(out));            \
        }                                                                      \
                                                                               \
        static void cleanup() {                                                \
            std::lock_guard<std::mutex> lock(fftw_plan_mutex<real_type>::m);   \
            PREFIX##_cleanup();                                                \
        }                                                                      \
    };

FFTW_IMPL_SPECIALIZATION_MACRO(float, fftwf);
FFTW_IMPL_SPECIALIZATION_MACRO(double, fftw);
FFTW_IMPL_SPECIALIZATION_MACRO(long double, fftwl);

#undef FFTW_IMPL_SPECIALIZATION_MACRO

} // namespace: detail

///
/// Fast Fourier transform (1D)
///
/// Wrapper for one-dimensional complex-to-complex or real-to-complex forward
/// transformation.
///
template <typename T>
struct FFT {
    using Impl = detail::fftw_impl<T>;

    /// Real scalar type
    using Real = T;
    /// Complex scalar type
    using Complex = std::complex<Real>;
    /// Pointer to the FFT plan
    using PlanPointer = std::shared_ptr<typename Impl::plan_s_type>;

    ///
    /// Create an FFT plan for complex-to-complex transform
    ///
    /// @param[in] n
    ///   size of transform dimension. `n` should be a positive integer.
    /// @param[in] howmany
    ///   number of transforms to compute
    /// @param[in] in
    ///   complex pointer that points the input array of transform
    /// @param[in,out] out
    ///   complex pointer that points the output array of transform
    ///
    /// @remark `in` and `out` need not be initialized, but they must be
    /// allocated. The `out` array is overwritten during planning.
    ///
    static PlanPointer make_plan(int n, int howmany, const Complex* in,
                                 Complex* out) {
        return PlanPointer(
            Impl::make_plan_fft(n, howmany, in, out, FFTW_MEASURE),
            [](typename Impl::plan_type p) { Impl::destroy_plan(p); });
    }
    ///
    /// Execute the transform
    ///
    /// @param[in] plan
    ///   The pointer to plan created by `FFT::make_plan`.
    /// @param[in] in
    ///   complex pointer that points the input array of transform
    /// @param[out] out
    ///   complex pointer that points the output array of transform
    ///
    static void run(const PlanPointer& plan, const Complex* in, Complex* out) {
        Impl::execute(plan.get(), in, out);
    }

    static PlanPointer make_plan(int n, int howmany, const Real* in,
                                 Complex* out) {
        return PlanPointer(
            Impl::make_plan_fft(n, howmany, in, out, FFTW_MEASURE),
            [](typename Impl::plan_type p) { Impl::destroy_plan(p); });
    }

    static void run(const PlanPointer& plan, const Real* in, Complex* out) {
        Impl::execute(plan.get(), in, out);
    }
};

///
/// Inverse Fast Fourier transform (1D)
///
/// Wrapper for one-dimensional complex-to-complex or complex-to-real inverse
/// transformation.
///
template <typename T>
struct IFFT {
    using Impl = detail::fftw_impl<T>;

    using Real        = T;
    using Complex     = std::complex<Real>;
    using PlanPointer = std::shared_ptr<typename Impl::plan_s_type>;

    //--------------------------------------------------------------------------
    // Complex-to-complex transform
    //--------------------------------------------------------------------------
    static PlanPointer make_plan(int n, int howmany, const Complex* in,
                                 Complex* out) {
        return PlanPointer(
            Impl::make_plan_ifft(n, howmany, in, out, FFTW_MEASURE),
            [](typename Impl::plan_type p) { Impl::destroy_plan(p); });
    }

    static void run(const PlanPointer& plan, const Complex* in, Complex* out) {
        Impl::execute(plan.get(), in, out);
    }

    //--------------------------------------------------------------------------
    // Complex-to-real transform
    //--------------------------------------------------------------------------
    static PlanPointer make_plan(int n, int howmany, Complex* in, Real* out) {
        return PlanPointer(
            Impl::make_plan_ifft(n, howmany, in, out, FFTW_MEASURE),
            [](typename Impl::plan_type p) { Impl::destroy_plan(p); });
    }

    static void run(const PlanPointer& plan, Complex* in, Real* out) {
        Impl::execute(plan.get(), in, out);
    }
};

//==============================================================================
// Real-to-real transform
//==============================================================================

namespace detail {

template <typename T, typename fftw_impl<T>::r2r_kind Kind>
struct transform_r2r_impl {
    using Impl = fftw_impl<T>;

    using Real        = T;
    using PlanPointer = std::shared_ptr<typename Impl::plan_s_type>;

    static PlanPointer make_plan(int n, int howmany, const Real* in,
                                 Real* out) {
        return PlanPointer(
            Impl::make_plan_r2r(n, howmany, in, out, Kind, FFTW_MEASURE),
            [](typename Impl::plan_type p) { Impl::destroy_plan(p); });
    }

    static void run(const PlanPointer& plan, Real* in, Real* out) {
        Impl::execute(plan.get(), in, out);
    }
};

} // namespace detail

///
/// ### DHT
///
/// Discrete Hartley transform (1D)
///
template <typename T>
using DHT = detail::transform_r2r_impl<T, FFTW_DHT>;
///
/// ### DCT
///
/// Discrete cosine transform (1D)
///
template <typename T>
using DCT1 = detail::transform_r2r_impl<T, FFTW_REDFT00>;

template <typename T>
using DCT2 = detail::transform_r2r_impl<T, FFTW_REDFT10>;

template <typename T>
using DCT3 = detail::transform_r2r_impl<T, FFTW_REDFT01>;

template <typename T>
using DCT4 = detail::transform_r2r_impl<T, FFTW_REDFT11>;

///
/// ### DST
///
/// Discrete sine transform
///
template <typename T>
using DST1 = detail::transform_r2r_impl<T, FFTW_RODFT00>;

template <typename T>
using DST2 = detail::transform_r2r_impl<T, FFTW_RODFT10>;

template <typename T>
using DST3 = detail::transform_r2r_impl<T, FFTW_RODFT01>;

template <typename T>
using DST4 = detail::transform_r2r_impl<T, FFTW_RODFT11>;

} // namespace: fftw3

#endif /* FFTW3_SHARED_PLAN_HPP */

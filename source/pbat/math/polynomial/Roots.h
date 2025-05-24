#ifndef PBAT_MATH_POLYNOMIAL_ROOTS_H
#define PBAT_MATH_POLYNOMIAL_ROOTS_H

/**
 * @file Roots.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Root-finders for polynomial of arbitrary degree
 *
 * We lightly wrap [Cem Yuksel](https://profiles.faculty.utah.edu/u0852752)'s
 * [cyCodeBase](https://github.com/cemyuksel/cyCodeBase/tree/master) polynomial root-finding
 * functions to work in our codebase.
 *
 * See \cite cem2022polyroot
 *
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wconversion"
#elif defined(__GNUC__) or defined(__GNUG__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wconversion"
#elif defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4365) // signed/unsigned mismatch
#endif

//-------------------------------------------------------------------------------

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

#if !defined(CY_NO_INTRIN_H) && !defined(CY_NO_EMMINTRIN_H) && !defined(CY_NO_IMMINTRIN_H)
    #include <immintrin.h>
#endif

namespace pbat::math::polynomial::detail {

// clang-format off

// Copyright (c) 2016, Cem Yuksel <cem@cemyuksel.com>
// All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//-------------------------------------------------------------------------------

#ifndef _CY_CORE_H_INCLUDED_
#define _CY_CORE_H_INCLUDED_

//-------------------------------------------------------------------------------

#ifndef _CY_CORE_MEMCPY_LIMIT
    #define _CY_CORE_MEMCPY_LIMIT 256
#endif

//-------------------------------------------------------------------------------
namespace cy {
//-------------------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////
// Compiler compatibility
//////////////////////////////////////////////////////////////////////////

#if defined(__INTEL_COMPILER)
    #define _CY_COMPILER_INTEL                             __INTEL_COMPILER
    #define _CY_COMPILER_VER_MEETS(msc, gcc, clang, intel) _CY_COMPILER_INTEL >= intel
    #define _CY_COMPILER_VER_BELOW(msc, gcc, clang, intel) _CY_COMPILER_INTEL < intel
#elif defined(__clang__)
    #define _CY_COMPILER_CLANG \
        (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
    #define _CY_COMPILER_VER_MEETS(msc, gcc, clang, intel) _CY_COMPILER_CLANG >= clang
    #define _CY_COMPILER_VER_BELOW(msc, gcc, clang, intel) _CY_COMPILER_CLANG < clang
#elif defined(_MSC_VER)
    #define _CY_COMPILER_MSC                               _MSC_VER
    #define _CY_COMPILER_VER_MEETS(msc, gcc, clang, intel) _CY_COMPILER_MSC >= msc
    #define _CY_COMPILER_VER_BELOW(msc, gcc, clang, intel) _CY_COMPILER_MSC < msc
#elif __GNUC__
    #define _CY_COMPILER_GCC                               (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
    #define _CY_COMPILER_VER_MEETS(msc, gcc, clang, intel) _CY_COMPILER_GCC >= gcc
    #define _CY_COMPILER_VER_BELOW(msc, gcc, clang, intel) _CY_COMPILER_GCC < gcc
#else
    #define _CY_COMPILER_UNKNOWN
    #define _CY_COMPILER_VER_MEETS(msc, gcc, clang, intel) false
    #define _CY_COMPILER_VER_BELOW(msc, gcc, clang, intel) false
#endif

// constexpr
#ifndef __cpp_constexpr
    #if _CY_COMPILER_VER_MEETS(1900, 40600, 30100, 1310)
        #define __cpp_constexpr
    #else
        #define constexpr
    #endif
#endif

// nullptr
#if _CY_COMPILER_VER_BELOW(1600, 40600, 20900, 1210)
class _cy_nullptr_t
{
  public:
    template <class T>
    operator T*() const
    {
        return 0;
    }
    template <class C, class T>
    operator T C::*() const
    {
        return 0;
    }

  private:
    void operator&() const {}
};
static _cy_nullptr_t nullptr;
#endif

// template aliases
#define _CY_TEMPLATE_ALIAS_UNPACK(...) __VA_ARGS__
#if _CY_COMPILER_VER_BELOW(1800, 40700, 30000, 1210)
    #define _CY_TEMPLATE_ALIAS(template_name, template_equivalent)                 \
        class template_name : public _CY_TEMPLATE_ALIAS_UNPACK template_equivalent \
        {                                                                          \
        }
#else
    #define _CY_TEMPLATE_ALIAS(template_name, template_equivalent) \
        using template_name = _CY_TEMPLATE_ALIAS_UNPACK template_equivalent
#endif

// std::is_trivially_copyable
#if _CY_COMPILER_VER_MEETS(1700, 50000, 30400, 1300)
    #define _cy_std_is_trivially_copyable 1
#endif

// restrict
#if defined(__INTEL_COMPILER)
// # define restrict restrict
#elif defined(__clang__)
    #define restrict __restrict__
#elif defined(_MSC_VER)
    #define restrict __restrict
#elif __GNUC__
    #define restrict __restrict__
#else
    #define restrict
#endif

// alignment
#if _CY_COMPILER_VER_BELOW(1900, 40800, 30000, 1500)
    #if defined(_MSC_VER)
        #define alignas(alignment_size) __declspec(align(alignment_size))
    #else
        #define alignas(alignment_size) __attribute__((aligned(alignment_size)))
    #endif
#endif

// final, override
#if _CY_COMPILER_VER_BELOW(1700, 40700, 20900, 1210)
    #define override
    #if defined(_MSC_VER)
        #define final sealed
    #else
        #define final
    #endif
#endif

// static_assert
#if _CY_COMPILER_VER_BELOW(1900, 60000, 20500, 1800)
    #define static_assert(condition, message) assert(condition&& message)
#endif

// unrestricted unions
#ifndef __cpp_unrestricted_unions
    #if _CY_COMPILER_VER_MEETS(1900, 40600, 30000, 1400)
        #define __cpp_unrestricted_unions
    #endif
#endif

// nodiscard
// #if _CY_COMPILER_VER_MEETS(1901,40800,30000,1500)
#if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
    #define CY_NODISCARD [[nodiscard]]
#else
    #define CY_NODISCARD
#endif

// default and deleted class member functions
#if _CY_COMPILER_VER_MEETS(1800, 40400, 30000, 1200)
    #define CY_CLASS_FUNCTION_DEFAULT = default;
    #define CY_CLASS_FUNCTION_DELETE  = delete;
#else
    #define CY_CLASS_FUNCTION_DEFAULT \
        {                             \
        }
    #define CY_CLASS_FUNCTION_DELETE                         \
        {                                                    \
            static_assert(false, "Calling deleted method."); \
        }
#endif

// switch statements where default cannot be reached
#if defined(__INTEL_COMPILER) || defined(__clang__) || defined(__GNUC__)
    #define nodefault \
        default: __builtin_unreachable()
#elif defined(_MSC_VER)
    #define nodefault \
        default: __assume(0)
#else
    #define nodefault default:
#endif

//////////////////////////////////////////////////////////////////////////
// Auto Vectorization
//////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
    #if _MSC_VER >= 1700
        #define _CY_IVDEP __pragma(loop(ivdep))
    #endif
#elif defined __GNUC__
    #if _CY_GCC_VER >= 40900
        #define _CY_IVDEP _Pragma("GCC ivdep");
    #endif
#elif defined __clang__
    #if _CY_CLANG_VER >= 30500
        #define _CY_IVDEP _Pragma("clang loop vectorize(enable) interleave(enable)");
    #endif
#else
    // # define _CY_IVDEP _Pragma("ivdep");
    #define _CY_IVDEP
#endif

#ifndef _CY_IVDEP
    #define _CY_IVDEP
#endif

#define _CY_IVDEP_FOR _CY_IVDEP for

//////////////////////////////////////////////////////////////////////////
// Disabling MSVC's non-standard depreciation warnings
//////////////////////////////////////////////////////////////////////////

#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
    #define _CY_CRT_SECURE_NO_WARNINGS     __pragma(warning(push)) __pragma(warning(disable : 4996))
    #define _CY_CRT_SECURE_RESUME_WARNINGS __pragma(warning(pop))
#else
    #define _CY_CRT_SECURE_NO_WARNINGS
    #define _CY_CRT_SECURE_RESUME_WARNINGS
#endif

//////////////////////////////////////////////////////////////////////////
// Math functions
//////////////////////////////////////////////////////////////////////////

//!@name Common math function templates

template <typename T>
CY_NODISCARD inline T Max(T v1, T v2)
{
    return v1 >= v2 ? v1 : v2;
}
template <typename T>
CY_NODISCARD inline T Min(T v1, T v2)
{
    return v1 <= v2 ? v1 : v2;
}
template <typename T>
CY_NODISCARD inline T Max(T v1, T v2, T v3)
{
    return Max(Max(v1, v2), v3);
}
template <typename T>
CY_NODISCARD inline T Min(T v1, T v2, T v3)
{
    return Min(Min(v1, v2), v3);
}
template <typename T>
CY_NODISCARD inline T Max(T v1, T v2, T v3, T const v4)
{
    return Max(Max(v1, v2), Max(v3, v4));
}
template <typename T>
CY_NODISCARD inline T Min(T v1, T v2, T v3, T const v4)
{
    return Min(Min(v1, v2), Min(v3, v4));
}
template <typename T>
CY_NODISCARD inline T Clamp(T v, T minVal = T(0), T maxVal = T(1))
{
    return Min(maxVal, Max(minVal, v));
}

template <typename T>
CY_NODISCARD inline T ACosSafe(T v)
{
    return (T)std::acos(Clamp(v, T(-1), T(1)));
}
template <typename T>
CY_NODISCARD inline T ASinSafe(T v)
{
    return (T)std::asin(Clamp(v, T(-1), T(1)));
}
template <typename T>
CY_NODISCARD inline T Sqrt(T v)
{
    return (T)std::sqrt(v);
}
template <typename T>
CY_NODISCARD inline T SqrtSafe(T v)
{
    return (T)std::sqrt(Max(v, T(0)));
}

#ifdef _INCLUDED_IMM
template <>
[[maybe_unused]] CY_NODISCARD inline float Sqrt<float>(float v)
{
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ps1(v)));
}
template <>
[[maybe_unused]] CY_NODISCARD inline float SqrtSafe<float>(float v)
{
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ps1(Max(v, 0.0f))));
}
template <>
[[maybe_unused]] CY_NODISCARD inline double Sqrt<double>(double v)
{
    __m128d t = _mm_set1_pd(v);
    return _mm_cvtsd_f64(_mm_sqrt_sd(t, t));
}
template <>
[[maybe_unused]] CY_NODISCARD inline double SqrtSafe<double>(double v)
{
    __m128d t = _mm_set1_pd(Max(v, 0.0));
    return _mm_cvtsd_f64(_mm_sqrt_sd(t, t));
}
#endif

template <typename T>
constexpr inline T Pi()
{
    return T(3.141592653589793238462643383279502884197169);
}

template <typename T>
CY_NODISCARD inline bool IsFinite(T v)
{
    return std::numeric_limits<T>::is_integer || std::isfinite(v);
}

//////////////////////////////////////////////////////////////////////////
// Memory Operations
//////////////////////////////////////////////////////////////////////////

template <typename T>
inline void MemCopy(T* restrict dest, T const* restrict src, size_t count)
{
#ifdef _cy_std_is_trivially_copyable
    if (std::is_trivially_copyable<T>())
    {
        memcpy(dest, src, (count) * sizeof(T));
    }
    else
#endif
        for (size_t i = 0; i < count; ++i)
            dest[i] = src[i];
}

template <typename T, typename S>
inline void MemConvert(T* restrict dest, S const* restrict src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dest[i] = reinterpret_cast<T>(src[i]);
}

template <typename T>
inline void MemClear(T* dest, size_t count)
{
    memset(dest, 0, count * sizeof(T));
}

template <typename T>
inline void SwapBytes(T& v1, T& v2)
{
    char t[sizeof(T)];
    memcpy(&t, &v1, sizeof(T));
    memcpy(&v1, &v2, sizeof(T));
    memcpy(&v2, &t, sizeof(T));
}
template <typename T>
inline void Swap(T& v1, T& v2)
{
    if (std::is_trivially_copyable<T>::value)
    {
        T t = v1;
        v1  = v2;
        v2  = t;
    }
    else
        SwapBytes(v1, v2);
}

/////////////////////////////////////////////////////////////////////////////////
// Sorting functions
/////////////////////////////////////////////////////////////////////////////////

template <bool ascending, typename T>
inline void Sort2(T& r0, T& r1, T const& v0, T const& v1)
{
    if (ascending)
    {
        r0 = Min(v0, v1);
        r1 = Max(v0, v1);
    }
    else
    {
        r0 = Max(v0, v1);
        r1 = Min(v0, v1);
    }
}

template <bool ascending, typename T>
inline void Sort2(T r[2], T const v[2])
{
    r[1 - ascending] = Min(v[0], v[1]);
    r[ascending]     = Max(v[0], v[1]);
}

template <bool ascending, typename T>
void Sort3(T& r0, T& r1, T& r2, T const& v0, T const& v1, T const& v2)
{
    T n01   = Min(v0, v1);
    T x01   = Max(v0, v1);
    T n2x01 = Min(v2, x01);
    r1      = Max(n01, n2x01);
    if (ascending)
    {
        r0 = Min(n2x01, n01);
        r2 = Max(x01, v2);
    }
    else
    {
        r0 = Max(x01, v2);
        r2 = Min(n2x01, n01);
    }
}

template <bool ascending, typename T>
void Sort3(T r[3], T const v[3])
{
    T n01   = Min(v[0], v[1]);
    T x01   = Max(v[0], v[1]);
    T n2x01 = Min(v[2], x01);
    T r0    = Min(n2x01, n01);
    T r1    = Max(n01, n2x01);
    T r2    = Max(x01, v[2]);
    if (ascending)
    {
        r[0] = r0;
        r[1] = r1;
        r[2] = r2;
    }
    else
    {
        r[0] = r2;
        r[1] = r1;
        r[2] = r0;
    }
}

template <bool ascending, typename T>
inline void Sort4(T& r0, T& r1, T& r2, T& r3, T const& v0, T const& v1, T const& v2, T const& v3)
{
    T n01 = Min(v0, v1);
    T x01 = Max(v0, v1);
    T n23 = Min(v2, v3);
    T x23 = Max(v2, v3);
    T x02 = Max(n23, n01);
    T n13 = Min(x01, x23);
    if (ascending)
    {
        r0 = Min(n01, n23);
        r1 = Min(x02, n13);
        r2 = Max(n13, x02);
        r3 = Max(x23, x01);
    }
    else
    {
        r0 = Max(x23, x01);
        r1 = Max(n13, x02);
        r2 = Min(x02, n13);
        r3 = Min(n01, n23);
    }
}

template <bool ascending, typename T>
inline void Sort4(T r[4], T const v[4])
{
    T n01 = Min(v[0], v[1]);
    T x01 = Max(v[0], v[1]);
    T n23 = Min(v[2], v[3]);
    T x23 = Max(v[2], v[3]);
    T x02 = Max(n23, n01);
    T n13 = Min(x01, x23);
    T r0  = Min(n01, n23);
    T r1  = Min(x02, n13);
    T r2  = Max(n13, x02);
    T r3  = Max(x23, x01);
    if (ascending)
    {
        r[0] = r0;
        r[1] = r1;
        r[2] = r2;
        r[3] = r3;
    }
    else
    {
        r[0] = r3;
        r[1] = r2;
        r[2] = r1;
        r[3] = r0;
    }
}

//////////////////////////////////////////////////////////////////////////

//-------------------------------------------------------------------------------
} // namespace cy
//-------------------------------------------------------------------------------

#endif // _CY_CORE_H_INCLUDED_

// Copyright (c) 2022, Cem Yuksel <cem@cemyuksel.com>
// All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//-------------------------------------------------------------------------------

#ifndef _CY_POLYNOMIAL_H_INCLUDED_
#define _CY_POLYNOMIAL_H_INCLUDED_

//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
namespace cy {
//-------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////
//
//! @name General Polynomial Functions
//!
//! These functions can be used for evaluating polynomials and their derivatives,
//! computing their derivatives, deflating them, and inflating them.
//!
/////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------

//! Evaluates the given polynomial of degree `N` at `x`.
//!
//! The coefficients in the `coef` array must be in the order of increasing degrees.
template <int N, typename ftype>
inline ftype PolynomialEval(ftype const coef[N + 1], ftype x)
{
    ftype r = coef[N];
    for (int i = N - 1; i >= 0; --i)
        r = r * x + coef[i];
    return r;
}

//-------------------------------------------------------------------------------

//! Evaluates the given polynomial and its derivative at `x`.
//!
//! This function does not require computing the derivative of the polynomial,
//! but it is slower than evaluating the polynomial and its precomputed derivative separately.
//! Therefore, it is not recommended when the polynomial and it derivative are computed repeatedly.
//! The coefficients in the `coef` array must be in the order of increasing degrees.
template <int N, typename ftype>
inline ftype PolynomialEvalWithDeriv(ftype& derivativeValue, ftype const coef[N + 1], ftype x)
{
    if constexpr (N < 1)
    {
        derivativeValue = 0;
        return coef[0];
    }
    else
    {
        ftype p  = coef[N] * x + coef[N - 1];
        ftype dp = coef[N];
        for (int i = N - 2; i >= 0; --i)
        {
            dp = dp * x + p;
            p  = p * x + coef[i];
        }
        derivativeValue = dp;
        return p;
    }
}

//-------------------------------------------------------------------------------

//! Computes the polynomial's derivative and sets the coefficients of the `deriv` array.
//!
//! Note that a degree N polynomial's derivative is degree `N-1` and has `N` coefficients.
//! The coefficients are in the order of increasing degrees.
template <int N, typename ftype>
inline void PolynomialDerivative(ftype deriv[N], ftype const coef[N + 1])
{
    deriv[0] = coef[1];
    for (int i = 2; i <= N; ++i)
        deriv[i - 1] = i * coef[i];
}

//-------------------------------------------------------------------------------

//! Deflates the given polynomial using one of its known roots.
//!
//! Stores the coefficients of the deflated polynomial in the `defPoly` array.
//! Let f(x) represent the given polynomial.
//! This computes the deflated polynomial g(x) of a lower degree such that
//!
//! `f(x) = (x - root) * g(x)`.
//!
//! The given root must be a valid root of the given polynomial.
//! Note that the deflated polynomial has degree `N-1` with `N` coefficients.
//! The coefficients are in the order of increasing degrees.
template <int N, typename ftype>
inline void PolynomialDeflate(ftype defPoly[N], ftype const coef[N + 1], ftype root)
{
    defPoly[N - 1] = coef[N];
    for (int i = N - 1; i > 0; --i)
        defPoly[i - 1] = coef[i] + root * defPoly[i];
}

//-------------------------------------------------------------------------------

//! Inflates the given polynomial using the given root.
//!
//! Stores the coefficients of the inflated polynomial in the `infPoly` array.
//! Let f(x) represent the given polynomial.
//! This computes the inflated polynomial g(x) of a higher degree such that
//!
//! `g(x) = (x - root) * f(x)`.
//!
//! Note that the inflated polynomial has degree `N+1` with `N+2` coefficients.
//! The coefficients are in the order of increasing degrees.
template <int N, typename ftype>
inline void PolynomialInflate(ftype infPoly[N + 2], ftype const coef[N + 1], ftype root)
{
    infPoly[N + 1] = coef[N];
    for (int i = N - 1; i >= 0; --i)
        infPoly[i + 1] = coef[i] - root * infPoly[i + 1];
    infPoly[0] = -root * coef[0];
}

//-------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////
//!@{
//!
//! @name Polynomial Root Finding Functions
//!
//! These functions find polynomial roots.
//! They can be limited to a finite bounds between `xMin` and `xMax`.
//! Functions for degree 3 and higher polynomials use numerical root finding.
//! By default, they use Newton iterations defined in `RootFinderNewton` as their
//! numerical root finding method, but they can also be used with a custom class
//! that provides the same interface.
//!
//! The given `xError` parameter is passed on to the numerical root finder.
//! The default value is 0, which aims to find the root up to numerical precision.
//! This might be too slow. Therefore, for a high-performance implementation,
//! it is recommended to use a non-zero error threshold for `xError`.
//!
//! If `boundError` is false (the default value), `RootFinderNewton` does not
//! guarantee satisfying the error bound `xError`, but it almost always does.
//! Keeping `boundError` false is recommended for improved performance.
//!
/////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------

template <typename ftype>
constexpr ftype PolynomialDefaultError()
{
    return ftype(0);
}
template <>
[[maybe_unused]] constexpr float PolynomialDefaultError<float>()
{
    return 6e-4f;
}
template <>
[[maybe_unused]] constexpr double PolynomialDefaultError<double>()
{
    return 6e-7;
}

//-------------------------------------------------------------------------------

class RootFinderNewton;

#define _CY_POLY_TEMPLATE_N                     \
    template <                                  \
        int N,                                  \
        typename ftype,                         \
        bool boundError     = false,            \
        typename RootFinder = RootFinderNewton> \
    CY_NODISCARD inline
#define _CY_POLY_TEMPLATE_R                                                                    \
    template <typename ftype, bool boundError = false, typename RootFinder = RootFinderNewton> \
    CY_NODISCARD inline
#define _CY_POLY_TEMPLATE_A   \
    template <typename ftype> \
    CY_NODISCARD inline
#define _CY_POLY_TEMPLATE_D   \
    template <typename ftype> \
    inline

#define _CY_POLY_TEMPLATE_NC                    \
    template <                                  \
        int N,                                  \
        typename ftype,                         \
        bool boundError     = false,            \
        typename RootFinder = RootFinderNewton, \
        typename RootCallback>                  \
    inline
#define _CY_POLY_TEMPLATE_RC                    \
    template <                                  \
        typename ftype,                         \
        bool boundError     = false,            \
        typename RootFinder = RootFinderNewton, \
        typename RootCallback>                  \
    inline
#define _CY_POLY_TEMPLATE_AC                         \
    template <typename ftype, typename RootCallback> \
    inline

//-------------------------------------------------------------------------------

//! Finds the roots of the given polynomial between `xMin` and `xMax` and returns the number of
//! roots found.
_CY_POLY_TEMPLATE_N int PolynomialRoots(
    ftype roots[N],
    ftype const coef[N + 1],
    ftype xMin,
    ftype xMax,
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_R int CubicRoots(
    ftype roots[3],
    ftype const coef[4],
    ftype xMin,
    ftype xMax,
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_A int QuadraticRoots(ftype roots[2], ftype const coef[3], ftype xMin, ftype xMax);
_CY_POLY_TEMPLATE_D int LinearRoot(ftype& root, ftype const coef[2], ftype xMin, ftype xMax);

//-------------------------------------------------------------------------------

//! Finds the roots of the given polynomial and returns the number of roots found.
_CY_POLY_TEMPLATE_N int PolynomialRoots(
    ftype roots[N],
    ftype const coef[N + 1],
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_R int
CubicRoots(ftype roots[3], ftype const coef[4], ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_A int QuadraticRoots(ftype roots[2], ftype const coef[3]);
_CY_POLY_TEMPLATE_D int LinearRoot(ftype& root, ftype const coef[2]);

//-------------------------------------------------------------------------------

//! Finds the first root of the given polynomial between `xMin` and `xMax` and returns true if a
//! root is found.
_CY_POLY_TEMPLATE_N bool PolynomialFirstRoot(
    ftype& root,
    ftype const coef[N + 1],
    ftype xMin,
    ftype xMax,
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_R bool CubicFirstRoot(
    ftype& root,
    ftype const coef[4],
    ftype xMin,
    ftype xMax,
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_A bool
QuadraticFirstRoot(ftype& root, ftype const coef[3], ftype xMin, ftype xMax);

//-------------------------------------------------------------------------------

//! Finds the first root of the given polynomial and returns true if a root is found.
_CY_POLY_TEMPLATE_N bool PolynomialFirstRoot(
    ftype& root,
    ftype const coef[N + 1],
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_R bool
CubicFirstRoot(ftype& root, ftype const coef[4], ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_A bool QuadraticFirstRoot(ftype& root, ftype const coef[3]);

//-------------------------------------------------------------------------------

//! Returns true if the given polynomial has a root between `xMin` and `xMax`.
_CY_POLY_TEMPLATE_N bool PolynomialHasRoot(
    ftype const coef[N + 1],
    ftype xMin,
    ftype xMax,
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_A bool CubicHasRoot(ftype const coef[4], ftype xMin, ftype xMax);
_CY_POLY_TEMPLATE_A bool QuadraticHasRoot(ftype const coef[3], ftype xMin, ftype xMax);

//-------------------------------------------------------------------------------

//! Returns true if the given polynomial has a root.
_CY_POLY_TEMPLATE_N bool
PolynomialHasRoot(ftype const coef[N + 1], ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_A bool CubicHasRoot(ftype const coef[4])
{
    return true;
}
_CY_POLY_TEMPLATE_A bool QuadraticHasRoot(ftype const coef[3])
{
    return coef[1] * coef[1] - ftype(4) * coef[0] * coef[2] >= 0;
}

//-------------------------------------------------------------------------------

//! Calls the given `callback` function for each root of the given polynomial between `xMin` and
//! `xMax` in increasing order. The `callback` function should have the following form:
//!
//! `bool RootCallback( ftype root );`
//!
//! If the `callback` function returns true, root finding is terminated without finding any
//! additional roots. If the `callback` function returns false, root finding continues. Returns true
//! if root finding is terminated by the `callback` function. Otherwise, returns false.
_CY_POLY_TEMPLATE_NC bool PolynomialForEachRoot(
    RootCallback callback,
    ftype const coef[N + 1],
    ftype xMin,
    ftype xMax,
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_RC bool CubicForEachRoot(
    RootCallback callback,
    ftype const coef[4],
    ftype xMin,
    ftype xMax,
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_AC bool
QuadraticForEachRoot(RootCallback callback, ftype const coef[3], ftype xMin, ftype xMax);

//-------------------------------------------------------------------------------

//! Calls the given `callback` function for each root of the given polynomial in increasing order.
//! The `callback` function should have the following form:
//!
//! `bool RootCallback( ftype root );`
//!
//! If the `callback` function returns true, root finding is terminated without finding any
//! additional roots. If the `callback` function returns false, root finding continues. Returns true
//! if root finding is terminated by the `callback` function. Otherwise, returns false.
_CY_POLY_TEMPLATE_NC bool PolynomialForEachRoot(
    RootCallback callback,
    ftype const coef[N + 1],
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_RC bool CubicForEachRoot(
    RootCallback callback,
    ftype const coef[4],
    ftype xError = PolynomialDefaultError<ftype>());
_CY_POLY_TEMPLATE_AC bool QuadraticForEachRoot(RootCallback callback, ftype const coef[3]);

//-------------------------------------------------------------------------------
//!@}
//-------------------------------------------------------------------------------

#define _CY_POLY_TEMPLATE_B            \
    template <bool boundError = false> \
    CY_NODISCARD
#define _CY_POLY_TEMPLATE_BC template <bool boundError = false, typename RootCallback>

//-------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////
//!
//! A general-purpose polynomial class.
//!
//! This class can be used for easily generating and manipulating polynomials.
//! It also offers interfaces to the polynomial evaluation and root finding functions.
//!
/////////////////////////////////////////////////////////////////////////////////
template <typename ftype, int N>
class Polynomial
{
  public:
    ftype coef[N + 1]; //!< The coefficients of the polynomial.

    CY_NODISCARD ftype const& operator[](int i) const
    {
        return coef[i];
    } //!< Access to the coefficients of the polynomial
    CY_NODISCARD ftype& operator[](int i)
    {
        return coef[i];
    } //!< Access to the coefficients of the polynomial
    CY_NODISCARD ftype operator()(ftype x) const
    {
        return Eval(x);
    } //!< Evaluates the polynomial at the given `x` value.

    CY_NODISCARD ftype Eval(ftype x) const
    {
        return PolynomialEval<N, ftype>(coef, x);
    } //!< Evaluates the polynomial at `x`.
    CY_NODISCARD ftype EvalWithDeriv(ftype& deriv, ftype x) const
    {
        return PolynomialEvalWithDeriv<N, ftype>(deriv, coef, x);
    } //!< Evaluates the polynomial and its derivative at `x`.

    CY_NODISCARD Polynomial<ftype, N> operator+(Polynomial<ftype, N> const& p) const
    {
        Polynomial<ftype, N> r;
        for (int i = 0; i <= N; ++i)
            r[i] = coef[i] + p[i];
        return r;
    } //!< Adds two polynomials.
    CY_NODISCARD Polynomial<ftype, N> operator-(Polynomial<ftype, N> const& p) const
    {
        Polynomial<ftype, N> r;
        for (int i = 0; i <= N; ++i)
            r[i] = coef[i] - p[i];
        return r;
    } //!< Subtracts two polynomials.
    CY_NODISCARD Polynomial<ftype, N> operator*(ftype s) const
    {
        Polynomial<ftype, N> r;
        for (int i = 0; i <= N; ++i)
            r[i] = coef[i] * s;
        return r;
    } //!< Multiplies the polynomial with a scalar.
    void operator+=(Polynomial<ftype, N> const& p)
    {
        for (int i = 0; i <= N; ++i)
            coef[i] += p[i];
    } //!< Adds another polynomial to this one.
    void operator-=(Polynomial<ftype, N> const& p)
    {
        for (int i = 0; i <= N; ++i)
            coef[i] -= p[i];
    } //!< Subtracts another polynomial from this one.
    void operator*=(ftype s)
    {
        for (int i = 0; i <= N; ++i)
            coef[i] *= s;
    } //!< Multiplies this polynomial with a scalar.

    //! Multiplies two polynomials and returns the resulting polynomial.
    template <int M>
    Polynomial<ftype, N + M> operator*(Polynomial<ftype, M> const& p) const
    {
        Polynomial<ftype, N + M> r;
        for (int i = 0; i <= N + M; ++i)
            r[i] = ftype(0);
        for (int i = 0; i <= N; ++i)
        {
            for (int j = 0; j <= M; ++j)
            {
                r[i + j] += coef[i] * p[j];
            }
        }
        return r;
    }

    //! Multiplies the polynomial with itself and returns the resulting polynomial.
    CY_NODISCARD Polynomial<ftype, 2 * N> Squared() const
    {
        Polynomial<ftype, 2 * N> r;
        for (int i = 0; i <= 2 * N; ++i)
            r[i] = ftype(0);
        for (int i = 0; i <= N; ++i)
        {
            r[2 * i] += coef[i] * coef[i];
            for (int j = i + 1; j <= N; ++j)
            {
                r[i + j] += 2 * coef[i] * coef[j];
            }
        }
        return r;
    }

    CY_NODISCARD Polynomial<ftype, N - 1> Derivative() const
    {
        Polynomial<ftype, N - 1> d;
        PolynomialDerivative<N, ftype>(d.coef, coef);
        return d;
    } //!< Returns the derivative of the polynomial.
    CY_NODISCARD Polynomial<ftype, N - 1> Deflate(ftype root) const
    {
        Polynomial<ftype, N - 1> p;
        PolynomialDeflate<N, ftype>(p.coef, coef, root);
        return p;
    } //!< Returns the deflation of the polynomial with the given root.
    CY_NODISCARD Polynomial<ftype, N + 1> Inflate(ftype root) const
    {
        Polynomial<ftype, N + 1> p;
        PolynomialInflate<N, ftype>(p.coef, coef, root);
        return p;
    } //!< Returns the inflated polynomial using the given root.

    _CY_POLY_TEMPLATE_B int Roots(ftype roots[N], ftype xError = DefaultError()) const
    {
        return PolynomialRoots<N, ftype, boundError>(roots, coef, xError);
    } //!< Finds all roots of the polynomial and returns the number of roots found.
    _CY_POLY_TEMPLATE_B bool FirstRoot(ftype& root, ftype xError = DefaultError()) const
    {
        return PolynomialFirstRoot<N, ftype, boundError>(root, coef, xError);
    } //!< Finds the first root of the polynomial and returns true if a root is found.
    _CY_POLY_TEMPLATE_B bool HasRoot(ftype xError = DefaultError()) const
    {
        return PolynomialHasRoot<N, ftype, boundError>(coef, xError);
    } //!< Returns true if the polynomial has a root.
    _CY_POLY_TEMPLATE_BC void ForEachRoot(RootCallback c, ftype xError = DefaultError()) const
    {
        return PolynomialForEachRoot<N, ftype, boundError>(c, coef, xError);
    } //!< Calls the given callback function for each root of the polynomial.

    _CY_POLY_TEMPLATE_B int
    Roots(ftype roots[N], ftype xMin, ftype xMax, ftype xError = DefaultError()) const
    {
        return PolynomialRoots<N, ftype, boundError>(roots, coef, xMin, xMax, xError);
    } //!< Finds all roots of the polynomial between `xMin` and `xMax` and returns the number of
      //!< roots found.
    _CY_POLY_TEMPLATE_B bool
    FirstRoot(ftype& root, ftype xMin, ftype xMax, ftype xError = DefaultError()) const
    {
        return PolynomialFirstRoot<N, ftype, boundError>(root, coef, xMin, xMax, xError);
    } //!< Finds the first root of the polynomial between `xMin` and `xMax` and returns true if a
      //!< root is found.
    _CY_POLY_TEMPLATE_B bool HasRoot(ftype xMin, ftype xMax, ftype xError = DefaultError()) const
    {
        return PolynomialHasRoot<N, ftype, boundError>(coef, xMin, xMax, xError);
    } //!< Returns true if the polynomial has a root between `xMin` and `xMax`.
    _CY_POLY_TEMPLATE_BC void
    ForEachRoot(RootCallback c, ftype xMin, ftype xMax, ftype xError = DefaultError()) const
    {
        return PolynomialForEachRoot<N, ftype, boundError>(c, coef, xMin, xMax, xError);
    } //!< Calls the given callback function for each root of the polynomial between `xMin` and
      //!< `xMax`.

    CY_NODISCARD bool IsFinite() const
    {
        for (int i = 0; i <= N; ++i)
            if (!cy::IsFinite(coef[i]))
                return false;
        return true;
    } //!< Returns true if all coefficients are finite real numbers.

  protected:
    static constexpr ftype DefaultError()
    {
        return PolynomialDefaultError<ftype>();
    } //!< Returns the default error threshold for numerical root finding.
};

//-------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////
//! @name Support Functions (Internal)
/////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------

template <typename T, typename S>
inline T MultSign(T v, S sign)
{
    return v * (sign < 0 ? T(-1) : T(1));
} //!< Multiplies the given value with the given sign
template <typename T, typename S>
inline bool IsDifferentSign(T a, S b)
{
    return a < 0 != b < 0;
} //!< Returns true if the sign bits are different

//-------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////
//! @name RootFinderNewton
/////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------

//! Numerical root finder using safe Newton iterations.
//!
//! This is the default numerical root finder used by the polynomial root finding functions.
//! The methods of this class are called after a single root is isolated within a closed
//! (finite) or open (infinite) interval.
//! It performs a combination of Newton iterations and bisection to ensure convergence
//! and achieve high-performance.
class RootFinderNewton
{
  public:
    template <int N, typename ftype, bool boundError = false>
    static inline ftype FindClosed(
        ftype const coef[N + 1],
        ftype const deriv[N],
        ftype x0,
        ftype x1,
        ftype y0,
        ftype y1,
        ftype xError); //!< @private Finds the single root within a closed interval between `x0` and
                       //!< `x1`.
    template <int N, typename ftype, bool boundError = false>
    static inline ftype FindOpen(
        ftype const coef[N + 1],
        ftype const deriv[N],
        ftype xError); //!< @private Finds the single root in an infinite interval.
    template <int N, typename ftype, bool boundError = false>
    static inline ftype FindOpenMin(
        ftype const coef[N + 1],
        ftype const deriv[N],
        ftype x1,
        ftype y1,
        ftype xError); //!< @private Finds the single root from negative infinity to the given x
                       //!< bound `x1`.
    template <int N, typename ftype, bool boundError = false>
    static inline ftype FindOpenMax(
        ftype const coef[N + 1],
        ftype const deriv[N],
        ftype x0,
        ftype y0,
        ftype xError); //!< @private Finds the single root from the given x bound `x0` to positive
                       //!< infinity.
  protected:
    template <int N, typename ftype, bool boundError, bool openMin>
    static inline ftype FindOpen(
        ftype const coef[N + 1],
        ftype const deriv[N],
        ftype xs,
        ftype ys,
        ftype xr,
        ftype xError); //!< @private The implementation of FindOpenMin and FindOpenMax
};

//-------------------------------------------------------------------------------

//! Finds the single root within a closed interval between `x0` and `x1`.
//!
//! It combines Newton iterations with bisection to ensure convergence.
//! It takes a polynomial's coefficients `coef` and its derivative's coefficients `deriv` along with
//! the x bounds `x0` and `x1` and the values of the polynomial `y0` and `y1` computed at `x0` and
//! `x1` respectively. It almost always satisfies the given error bound `xError` but this is not
//! guaranteed unless `boundError` is set to `true`. If `boundError` is `true`, it performs
//! additional operations to bound the error.
template <int N, typename ftype, bool boundError>
inline ftype RootFinderNewton::FindClosed(
    ftype const coef[N + 1],
    ftype const deriv[N],
    ftype x0,
    ftype x1,
    ftype y0,
    [[maybe_unused]] ftype y1,
    ftype xError)
{
    ftype ep2 = 2 * xError;
    ftype xr  = (x0 + x1) / 2; // mid point
    if (x1 - x0 <= ep2)
        return xr;

    if constexpr (N <= 3)
    {
        ftype xr0 = xr;
        for (int safetyCounter = 0; safetyCounter < 16; ++safetyCounter)
        {
            ftype xn =
                xr - PolynomialEval<N, ftype>(coef, xr) / PolynomialEval<2, ftype>(deriv, xr);
            xn = Clamp(xn, x0, x1);
            if (std::abs(xr - xn) <= xError)
                return xn;
            xr = xn;
        }
        if (!IsFinite(xr))
            xr = xr0;
    }

    ftype yr  = PolynomialEval<N, ftype>(coef, xr);
    ftype xb0 = x0;
    ftype xb1 = x1;

    while (true)
    {
        int side = IsDifferentSign(y0, yr);
        if (side)
            xb1 = xr;
        else
            xb0 = xr;
        ftype dy = PolynomialEval<N - 1, ftype>(deriv, xr);
        ftype dx = yr / dy;
        ftype xn = xr - dx;
        if (xn > xb0 && xn < xb1)
        { // valid Newton step
            ftype stepsize = std::abs(xr - xn);
            xr             = xn;
            if (stepsize > xError)
            {
                yr = PolynomialEval<N, ftype>(coef, xr);
            }
            else
            {
                if constexpr (boundError)
                {
                    ftype xs;
                    if (xError == 0)
                    {
                        xs = std::nextafter(side ? xb1 : xb0, side ? xb0 : xb1);
                    }
                    else
                    {
                        xs = xn - MultSign(xError, side - 1);
                        if (xs == xn)
                            xs = std::nextafter(side ? xb1 : xb0, side ? xb0 : xb1);
                    }
                    ftype ys = PolynomialEval<N, ftype>(coef, xs);
                    int s    = IsDifferentSign(y0, ys);
                    if (side != s)
                        return xn;
                    xr = xs;
                    yr = ys;
                }
                else
                    break;
            }
        }
        else
        { // Newton step failed
            xr = (xb0 + xb1) / 2;
            if (xr == xb0 || xr == xb1 || xb1 - xb0 <= ep2)
            {
                if constexpr (boundError)
                {
                    if (xError == 0)
                    {
                        ftype xm = side ? xb0 : xb1;
                        ftype ym = PolynomialEval<N, ftype>(coef, xm);
                        if (std::abs(ym) < std::abs(yr))
                            xr = xm;
                    }
                }
                break;
            }
            yr = PolynomialEval<N, ftype>(coef, xr);
        }
    }
    return xr;
}

//-------------------------------------------------------------------------------

//! Finds the single root in an infinite interval.
//!
//! This is intended for root finding in an open interval for polynomials with odd degrees.
//! Polynomials with odd degrees always have at least one real root.
//! If that is the only root, this function can be used to find it.
template <int N, typename ftype, bool boundError>
inline ftype RootFinderNewton::FindOpen(ftype const coef[N + 1], ftype const deriv[N], ftype xError)
{
    static_assert(
        (N & 1) == 1,
        "RootFinderNewton::FindOpen only works for polynomials with odd degrees.");
    const ftype xr = 0;
    const ftype yr = coef[0]; // PolynomialEval<N,ftype>( coef, xr );
    if (IsDifferentSign(coef[N], yr))
    {
        return FindOpenMax<N, ftype, boundError>(coef, deriv, xr, yr, xError);
    }
    else
    {
        return FindOpenMin<N, ftype, boundError>(coef, deriv, xr, yr, xError);
    }
}

//-------------------------------------------------------------------------------

//! Finds the single root from negative infinity to the given x bound `x1`.
template <int N, typename ftype, bool boundError>
inline ftype RootFinderNewton::FindOpenMin(
    ftype const coef[N + 1],
    ftype const deriv[N],
    ftype x1,
    ftype y1,
    ftype xError)
{
    return FindOpen<N, ftype, boundError, true>(coef, deriv, x1, y1, x1 - ftype(1), xError);
}

//-------------------------------------------------------------------------------

//! Finds the single root from the given x bound `x0` to positive infinity.
template <int N, typename ftype, bool boundError>
inline ftype RootFinderNewton::FindOpenMax(
    ftype const coef[N + 1],
    ftype const deriv[N],
    ftype x0,
    ftype y0,
    ftype xError)
{
    return FindOpen<N, ftype, boundError, false>(coef, deriv, x0, y0, x0 + ftype(1), xError);
}

//-------------------------------------------------------------------------------

//! @private The implementation of FindOpenMin and FindOpenMax
template <int N, typename ftype, bool boundError, bool openMin>
inline ftype RootFinderNewton::FindOpen(
    ftype const coef[N + 1],
    ftype const deriv[N],
    ftype xm,
    ftype ym,
    ftype xr,
    ftype xError)
{
    ftype delta = ftype(1);
    ftype yr    = PolynomialEval<N, ftype>(coef, xr);

    bool otherside = IsDifferentSign(ym, yr);

    while (yr != 0)
    {
        if (otherside)
        {
            if constexpr (openMin)
            {
                return FindClosed<N, ftype, boundError>(coef, deriv, xr, xm, yr, ym, xError);
            }
            else
            {
                return FindClosed<N, ftype, boundError>(coef, deriv, xm, xr, ym, yr, xError);
            }
        }
        else
        {
        open_interval:
            xm        = xr;
            ym        = yr;
            ftype dy  = PolynomialEval<N - 1, ftype>(deriv, xr);
            ftype dx  = yr / dy;
            ftype xn  = xr - dx;
            ftype dif = openMin ? xr - xn : xn - xr;
            if (dif <= 0 && std::isfinite(xn))
            { // valid Newton step
                xr = xn;
                if (dif <= xError)
                { // we might have converged
                    if (xr == xm)
                        break;
                    ftype xs = xn - MultSign(xError, -ftype(openMin));
                    ftype ys = PolynomialEval<N, ftype>(coef, xs);
                    bool s   = IsDifferentSign(ym, ys);
                    if (s)
                        break;
                    xr = xs;
                    yr = ys;
                    goto open_interval;
                }
            }
            else
            { // Newton step failed
                xr = openMin ? xr - delta : xr + delta;
                delta *= 2;
            }
            yr        = PolynomialEval<N, ftype>(coef, xr);
            otherside = IsDifferentSign(ym, yr);
        }
    }
    return xr;
}

//-------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////
// Implementations of the root finding functions declared above
/////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
// Linear
//-------------------------------------------------------------------------------

template <typename ftype>
inline int LinearRoot(ftype& root, ftype const coef[2], ftype x0, ftype x1)
{
    if (coef[1] != ftype(0))
    {
        ftype r = -coef[0] / coef[1];
        root    = r;
        return (r >= x0 && r <= x1);
    }
    else
    {
        root = (x0 + x1) / 2;
        return coef[0] == 0;
    }
}

template <typename ftype>
inline int LinearRoot(ftype& root, ftype const coef[2])
{
    root = -coef[0] / coef[1];
    return coef[1] != 0;
}

//-------------------------------------------------------------------------------
// Quadratics
//-------------------------------------------------------------------------------

template <typename ftype>
inline int QuadraticRoots(ftype roots[2], ftype const coef[3])
{
    const ftype c     = coef[0];
    const ftype b     = coef[1];
    const ftype a     = coef[2];
    const ftype delta = b * b - 4 * a * c;
    if (delta > 0)
    {
        const ftype d   = Sqrt(delta);
        const ftype q   = ftype(-0.5) * (b + MultSign(d, b));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        roots[0]        = Min(rv0, rv1);
        roots[1]        = Max(rv0, rv1);
        return 2;
    }
    else if (delta < 0)
        return 0;
    roots[0] = ftype(-0.5) * b / a;
    return a != 0;
}

template <typename ftype>
inline int QuadraticRoots(ftype roots[2], ftype const coef[3], ftype x0, ftype x1)
{
    const ftype c     = coef[0];
    const ftype b     = coef[1];
    const ftype a     = coef[2];
    const ftype delta = b * b - 4 * a * c;
    if (delta > 0)
    {
        const ftype d   = Sqrt(delta);
        const ftype q   = ftype(-0.5) * (b + MultSign(d, b));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        const ftype r0  = Min(rv0, rv1);
        const ftype r1  = Max(rv0, rv1);
        int r0i         = (r0 >= x0) & (r0 <= x1);
        int r1i         = (r1 >= x0) & (r1 <= x1);
        roots[0]        = r0;
        roots[r0i]      = r1;
        return r0i + r1i;
    }
    else if (delta < 0)
        return 0;
    const ftype r0 = ftype(-0.5) * b / a;
    roots[0]       = r0;
    return (r0 >= x0) & (r0 <= x1);
}

//-------------------------------------------------------------------------------
#ifdef _INCLUDED_IMM

template <typename RootCallback>
inline int QuadraticRoots([[maybe_unused]] float roots[2], float const* coef, RootCallback callback)
{
    //__m128 _0abc    = _mm_set_ps( 0.0f, coef[2], coef[1], coef[0] );
    __m128 _0abc    = _mm_load_ps(coef);
    __m128 _02a2b2c = _mm_add_ps(_0abc, _0abc);
    __m128 _2a2c_bb = _mm_shuffle_ps(_0abc, _02a2b2c, _MM_SHUFFLE(2, 0, 1, 1));
    __m128 _2c2a_bb = _mm_shuffle_ps(_0abc, _02a2b2c, _MM_SHUFFLE(0, 2, 1, 1));
    __m128 _4ac_b2  = _mm_mul_ps(_2a2c_bb, _2c2a_bb);
    __m128 _4ac     = _mm_shuffle_ps(_4ac_b2, _4ac_b2, _MM_SHUFFLE(2, 2, 2, 2));
    if (_mm_comigt_ss(_4ac_b2, _4ac))
    {
        __m128 delta  = _mm_sub_ps(_4ac_b2, _4ac);
        __m128 sqrtd  = _mm_sqrt_ss(delta);
        __m128 signb  = _mm_set_ps(-0.0f, -0.0f, -0.0f, -0.0f);
        __m128 db     = _mm_xor_ps(sqrtd, _mm_and_ps(_2a2c_bb, signb));
        __m128 b_db   = _mm_add_ss(_2a2c_bb, db);
        __m128 _2q    = _mm_xor_ps(b_db, signb);
        __m128 _2c_2q = _mm_shuffle_ps(_2q, _02a2b2c, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 _2q_2a = _mm_shuffle_ps(_02a2b2c, _2q, _MM_SHUFFLE(0, 0, 2, 2));
        __m128 rv     = _mm_div_ps(_2c_2q, _2q_2a);
        __m128 r0     = _mm_min_ps(rv, _mm_shuffle_ps(rv, rv, _MM_SHUFFLE(3, 2, 1, 2)));
        __m128 r      = _mm_max_ps(r0, _mm_shuffle_ps(r0, r0, _MM_SHUFFLE(3, 2, 2, 0)));
        return callback(r);
    }
    else if (_mm_comilt_ss(_4ac_b2, _4ac))
        return 0;
    __m128 r = _mm_div_ps(_2a2c_bb, _mm_shuffle_ps(_2a2c_bb, _2a2c_bb, _MM_SHUFFLE(1, 0, 3, 3)));
    return callback(r) * (coef[2] != 0);
}

template <>
[[maybe_unused]] inline int QuadraticRoots<float>(float roots[2], float const* coef)
{
    return QuadraticRoots(roots, coef, [&](__m128 r) {
        roots[0] = _mm_cvtss_f32(r);
        roots[1] = _mm_cvtss_f32(_mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 2, 0, 1)));
        return 2;
    });
}

template <>
[[maybe_unused]] inline int QuadraticRoots<float>(float roots[2], float const* coef, float x0, float x1)
{
    return QuadraticRoots(roots, coef, [&](__m128 r) {
        __m128 range = _mm_set_ps(x1, x1, x0, x0);
        __m128 minT  = _mm_cmpge_ps(r, range);
        __m128 maxT  = _mm_cmple_ps(r, _mm_shuffle_ps(range, range, _MM_SHUFFLE(3, 2, 2, 2)));
        __m128 valid =
            _mm_and_ps(minT, _mm_shuffle_ps(maxT, _mm_setzero_ps(), _MM_SHUFFLE(3, 2, 1, 0)));
        __m128 rr = _mm_blendv_ps(_mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 2, 0, 1)), r, valid);
        roots[0]  = _mm_cvtss_f32(rr);
        roots[1]  = _mm_cvtss_f32(_mm_shuffle_ps(rr, rr, _MM_SHUFFLE(3, 2, 0, 1)));
        return _mm_popcnt_u32(_mm_movemask_ps(valid));
    });
}

#endif // _INCLUDED_IMM
//-------------------------------------------------------------------------------

template <typename ftype>
inline bool QuadraticFirstRoot(ftype& root, ftype const coef[3], ftype x0, ftype x1)
{
    const ftype c     = coef[0];
    const ftype b     = coef[1];
    const ftype a     = coef[2];
    const ftype delta = b * b - 4 * a * c;
    if (delta >= 0)
    {
        const ftype d   = Sqrt(delta);
        const ftype q   = ftype(-0.5) * (b + MultSign(d, b));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        const ftype r0  = Min(rv0, rv1);
        if (r0 >= x0)
        {
            root = r0;
            return r0 <= x1;
        }
        else
        {
            const ftype r1 = Max(rv0, rv1);
            root           = r1;
            return (r1 >= x0) & (r1 <= x1);
        }
    }
    return false;
}

//-------------------------------------------------------------------------------

template <typename ftype>
inline bool QuadraticFirstRoot(ftype& root, ftype const coef[3])
{
    const ftype c     = coef[0];
    const ftype b     = coef[1];
    const ftype a     = coef[2];
    const ftype delta = b * b - 4 * a * c;
    if (delta >= 0)
    {
        const ftype d   = Sqrt(delta);
        const ftype q   = ftype(-0.5) * (b + MultSign(d, b));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        root            = Min(rv0, rv1);
        return true;
    }
    return false;
}

//-------------------------------------------------------------------------------

template <typename ftype>
inline bool QuadraticHasRoot(ftype const coef[3], ftype x0, ftype x1)
{
    const ftype c     = coef[0];
    const ftype b     = coef[1];
    const ftype a     = coef[2];
    const ftype delta = b * b - 4 * a * c;
    if (delta >= 0)
    {
        const ftype d   = Sqrt(delta);
        const ftype q   = ftype(-0.5) * (b + MultSign(d, b));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        const ftype r0  = Min(rv0, rv1);
        const ftype r1  = Max(rv0, rv1);
        if (r0 >= x0 && r0 <= x1)
            return true;
        if (r1 >= x0 && r1 <= x1)
            return true;
    }
    return false;
}

//-------------------------------------------------------------------------------

template <typename ftype, typename RootCallback>
inline bool QuadraticForEachRoot(RootCallback callback, ftype const coef[3], ftype x0, ftype x1)
{
    const ftype c     = coef[0];
    const ftype b     = coef[1];
    const ftype a     = coef[2];
    const ftype delta = b * b - 4 * a * c;
    if (delta >= 0)
    {
        const ftype d   = Sqrt(delta);
        const ftype q   = ftype(-0.5) * (b + MultSign(d, b));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        const ftype r0  = Min(rv0, rv1);
        const ftype r1  = Max(rv0, rv1);
        if (r0 >= x0 && r0 <= x1)
            if (callback(r0))
                return true;
        if (r1 >= x0 && r1 <= x1)
            return callback(r1);
    }
    return false;
}

//-------------------------------------------------------------------------------

template <typename ftype, typename RootCallback>
inline bool QuadraticForEachRoot(RootCallback callback, ftype const coef[3])
{
    const ftype c     = coef[0];
    const ftype b     = coef[1];
    const ftype a     = coef[2];
    const ftype delta = b * b - 4 * a * c;
    if (delta >= 0)
    {
        const ftype d   = Sqrt(delta);
        const ftype q   = ftype(-0.5) * (b + MultSign(d, b));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        const ftype r0  = Min(rv0, rv1);
        const ftype r1  = Max(rv0, rv1);
        if (callback(r0))
            return true;
        return callback(r1);
    }
    return false;
}

//-------------------------------------------------------------------------------
// Cubics
//-------------------------------------------------------------------------------

template <typename ftype, bool boundError, typename RootFinder>
inline int CubicRoots(ftype roots[3], ftype const coef[4], ftype x0, ftype x1, ftype xError)
{
    const ftype y0 = PolynomialEval<3, ftype>(coef, x0);
    const ftype y1 = PolynomialEval<3, ftype>(coef, x1);

    const ftype a   = coef[3] * 3;
    const ftype b_2 = coef[2];
    const ftype c   = coef[1];

    const ftype deriv[4] = {c, 2 * b_2, a, 0};

    const ftype delta_4 = b_2 * b_2 - a * c;

    if (delta_4 > 0)
    {
        const ftype d_2 = Sqrt(delta_4);
        const ftype q   = -(b_2 + MultSign(d_2, b_2));
        ftype rv0       = q / a;
        ftype rv1       = c / q;
        const ftype xa  = Min(rv0, rv1);
        const ftype xb  = Max(rv0, rv1);

        if (IsDifferentSign(y0, y1))
        {
            if (xa >= x1 || xb <= x0 || (xa <= x0 && xb >= x1))
            { // first, last, or middle interval only
                roots[0] = RootFinder::template FindClosed<3, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    x1,
                    y0,
                    y1,
                    xError);
                return 1;
            }
        }
        else
        {
            if ((xa >= x1 || xb <= x0) || (xa <= x0 && xb >= x1))
                return 0;
        }

        int numRoots = 0;
        if (xa > x0)
        {
            const ftype ya = PolynomialEval<3, ftype>(coef, xa);
            if (IsDifferentSign(y0, ya))
            {
                roots[0] = RootFinder::template FindClosed<3, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    xa,
                    y0,
                    ya,
                    xError); // first interval
                if constexpr (!boundError)
                {
                    if (IsDifferentSign(ya, y1) ||
                        (xb < x1 && IsDifferentSign(ya, PolynomialEval<3, ftype>(coef, xb))))
                    {
                        ftype defPoly[4];
                        PolynomialDeflate<3>(defPoly, coef, roots[0]);
                        return QuadraticRoots(roots + 1, defPoly, xa, x1) + 1;
                    }
                    else
                        return 1;
                }
                else
                    numRoots++;
            }
            if (xb < x1)
            {
                const ftype yb = PolynomialEval<3, ftype>(coef, xb);
                if (IsDifferentSign(ya, yb))
                {
                    roots[!boundError ? 0 : numRoots++] =
                        RootFinder::template FindClosed<3, ftype, boundError>(
                            coef,
                            deriv,
                            xa,
                            xb,
                            ya,
                            yb,
                            xError);
                    if constexpr (!boundError)
                    {
                        if (IsDifferentSign(yb, y1))
                        {
                            ftype defPoly[4];
                            PolynomialDeflate<3>(defPoly, coef, roots[0]);
                            return QuadraticRoots(roots + 1, defPoly, xb, x1) + 1;
                        }
                        else
                            return 1;
                    }
                }
                if (IsDifferentSign(yb, y1))
                {
                    roots[!boundError ? 0 : numRoots++] =
                        RootFinder::template FindClosed<3, ftype, boundError>(
                            coef,
                            deriv,
                            xb,
                            x1,
                            yb,
                            y1,
                            xError); // last interval
                    if constexpr (!boundError)
                        return 1;
                }
            }
            else
            {
                if (IsDifferentSign(ya, y1))
                {
                    roots[!boundError ? 0 : numRoots++] =
                        RootFinder::template FindClosed<3, ftype, boundError>(
                            coef,
                            deriv,
                            xa,
                            x1,
                            ya,
                            y1,
                            xError);
                    if (!boundError)
                        return 1;
                }
            }
        }
        else
        {
            const ftype yb = PolynomialEval<3, ftype>(coef, xb);
            if (IsDifferentSign(y0, yb))
            {
                roots[0] = RootFinder::template FindClosed<3, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    xb,
                    y0,
                    yb,
                    xError);
                if constexpr (!boundError)
                {
                    if (IsDifferentSign(yb, y1))
                    {
                        ftype defPoly[4];
                        PolynomialDeflate<3>(defPoly, coef, roots[0]);
                        return QuadraticRoots(roots + 1, defPoly, xb, x1) + 1;
                    }
                    else
                        return 1;
                }
                else
                    numRoots++;
            }
            if (IsDifferentSign(yb, y1))
            {
                roots[!boundError ? 0 : numRoots++] =
                    RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        xb,
                        x1,
                        yb,
                        y1,
                        xError); // last interval
                if constexpr (!boundError)
                    return 1;
            }
        }
        return numRoots;
    }
    else
    {
        if (IsDifferentSign(y0, y1))
        {
            roots[0] = RootFinder::template FindClosed<3, ftype, boundError>(
                coef,
                deriv,
                x0,
                x1,
                y0,
                y1,
                xError);
            return 1;
        }
        return 0;
    }
}

//-------------------------------------------------------------------------------

template <typename ftype, bool boundError, typename RootFinder>
inline int CubicRoots(ftype roots[3], ftype const coef[4], ftype xError)
{
    if (coef[3] != 0)
    {
        const ftype a   = coef[3] * 3;
        const ftype b_2 = coef[2];
        const ftype c   = coef[1];

        const ftype deriv[4] = {c, 2 * b_2, a, 0};

        const ftype delta_4 = b_2 * b_2 - a * c;

        if (delta_4 > 0)
        {
            const ftype d_2 = Sqrt(delta_4);
            const ftype q   = -(b_2 + MultSign(d_2, b_2));
            const ftype rv0 = q / a;
            const ftype rv1 = c / q;
            const ftype xa  = Min(rv0, rv1);
            const ftype xb  = Max(rv0, rv1);

            const ftype ya = PolynomialEval<3, ftype>(coef, xa);
            const ftype yb = PolynomialEval<3, ftype>(coef, xb);

            if (!IsDifferentSign(coef[3], ya))
            {
                roots[0] = RootFinder::template FindOpenMin<3, ftype, boundError>(
                    coef,
                    deriv,
                    xa,
                    ya,
                    xError);
                if constexpr (!boundError)
                {
                    if (IsDifferentSign(ya, yb))
                    {
                        ftype defPoly[4];
                        PolynomialDeflate<3>(defPoly, coef, roots[0]);
                        return QuadraticRoots(roots + 1, defPoly) + 1;
                    }
                }
                else
                {
                    if (IsDifferentSign(ya, yb))
                    {
                        roots[1] = RootFinder::template FindClosed<3, ftype, boundError>(
                            coef,
                            deriv,
                            xa,
                            xb,
                            ya,
                            yb,
                            xError);
                        roots[2] = RootFinder::template FindOpenMax<3, ftype, boundError>(
                            coef,
                            deriv,
                            xb,
                            yb,
                            xError);
                        return 3;
                    }
                }
            }
            else
            {
                roots[0] = RootFinder::template FindOpenMax<3, ftype, boundError>(
                    coef,
                    deriv,
                    xb,
                    yb,
                    xError);
            }
            return 1;
        }
        else
        {
            ftype x_inf = -b_2 / a;
            ftype y_inf = PolynomialEval<3, ftype>(coef, x_inf);
            if (IsDifferentSign(coef[3], y_inf))
            {
                roots[0] = RootFinder::template FindOpenMax<3, ftype, boundError>(
                    coef,
                    deriv,
                    x_inf,
                    y_inf,
                    xError);
            }
            else
            {
                roots[0] = RootFinder::template FindOpenMin<3, ftype, boundError>(
                    coef,
                    deriv,
                    x_inf,
                    y_inf,
                    xError);
            }
            return 1;
        }
    }
    else
        return QuadraticRoots<ftype>(roots, coef);
}

//-------------------------------------------------------------------------------

template <typename ftype, bool boundError, typename RootFinder>
inline bool CubicFirstRoot(ftype& root, ftype const coef[4], ftype x0, ftype x1, ftype xError)
{
    const ftype y0 = PolynomialEval<3, ftype>(coef, x0);
    const ftype y1 = PolynomialEval<3, ftype>(coef, x1);

    const ftype a   = coef[3] * 3;
    const ftype b_2 = coef[2];
    const ftype c   = coef[1];

    const ftype deriv[4] = {c, 2 * b_2, a, 0};

    const ftype delta_4 = b_2 * b_2 - a * c;

    if (delta_4 > 0)
    {
        const ftype d_2 = Sqrt(delta_4);
        const ftype q   = -(b_2 + MultSign(d_2, b_2));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        const ftype xa  = Min(rv0, rv1);
        const ftype xb  = Max(rv0, rv1);

        if (IsDifferentSign(y0, y1))
        {
            if (xa >= x1 || xb <= x0 || (xa <= x0 && xb >= x1))
            { // first, last, or middle interval only
                root = RootFinder::template FindClosed<3, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    x1,
                    y0,
                    y1,
                    xError); // first/last interval
                return true;
            }
        }
        else
        {
            if ((xa >= x1 || xb <= x0) || (xa <= x0 && xb >= x1))
                return false;
        }

        if (xa > x0)
        {
            const ftype ya = PolynomialEval<3, ftype>(coef, xa);
            if (IsDifferentSign(y0, ya))
            {
                root = RootFinder::template FindClosed<3, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    xa,
                    y0,
                    ya,
                    xError); // first interval
                return true;
            }
            if (xb < x1)
            {
                const ftype yb = PolynomialEval<3, ftype>(coef, xb);
                if (IsDifferentSign(ya, yb))
                {
                    root = RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        xa,
                        xb,
                        ya,
                        yb,
                        xError);
                    return true;
                }
                if (IsDifferentSign(yb, y1))
                {
                    root = RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        xb,
                        x1,
                        yb,
                        y1,
                        xError); // last interval
                    return true;
                }
            }
            else
            {
                if (IsDifferentSign(ya, y1))
                {
                    root = RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        xa,
                        x1,
                        ya,
                        y1,
                        xError);
                    return true;
                }
            }
        }
        else
        {
            const ftype yb = PolynomialEval<3, ftype>(coef, xb);
            if (IsDifferentSign(y0, yb))
            {
                root = RootFinder::template FindClosed<3, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    xb,
                    y0,
                    yb,
                    xError);
                return true;
            }
            if (IsDifferentSign(yb, y1))
            {
                root = RootFinder::template FindClosed<3, ftype, boundError>(
                    coef,
                    deriv,
                    xb,
                    x1,
                    yb,
                    y1,
                    xError); // last interval
                return true;
            }
        }
    }
    else
    {
        if (IsDifferentSign(y0, y1))
        {
            root = RootFinder::template FindClosed<3, ftype, boundError>(
                coef,
                deriv,
                x0,
                x1,
                y0,
                y1,
                xError);
            return true;
        }
    }
    return false;
}

//-------------------------------------------------------------------------------

template <typename ftype, bool boundError, typename RootFinder>
inline bool CubicFirstRoot(ftype& root, ftype const coef[4], ftype xError)
{
    if (coef[3] != 0)
    {
        const ftype a   = coef[3] * 3;
        const ftype b_2 = coef[2];
        const ftype c   = coef[1];

        const ftype deriv[4] = {c, 2 * b_2, a, 0};

        const ftype delta_4 = b_2 * b_2 - a * c;

        if (delta_4 > 0)
        {
            const ftype d_2 = Sqrt(delta_4);
            const ftype q   = -(b_2 + MultSign(d_2, b_2));
            const ftype rv0 = q / a;
            const ftype rv1 = c / q;
            const ftype xa  = Min(rv0, rv1);
            const ftype xb  = Max(rv0, rv1);

            const ftype ya = PolynomialEval<3, ftype>(coef, xa);
            if (!IsDifferentSign(coef[3], ya))
            {
                root = RootFinder::template FindOpenMin<3, ftype, boundError>(
                    coef,
                    deriv,
                    xa,
                    ya,
                    xError);
            }
            else
            {
                const ftype yb = PolynomialEval<3, ftype>(coef, xb);
                root           = RootFinder::template FindOpenMax<3, ftype, boundError>(
                    coef,
                    deriv,
                    xb,
                    yb,
                    xError);
            }
        }
        else
        {
            ftype x_inf = -b_2 / a;
            ftype y_inf = PolynomialEval<3, ftype>(coef, x_inf);
            if (IsDifferentSign(coef[3], y_inf))
            {
                root = RootFinder::template FindOpenMax<3, ftype, boundError>(
                    coef,
                    deriv,
                    x_inf,
                    y_inf,
                    xError);
            }
            else
            {
                root = RootFinder::template FindOpenMin<3, ftype, boundError>(
                    coef,
                    deriv,
                    x_inf,
                    y_inf,
                    xError);
            }
        }
        return true;
    }
    else
        return QuadraticFirstRoot<ftype>(root, coef);
}

//-------------------------------------------------------------------------------

template <typename ftype>
inline bool CubicHasRoot(ftype const coef[4], ftype x0, ftype x1)
{
    const ftype y0 = PolynomialEval<3, ftype>(coef, x0);
    const ftype y1 = PolynomialEval<3, ftype>(coef, x1);
    if (IsDifferentSign(y0, y1))
        return true;

    const ftype a   = coef[3] * 3;
    const ftype b_2 = coef[2];
    const ftype c   = coef[1];

    const ftype delta_4 = b_2 * b_2 - a * c;

    if (delta_4 > 0)
    {
        const ftype d_2 = Sqrt(delta_4);
        const ftype q   = -(b_2 + MultSign(d_2, b_2));
        const ftype rv0 = q / a;
        const ftype rv1 = c / q;
        const ftype xa  = Min(rv0, rv1);
        const ftype xb  = Max(rv0, rv1);

        if ((xa >= x1 || xb <= x0) || (xa <= x0 && xb >= x1))
            return false;

        if (xa > x0)
        {
            const ftype ya = PolynomialEval<3, ftype>(coef, xa);
            if (IsDifferentSign(y0, ya))
                return true;
            if (xb < x1)
            {
                const ftype yb = PolynomialEval<3, ftype>(coef, xb);
                if (IsDifferentSign(y0, yb))
                    return true;
            }
        }
        else
        {
            const ftype yb = PolynomialEval<3, ftype>(coef, xb);
            if (IsDifferentSign(y0, yb))
                return true;
        }
    }
    return false;
}

//-------------------------------------------------------------------------------

template <typename ftype, bool boundError, typename RootFinder, typename RootCallback>
inline bool
CubicForEachRoot(RootCallback callback, ftype const coef[4], ftype x0, ftype x1, ftype xError)
{
    const ftype y0 = PolynomialEval<3, ftype>(coef, x0);
    const ftype y1 = PolynomialEval<3, ftype>(coef, x1);

    const ftype a   = coef[3] * 3;
    const ftype b_2 = coef[2];
    const ftype c   = coef[1];

    const ftype deriv[4] = {c, 2 * b_2, a, 0};

    const ftype delta_4 = b_2 * b_2 - a * c;

    if (delta_4 > 0)
    {
        const ftype d_2 = Sqrt(delta_4);
        const ftype t   = -(b_2 + MultSign(d_2, b_2));
        const ftype rv0 = t / a;
        const ftype rv1 = c / t;
        const ftype xa  = Min(rv0, rv1);
        const ftype xb  = Max(rv0, rv1);

        if (xa >= x1 || xb <= x0)
        {
            if (IsDifferentSign(y0, y1))
            {
                if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        x0,
                        x1,
                        y0,
                        y1,
                        xError)))
                    return true; // first/last interval
            }
        }
        else if (xa <= x0 && xb >= x1)
        {
            if (IsDifferentSign(y0, y1))
            {
                if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        x0,
                        x1,
                        y0,
                        y1,
                        xError)))
                    return true;
            }
        }
        else if (xa > x0)
        {
            const ftype ya = PolynomialEval<3, ftype>(coef, xa);
            if (IsDifferentSign(y0, ya))
            {
                if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        x0,
                        xa,
                        y0,
                        ya,
                        xError)))
                    return true; // first interval
            }
            if (xb < x1)
            {
                const ftype yb = PolynomialEval<3, ftype>(coef, xb);
                if (IsDifferentSign(ya, yb))
                {
                    if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                            coef,
                            deriv,
                            xa,
                            xb,
                            ya,
                            yb,
                            xError)))
                        return true;
                }
                if (IsDifferentSign(yb, y1))
                {
                    if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                            coef,
                            deriv,
                            xb,
                            x1,
                            yb,
                            y1,
                            xError)))
                        return true; // last interval
                }
            }
            else
            {
                if (IsDifferentSign(ya, y1))
                {
                    if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                            coef,
                            deriv,
                            xa,
                            x1,
                            ya,
                            y1,
                            xError)))
                        return true;
                }
            }
        }
        else
        {
            const ftype yb = PolynomialEval<3, ftype>(coef, xb);
            if (IsDifferentSign(y0, yb))
            {
                if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        x0,
                        xb,
                        y0,
                        yb,
                        xError)))
                    return true;
            }
            if (IsDifferentSign(yb, y1))
            {
                if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                        coef,
                        deriv,
                        xb,
                        x1,
                        yb,
                        y1,
                        xError)))
                    return true; // last interval
            }
        }
    }
    else
    {
        if (IsDifferentSign(y0, y1))
        {
            if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    x1,
                    y0,
                    y1,
                    xError)))
                return true;
        }
    }
    return false;
}

//-------------------------------------------------------------------------------

template <typename ftype, bool boundError, typename RootFinder, typename RootCallback>
inline bool CubicForEachRoot(RootCallback callback, ftype const coef[4], ftype xError)
{
    if (coef[3] != 0)
    {
        const ftype a   = coef[3] * 3;
        const ftype b_2 = coef[2];
        const ftype c   = coef[1];

        const ftype deriv[4] = {c, 2 * b_2, a, 0};

        const ftype delta_4 = b_2 * b_2 - a * c;

        if (delta_4 > 0)
        {
            const ftype d_2 = Sqrt(delta_4);
            const ftype q   = -(b_2 + MultSign(d_2, b_2));
            const ftype rv0 = q / a;
            const ftype rv1 = c / q;
            const ftype xa  = Min(rv0, rv1);
            const ftype xb  = Max(rv0, rv1);

            const ftype ya = PolynomialEval<3, ftype>(coef, xa);
            const ftype yb = PolynomialEval<3, ftype>(coef, xb);

            if (!IsDifferentSign(coef[3], ya))
            {
                ftype root;
                root = RootFinder::template FindOpenMin<3, ftype, boundError>(
                    coef,
                    deriv,
                    xa,
                    ya,
                    xError);
                if (callback(root))
                    return true;
                if constexpr (!boundError)
                {
                    if (IsDifferentSign(ya, yb))
                    {
                        ftype defPoly[4];
                        PolynomialDeflate<3>(defPoly, coef, root);
                        return QuadraticForEachRoot(callback, defPoly);
                    }
                }
                else
                {
                    if (IsDifferentSign(ya, yb))
                    {
                        if (callback(RootFinder::template FindClosed<3, ftype, boundError>(
                                coef,
                                deriv,
                                xa,
                                xb,
                                ya,
                                yb,
                                xError)))
                            return true;
                        if (callback(RootFinder::template FindOpenMax<3, ftype, boundError>(
                                coef,
                                deriv,
                                xb,
                                yb,
                                xError)))
                            return true;
                        return false;
                    }
                }
            }
            else
            {
                if (callback(RootFinder::template FindOpenMax<3, ftype, boundError>(
                        coef,
                        deriv,
                        xb,
                        yb,
                        xError)))
                    return true;
            }
        }
        else
        {
            ftype x_inf = -b_2 / a;
            ftype y_inf = PolynomialEval<3, ftype>(coef, x_inf);
            if (IsDifferentSign(coef[3], y_inf))
            {
                if (callback(RootFinder::template FindOpenMax<3, ftype, boundError>(
                        coef,
                        deriv,
                        x_inf,
                        y_inf,
                        xError)))
                    return true;
            }
            else
            {
                if (callback(RootFinder::template FindOpenMin<3, ftype, boundError>(
                        coef,
                        deriv,
                        x_inf,
                        y_inf,
                        xError)))
                    return true;
            }
        }
        return false;
    }
    else
        return QuadraticForEachRoot<ftype>(callback, coef);
}

//-------------------------------------------------------------------------------
// Higher Order Polynomials
//-------------------------------------------------------------------------------

template <int N, typename ftype, bool boundError, typename RootFinder>
inline int
PolynomialRoots(ftype roots[N], ftype const coef[N + 1], ftype x0, ftype x1, ftype xError)
{
    if constexpr (N == 1)
        return LinearRoot<ftype>(*roots, coef, x0, x1);
    else if constexpr (N == 2)
        return QuadraticRoots<ftype>(roots, coef, x0, x1);
    else if constexpr (N == 3)
        return CubicRoots<ftype, boundError, RootFinder>(roots, coef, x0, x1, xError);
    else if (coef[N] == 0)
        return PolynomialRoots<N - 1, ftype, boundError, RootFinder>(roots, coef, x0, x1, xError);
    else
    {
        ftype y0 = PolynomialEval<N, ftype>(coef, x0);
        ftype deriv[N];
        PolynomialDerivative<N, ftype>(deriv, coef);
        ftype derivRoots[N - 1];
        int nd = PolynomialRoots<N - 1, ftype, boundError, RootFinder>(
            derivRoots,
            deriv,
            x0,
            x1,
            xError);
        ftype x[N + 1] = {x0};
        ftype y[N + 1] = {y0};
        for (int i = 0; i < nd; ++i)
        {
            x[i + 1] = derivRoots[i];
            y[i + 1] = PolynomialEval<N, ftype>(coef, derivRoots[i]);
        }
        x[nd + 1] = x1;
        y[nd + 1] = PolynomialEval<N, ftype>(coef, x1);
        int nr    = 0;
        for (int i = 0; i <= nd; ++i)
        {
            if (IsDifferentSign(y[i], y[i + 1]))
            {
                roots[nr++] = RootFinder::template FindClosed<N, ftype, boundError>(
                    coef,
                    deriv,
                    x[i],
                    x[i + 1],
                    y[i],
                    y[i + 1],
                    xError);
            }
        }
        return nr;
    }
}

//-------------------------------------------------------------------------------

template <int N, typename ftype, bool boundError, typename RootFinder>
inline int PolynomialRoots(ftype roots[N], ftype const coef[N + 1], ftype xError)
{
    if constexpr (N == 1)
        return LinearRoot<ftype>(*roots, coef);
    else if constexpr (N == 2)
        return QuadraticRoots<ftype>(roots, coef);
    else if constexpr (N == 3)
        return CubicRoots<ftype, boundError, RootFinder>(roots, coef, xError);
    else if (coef[N] == 0)
        return PolynomialRoots<N - 1, ftype, boundError, RootFinder>(roots, coef, xError);
    else
    {
        ftype deriv[N];
        PolynomialDerivative<N, ftype>(deriv, coef);
        ftype derivRoots[N - 1];
        int nd = PolynomialRoots<N - 1, ftype, boundError, RootFinder>(derivRoots, deriv, xError);
        if ((N & 1) || ((N & 1) == 0 && nd > 0))
        {
            int nr   = 0;
            ftype xa = derivRoots[0];
            ftype ya = PolynomialEval<N, ftype>(coef, xa);
            if (IsDifferentSign(coef[N], ya) != (N & 1))
            {
                roots[0] = RootFinder::template FindOpenMin<N, ftype, boundError>(
                    coef,
                    deriv,
                    xa,
                    ya,
                    xError);
                nr = 1;
            }
            for (int i = 1; i < nd; ++i)
            {
                ftype xb = derivRoots[i];
                ftype yb = PolynomialEval<N, ftype>(coef, xb);
                if (IsDifferentSign(ya, yb))
                {
                    roots[nr++] = RootFinder::template FindClosed<N, ftype, boundError>(
                        coef,
                        deriv,
                        xa,
                        xb,
                        ya,
                        yb,
                        xError);
                }
                xa = xb;
                ya = yb;
            }
            if (IsDifferentSign(coef[N], ya))
            {
                roots[nr++] = RootFinder::template FindOpenMax<N, ftype, boundError>(
                    coef,
                    deriv,
                    xa,
                    ya,
                    xError);
            }
            return nr;
        }
        else
        {
            if constexpr (N & 1)
            {
                roots[0] = RootFinder::template FindOpen<N, ftype, boundError>(coef, deriv, xError);
                return 1;
            }
            else
                return 0; // this should not happen
        }
    }
}

//-------------------------------------------------------------------------------

template <int N, typename ftype, bool boundError, typename RootFinder>
inline bool
PolynomialFirstRoot(ftype& root, ftype const coef[N + 1], ftype x0, ftype x1, ftype xError)
{
    if constexpr (N == 1)
        return LinearRoot<ftype>(root, coef, x0, x1);
    if constexpr (N == 2)
        return QuadraticFirstRoot<ftype>(root, coef, x0, x1);
    else if constexpr (N == 3)
        return CubicFirstRoot<ftype, boundError, RootFinder>(root, coef, x0, x1, xError);
    else if (coef[N] == 0)
        return PolynomialFirstRoot<N - 1, ftype, boundError, RootFinder>(
            root,
            coef,
            x0,
            x1,
            xError);
    else
    {
        ftype y0 = PolynomialEval<N, ftype>(coef, x0);
        ftype deriv[N];
        PolynomialDerivative<N, ftype>(deriv, coef);
        bool done = PolynomialForEachRoot<N - 1, ftype, boundError, RootFinder>(
            [&](ftype xa) {
                ftype ya = PolynomialEval<N, ftype>(coef, xa);
                if (IsDifferentSign(y0, ya))
                {
                    root = RootFinder::template FindClosed<N, ftype, boundError>(
                        coef,
                        deriv,
                        x0,
                        xa,
                        y0,
                        ya,
                        xError);
                    return true;
                }
                x0 = xa;
                y0 = ya;
                return false;
            },
            deriv,
            x0,
            x1,
            xError);
        if (done)
            return true;
        else
        {
            ftype y1 = PolynomialEval<N, ftype>(coef, x1);
            if (IsDifferentSign(y0, y1))
            {
                root = RootFinder::template FindClosed<N, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    x1,
                    y0,
                    y1,
                    xError);
                return true;
            }
            else
                return false;
        }
    }
}

//-------------------------------------------------------------------------------

template <int N, typename ftype, bool boundError, typename RootFinder>
inline bool PolynomialFirstRoot(ftype& root, ftype const coef[N + 1], ftype xError)
{
    if constexpr (N == 1)
        return LinearRoot<ftype>(root, coef);
    if constexpr (N == 2)
        return QuadraticFirstRoot<ftype>(root, coef);
    else if constexpr (N == 3)
        return CubicFirstRoot<ftype, boundError, RootFinder>(root, coef, xError);
    else if (coef[N] == 0)
        return PolynomialFirstRoot<N - 1, ftype, boundError, RootFinder>(root, coef, xError);
    else
    {
        ftype x0 = -std::numeric_limits<ftype>::infinity();
        ftype y0 = (((N & 1) == 0) ^ (coef[N] < 0)) ? std::numeric_limits<ftype>::infinity() :
                                                      -std::numeric_limits<ftype>::infinity();
        bool firstInterval = true;
        ftype deriv[N];
        PolynomialDerivative<N, ftype>(deriv, coef);
        bool done = PolynomialForEachRoot<N - 1, ftype, boundError, RootFinder>(
            [&](ftype xa) {
                ftype ya = PolynomialEval<N, ftype>(coef, xa);
                if (IsDifferentSign(y0, ya))
                {
                    if (firstInterval)
                    {
                        root = RootFinder::template FindOpenMin<N, ftype, boundError>(
                            coef,
                            deriv,
                            xa,
                            ya,
                            xError);
                    }
                    else
                    {
                        root = RootFinder::template FindClosed<N, ftype, boundError>(
                            coef,
                            deriv,
                            x0,
                            xa,
                            y0,
                            ya,
                            xError);
                    }
                    return true;
                }
                firstInterval = false;
                x0            = xa;
                y0            = ya;
                return false;
            },
            deriv,
            xError);
        if (done)
            return true;
        else
        {
            if constexpr ((N & 1) == 1)
            {
                if (firstInterval)
                {
                    root = RootFinder::template FindOpen<N, ftype, boundError>(coef, deriv, xError);
                }
                else
                {
                    root = RootFinder::template FindOpenMax<N, ftype, boundError>(
                        coef,
                        deriv,
                        x0,
                        y0,
                        xError);
                }
                return true;
            }
            else
                return false;
        }
    }
}

//-------------------------------------------------------------------------------

template <int N, typename ftype, bool boundError, typename RootFinder>
inline bool PolynomialHasRoot(ftype const coef[N + 1], ftype x0, ftype x1, ftype xError)
{
    if constexpr (N == 2)
        return QuadraticHasRoot<ftype>(coef, x0, x1);
    else if constexpr (N == 3)
        return CubicHasRoot<ftype>(coef, x0, x1);
    else if (coef[N] == 0)
        return PolynomialHasRoot<N - 1, ftype, boundError, RootFinder>(coef, x0, x1, xError);
    else
    {
        ftype y0 = PolynomialEval<N, ftype>(coef, x0);
        ftype y1 = PolynomialEval<N, ftype>(coef, x1);
        if (IsDifferentSign(y0, y1))
            return true;
        [[maybe_unused]] bool foundRoot = false;
        ftype deriv[N];
        PolynomialDerivative<N, ftype>(deriv, coef);
        return PolynomialForEachRoot<N - 1, ftype, boundError, RootFinder>(
            [&](ftype xa) {
                ftype ya = PolynomialEval<N, ftype>(coef, xa);
                return (IsDifferentSign(y0, ya));
            },
            deriv,
            x0,
            x1,
            xError);
    }
}

//-------------------------------------------------------------------------------

template <int N, typename ftype, bool boundError, typename RootFinder>
inline bool PolynomialHasRoot(ftype const coef[N + 1], ftype xError)
{
    if constexpr (N == 2)
        return QuadraticHasRoot<ftype>(coef);
    else if constexpr (N == 3)
        return CubicHasRoot<ftype>(coef);
    else if (coef[N] == 0)
        return PolynomialHasRoot<N - 1, ftype, boundError, RootFinder>(coef);
    else if constexpr ((N & 1) == 1)
        return true;
    else
    {
        ftype y0 = (coef[N] < 0) ? -std::numeric_limits<ftype>::infinity() :
                                   std::numeric_limits<ftype>::infinity();
        ftype deriv[N];
        PolynomialDerivative<N, ftype>(deriv, coef);
        return PolynomialForEachRoot<N - 1, ftype, boundError, RootFinder>(
            [&](ftype xa) {
                ftype ya = PolynomialEval<N, ftype>(coef, xa);
                if (IsDifferentSign(y0, ya))
                    return true;
                return false;
            },
            deriv,
            xError);
    }
}

//-------------------------------------------------------------------------------

template <int N, typename ftype, bool boundError, typename RootFinder, typename RootCallback>
inline bool PolynomialForEachRoot(
    RootCallback callback,
    ftype const coef[N + 1],
    ftype x0,
    ftype x1,
    ftype xError)
{
    if constexpr (N == 2)
        return QuadraticForEachRoot<ftype, RootCallback>(callback, coef, x0, x1);
    else if constexpr (N == 3)
        return CubicForEachRoot<ftype, boundError, RootFinder, RootCallback>(
            callback,
            coef,
            x0,
            x1,
            xError);
    else if (coef[N] == 0)
        return PolynomialForEachRoot<N - 1, ftype, boundError, RootFinder, RootCallback>(
            callback,
            coef,
            x0,
            x1,
            xError);
    else
    {
        ftype y0 = PolynomialEval<N, ftype>(coef, x0);
        ftype deriv[N];
        PolynomialDerivative<N, ftype>(deriv, coef);
        bool done = PolynomialForEachRoot<N - 1, ftype, boundError, RootFinder>(
            [&](ftype xa) {
                ftype ya = PolynomialEval<N, ftype>(coef, xa);
                if (IsDifferentSign(y0, ya))
                {
                    ftype root;
                    root = RootFinder::template FindClosed<N, ftype, boundError>(
                        coef,
                        deriv,
                        x0,
                        xa,
                        y0,
                        ya,
                        xError);
                    if (callback(root))
                        return true;
                }
                x0 = xa;
                y0 = ya;
                return false;
            },
            deriv,
            x0,
            x1,
            xError);
        if (done)
            return true;
        else
        {
            ftype y1 = PolynomialEval<N, ftype>(coef, x1);
            if (IsDifferentSign(y0, y1))
            {
                return callback(RootFinder::template FindClosed<N, ftype, boundError>(
                    coef,
                    deriv,
                    x0,
                    x1,
                    y0,
                    y1,
                    xError));
            }
            return false;
        }
    }
}

//-------------------------------------------------------------------------------

template <int N, typename ftype, bool boundError, typename RootFinder, typename RootCallback>
inline bool PolynomialForEachRoot(RootCallback callback, ftype const coef[N + 1], ftype xError)
{
    if constexpr (N == 2)
        return QuadraticForEachRoot<ftype, RootCallback>(callback, coef);
    else if constexpr (N == 3)
        return CubicForEachRoot<ftype, boundError, RootFinder, RootCallback>(
            callback,
            coef,
            xError);
    else if (coef[N] == 0)
        return PolynomialForEachRoot<N - 1, ftype, boundError, RootFinder, RootCallback>(
            callback,
            coef,
            xError);
    else
    {
        ftype x0 = -std::numeric_limits<ftype>::infinity();
        ftype y0 = (((N & 1) == 0) ^ (coef[N] < 0)) ? std::numeric_limits<ftype>::infinity() :
                                                      -std::numeric_limits<ftype>::infinity();
        bool firstInterval = true;
        ftype deriv[N];
        PolynomialDerivative<N, ftype>(deriv, coef);
        bool done = PolynomialForEachRoot<N - 1, ftype, boundError, RootFinder>(
            [&](ftype xa) {
                ftype ya = PolynomialEval<N, ftype>(coef, xa);
                if (IsDifferentSign(y0, ya))
                {
                    ftype root;
                    if (firstInterval)
                    {
                        root = RootFinder::template FindOpenMin<N, ftype, boundError>(
                            coef,
                            deriv,
                            xa,
                            ya,
                            xError);
                    }
                    else
                    {
                        root = RootFinder::template FindClosed<N, ftype, boundError>(
                            coef,
                            deriv,
                            x0,
                            xa,
                            y0,
                            ya,
                            xError);
                    }
                    if (callback(root))
                        return true;
                }
                firstInterval = false;
                x0            = xa;
                y0            = ya;
                return false;
            },
            deriv,
            xError);
        if (done)
            return true;
        else
        {
            if (IsDifferentSign(y0, coef[N]))
            {
                ftype root;
                if constexpr ((N & 1) == 1)
                {
                    if (firstInterval)
                    {
                        root = RootFinder::template FindOpen<N, ftype, boundError>(
                            coef,
                            deriv,
                            xError);
                    }
                    else
                    {
                        root = RootFinder::template FindOpenMax<N, ftype, boundError>(
                            coef,
                            deriv,
                            x0,
                            y0,
                            xError);
                    }
                }
                else
                {
                    root = RootFinder::template FindOpenMax<N, ftype, boundError>(
                        coef,
                        deriv,
                        x0,
                        y0,
                        xError);
                }
                return callback(root);
            }
            return false;
        }
    }
}

//-------------------------------------------------------------------------------
} // namespace cy

#endif // _CY_POLYNOMIAL_H_INCLUDED_

// clang-format on

} // namespace pbat::math::polynomial::detail

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__) or defined(__GNUG__)
    #pragma GCC diagnostic pop
#elif defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include "pbat/Aliases.h"

#include <array>
#include <utility>

namespace pbat::math::polynomial {

namespace detail {
template <class T, auto N>
using CArray = T[N]; ///< C-style array alias.
} // namespace detail

/**
 * @brief Check if a polynomial has a root in the range [min,max].
 *
 * @note We use the expensive but most accurate method from \cite cem2022polyroot
 * @tparam N The degree of the polynomial.
 * @tparam TScalar The scalar type of the polynomial.
 * @param coeffs The coefficients of the polynomial.
 * @param min The minimum value of the search range.
 * @param max The maximum value of the search range.
 * @return true if the polynomial has a root, false otherwise.
 */
template <auto N, class TScalar = Scalar>
inline bool HasRoot(
    std::array<TScalar, N + 1> const& coeffs,
    TScalar min = std::numeric_limits<TScalar>::lowest(),
    TScalar max = std::numeric_limits<TScalar>::max())
{
    return detail::cy::PolynomialHasRoot<N, TScalar, true>(
        *reinterpret_cast<detail::CArray<TScalar, N + 1> const*>(coeffs.data()),
        min,
        max,
        TScalar(0));
}

/**
 * @brief Computes all real roots of a degree N polynomial in the range [min,max].
 *
 * @note We use the expensive but most accurate method from @cite cem2022polyroot
 * @tparam N Degree of the polynomial.
 * @tparam TScalar Scalar type of the polynomial.
 * @param coeffs Coefficients of the polynomial.
 * @param min Minimum value of the search range.
 * @param max Maximum value of the search range.
 * @return The roots of the polynomial at the head of the returned array, NaNs at the tail.
 */
template <auto N, class TScalar = Scalar>
inline std::array<TScalar, N> Roots(
    std::array<TScalar, N + 1> const& coeffs,
    TScalar min = std::numeric_limits<TScalar>::lowest(),
    TScalar max = std::numeric_limits<TScalar>::max())
{
    std::array<TScalar, N> roots;
    roots.fill(std::numeric_limits<TScalar>::quiet_NaN());
    [[maybe_unused]] int const nRoots = detail::cy::PolynomialRoots<N, TScalar, true>(
        *reinterpret_cast<detail::CArray<TScalar, N>*>(roots.data()),
        *reinterpret_cast<detail::CArray<TScalar, N + 1> const*>(coeffs.data()),
        min,
        max,
        TScalar(0));
    return roots;
}

/**
 * @brief Computes the each real root of a degree N polynomial in the range [min,max] in increasing
 * order.
 *
 *
 *
 * @note We use the expensive but most accurate method from @cite cem2022polyroot
 *
 * @tparam N Degree of the polynomial.
 * @tparam FOnRoot Callable type with signature `bool(TScalar root)`.
 * @tparam TScalar Scalar type of the polynomial.
 * @param fOnRoot Callable object to be called for each root. If the callable returns true, the
 * iteration stops.
 * @param coeffs Coefficients of the polynomial.
 * @param min Minimum value of the search range.
 * @param max Maximum value of the search range.
 * @return true if root finding is terminated by fOnRoot. Otherwise, returns false.
 */
template <auto N, class FOnRoot, class TScalar = Scalar>
inline bool ForEachRoot(
    FOnRoot&& fOnRoot,
    std::array<TScalar, N + 1> const& coeffs,
    TScalar min = std::numeric_limits<TScalar>::lowest(),
    TScalar max = std::numeric_limits<TScalar>::max())
{
    return detail::cy::PolynomialForEachRoot<N, TScalar, true>(
        std::forward<FOnRoot>(fOnRoot),
        *reinterpret_cast<detail::CArray<TScalar, N + 1> const*>(coeffs.data()),
        min,
        max,
        TScalar(0));
}

} // namespace pbat::math::polynomial

#endif // PBAT_MATH_POLYNOMIAL_ROOTS_H

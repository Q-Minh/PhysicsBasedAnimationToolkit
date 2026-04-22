#ifndef PBAT_COMMON_ATOMIC_H
#define PBAT_COMMON_ATOMIC_H

#include <atomic>
#include <bit>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>

namespace pbat::common {

namespace detail {

/**
 * @brief Maps a floating-point type to an unsigned integer type of the same size.
 */
template <class T>
struct UintFromFloat;

template <>
struct UintFromFloat<float>
{
    using type = std::uint32_t;
};

template <>
struct UintFromFloat<double>
{
    using type = std::uint64_t;
};

/**
 * @brief Lock-free atomic compare-and-swap for IEEE 754 floating-point values, implemented via
 * integer atomics.
 *
 * # Motivation
 *
 * `std::atomic_ref<float>` is not guaranteed to be lock-free (and is NOT lock-free on MSVC).
 * When it's not lock-free, the implementation uses an internal hash-based lock table keyed on
 * the object's address. Creating temporary `atomic_ref<float>` instances with non-overlapping
 * lifetimes in concurrent threads is then UB (and crashes in practice on MSVC).
 *
 * # Approach
 *
 * `std::atomic_ref<uint32_t>` (and `uint64_t`) IS always lock-free on all major platforms.
 * Since `float` and `uint32_t` share size and alignment, we reinterpret the float storage as
 * an unsigned integer and perform a CAS loop on that.
 *
 * For IEEE 754, we transform the bit representation into a sign-magnitude-aware unsigned
 * integer that preserves the total ordering of finite floats:
 *   - Non-negative floats: `bit_cast<uint>(a)` is already order-preserving.
 *   - Negative floats: flip all bits to reverse the ordering.
 *
 * # Aliasing note
 *
 * Strictly speaking, creating `std::atomic_ref<uint32_t>` over storage that contains a `float`
 * object is not sanctioned by the C++ standard. However, this technique is universally used
 * (CUDA atomicMin, folly, abseil, etc.) and is safe on all real platforms because:
 *   1. `float` and `uint32_t` share size, alignment, and have no trap representations.
 *   2. Lock-free `atomic_ref` compiles to hardware CAS instructions that operate on raw bytes.
 *   3. No compiler optimizes across `atomic_ref` CAS boundaries in a way that would break this.
 *
 * If C++23 `std::start_lifetime_as<TUint>` becomes available, it can be used to make this
 * fully standards-compliant.
 *
 * @tparam TFloat Floating-point type (float or double)
 * @tparam FCompare Binary predicate on ordered unsigned integers; `a` is replaced by `b` when
 *         `fCompare(orderedB, orderedA)` returns true.
 * @param a Reference to the floating-point value to update
 * @param b The candidate value
 * @param fCompare Comparison predicate
 */
template <std::floating_point TFloat, class FCompare>
void AtomicCmpXchgFloat(TFloat& a, TFloat b, FCompare&& fCompare) noexcept
{
    using TUint                  = typename UintFromFloat<TFloat>::type;
    static constexpr TUint kSign = TUint{1} << (sizeof(TUint) * 8 - 1);
    static_assert(sizeof(TFloat) == sizeof(TUint));
    static_assert(alignof(TFloat) >= alignof(TUint));
    static_assert(
        std::atomic_ref<TUint>::is_always_lock_free,
        "Integer atomic_ref must be lock-free for the float-via-integer-atomics trick to work");

    // Transform float bits into an unsigned integer that preserves the total float ordering.
    auto fToOrderedUint = [](TFloat v) noexcept -> TUint {
        TUint bits = std::bit_cast<TUint>(v);
        // If the sign bit is set (negative), flip all bits.
        // If the sign bit is clear (non-negative), flip only the sign bit.
        // This maps the float range [-inf, +inf] to [0, UINT_MAX] monotonically.
        return (bits & kSign) ? ~bits : (bits | kSign);
    };

    // NOTE: We use reinterpret_cast to obtain a TUint& alias to the float storage.
    // See the aliasing note in the doc comment above.
    auto& aAsUint = reinterpret_cast<TUint&>(a);
    std::atomic_ref<TUint> aAtomic(aAsUint);
    TUint bOrdered   = fToOrderedUint(b);
    TUint old        = aAtomic.load(std::memory_order_relaxed);
    TUint oldOrdered = fToOrderedUint(std::bit_cast<TFloat>(old));
    while (fCompare(bOrdered, oldOrdered) and not aAtomic.compare_exchange_weak(
                                                  old,
                                                  std::bit_cast<TUint>(b),
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed))
    {
        oldOrdered = fToOrderedUint(std::bit_cast<TFloat>(old));
    }
}

/**
 * @brief Atomic compare-and-swap loop for `std::atomic<T>` or `std::atomic_ref<T>`.
 *
 * Replaces the value in `a` with `b` when `fCompare(b, old)` returns true.
 *
 * @tparam TAtomic `std::atomic<T>` or `std::atomic_ref<T>`
 * @tparam T Value type
 * @tparam FCompare Binary predicate
 */
template <class TAtomic, class T, class FCompare>
void AtomicCmpXchg(TAtomic& a, T b, FCompare&& fCompare) noexcept
{
    T old = a.load(std::memory_order_relaxed);
    while (
        fCompare(b, old) and
        not a.compare_exchange_weak(old, b, std::memory_order_release, std::memory_order_relaxed))
        ;
}

} // namespace detail

/**
 * @brief Order-independent atomic minimum operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 * @post The minimum of the original value of `a` and `b` is stored in `a`.
 */
template <class T>
    requires std::is_arithmetic_v<T>
void AtomicMin(std::atomic<T>& a, T b) noexcept
{
    detail::AtomicCmpXchg(a, b, std::less<T>{});
}

/**
 * @brief Order-independent atomic minimum operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 * @post The minimum of the original value of `a` and `b` is stored in `a`.
 */
template <class T>
    requires std::is_arithmetic_v<T>
void AtomicMin(std::atomic_ref<T>& a, T b) noexcept
{
    detail::AtomicCmpXchg(a, b, std::less<T>{});
}

/**
 * @brief Order-independent atomic minimum for floating-point values (`float` or `double`).
 *
 * Uses integer atomics on the IEEE 754 bit representation to guarantee lock-free
 * operation and avoid `std::atomic_ref<float/double>` lifetime/lock-freedom pitfalls on MSVC.
 *
 * @tparam T Floating-point type
 * @param a Left operand
 * @param b Right operand
 */
template <std::floating_point T>
void AtomicMin(T& a, T b) noexcept
{
    detail::AtomicCmpXchgFloat(a, b, std::less<>{});
}

/**
 * @brief Order-independent atomic minimum operation for integral types.
 *
 * @pre `std::atomic_ref<T>` is lock-free, which guarantees that temporary `std::atomic_ref`
 * instances are safe even with non-overlapping lifetimes across threads (no internal lock-table
 * bookkeeping).
 *
 * @tparam T Integral type
 * @param a Left operand
 * @param b Right operand
 */
template <std::integral T>
void AtomicMin(T& a, T b) noexcept
{
    static_assert(
        std::atomic_ref<T>::is_always_lock_free,
        "AtomicMin(T&,T) requires lock-free std::atomic_ref<T> to be safe with "
        "non-overlapping temporary lifetimes across threads");
    std::atomic_ref<T> aa(a);
    AtomicMin(aa, b);
}

/**
 * @brief Order-independent atomic maximum operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 * @post The maximum of the original value of `a` and `b` is stored in `a`.
 */
template <class T>
    requires std::is_arithmetic_v<T>
void AtomicMax(std::atomic<T>& a, T b) noexcept
{
    detail::AtomicCmpXchg(a, b, std::greater<T>{});
}

/**
 * @brief Order-independent atomic maximum operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 * @post The maximum of the original value of `a` and `b` is stored in `a`.
 */
template <class T>
    requires std::is_arithmetic_v<T>
void AtomicMax(std::atomic_ref<T>& a, T b) noexcept
{
    detail::AtomicCmpXchg(a, b, std::greater<T>{});
}

/**
 * @brief Order-independent atomic maximum for floating-point values (`float` or `double`).
 *
 * Uses integer atomics on the IEEE 754 bit representation to guarantee lock-free
 * operation and avoid `std::atomic_ref<float/double>` lifetime/lock-freedom pitfalls on MSVC.
 *
 * @tparam T Floating-point type
 * @param a Left operand
 * @param b Right operand
 */
template <std::floating_point T>
void AtomicMax(T& a, T b) noexcept
{
    detail::AtomicCmpXchgFloat(a, b, std::greater<>{});
}

/**
 * @brief Order-independent atomic maximum operation for integral types.
 *
 * @pre `std::atomic_ref<T>` is lock-free, which guarantees that temporary `std::atomic_ref`
 * instances are safe even with non-overlapping lifetimes across threads (no internal lock-table
 * bookkeeping).
 *
 * @tparam T Integral type
 * @param a Left operand
 * @param b Right operand
 */
template <std::integral T>
void AtomicMax(T& a, T b) noexcept
{
    static_assert(
        std::atomic_ref<T>::is_always_lock_free,
        "AtomicMax(T&,T) requires lock-free std::atomic_ref<T> to be safe with "
        "non-overlapping temporary lifetimes across threads");
    std::atomic_ref<T> aa(a);
    AtomicMax(aa, b);
}

/**
 * @brief Execute a function atomically with respect to a boolean lock.
 *
 * @tparam Func Callable type with signature `void()`
 * @param lock Lock variable
 * @param f Function to execute
 */
template <class Func>
void AtomicExecute(std::atomic<bool>& lock, Func&& f) noexcept
{
    while (std::atomic_exchange_explicit(&lock, true, std::memory_order_acquire))
        ;
    f();
    std::atomic_store_explicit(&lock, false, std::memory_order_release);
}

/**
 * @brief Execute a function atomically with respect to a boolean lock.
 *
 * @tparam Func Callable type with signature `void()`
 * @param lock Lock variable
 * @param f Function to execute
 */
template <class Func>
void AtomicExecute(std::atomic_ref<bool>& lock, Func&& f) noexcept
{
    while (lock.exchange(true, std::memory_order_acquire))
        ;
    f();
    lock.store(false, std::memory_order_release);
}

/**
 * @brief Execute a function atomically with respect to a boolean lock.
 *
 * @pre `std::atomic_ref<bool>` is lock-free, which guarantees that temporary `std::atomic_ref`
 * instances are safe even with non-overlapping lifetimes across threads.
 *
 * @tparam Func Callable type with signature `void()`
 * @param lock Lock variable
 * @param f Function to execute
 */
template <class Func>
void AtomicExecute(bool& lock, Func&& f) noexcept
{
    static_assert(
        std::atomic_ref<bool>::is_always_lock_free,
        "AtomicExecute(bool&,Func) requires lock-free std::atomic_ref<bool> to be safe with "
        "non-overlapping temporary lifetimes across threads");
    std::atomic_ref<bool> alock(lock);
    AtomicExecute(alock, std::forward<Func>(f));
}

/**
 * @brief Atomic addition operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 * @post The sum of the original value of `a` and `b` is stored in `a`.
 */
template <class T>
    requires std::is_arithmetic_v<T>
T AtomicAdd(std::atomic<T>& a, T b) noexcept
{
    return a.fetch_add(b, std::memory_order_relaxed);
}

/**
 * @brief Atomic addition operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 * @post The sum of the original value of `a` and `b` is stored in `a`.
 */
template <class T>
    requires std::is_arithmetic_v<T>
T AtomicAdd(std::atomic_ref<T>& a, T b) noexcept
{
    return a.fetch_add(b, std::memory_order_relaxed);
}

/**
 * @brief Atomic addition operation.
 *
 * @pre `std::atomic_ref<T>` is lock-free, which guarantees that temporary `std::atomic_ref`
 * instances are safe even with non-overlapping lifetimes across threads.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 */
template <class T>
    requires std::is_arithmetic_v<T>
T AtomicAdd(T& a, T b) noexcept
{
    static_assert(
        std::atomic_ref<T>::is_always_lock_free,
        "AtomicAdd(T&,T) requires lock-free std::atomic_ref<T> to be safe with "
        "non-overlapping temporary lifetimes across threads");
    std::atomic_ref<T> aa(a);
    return AtomicAdd(aa, b);
}

} // namespace pbat::common

#endif // PBAT_COMMON_ATOMIC_H

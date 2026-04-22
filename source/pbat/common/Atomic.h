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

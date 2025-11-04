#ifndef PBAT_COMMON_ATOMIC_H
#define PBAT_COMMON_ATOMIC_H

#include <atomic>
#include <type_traits>

namespace pbat::common {

/**
 * @brief Order-independent atomic minimum operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 * @post The minimum of the original value of `a` and `b` is stored in `a`.
 */
template <class T>
void AtomicMin(std::atomic<T>& a, T b) noexcept
{
    static_assert(std::is_arithmetic_v<T>, "AtomicMin requires arithmetic type T");
    T old = a.load(std::memory_order_relaxed);
    while (
        b < old and
        not a.compare_exchange_weak(old, b, std::memory_order_release, std::memory_order_relaxed))
        ;
}

/**
 * @brief Order-independent atomic minimum operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 * @post The minimum of the original value of `a` and `b` is stored
 */
template <class T>
void AtomicMin(std::atomic_ref<T>& a, T b) noexcept
{
    static_assert(std::is_arithmetic_v<T>, "AtomicMin requires arithmetic type T");
    T old = a.load(std::memory_order_relaxed);
    while (
        b < old and
        not a.compare_exchange_weak(old, b, std::memory_order_release, std::memory_order_relaxed))
        ;
}

/**
 * @brief Order-independent atomic minimum operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 */
template <class T>
void AtomicMin(T& a, T b) noexcept
{
    std::atomic_ref<T> aa(a);
    AtomicMin(aa, b);
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
 * @tparam Func Callable type with signature `void()`
 * @param lock Lock variable
 * @param f Function to execute
 */
template <class Func>
void AtomicExecute(bool& lock, Func&& f) noexcept
{
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
T AtomicAdd(std::atomic<T>& a, T b) noexcept
{
    static_assert(std::is_arithmetic_v<T>, "AtomicIncrement requires arithmetic type T");
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
T AtomicAdd(std::atomic_ref<T>& a, T b) noexcept
{
    static_assert(std::is_arithmetic_v<T>, "AtomicIncrement requires arithmetic type T");
    return a.fetch_add(b, std::memory_order_relaxed);
}

/**
 * @brief Atomic addition operation.
 *
 * @tparam T Arithmetic type
 * @param a Left operand
 * @param b Right operand
 */
template <class T>
T AtomicAdd(T& a, T b) noexcept
{
    std::atomic_ref<T> aa(a);
    return AtomicAdd(aa, b);
}

} // namespace pbat::common

#endif // PBAT_COMMON_ATOMIC_H

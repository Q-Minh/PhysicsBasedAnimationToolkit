/**
 * @file Stack.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Fixed-size stack implementation usable in both host and device code
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup common
 */

#ifndef PBAT_COMMON_STACK_H
#define PBAT_COMMON_STACK_H

#include "pbat/HostDevice.h"

namespace pbat {
namespace common {

/**
 * @brief Fixed-size stack implementation
 *
 * @tparam T Type of the elements in the stack
 * @tparam kCapacity Maximum number of elements in the stack
 * @ingroup common
 */
template <class T, auto kCapacity = 64>
class Stack
{
  public:
    /**
     * @brief Construct empty Stack
     */
    PBAT_HOST_DEVICE Stack() : stack{}, size{0} {}
    /**
     * @brief Add element to the stack
     *
     * @param value Element to add
     */
    PBAT_HOST_DEVICE void Push(T value) { stack[size++] = value; }
    /**
     * @brief Remove the top element from the stack
     *
     * @return T Top element of the stack
     */
    PBAT_HOST_DEVICE T Pop() { return stack[--size]; }
    /**
     * @brief Get the top element of the stack
     *
     * @return const& T Top element of the stack
     */
    PBAT_HOST_DEVICE T const& Top() const { return stack[size - 1]; }
    /**
     * @brief Get the number of elements in the stack
     *
     * @return auto Number of elements in the stack
     */
    PBAT_HOST_DEVICE auto Size() const { return size; }
    /**
     * @brief Check if the stack is empty
     *
     * @return true If the stack is empty
     */
    PBAT_HOST_DEVICE bool IsEmpty() const { return size == 0; }
    /**
     * @brief Check if the stack is full
     *
     * @return true If the stack is full
     */
    PBAT_HOST_DEVICE bool IsFull() const { return size == kCapacity; }
    /**
     * @brief Clear the stack
     */
    PBAT_HOST_DEVICE void Clear() { size = 0; }
    /**
     * @brief Access element at index i
     *
     * @note No bounds checking
     * @pre i < size and i >= 0
     * @param i Index of the element
     * @return T& Reference to the element at index i
     */
    PBAT_HOST_DEVICE T& operator[](auto i) { return stack[i]; }
    /**
     * @brief Read-only access element at index i
     *
     * @note No bounds checking
     * @pre i < size and i >= 0
     * @param i Index of the element
     * @return const& T Reference to the element at index i
     */
    PBAT_HOST_DEVICE T const& operator[](auto i) const { return stack[i]; }
    /**
     * @brief Pointer to the beginning of the stack
     *
     * @return T* Pointer to the beginning of the stack
     */
    PBAT_HOST_DEVICE T* begin() { return stack; }
    /**
     * @brief Pointer to the end of the stack
     *
     * @return T* Pointer to the end of the stack
     */
    PBAT_HOST_DEVICE T* end() { return stack + size; }

  private:
    T stack[kCapacity];
    int size; ///< Serves as both the stack pointer and the number of elements in the stack
};

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_STACK_H

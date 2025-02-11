/**
 * @file Queue.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Fixed-size queue implementation usable in both host and device code
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_COMMON_QUEUE_H
#define PBAT_COMMON_QUEUE_H

#include "pbat/HostDevice.h"

namespace pbat {
namespace common {

/**
 * @brief Fixed-size queue implementation
 *
 * @tparam T Type of the elements in the queue
 * @tparam kCapacity Maximum number of elements in the queue
 */
template <class T, auto kCapacity = 64>
class Queue
{
  public:
    /**
     * @brief Construct empty Queue
     */
    PBAT_HOST_DEVICE Queue() : queue{}, begin{0}, end{0}, n{0} {}
    /**
     * @brief Add element to the queue
     *
     * @param value Element to add
     */
    PBAT_HOST_DEVICE void Push(T value)
    {
        queue[end] = value;
        end        = (end + 1) % Capacity();
        ++n;
    }
    /**
     * @brief Get the next element in the queue
     *
     * @return const& T Next element in the queue
     */
    PBAT_HOST_DEVICE T const& Top() const { return queue[begin]; }
    /**
     * @brief Remove the next element in the queue
     */
    PBAT_HOST_DEVICE void Pop()
    {
        begin = (begin + 1) % Capacity();
        --n;
    }
    /**
     * @brief Check if the queue is full
     *
     * @return bool True if the queue is full
     */
    PBAT_HOST_DEVICE bool IsFull() const { return n == Capacity(); }
    /**
     * @brief Check if the queue is empty
     *
     * @return bool True if the queue is empty
     */
    PBAT_HOST_DEVICE bool IsEmpty() const { return n == 0; }
    /**
     * @brief Get the number of elements in the queue
     *
     * @return auto Number of elements in the queue
     */
    PBAT_HOST_DEVICE auto Size() const { return n; }
    /**
     * @brief Clear the queue
     */
    PBAT_HOST_DEVICE void Clear() { begin = end = n = 0; }
    /**
     * @brief Get the maximum number of elements in the queue
     *
     * @return constexpr auto Maximum number of elements in the queue
     */
    PBAT_HOST_DEVICE constexpr auto Capacity() const { return kCapacity; }

  private:
    T queue[kCapacity];
    int begin, end, n;
};

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_QUEUE_H

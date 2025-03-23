/**
 * @file Heap.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Fixed-size heap implementation usable in both host and device code
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_COMMON_HEAP_H
#define PBAT_COMMON_HEAP_H

#include "pbat/HostDevice.h"

#include <algorithm>
#include <cassert>

namespace pbat::common {

/**
 * @brief Fixed-size max heap
 *
 * @tparam T Type of the elements
 * @tparam Less Less-than comparator
 * @tparam kCapacity Capacity of the heap
 */
template <class T, class Less = std::less<T>, auto kCapacity = 64>
class Heap
{
  public:
    /**
     * @brief Construct a new Heap object
     *
     * @param less Comparator for the heap
     */
    PBAT_HOST_DEVICE Heap(Less less = Less{}) : heap{}, less(less), size{0} {}
    /**
     * @brief Push an element to the heap
     *
     * @param value Element to push
     */
    PBAT_HOST_DEVICE void Push(T value)
    {
        assert(size < kCapacity);
        heap[size++] = value;
        std::push_heap(heap, heap + size, less);
    }
    /**
     * @brief Pop the top element from the heap
     *
     * @return Top element
     */
    PBAT_HOST_DEVICE T Pop()
    {
        assert(not IsEmpty());
        std::pop_heap(heap, heap + size--, less);
        return heap[size];
    }
    /**
     * @brief Get the top element of the heap
     *
     * @return Top element
     */
    PBAT_HOST_DEVICE T const& Top() const { return heap[0]; }
    /**
     * @brief Get the size of the heap
     *
     * @return Size of the heap
     */
    PBAT_HOST_DEVICE auto Size() const { return size; }
    /**
     * @brief Check if the heap is empty
     *
     * @return true if the heap is empty, false otherwise
     */
    PBAT_HOST_DEVICE bool IsEmpty() const { return size == 0; }

  private:
    T heap[kCapacity]; ///< Buffer allocated for storing elements
    Less less;         ///< Comparator
    int size;          ///< Size of the heap
};

} // namespace pbat::common

#endif // PBAT_COMMON_HEAP_H

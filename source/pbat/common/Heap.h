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

template <class T, class Less = std::less<T>, auto kCapacity = 64>
class Heap
{
  public:
    PBAT_HOST_DEVICE Heap(Less less = Less{}) : heap{}, less(less), size{0} {}
    PBAT_HOST_DEVICE void Push(T value)
    {
        assert(size < kCapacity);
        heap[size++] = value;
        std::push_heap(heap, heap + size, less);
    }
    PBAT_HOST_DEVICE T Pop()
    {
        assert(not IsEmpty());
        std::pop_heap(heap, heap + size--, less);
        return heap[size];
    }
    PBAT_HOST_DEVICE T const& Top() const { return heap[0]; }
    PBAT_HOST_DEVICE auto Size() const { return size; }
    PBAT_HOST_DEVICE bool IsEmpty() const { return size == 0; }

  private:
    T heap[kCapacity];
    Less less;
    int size;
};

} // namespace pbat::common

#endif // PBAT_COMMON_HEAP_H

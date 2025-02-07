#ifndef PBAT_COMMON_QUEUE_H
#define PBAT_COMMON_QUEUE_H

#include "pbat/HostDevice.h"

namespace pbat {
namespace common {

template <class T, auto kCapacity = 64>
class Queue
{
  public:
    PBAT_HOST_DEVICE Queue() : queue{}, begin{0}, end{0}, n{0} {}
    PBAT_HOST_DEVICE void Push(T value)
    {
        queue[end] = value;
        end        = (end + 1) % Capacity();
        ++n;
    }
    PBAT_HOST_DEVICE T const& Top() const { return queue[begin]; }
    PBAT_HOST_DEVICE void Pop()
    {
        begin = (begin + 1) % Capacity();
        --n;
    }
    PBAT_HOST_DEVICE bool IsFull() const { return n == Capacity(); }
    PBAT_HOST_DEVICE bool IsEmpty() const { return n == 0; }
    PBAT_HOST_DEVICE auto Size() const { return n; }
    PBAT_HOST_DEVICE void Clear() { begin = end = n = 0; }
    PBAT_HOST_DEVICE constexpr auto Capacity() const { return kCapacity; }

  private:
    T queue[kCapacity];
    int begin, end, n;
};

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_QUEUE_H

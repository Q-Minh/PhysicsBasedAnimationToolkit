from . import barrier, buffer

import warp as wp


@wp.func
def clz(x: wp.int32) -> wp.int32:  # type: ignore
    # Handle the case where x is 0 (all 32 bits are zero)
    if x == wp.int32(0):
        return wp.int32(32)

    # If x is negative, bit 31 is 1, so there are 0 leading zeros
    if x < wp.int32(0):
        return wp.int32(0)

    n = wp.int32(0)

    # Binary search the leading zeros
    # Check if the top 16 bits are all zero
    if wp.bit_and(x, wp.int32(0xFFFF0000)) == wp.int32(0):
        n += wp.int32(16)
        x = wp.lshift(x, wp.int32(16))

    # Check if the top 8 bits (of the remaining) are zero
    if wp.bit_and(x, wp.int32(0xFF000000)) == wp.int32(0):
        n += wp.int32(8)
        x = wp.lshift(x, wp.int32(8))

    # Check top 4 bits
    if wp.bit_and(x, wp.int32(0xF0000000)) == wp.int32(0):
        n += wp.int32(4)
        x = wp.lshift(x, wp.int32(4))

    # Check top 2 bits
    if wp.bit_and(x, wp.int32(0xC0000000)) == wp.int32(0):
        n += wp.int32(2)
        x = wp.lshift(x, wp.int32(2))

    # Check the very last bit
    if wp.bit_and(x, wp.int32(0x80000000)) == wp.int32(0):
        n += wp.int32(1)

    return n


@wp.func
def clz(x: wp.uint32) -> wp.uint32:  # type: ignore
    """Count leading zeros for unsigned 32-bit integers."""
    if x == wp.uint32(0):
        return wp.uint32(32)

    n = wp.uint32(0)

    if wp.bit_and(x, wp.uint32(0xFFFF0000)) == wp.uint32(0):
        n += wp.uint32(16)
        x = wp.lshift(x, wp.uint32(16))

    if wp.bit_and(x, wp.uint32(0xFF000000)) == wp.uint32(0):
        n += wp.uint32(8)
        x = wp.lshift(x, wp.uint32(8))

    if wp.bit_and(x, wp.uint32(0xF0000000)) == wp.uint32(0):
        n += wp.uint32(4)
        x = wp.lshift(x, wp.uint32(4))

    if wp.bit_and(x, wp.uint32(0xC0000000)) == wp.uint32(0):
        n += wp.uint32(2)
        x = wp.lshift(x, wp.uint32(2))

    if wp.bit_and(x, wp.uint32(0x80000000)) == wp.uint32(0):
        n += wp.uint32(1)

    return n


@wp.func
def clz(x: wp.uint64) -> wp.uint64:  # type: ignore
    """Count leading zeros for unsigned 64-bit integers."""
    if x == wp.uint64(0):
        return wp.uint64(64)

    n = wp.uint64(0)

    if wp.bit_and(x, wp.uint64(0xFFFFFFFF00000000)) == wp.uint64(0):
        n += wp.uint64(32)
        x = wp.lshift(x, wp.uint64(32))

    if wp.bit_and(x, wp.uint64(0xFFFF000000000000)) == wp.uint64(0):
        n += wp.uint64(16)
        x = wp.lshift(x, wp.uint64(16))

    if wp.bit_and(x, wp.uint64(0xFF00000000000000)) == wp.uint64(0):
        n += wp.uint64(8)
        x = wp.lshift(x, wp.uint64(8))

    if wp.bit_and(x, wp.uint64(0xF000000000000000)) == wp.uint64(0):
        n += wp.uint64(4)
        x = wp.lshift(x, wp.uint64(4))

    if wp.bit_and(x, wp.uint64(0xC000000000000000)) == wp.uint64(0):
        n += wp.uint64(2)
        x = wp.lshift(x, wp.uint64(2))

    if wp.bit_and(x, wp.uint64(0x8000000000000000)) == wp.uint64(0):
        n += wp.uint64(1)

    return n


@wp.func
def lower_bound(arr: wp.array[wp.int32], n: wp.int32, key: wp.int32) -> wp.int32:  # type: ignore
    """Branchless lower bound: returns the index of the first element >= key in arr[0:n].
    Uses clz to compute the exact number of iterations needed (no wasted steps).
    """
    lo = wp.int32(0)
    # Number of binary-search iterations = 32 - clz(n) for n > 0, 0 for n == 0.
    iters = wp.int32(32) - clz(n)  # type: ignore
    length = n
    while iters > wp.int32(0):
        half = length >> wp.int32(1)
        go_right = wp.int32(arr[lo + half] < key)
        lo += go_right * (half + wp.int32(1))
        length = (
            go_right * (length - half - wp.int32(1)) + (wp.int32(1) - go_right) * half
        )
        iters -= wp.int32(1)
    return lo


@wp.func
def lower_bound(u: wp.array[wp.int32], v: wp.array[wp.int32], n: wp.int32, keyu: wp.int32, keyv: wp.int32) -> wp.int32:  # type: ignore
    """Branchless lower bound: returns the index of the first element >= (keyu, keyv) in (u, v)[0:n].
    Uses lexicographic order: (a, b) < (c, d) if a < c or (a == c and b < d).
    Uses clz to compute the exact number of iterations needed (no wasted steps).
    """
    lo = wp.int32(0)
    # Number of binary-search iterations = 32 - clz(n) for n > 0, 0 for n == 0.
    iters = wp.int32(32) - clz(n)  # type: ignore
    length = n
    while iters > wp.int32(0):
        half = length >> wp.int32(1)
        go_right = wp.int32(
            u[lo + half] < keyu or (u[lo + half] == keyu and v[lo + half] < keyv)
        )
        lo += go_right * (half + wp.int32(1))
        length = (
            go_right * (length - half - wp.int32(1)) + (wp.int32(1) - go_right) * half
        )
        iters -= wp.int32(1)
    return lo


@wp.func
def lower_bound(arr: wp.array[wp.uint32], n: wp.uint32, key: wp.uint32) -> wp.uint32:  # type: ignore
    """Branchless lower bound on a uint32 array: returns the index of the first element >= key in arr[0:n]."""
    lo = wp.uint32(0)
    iters = wp.uint32(32) - clz(n)  # type: ignore
    length = n
    while iters > wp.uint32(0):
        half = length >> wp.uint32(1)
        go_right = wp.uint32(arr[lo + half] < key)
        lo += go_right * (half + wp.uint32(1))
        length = (
            go_right * (length - half - wp.uint32(1)) + (wp.uint32(1) - go_right) * half
        )
        iters -= wp.uint32(1)
    return lo


@wp.func
def lower_bound(u: wp.array[wp.uint32], v: wp.array[wp.uint32], n: wp.uint32, keyu: wp.uint32, keyv: wp.uint32) -> wp.uint32:  # type: ignore
    """Branchless lower bound on uint32 pair arrays: returns the index of the first element >= (keyu, keyv) in (u,v)[0:n]."""
    lo = wp.uint32(0)
    iters = wp.uint32(32) - clz(n)  # type: ignore
    length = n
    while iters > wp.uint32(0):
        half = length >> wp.uint32(1)
        go_right = wp.uint32(
            u[lo + half] < keyu or (u[lo + half] == keyu and v[lo + half] < keyv)
        )
        lo += go_right * (half + wp.uint32(1))
        length = (
            go_right * (length - half - wp.uint32(1)) + (wp.uint32(1) - go_right) * half
        )
        iters -= wp.uint32(1)
    return lo


@wp.func
def lower_bound(u: wp.array[wp.uint32], v: wp.array[wp.uint32], n: wp.uint64, keyu: wp.uint32, keyv: wp.uint32) -> wp.uint64:  # type: ignore
    """Branchless lower bound on uint32 pair arrays with uint64 count: returns the index of the first element >= (keyu, keyv) in (u,v)[0:n]."""
    lo = wp.uint64(0)
    iters = wp.uint64(64) - clz(n)  # type: ignore
    length = n
    while iters > wp.uint64(0):
        half = length >> wp.uint64(1)
        go_right = wp.uint64(
            u[lo + half] < keyu or (u[lo + half] == keyu and v[lo + half] < keyv)
        )
        lo += go_right * (half + wp.uint64(1))
        length = (
            go_right * (length - half - wp.uint64(1)) + (wp.uint64(1) - go_right) * half
        )
        iters -= wp.uint64(1)
    return lo


class Stream:

    def __init__(self, stream: wp.Stream):
        self._stream = stream

    def __cuda_stream__(self):
        return (0, self._stream.cuda_stream)

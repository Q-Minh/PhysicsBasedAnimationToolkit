"""
Device-side wall-clock timestamps.

``globaltimer()`` reads the GPU's global nanosecond timer (PTX ``%globaltimer``) from inside a kernel.
Unlike CUDA events, it can be recorded any number of times within one captured graph, including
inside ``wp.capture_while`` bodies, which replay the same nodes on every iteration. Its resolution
is about 1 microsecond.
"""

import warp as wp


@wp.func_native(
    """
unsigned long long t;
asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
return (int64_t)t;
"""
)
def globaltimer() -> wp.int64:  # pyright: ignore[reportReturnType]
    """GPU global timer in nanoseconds (CUDA only)."""
    ...

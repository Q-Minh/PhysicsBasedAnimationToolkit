from typing import Any

import warp as wp


class DoubleBuffer:
    """A pair of wp.array buffers supporting ping-pong swap patterns.

    One buffer is designated *current* (output) and the other *alternate*
    (input).  Calling :meth:`swap` exchanges their roles so the previous
    current becomes the new alternate and vice-versa.

    For contexts that require CUDA graph capture compatibility, use
    :meth:`copy_to_alternate` instead of :meth:`swap`.  The copy submits a
    real ``wp.copy`` on the supplied stream so the operation is recorded by the
    graph; Python-level pointer swaps (:meth:`swap`) are **not** graph-safe.
    """

    def __init__(self, current: wp.array[Any], alternate: wp.array[Any]):
        """Create both buffers with the same arguments forwarded to ``wp.array``."""
        self._buffers = [current, alternate]
        self._current = 0

    @property
    def current(self) -> wp.array[Any]:
        """The active (output) buffer."""
        return self._buffers[self._current]

    @property
    def alternate(self) -> wp.array[Any]:
        """The inactive (input) buffer."""
        return self._buffers[1 - self._current]

    def swap(self):
        """Swap current and alternate buffers."""
        self._current = 1 - self._current

    def copy_to_alternate(self, stream: wp.Stream | None = None):
        """Async-copy current into alternate on *stream*.

        Unlike :meth:`swap`, this submits a ``wp.copy`` GPU operation so the
        transfer is recorded during CUDA graph capture.  The roles of current
        and alternate do **not** change; alternate simply receives a snapshot of
        current as of this call.

        Args:
            stream: Stream to submit the copy on.  Defaults to the Warp
                default stream (``wp.get_stream()``) when *None*.
        """
        wp.copy(dest=self.alternate, src=self.current, stream=stream)

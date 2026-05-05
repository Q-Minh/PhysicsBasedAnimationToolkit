from typing import Any

import warp as wp


class DoubleBuffer:
    """A pair of wp.array buffers supporting ping-pong swap patterns.

    One buffer is designated *current* (output) and the other *alternate*
    (input).  Calling :meth:`swap` exchanges their roles so the previous
    current becomes the new alternate and vice-versa.
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

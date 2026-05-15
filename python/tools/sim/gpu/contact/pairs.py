import warp as wp
import cupy as cp
import numpy as np


@wp.struct
class PairsData:
    """Data structure for contact pairs."""

    prefix: wp.array[
        wp.uint64
    ]  # (# u + 1,) contact prefix sums per u primitive, count is in last prefix element
    u: wp.array[wp.uint32]  # (capacity,) u indices
    v: wp.array[wp.uint32]  # (capacity,) v indices


class Pairs:
    """Data structure for contact pairs."""

    nu: int
    nv: int
    capacity: int
    data: PairsData  # type: ignore

    _prefix: cp.ndarray
    _u: cp.ndarray
    _v: cp.ndarray

    def __init__(self, nu: int, nv: int, capacity: int):
        self.nu = nu
        self.nv = nv
        self.capacity = capacity
        self._prefix = cp.zeros(nu + 1, dtype=np.uint64)
        self._u = cp.full(shape=(capacity,), fill_value=nu, dtype=np.uint32)
        self._v = cp.full(shape=(capacity,), fill_value=nv, dtype=np.uint32)
        self.data = PairsData()
        self.data.prefix = wp.array(
            ptr=self._prefix.data.ptr,
            dtype=wp.uint64,
            shape=(nu + 1,),
            ndim=1,
            copy=False,
        )
        self.data.u = wp.array(
            ptr=self._u.data.ptr,
            dtype=wp.uint32,
            shape=(capacity,),
            ndim=1,
            copy=False,
        )
        self.data.v = wp.array(
            ptr=self._v.data.ptr,
            dtype=wp.uint32,
            shape=(capacity,),
            ndim=1,
            copy=False,
        )

    @property
    def prefix(self) -> cp.ndarray:
        """Only exposed to interoperate well with cuda.compute which sometimes
        refuses to take wp.array, which can be non-contiguous"""
        return self._prefix

    @property
    def u(self) -> cp.ndarray:
        """Only exposed to interoperate well with cuda.compute which sometimes
        refuses to take wp.array, which can be non-contiguous"""
        return self._u

    @property
    def v(self) -> cp.ndarray:
        """Only exposed to interoperate well with cuda.compute which sometimes
        refuses to take wp.array, which can be non-contiguous"""
        return self._v

    def clear(self):
        # NOTE: We need to go through the warp interface for memset, because 
        # warp and CuPy use different main streams.
        self.data.prefix.fill_(wp.uint64(0))
        self.data.u.fill_(wp.uint32(self.nu))
        self.data.v.fill_(wp.uint32(self.nv))

    def uv(self):
        """Get the pairs (u,v) on CPU

        Preconditions:
        - wp.synchronize() must have been called prior.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (u, v) pairs
        """
        nuv = self._prefix[-1].get()
        u, v = self._u[:nuv].get(), self._v[:nuv].get()
        return u, v

    def size(self):
        """Get the number of contact pairs.

        Preconditions:
        - wp.synchronize() must have been called prior.

        Returns:
            int: The number of contact pairs.
        """
        return self._prefix[-1].get()

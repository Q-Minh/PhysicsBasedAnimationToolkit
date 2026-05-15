import warp as wp
import cupy as cp


@wp.struct
class PairsData:
    """Data structure for contact pairs."""

    prefix: wp.array[
        wp.int32
    ]  # (# u + 1,) contact prefix sums per u primitive, count is in last prefix element
    u: wp.array[wp.int32]  # (capacity,) u indices
    v: wp.array[wp.int32]  # (capacity,) v indices


class Pairs:
    """Data structure for contact pairs."""

    nu: int
    nv: int
    capacity: int
    data: PairsData  # type: ignore

    def __init__(self, nu: int, nv: int, capacity: int):
        self.nu = nu
        self.nv = nv
        self.capacity = capacity
        self.data = PairsData()
        self.data.prefix = wp.zeros((nu + 1,), dtype=wp.int32)
        self.data.u = wp.full(shape=(capacity,), value=nu, dtype=wp.int32)
        self.data.v = wp.full(shape=(capacity,), value=nv, dtype=wp.int32)

    def clear(self):
        self.data.prefix.fill_(wp.int32(0))
        self.data.u.fill_(self.nu)
        self.data.v.fill_(self.nv)

    def uv(self):
        """Get the pairs (u,v) on CPU

        Returns:
            Tuple[np.ndarray, np.ndarray]: (u, v) pairs
        """
        nuv = cp.asarray(self.data.prefix)[-1].get()
        u, v = self.data.u.numpy()[:nuv], self.data.v.numpy()[:nuv]
        return u, v

    def size(self):
        """Get the number of contact pairs."""
        return cp.asarray(self.data.prefix)[-1].get()

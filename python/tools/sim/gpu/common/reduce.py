from typing import Any
import cuda.compute
import cuda.compute.typing
import cupy as cp
import numpy as np
import warp as wp
from ..common import Stream


class Reduce:
    d_in: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT  # type: ignore
    d_out: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT  # type: ignore
    num_items: int
    op: cuda.compute.typing.Operator
    h_init: np.ndarray
    storage: cp.ndarray

    _reduce: Any

    def __init__(
        self,
        d_in: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT,
        d_out: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT,
        num_items: int,
        op: cuda.compute.typing.Operator = cuda.compute.OpKind.PLUS,
        h_init: np.ndarray = np.zeros(shape=(1,), dtype=np.float32),
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.num_items = num_items
        self.op = op
        self.h_init = h_init

        self._reduce = cuda.compute.make_reduce_into(
            d_in=self.d_in,
            d_out=self.d_out,
            num_items=self.num_items,
            op=self.op,
            h_init=self.h_init,
        )
        temp_storage_size = self._reduce(
            temp_storage=None,
            d_in=self.d_in,
            d_out=self.d_out,
            num_items=self.num_items,
            op=self.op,
            h_init=self.h_init,
        )
        self.storage = cp.empty(shape=(temp_storage_size,), dtype=np.uint8)

    def __call__(self, stream: wp.Stream):
        self._reduce(
            temp_storage=self.storage,
            d_in=self.d_in,
            d_out=self.d_out,
            num_items=self.num_items,
            op=self.op,
            h_init=self.h_init,
            stream=Stream(stream),
        )

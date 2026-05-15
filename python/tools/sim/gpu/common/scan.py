from typing import Any
import cuda.compute
import cuda.compute.typing
import cupy as cp
import numpy as np
import warp as wp
from ..common import Stream


class ExclusiveScan:
    d_in: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT  # type: ignore
    d_out: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT  # type: ignore
    op: cuda.compute.typing.Operator
    init_value: np.ndarray
    num_items: int
    storage: cp.ndarray

    _scanner: Any

    def __init__(
        self,
        d_in: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT,
        d_out: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT,
        num_items: int,
        init_value: np.ndarray = np.zeros(shape=1, dtype=np.int32),
        op: cuda.compute.typing.Operator = cuda.compute.OpKind.PLUS,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.op = op
        self.init_value = init_value
        self.num_items = num_items

        self._scanner = cuda.compute.make_exclusive_scan(
            d_in=self.d_in,
            d_out=self.d_out,
            op=self.op,
            init_value=self.init_value,
        )
        temp_storage_size = self._scanner(
            temp_storage=None,
            d_in=self.d_in,
            d_out=self.d_out,
            op=self.op,
            num_items=self.num_items,
            init_value=self.init_value,
        )
        self.storage = cp.empty(temp_storage_size, dtype=np.uint8)

    def __call__(self, stream: wp.Stream):
        self._scanner(
            temp_storage=self.storage,
            d_in=self.d_in,
            d_out=self.d_out,
            op=self.op,
            num_items=self.num_items,
            init_value=self.init_value,
            stream=Stream(stream),
        )


class InclusiveScan:
    d_in: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT  # type: ignore
    d_out: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT  # type: ignore
    op: cuda.compute.typing.Operator
    init_value: np.ndarray
    num_items: int
    storage: cp.ndarray

    _scanner: Any

    def __init__(
        self,
        d_in: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT,
        d_out: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT,
        num_items: int,
        init_value: np.ndarray = np.zeros(shape=1, dtype=np.int32),
        op: cuda.compute.typing.Operator = cuda.compute.OpKind.PLUS,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.op = op
        self.init_value = init_value
        self.num_items = num_items

        self._scanner = cuda.compute.make_inclusive_scan(
            d_in=self.d_in,
            d_out=self.d_out,
            op=self.op,
            init_value=self.init_value,
        )
        temp_storage_size = self._scanner(
            temp_storage=None,
            d_in=self.d_in,
            d_out=self.d_out,
            op=self.op,
            num_items=self.num_items,
            init_value=self.init_value,
        )
        self.storage = cp.empty(temp_storage_size, dtype=np.uint8)

    def __call__(self, stream: wp.Stream):
        self._scanner(
            temp_storage=self.storage,
            d_in=self.d_in,
            d_out=self.d_out,
            op=self.op,
            num_items=self.num_items,
            init_value=self.init_value,
            stream=Stream(stream),
        )

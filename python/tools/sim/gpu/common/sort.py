from typing import Any
import cuda.compute
import cupy as cp
import numpy as np
import warp as wp
from ..common import Stream


class Sort:
    d_in_keys: cp.ndarray
    d_in_values: cp.ndarray
    d_out_keys: cp.ndarray
    d_out_values: cp.ndarray
    storage: cp.ndarray
    order: cuda.compute.SortOrder
    n_items: int

    _sorter: Any

    def __init__(
        self,
        d_in_keys: cp.ndarray,
        d_in_values: cp.ndarray,
        d_out_keys: cp.ndarray,
        d_out_values: cp.ndarray,
        n_items: int,
        order: cuda.compute.SortOrder = cuda.compute.SortOrder.ASCENDING,
    ):
        self.d_in_keys = d_in_keys
        self.d_in_values = d_in_values
        self.d_out_keys = d_out_keys
        self.d_out_values = d_out_values
        self.order = order
        self.n_items = n_items

        self._sorter = cuda.compute.make_radix_sort(
            d_in_keys=d_in_keys,
            d_out_keys=d_out_keys,
            d_in_values=d_in_values,
            d_out_values=d_out_values,
            order=order,
        )

        temp_storage_size = self._sorter(
            temp_storage=None,
            d_in_keys=self.d_in_keys,
            d_out_keys=self.d_out_keys,
            d_in_values=self.d_in_values,
            d_out_values=self.d_out_values,
            num_items=self.n_items,
        )
        self.storage = cp.empty(temp_storage_size, dtype=np.uint8)

    def __call__(self, stream: wp.Stream, *args: Any, **kwds: Any):
        self._sorter(
            temp_storage=self.storage,
            d_in_keys=self.d_in_keys,
            d_out_keys=self.d_out_keys,
            d_in_values=self.d_in_values,
            d_out_values=self.d_out_values,
            num_items=self.n_items,
            stream=Stream(stream),
        )

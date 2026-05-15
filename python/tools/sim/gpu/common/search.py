from typing import Any
import cuda.compute
import cuda.compute.typing
import warp as wp
from ..common import Stream


class LowerBound:
    d_data: cuda.compute.typing.DeviceArrayLike
    num_items: int
    d_values: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT  # type: ignore
    num_values: int
    d_out: cuda.compute.typing.DeviceArrayLike
    comp: cuda.compute.typing.Operator | None

    _lower_bound: Any

    def __init__(
        self,
        d_data: cuda.compute.typing.DeviceArrayLike,
        num_items: int,
        d_values: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT,  # type: ignore
        num_values: int,
        d_out: cuda.compute.typing.DeviceArrayLike,
        comp: cuda.compute.typing.Operator | None = None,
    ):
        self.d_data = d_data
        self.num_items = num_items
        self.d_values = d_values
        self.num_values = num_values
        self.d_out = d_out
        self.comp = comp

        self._lower_bound = cuda.compute.make_lower_bound(
            d_data=self.d_data, d_values=self.d_values, d_out=self.d_out, comp=self.comp
        )

    def __call__(self, stream: wp.Stream):
        self._lower_bound(
            d_data=self.d_data,
            num_items=self.num_items,
            d_values=self.d_values,
            num_values=self.num_values,
            d_out=self.d_out,
            comp=self.comp,
            stream=Stream(stream),
        )


class UpperBound:
    d_data: cuda.compute.typing.DeviceArrayLike
    num_items: int
    d_values: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT  # type: ignore
    num_values: int
    d_out: cuda.compute.typing.DeviceArrayLike
    comp: cuda.compute.typing.Operator

    _upper_bound: Any

    def __init__(
        self,
        d_data: cuda.compute.typing.DeviceArrayLike,
        num_items: int,
        d_values: cuda.compute.typing.DeviceArrayLike | cuda.compute.typing.IteratorT,  # type: ignore
        num_values: int,
        d_out: cuda.compute.typing.DeviceArrayLike,
        comp: cuda.compute.typing.Operator = cuda.compute.OpKind.LESS,
    ):
        self.d_data = d_data
        self.num_items = num_items
        self.d_values = d_values
        self.num_values = num_values
        self.d_out = d_out
        self.comp = comp

        self._upper_bound = cuda.compute.make_upper_bound(
            d_data=self.d_data, d_values=self.d_values, d_out=self.d_out, comp=self.comp
        )

    def __call__(self, stream: wp.Stream):
        self._upper_bound(
            d_data=self.d_data,
            num_items=self.num_items,
            d_values=self.d_values,
            num_values=self.num_values,
            d_out=self.d_out,
            comp=self.comp,
            stream=Stream(stream),
        )

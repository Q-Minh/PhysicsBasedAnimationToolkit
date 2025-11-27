# type: ignore
import inspect
import polyscope as ps
import polyscope.imgui as imgui
import enum
import typing


def try_draw_tooltip(obj, name):
    if imgui.IsItemHovered():
        imgui.BeginTooltip()
        imgui.SetTooltip(getattr(type(obj), name).__doc__ or "")
        imgui.EndTooltip()


def draw_params(obj):
    for name, value in inspect.getmembers(obj):
        if (
            isinstance(getattr(type(obj), name, None), property)
            and getattr(type(obj), name).fset is None
        ):
            continue
        if isinstance(value, float):
            _, new_value = imgui.InputFloat(name, value, format="%.10f")
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, int):
            _, new_value = imgui.InputInt(name, value)
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, bool):
            _, new_value = imgui.Checkbox(name, value)
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, enum.Enum):
            enum_values = list(type(value))
            selected_idx = enum_values.index(value)
            _, selected_idx = imgui.Combo(
                name,
                selected_idx,
                [enum_value.name for enum_value in enum_values],
            )
            try_draw_tooltip(obj, name)
            setattr(obj, name, enum_values[selected_idx])


def get_nested_attr(obj, attr_path, default=None):
    attrs = attr_path.split(".")
    current_obj = obj
    for attr in attrs:
        try:
            current_obj = getattr(current_obj, attr)
        except AttributeError:
            return default  # Return default if any attribute in the path is missing
    return attrs, current_obj


class ParameterObject:
    params: typing.Any  # The parameter object
    sub_params: dict  # List of sub-parameters as attribute paths from self.params

    def __init__(self, params, sub_params: dict = {}):
        self.params = params
        self.sub_params = sub_params

    def draw(self):
        draw_params(self.params)
        self._draw_sub_params(self.params, self.sub_params)

    def _draw_sub_params(self, params: typing.Any, sub_params: dict):
        for sub_param_name, sub_param_entry in sub_params.items():
            sub_param = getattr(params, sub_param_name)
            if imgui.TreeNode(sub_param_name):
                imgui.PushID(sub_param_name)
                draw_params(sub_param)
                if isinstance(sub_param_entry, dict):
                    self._draw_sub_params(sub_param, sub_param_entry)
                imgui.PopID()
                imgui.TreePop()

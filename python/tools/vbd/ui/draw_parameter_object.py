# type: ignore
import inspect
import polyscope as ps
import polyscope.imgui as imgui
import enum


def draw(obj):
    for name, value in inspect.getmembers(obj):
        if not inspect.isdatadescriptor(value) or name.startswith("_"):
            continue
        if isinstance(value, float):
            _, new_value = imgui.InputFloat(name, value)
            setattr(obj, name, new_value)
        elif isinstance(value, int):
            _, new_value = imgui.InputInt(name, value)
            setattr(obj, name, new_value)
        elif isinstance(value, bool):
            _, new_value = imgui.Checkbox(name, value)
            setattr(obj, name, new_value)
        elif isinstance(value, enum.Enum):
            enum_values = list(type(value))
            selected_idx = enum_values.index(value)
            _, selected_idx = imgui.Combo(
                name,
                selected_idx,
                enum_values,
            )
            setattr(obj, name, enum_values[selected_idx])

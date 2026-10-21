"""Widget helpers shared by ui.py and the feature panels."""

import enum
import inspect
import numpy as np
import polyscope.imgui as imgui


def try_draw_tooltip(obj, name):
    if imgui.IsItemHovered():
        imgui.BeginTooltip()
        imgui.SetTooltip(getattr(type(obj), name, "").__doc__ or "")
        imgui.EndTooltip()


def draw_params(obj, sub_params: dict | None = None, _depth=0, exclude=frozenset(), only=None):
    """Draw widgets for the settable scalar and enum attributes of ``obj``, recursing into nested
    objects (or only into ``sub_params``, if given). Attributes named in ``exclude`` are skipped at
    every depth. If ``only`` is given, ``obj``'s own attributes outside it are skipped too."""
    for name, value in inspect.getmembers(obj):
        if name.startswith("_") or name in exclude or (only is not None and name not in only):
            continue
        if (
            isinstance(getattr(type(obj), name, None), property)
            and getattr(type(obj), name).fset is None
        ):
            continue
        if value is None or callable(value):
            continue
        if isinstance(value, float):
            # 8 significant digits, in scientific notation when small (e.g. rtol 1e-08), not 6 fixed decimals
            _, new_value = imgui.InputFloat(name, value, format="%.8g")
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, bool):
            _, new_value = imgui.Checkbox(name, value)
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, int):
            _, new_value = imgui.InputInt(name, value)
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, enum.Enum):
            enum_values = list(type(value))
            selected_idx = enum_values.index(value)
            _, selected_idx = imgui.Combo(
                name, selected_idx, [ev.name for ev in enum_values]
            )
            try_draw_tooltip(obj, name)
            setattr(obj, name, enum_values[selected_idx])
        elif sub_params is None and _depth < 4 and not isinstance(value, np.ndarray):
            if imgui.TreeNode(name):
                imgui.PushID(name)
                draw_params(value, None, _depth + 1, exclude)
                imgui.PopID()
                imgui.TreePop()
    if sub_params:
        for sub_name, nested_sub_params in sub_params.items():
            sub_value = getattr(obj, sub_name, None)
            if sub_value is None:
                continue
            if imgui.TreeNode(sub_name):
                imgui.PushID(sub_name)
                draw_params(sub_value, nested_sub_params or None, _depth + 1, exclude)
                imgui.PopID()
                imgui.TreePop()

"""
GUI panel for live per-phase GPU profiling (CUPTI-based ``kernels`` backend).

While enabled, steps run eagerly and every step's GPU activities are attributed to profiling scopes.
The panel shows the mean GPU busy time per step over a rolling window of steps, by phase (innermost
scope) or by full scope path, and can export the window in the result schema of
``profiling/results.py``, so ``profiling/report.py`` can render it.
"""

import collections
import contextlib
import tkinter as tk
from tkinter import filedialog

import polyscope.imgui as imgui

from ..gpu.common import profiling
from ..profiling import results


class _PanelState:
    def __init__(self):
        self.window = 20
        self.by_leaf = True
        self.steps: collections.deque = collections.deque(maxlen=self.window)
        self.error: str | None = None


_panel = _PanelState()


@contextlib.contextmanager
def profile_step(state):
    """Wrap ``state.step()`` to profile the step while the panel has profiling enabled."""
    if profiling.get_backend() != profiling.Backend.KERNELS:
        yield
        return
    profiling.begin_step()
    try:
        yield
    finally:
        profile = profiling.end_step()
    _panel.steps.append({"t": state.t - 1, **_step_dict(profile)})


def _step_dict(profile: profiling.StepProfile) -> dict:
    return {
        "busy_ms": profile.busy_ms,
        "launches": profile.launches,
        "calls": profile.calls,
        "union_ms": profile.union_ms,
        "step_union_ms": profile.step_union_ms,
        "step_span_ms": profile.step_span_ms,
    }


def _window_result(state) -> dict:
    return {
        "schema": results.SCHEMA,
        "source": "kernels",
        "metadata": results.collect_metadata(state, "(GUI)", warmup_steps=0),
        "steps": list(_panel.steps),
        "kernels": [],
    }


def _export(state):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(
        defaultextension=".json", filetypes=[("Profile", "*.json")], title="Export profile"
    )
    root.destroy()
    if path:
        results.save(_window_result(state), path)


def draw_profiling_panel(state, ui_state):
    if not imgui.TreeNode("Profiling"):
        return
    enabled = profiling.get_backend() == profiling.Backend.KERNELS
    changed, enabled = imgui.Checkbox("Profile GPU phases (steps run eagerly)", enabled)
    if changed:
        try:
            profiling.set_backend(profiling.Backend.KERNELS if enabled else profiling.Backend.OFF)
            _panel.error = None
        except RuntimeError as e:  # e.g. CUPTI missing
            profiling.set_backend(profiling.Backend.OFF)
            _panel.error = str(e)
    if _panel.error:
        imgui.TextColored((1.0, 0.4, 0.4, 1.0), _panel.error)
    changed, window = imgui.InputInt("Window (steps)", _panel.window)
    if changed and window > 0:
        _panel.window = window
        _panel.steps = collections.deque(_panel.steps, maxlen=window)
    if imgui.RadioButton("By phase", _panel.by_leaf):
        _panel.by_leaf = True
    imgui.SameLine()
    if imgui.RadioButton("By scope path", not _panel.by_leaf):
        _panel.by_leaf = False
    imgui.SameLine()
    if imgui.Button("Clear##profiling"):
        _panel.steps.clear()
    imgui.SameLine()
    if imgui.Button("Export...##profiling") and _panel.steps:
        _export(state)

    n = len(_panel.steps)
    if n == 0:
        imgui.TextDisabled("No profiled steps yet. Enable profiling and step the simulation.")
        imgui.TreePop()
        return
    window = {"steps": list(_panel.steps)}
    busy = results.mean_busy_ms_per_step(window, by_leaf=_panel.by_leaf)
    total = sum(busy.values())
    union = sum(s["step_union_ms"] for s in _panel.steps) / n
    imgui.Text(f"{n} steps: {total:.3f} ms/step GPU busy, {union:.3f} ms/step union over streams")

    launches: dict[str, float] = {}
    calls: dict[str, float] = {}
    for step in _panel.steps:
        for path, count in step["launches"].items():
            key = results.leaf(path) if _panel.by_leaf else (path or results.UNSCOPED)
            launches[key] = launches.get(key, 0.0) + count / n
        for path, count in step["calls"].items():
            key = results.leaf(path) if _panel.by_leaf else path
            calls[key] = calls.get(key, 0.0) + count / n

    imgui.Columns(5, "profiling_table", True)
    for header in ("Phase" if _panel.by_leaf else "Scope path", "ms/step", "Share", "Launches", "Calls"):
        imgui.Text(header)
        imgui.NextColumn()
    imgui.Separator()
    for key, ms in sorted(busy.items(), key=lambda kv: -kv[1]):
        label = key or results.UNSCOPED
        share = ms / total if total > 0.0 else 0.0
        imgui.Text(label)
        imgui.NextColumn()
        imgui.Text(f"{ms:.3f}")
        imgui.NextColumn()
        imgui.ProgressBar(share, (-1.0, 0.0))
        imgui.NextColumn()
        imgui.Text(f"{launches.get(label, 0.0):.0f}")
        imgui.NextColumn()
        imgui.Text(f"{calls.get(key, 0.0):.0f}" if key in calls else "")
        imgui.NextColumn()
    imgui.Columns(1)
    imgui.TreePop()

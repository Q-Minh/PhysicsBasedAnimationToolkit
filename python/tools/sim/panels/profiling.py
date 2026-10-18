"""
GUI panel for live per-phase GPU profiling (CUPTI-based ``kernels`` backend).

While enabled, steps run eagerly and every step's GPU activities are attributed to profiling scopes.
The panel shows the mean GPU busy time per step over a rolling window of steps, by phase (innermost
scope), by full scope path, or as a scope tree with total (inclusive) and self times, and can export the window in the result schema of
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
    """Persistent panel settings and the rolling window of profiled steps."""
    def __init__(self):
        self.window = 20
        self.by_leaf = True
        self.tree = False
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
    """The per-step fields of a StepProfile, as stored in the profiling result schema."""
    return {
        "busy_ms": profile.busy_ms,
        "launches": profile.launches,
        "calls": profile.calls,
        "union_ms": profile.union_ms,
        "union_incl_ms": profile.union_incl_ms,
        "step_union_ms": profile.step_union_ms,
        "step_span_ms": profile.step_span_ms,
    }


def _window_result(state) -> dict:
    """The profiled window as a profiling result (schema of profiling/results.py)."""
    return {
        "schema": results.SCHEMA,
        "source": "kernels",
        "metadata": results.collect_metadata(state, "(GUI)", warmup_steps=0),
        "steps": list(_panel.steps),
        "kernels": [],
    }


def _export(state):
    """Ask for a file name and save the profiled window as JSON."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(
        defaultextension=".json", filetypes=[("Profile", "*.json")], title="Export profile"
    )
    root.destroy()
    if path:
        results.save(_window_result(state), path)


def _tooltip(text: str):
    """Show ``text`` as a tooltip while the last drawn item is hovered."""
    if imgui.IsItemHovered():
        imgui.SetTooltip(text)


_FLAT_HEADERS = {
    "Phase": "Innermost scope name, summed over every place it occurs.\n"
    "(unscoped) is work launched outside any scope.",
    "Scope path": "Full scope path, so a phase appears once per caller.\n"
    "(unscoped) is work launched outside any scope.",
    "ms/step": "Mean GPU busy time per step of the phase's own work, excluding nested scopes.",
    "Share": "Fraction of the step's total GPU busy time.",
    "Launches": "Mean kernel launches, memsets and memcpys per step.",
    "Calls": "Mean number of times the scope was entered per step (e.g. VBD sweeps per step).",
}
_TREE_HEADERS = {
    "Scope": "Scope name, indented under the scope that encloses it. Hover a name for its full path.",
    "Total ms": "Mean GPU busy time per step, including all nested scopes.",
    "Self ms": "Mean GPU busy time per step of the scope's own work, excluding nested scopes.",
    "Share of step": "Total time as a fraction of the step's GPU busy time.",
    "Total union ms": "Like Total, but work running concurrently on several streams counts once\n"
    "(union of activity intervals).",
    "Calls": _FLAT_HEADERS["Calls"],
}


def draw_profiling_panel(state, ui_state):
    """Draw the Profiling tree node: backend toggle, window controls and the phase, path or tree
    table."""
    if not imgui.TreeNode("Profiling"):
        return
    enabled = profiling.get_backend() == profiling.Backend.KERNELS
    changed, enabled = imgui.Checkbox("Profile GPU phases (steps run eagerly)", enabled)
    _tooltip(
        "Time every GPU kernel, memset and memcpy of each step with CUPTI and attribute it to\n"
        "the profiling scope that launched it. While on, steps run eagerly (without the CUDA\n"
        "graph), so they are slower than normal; kernel times stay representative."
    )
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
    _tooltip(
        "Number of most recent profiled steps the table averages over. Older steps drop out\n"
        "as new ones arrive; changing it keeps the newest steps."
    )
    if changed and window > 0:
        _panel.window = window
        _panel.steps = collections.deque(_panel.steps, maxlen=window)
    if imgui.RadioButton("By phase", _panel.by_leaf and not _panel.tree):
        _panel.by_leaf, _panel.tree = True, False
    _tooltip("One row per innermost scope name (e.g. aaavbd.sweep), summed over every place it occurs.")
    imgui.SameLine()
    if imgui.RadioButton("By scope path", not _panel.by_leaf and not _panel.tree):
        _panel.by_leaf, _panel.tree = False, False
    _tooltip("One row per full scope path (e.g. solve/aaavbd.subproblem/aaavbd.sweep).")
    imgui.SameLine()
    if imgui.RadioButton("Tree", _panel.tree):
        _panel.tree = True
    _tooltip("Nested scopes as a tree, with total (including nested scopes) and self (own work) times.")
    imgui.SameLine()
    if imgui.Button("Clear##profiling"):
        _panel.steps.clear()
    _tooltip("Drop all profiled steps from the window.")
    imgui.SameLine()
    if imgui.Button("Export...##profiling") and _panel.steps:
        _export(state)
    _tooltip(
        "Save the profiled steps as JSON. python -m python.tools.sim.profiling.report turns it into an\n"
        "HTML report."
    )

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
    _tooltip(
        "GPU busy sums the duration of every activity, so work running concurrently on several streams\n"
        "counts once per stream. The union over streams counts overlapping work once."
    )
    if _panel.tree:
        _draw_tree(window)
        imgui.TreePop()
        return

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
        _tooltip(_FLAT_HEADERS[header])
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


def _draw_tree(window: dict):
    """Scope tree with total (inclusive) and self busy time per step, share of the step, total union and calls."""
    rows = results.scope_tree(window)
    step_total = sum(r["total_ms"] for r in rows if r["depth"] == 0) or 1.0
    imgui.Columns(6, "profiling_tree", True)
    for header in ("Scope", "Total ms", "Self ms", "Share of step", "Total union ms", "Calls"):
        imgui.Text(header)
        _tooltip(_TREE_HEADERS[header])
        imgui.NextColumn()
    imgui.Separator()
    for r in rows:
        imgui.Text("  " * r["depth"] + (r["name"] if r["path"] else results.UNSCOPED))
        _tooltip(r["path"] or "Work launched outside any profiling scope.")
        imgui.NextColumn()
        imgui.Text(f"{r['total_ms']:.3f}")
        imgui.NextColumn()
        imgui.Text(f"{r['self_ms']:.3f}")
        imgui.NextColumn()
        imgui.ProgressBar(r["total_ms"] / step_total, (-1.0, 0.0))
        imgui.NextColumn()
        imgui.Text("" if r["union_incl_ms"] is None else f"{r['union_incl_ms']:.3f}")
        imgui.NextColumn()
        imgui.Text(f"{r['calls']:.0f}" if r["calls"] else "")
        imgui.NextColumn()
    imgui.Columns(1)


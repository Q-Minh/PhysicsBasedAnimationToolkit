# type: ignore
"""GUI panel for static equilibrium solves (see ``static.py``).

Open the GUI, e.g. ``python -m python.tools.sim.ui --fem-elasto-dynamics dump.h5:fem [--sim-params
a.toml]``, then the "Static Equilibrium" node in the Simulation tab:

- **Static equilibrium**: switches the FEM to static mode (``fem.static``). Every simulation step then
  runs the selected solver on min_x Psi(x) - s fext^T x from the current positions, with no inertia and
  kinematic constraints frozen, so stepping watches the solve converge. Turning it off restarts the
  dynamics at rest from the current positions.
- **External load scale**: s, the factor on the external load in static mode (0 = no loads).
- **Load samples...**: loads an MCMC sample batch (.h5) and checks it against the mesh. Its warnings
  (unmixed chains, sampling gravity) are shown.
- **Sample / Apply sample**: sets the positions to one sample and remembers its error to the rest shape.
- **Error to rest shape**: e/e0, with e = |x - X|_M over free nodes and e0 the error of the applied sample.

Both settings survive a GUI reset (they're re-applied to the new FEM), and any change drops the
captured CUDA graph so the next step records a new one. For timed runs, use ``benchmark.py``.
"""

import tkinter as tk
from tkinter import filedialog

import numpy as np
import polyscope.imgui as imgui

from .. import static


class _StaticPanelState:
    """The panel's settings and loaded batch, kept on ``ui_state.static_panel``."""

    def __init__(self):
        self.enabled = False
        self.fext_scale = 0.0
        self.batch: static.SampleBatch | None = None
        self.sample = 0
        self.error_message = ""
        self.e0: float | None = None
        self._fem = None  # FEM the settings were last applied to
        self._fext0: np.ndarray | None = None

    def apply(self, state):
        """Apply static mode and the load scale to ``state.fem``, also after a reset replaced it."""
        fem = state.fem
        if fem is not self._fem:
            self._fem = fem
            self._fext0 = fem.data.fext.numpy().copy()
            self.e0 = None
            if self.enabled:
                self._set_static(state, True)
        elif fem.static != self.enabled:
            self._set_static(state, self.enabled)

    def _set_static(self, state, enabled: bool):
        fem = state.fem
        fem.static = enabled
        scale = self.fext_scale if enabled else 1.0
        fem.data.fext.assign((self._fext0 * np.float32(scale)).astype(np.float32))
        state.capture = None

    def relative_error(self, state) -> float | None:
        """e/e0 for the current positions, or None until a sample has been applied."""
        if self.e0 is None or self.e0 <= 0:
            return None
        return _error(state.fem) / self.e0


def _error(fem) -> float:
    """Mass-weighted error to the rest shape over free nodes, |x - X|_M."""
    x = fem.data.x.numpy().astype(np.float64)
    X = fem.data.X.numpy().astype(np.float64)
    m = fem.data.m.numpy().astype(np.float64)
    free = fem.data.dmask.numpy() == 0
    return float(np.sqrt(np.sum(m[free, None] * (x[free] - X[free]) ** 2)))


def _panel_state(ui_state) -> _StaticPanelState:
    panel = getattr(ui_state, "static_panel", None)
    if panel is None:
        panel = _StaticPanelState()
        ui_state.static_panel = panel
    return panel


def draw_static_panel(state, ui_state):
    """Draw the "Static Equilibrium" panel and apply its settings to ``state.fem``.

    Args:
        state (SimulationState): The simulation; its FEM's static flag and external load are set here.
        ui_state (UIState): GUI state; the panel keeps its own settings on it as ``static_panel``.
    """
    panel = _panel_state(ui_state)
    panel.apply(state)
    if not imgui.TreeNode("Static Equilibrium"):
        return
    changed, panel.enabled = imgui.Checkbox("Static equilibrium", panel.enabled)
    if imgui.IsItemHovered():
        imgui.SetTooltip(
            "Solve min Psi(x) - s fext^T x (no inertia) instead of time steps. Each step continues the solve."
        )
    if changed:
        panel.apply(state)
    if panel.enabled:
        changed, panel.fext_scale = imgui.InputFloat("External load scale", panel.fext_scale, format="%.3f")
        if changed:
            panel._set_static(state, True)

    if imgui.Button("Load samples..."):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Load MCMC sample batch",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            try:
                batch = static.load_sample_batch(path)
                static.check_sample_batch(batch, state.fem)
                panel.batch, panel.sample, panel.error_message = batch, 0, ""
            except (OSError, KeyError, ValueError) as e:
                panel.error_message = str(e)
    if panel.error_message:
        imgui.TextWrapped(panel.error_message)
    if panel.batch is not None:
        batch = panel.batch
        imgui.Text(f"{batch.n_samples} samples at E* = {batch.target_energy:g} J")
        for w in batch.warnings():
            imgui.TextWrapped(f"Warning: {w}")
        _, panel.sample = imgui.InputInt("Sample", panel.sample)
        panel.sample = max(0, min(batch.n_samples - 1, panel.sample))
        if imgui.Button("Apply sample"):
            state.fem.data.x.assign(batch.x[panel.sample])
            state.fem.data.v.zero_()
            panel.e0 = _error(state.fem)
    rel = panel.relative_error(state)
    if rel is not None:
        imgui.Text(f"Error to rest shape: e/e0 = {rel:.4e}")
    imgui.TreePop()

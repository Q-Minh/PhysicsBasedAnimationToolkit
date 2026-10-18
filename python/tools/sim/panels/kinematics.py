"""
GUI panel for kinematic constraints (see gpu/kinematics/constraints.py).

Lists the constraints loaded with the problem (from a Dump's kinematics group, its dmask, or a scene's
transform library and static colliders) with their kind, time window and parameters, and marks those
active at the current device time step. A checkbox per constraint enables or disables it on the device
(no recapture). Nodes can be shown per constraint, and all currently constrained nodes highlighted, as
Polyscope point clouds that follow the deforming mesh. Constraints are authored in the CPU editor, not here.
"""

import numpy as np
import polyscope as ps
import polyscope.imgui as imgui

# Constraint indices whose nodes are shown, and whether the constrained nodes are highlighted
_shown: set[int] = set()
_highlight_constrained = {"enabled": False}
_CONSTRAINED_CLOUD = "Constrained nodes"
_RADIUS = 0.004


def _cloud_name(k: int, name: str) -> str:
    return f"Kinematic constraint {k}: {name}"


def _update_cloud(name: str, points: np.ndarray, color: tuple[float, float, float]):
    if points.shape[0] == 0:
        _remove_cloud(name)
        return
    if ps.has_point_cloud(name) and ps.get_point_cloud(name).n_points() == points.shape[0]:
        ps.get_point_cloud(name).update_point_positions(points)
    else:
        if ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
        cloud = ps.register_point_cloud(name, points, radius=_RADIUS)
        cloud.set_color(color)


def _remove_cloud(name: str):
    if ps.has_point_cloud(name):
        ps.remove_point_cloud(name)


def draw_kinematics_panel(state, ui_state):
    """Draw the kinematic constraints tree node and update the node point clouds.

    Reads the device step and, when point clouds are shown, the positions (and dmask), so it syncs with the
    device once per frame.

    Args:
        state: The `state.SimulationState`.
        ui_state: The `ui.UIState` (unused).
    """
    kcs = state.kinematics
    constraints = state.kinematic_constraints
    x = None
    if len(_shown) > 0 or _highlight_constrained["enabled"]:
        x = state.fem.data.x.numpy()
    if imgui.TreeNode(f"Kinematic constraints ({len(constraints)})"):
        n = kcs.step
        imgui.Text(f"Step {n}, t = {n * kcs.h:.4f} s")
        changed, _highlight_constrained["enabled"] = imgui.Checkbox(
            "Highlight constrained nodes", _highlight_constrained["enabled"]
        )
        if changed and not _highlight_constrained["enabled"]:
            _remove_cloud(_CONSTRAINED_CLOUD)
        for k, c in enumerate(constraints):
            imgui.PushID(f"kinematic_constraint_{k}")
            n_begin, n_end = c.window_steps(kcs.h)
            end = c.begin + c.duration
            window = f"[{c.begin:g}, {end:g}) s" if np.isfinite(end) else f"[{c.begin:g}, inf) s"
            active = c.enabled and n_begin <= n < n_end
            changed, enabled = imgui.Checkbox("##enabled", c.enabled)
            if changed:
                kcs.set_enabled(k, enabled)
            if imgui.IsItemHovered():
                imgui.SetTooltip("Enable or disable this constraint")
            imgui.SameLine()
            label = f"{c.name} ({c.kind.name}) {window}{'  [active]' if active else ''}"
            if imgui.TreeNode(f"{label}###constraint"):
                imgui.Text(f"Nodes: {c.nodes.size}")
                if c.kind.name in ("Translate", "Rotate"):
                    d = c.direction
                    imgui.Text(
                        f"{'Direction' if c.kind.name == 'Translate' else 'Axis'}: "
                        f"({d[0]:.3g}, {d[1]:.3g}, {d[2]:.3g})"
                    )
                if c.kind.name in ("Rotate", "RadialTranslate"):
                    p = c.point
                    imgui.Text(
                        f"{'Pivot' if c.kind.name == 'Rotate' else 'Origin'}: "
                        f"({p[0]:.3g}, {p[1]:.3g}, {p[2]:.3g})"
                    )
                if c.kind.name == "Rotate":
                    imgui.Text(f"Angular velocity: {np.rad2deg(c.rate):.4g} deg/s")
                elif c.kind.name != "Fixed":
                    imgui.Text(f"Speed: {c.rate:.4g} units/s")
                shown = k in _shown
                changed, shown = imgui.Checkbox("Show nodes", shown)
                if changed:
                    if shown:
                        _shown.add(k)
                        x = state.fem.data.x.numpy()
                    else:
                        _shown.discard(k)
                        _remove_cloud(_cloud_name(k, c.name))
                imgui.TreePop()
            imgui.PopID()
        imgui.TreePop()
    # Keep the highlighted nodes on the deforming mesh. Checkboxes toggled during this frame may have
    # enabled drawing after the positions were (not) fetched above, so fetch them here if needed.
    if x is None and (len(_shown) > 0 or _highlight_constrained["enabled"]):
        x = state.fem.data.x.numpy()
    for k in _shown:
        c = constraints[k]
        _update_cloud(_cloud_name(k, c.name), x[c.nodes], (0.95, 0.55, 0.1))
    if _highlight_constrained["enabled"]:
        dmask = state.fem.data.dmask.numpy()
        _update_cloud(_CONSTRAINED_CLOUD, x[dmask != 0], (0.85, 0.15, 0.15))

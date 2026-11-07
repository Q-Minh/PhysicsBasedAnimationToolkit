"""
GUI panel for per-solve statuses (see status.py), for interactive tuning.

Two tree nodes, read top to bottom:

- **Last step**: per substep, how the solve ended and what it used, one fact per row. VBD family: result,
  outer iterations and sweeps against their budgets, and the final ratios of the stationarity test
  (``status_ratio.stationarity_test_ratio``, then its parts) and the penetration to their tolerances (at most 1
  passes). Newton: the outer test with the same two ratios
  (at the end and at the first test), counts of subproblems that converged, hit the CG cap or rejected a
  step, and, in a "Subproblems" node (closed by default), one subproblem at a time, picked with a slider: its
  status, Newton steps, CG iterations, line-search backtracks and the outer test after it. With substeps,
  a summary (substeps that passed, worst ratios) comes first, then one substep at a time, picked with a
  slider (by default the last one; a picked index is kept across steps).
- **Last N steps**: the same counts summarized over a rolling window (``StatusLog.summary``).

Visible text stays short; tooltips explain each row. ``SimulationState.step`` records the statuses with
data it already synchronizes on, so the panel reads host memory only.
"""

import numpy as np
import polyscope.imgui as imgui

from ..gpu.newton.solver import SubproblemStatus
from ..gpu.vbd.convergence import SolveEnd
from ..status_ratio import stationarity_test_ratio

_window = {"steps": 10}
_subproblem = {}  # substep -> the subproblem picked in its browser (else the last one)
_substep = {}  # "picked": the substep picked in the last step's browser (else the last one)
_GOOD = (0.4, 0.85, 0.4, 1.0)
_WARN = (1.0, 0.6, 0.2, 1.0)

_RESULT = {
    SolveEnd.CONVERGED: ("Converged", "The gradient test and feasibility were met within the outer-iteration budget."),
    SolveEnd.GRADIENT_UNMET: (
        "Budget: gradient unmet",
        "The outer iterations ran out with the gradient test unmet (the step is feasible).\n"
        "More sweeps or outer iterations, or a looser Solver.Convergence.rstol, would help.",
    ),
    SolveEnd.FEASIBILITY_UNMET: (
        "Budget: feasibility unmet",
        "The outer iterations ran out with the gradient test met but a contact penetrating its margin\n"
        "by more than ftol * max(|dmin|, 1e-6 m). Look at the contact penalty and detection.",
    ),
    SolveEnd.BOTH_UNMET: (
        "Budget: both unmet",
        "The outer iterations ran out with neither the gradient test nor feasibility met.",
    ),
    SolveEnd.RUNNING: ("Unfinished", "No result recorded for this solve."),
}

_SUBPROBLEM = {
    SubproblemStatus.CONVERGED: ("Converged", "The subproblem's augmented-Lagrangian gradient passed its test."),
    SubproblemStatus.STEPPED: (
        "Budget hit",
        "The subproblem took all its Newton steps (Solver.Newton n_subproblem_max_iters) without passing its test.",
    ),
    SubproblemStatus.NON_DESCENT: (
        "Non-descent",
        "The Newton direction wasn't a descent direction; the step was rejected and x kept.",
    ),
    SubproblemStatus.LINE_SEARCH_FAILED: (
        "Line-search failed",
        "No step length passed the Armijo test; x was restored to the last accepted iterate.",
    ),
}

_NORM_TIP = (
    "||.|| is the norm over free nodes, Solver.Convergence.node_norm: RMS (default), the mass-weighted RMS,\n"
    "or MAX, the largest node value.\n"
    "With Solver.Convergence.smoothing_passes k > 0 the test must also pass with r_i and phi_i smoothed\n"
    "(k passes of mass-weighted 1-ring averaging; the floor from a +-1 float32 spacing probe), and these\n"
    "rows show the stricter of the two tests."
)
_STATIONARITY_TIP = (
    "The stationarity test's ratio at the end of the solve: at most 1 exactly when the test passed.\n"
    "Stationarity test (Solver.Convergence.rstol or astol > 0): ||q|| <= 1 and (||q0|| <= 1 or\n"
    "||r|| <= floor_kappa * ||phi||), so this is max(||q||, min(||q0||, ||r|| / (floor_kappa * ||phi||))),\n"
    "from the rows below (Node norm, Without floor, Residual / floor). ||q|| alone can be below 1\n"
    "on a solve that failed the second condition. Absolute test (both 0, or static): |g| / gtol.\n"
    "Solves recorded without the other ratios, and GPU Newton's (no floor): ||q|| (Newton: ||q0||).\n" + _NORM_TIP
)
_RESIDUAL_TIP = (
    "||q||, the stationarity test's first condition (at most 1 meets it; the test's ratio is the row above).\n"
    "Stationarity test (Solver.Convergence.rstol or astol > 0): ||q|| <= 1 and (||q0|| <= 1 or\n"
    "||r|| <= floor_kappa * ||phi||). This row is ||q|| over free nodes of\n"
    "q_i = |r_i| / max(rstol * s_i / m_i, alpha_i, floor_local_kappa * phi_i): r_i node i's residual\n"
    "acceleration, s_i the largest of its forces (inertial, elastic, contact, external) in the step's Lagrangian\n"
    "gradient, alpha_i = max(astol * |f_ext,i| / m_i, 1e-6 m/s^2) its own absolute allowance, phi_i its\n"
    "float32 floor |H_i|_F eps32 |x_i| / (m_i h^2).\n"
    "Absolute test (both 0, or static): |g| / gtol.\n" + _NORM_TIP
)
_MAX_TIP = (
    "The largest node ratio max_i q_i (information only; single nodes at their float32 floor exceed 1).\n"
    "0: absolute test."
)
_NO_FLOOR_TIP = (
    "||q0||, the node ratios without the float32 floor, q0_i = |r_i| / max(rstol * s_i / m_i, alpha_i).\n"
    "At most 1 passes without the floor clause. 0: absolute test.\n" + _NORM_TIP
)
_FLOOR_CLAUSE_TIP = (
    "||r|| / (floor_kappa * ||phi||): the floor clause, needed when Without floor / tol is above 1,\n"
    "passes at most 1 (the residual near the float32 floor overall). inf: no floor (GPU Newton, floor_kappa 0).\n"
    "0: absolute test.\n" + _NORM_TIP
)
_PENETRATION_TIP = (
    "Largest penetration into the contact margin over its tolerance ftol * max(|dmin|, 1e-6 m),\n"
    "at the end of the solve; at most 1 is feasible."
)
_NEWTON_RESIDUAL_TIP = (
    "The outer test's stationarity ratio, {when}; at most 1 exactly when the gradient test passes.\n"
    "Stationarity test (Solver.Convergence.rstol or astol > 0, dynamic problems), as the VBD family's without\n"
    "the float32 floor, so its ratio is ||q0|| over free nodes of q0_i = |r_i| / max(rstol * s_i / m_i, alpha_i),\n"
    "r_i node i's residual acceleration, s_i the largest of its forces in the step's Lagrangian gradient,\n"
    "alpha_i = max(astol * |f_ext,i| / m_i, 1e-6 m/s^2) its own absolute allowance (no float32 floor).\n"
    "Absolute test (both 0, or static problems): |g| / sqrt(gtol2), g the augmented-Lagrangian gradient.\n"
    "||.|| is the norm over free nodes, Solver.Convergence.node_norm: RMS (default), the mass-weighted RMS,\n"
    "or MAX, the largest node value.\n"
    "With Solver.Convergence.smoothing_passes k > 0 the test must also pass with r_i smoothed (k passes of\n"
    "mass-weighted 1-ring averaging), and this shows the stricter of the two tests."
)
_NEWTON_PENETRATION_TIP = (
    "Largest penetration into the contact margin over its tolerance ftol * max(|dmin|, 1e-6 m),\n"
    "{when}; at most 1 is feasible. Part of the stationarity outer test only (the absolute test, as C++'s,\n"
    "checks the gradient alone)."
)


def _tooltip(text: str):
    if text and imgui.IsItemHovered():
        imgui.SetTooltip(text)


class _Rows:
    """Label/value rows in two aligned columns. Each row's tooltip shows on both cells."""

    def __init__(self, id: str):
        self._id = id

    def __enter__(self):
        imgui.Columns(2, self._id, False)
        return self

    def __exit__(self, *exc):
        imgui.Columns(1)
        return False

    def row(self, label: str, value: str, tooltip: str = "", color=None):
        imgui.Text(label)
        _tooltip(tooltip)
        imgui.NextColumn()
        if color is None:
            imgui.Text(value)
        else:
            imgui.TextColored(color, value)
        _tooltip(tooltip)
        imgui.NextColumn()


def _ratio(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.3g}"


def _ratio_rows(rows: _Rows, prefix: str, gradient: float, feasibility: float, when: str):
    """Stationarity and penetration rows of one Newton outer test, ``when`` naming it in the tooltips."""
    rows.row(
        f"{prefix}Stationarity / tol",
        _ratio(gradient),
        _NEWTON_RESIDUAL_TIP.format(when=when),
        _WARN if gradient > 1 else None,
    )
    rows.row(
        f"{prefix}Penetration / tol",
        _ratio(feasibility),
        _NEWTON_PENETRATION_TIP.format(when=when),
        _WARN if feasibility > 1 else None,
    )


def _draw_vbd_solve(r, rows: _Rows, norm: str = "RMS"):
    reason = SolveEnd(int(r["reason"]))
    label, tip = _RESULT[reason]
    rows.row("Result", label, tip, _GOOD if reason == SolveEnd.CONVERGED else _WARN)
    rows.row(
        "Outer iterations",
        f"{int(r['outer'])} / {int(r['max_outer'])}",
        "Augmented-Lagrangian outer iterations used / budget (Solver.<solver>.n_max_iters).",
    )
    rows.row(
        "Sweeps",
        f"{int(r['sweeps'])} / {int(r['max_sweeps'])}",
        "VBD sweeps used / budget (outer iterations x sweeps per subproblem).\n"
        "A subproblem stops early once its gradient test passes (tested every check_interval sweeps).",
    )
    ratio = float(stationarity_test_ratio(r))
    passed = reason in (SolveEnd.CONVERGED, SolveEnd.FEASIBILITY_UNMET)
    rows.row("Stationarity / tol", _ratio(ratio), _STATIONARITY_TIP, None if ratio <= 1 or passed else _WARN)
    if "max_ratio" in r.dtype.names:
        rows.row(f"{norm} node / tol", _ratio(float(r["gradient_ratio"])), _RESIDUAL_TIP)
        rows.row("Largest node / tol", _ratio(float(r["max_ratio"])), _MAX_TIP)
        rows.row("Without floor / tol", _ratio(float(r["no_floor_ratio"])), _NO_FLOOR_TIP)
        rows.row("Residual / floor", _ratio(float(r["floor_clause_ratio"])), _FLOOR_CLAUSE_TIP)
    ratio = float(r["feasibility_ratio"])
    rows.row("Penetration / tol", _ratio(ratio), _PENETRATION_TIP, None if ratio <= 1 else _WARN)


def _draw_newton_solve(s, subproblems, rows: _Rows):
    passed = bool(s["converged"])
    rows.row(
        "Outer test",
        "Passed" if passed else "Not passed",
        "The Lagrangian gradient test and feasibility, tested before each subproblem and after the last one.",
        _GOOD if passed else _WARN,
    )
    _ratio_rows(rows, "", float(s["gradient_ratio"]), float(s["feasibility_ratio"]), "at the end of the solve")
    rows.row(
        "Outer iterations",
        f"{int(s['outer'])} / {int(s['max_outer'])}",
        "Augmented-Lagrangian outer iterations run / budget (Solver.Newton.n_max_iters).",
    )
    if subproblems.shape[0]:
        _ratio_rows(
            rows,
            "Initial ",
            float(s["initial_gradient_ratio"]),
            float(s["initial_feasibility_ratio"]),
            "at the first outer test (before subproblem 0)",
        )
        n = subproblems.shape[0]
        statuses = [SubproblemStatus(int(v)) for v in subproblems["status"]]
        converged = sum(v == SubproblemStatus.CONVERGED for v in statuses)
        rejected = sum(v in (SubproblemStatus.NON_DESCENT, SubproblemStatus.LINE_SEARCH_FAILED) for v in statuses)
        capped = int(np.sum(subproblems["cg_capped"]))
        rows.row("Subproblems", f"{n}", "Newton subproblems solved (one per outer iteration); browse them below.")
        rows.row("  Converged", f"{converged} / {n}", "Subproblems whose augmented-Lagrangian gradient passed its test.")
        rows.row(
            "  CG cap hit",
            f"{capped} / {n}",
            "Subproblems with a Newton step whose CG stopped at its cap (an inexact direction).",
            _WARN if capped else None,
        )
        rows.row(
            "  Rejected step",
            f"{rejected} / {n}",
            "Subproblems that ended on a rejected step (non-descent direction or failed line search).",
            _WARN if rejected else None,
        )


def _draw_subproblem(p, rows: _Rows):
    k = int(p["outer"])
    status = SubproblemStatus(int(p["status"]))
    label, tip = _SUBPROBLEM[status]
    ok = status in (SubproblemStatus.CONVERGED, SubproblemStatus.STEPPED)
    rows.row(f"Subproblem {k}", label, tip, None if ok else _WARN)
    rows.row(
        "  Newton steps",
        f"{int(p['newton_steps'])} / {int(p['max_newton_steps'])}",
        "Newton directions computed (one CG solve each) / budget (n_subproblem_max_iters).",
    )
    capped = bool(p["cg_capped"])
    rows.row(
        "  CG iterations (max)",
        f"{int(p['max_cg_iters'])} / {int(p['cg_cap'])}" + ("  cap hit" if capped else ""),
        "Most CG iterations of one Newton step / cap (Solver.Newton.LinearSolver.n_max_iters).\n"
        "At the cap, the Newton direction is inexact.",
        _WARN if capped else None,
    )
    rows.row(
        "  Line-search backtracks",
        f"{int(p['backtracks'])}",
        "Step-size reductions over all of the subproblem's line searches (0: every full step was accepted).",
    )
    _ratio_rows(
        rows,
        "  ",
        float(p["gradient_ratio"]),
        float(p["feasibility_ratio"]),
        f"at the outer test after subproblem {k} (after its contact detection and dual update)",
    )


def _draw_subproblem_browser(subproblems, key: int):
    """A closed-by-default "Subproblems" node showing one subproblem at a time, picked with a slider (by
    default the last one; a picked index is kept across steps)."""
    n = subproblems.shape[0]
    if n == 0 or not imgui.TreeNode(f"Subproblems ({n})###subproblems_{key}"):
        return
    k = min(_subproblem.get(key, n - 1), n - 1)
    if n > 1:
        changed, k = imgui.SliderInt(f"Subproblem##{key}", k, 0, n - 1)
        _tooltip("Which subproblem (outer iteration) to show.")
        if changed:
            _subproblem[key] = k
    with _Rows(f"subproblem_{key}") as rows:
        _draw_subproblem(subproblems[k], rows)
    imgui.TreePop()


def _draw_substeps_summary(vbd, solves, n: int):
    """The time step's substeps at a glance: how many passed their tests and the worst final ratios."""
    passed = int(np.sum(vbd["reason"] == int(SolveEnd.CONVERGED))) + int(np.sum(solves["converged"]))
    # The stationarity test's ratio (GPU Newton's gradient ratio is its own, without a floor)
    gradient = max(np.max(stationarity_test_ratio(vbd), initial=0.0), np.max(solves["gradient_ratio"], initial=0.0))
    feasibility = max(
        np.max(vbd["feasibility_ratio"], initial=0.0), np.max(solves["feasibility_ratio"], initial=0.0)
    )
    with _Rows("substeps") as rows:
        rows.row("Substeps", f"{n}", "Solves of this time step, one per substep; browse them below.")
        rows.row(
            "  Passed",
            f"{passed} / {n}",
            "Substeps whose solve passed its tests (gradient and feasibility) within the budget.",
            _GOOD if passed == n else _WARN,
        )
        rows.row(
            "  Worst stationarity / tol", _ratio(float(gradient)), _STATIONARITY_TIP, None if gradient <= 1 else _WARN
        )
        rows.row(
            "  Worst penetration / tol",
            _ratio(float(feasibility)),
            _PENETRATION_TIP,
            None if feasibility <= 1 else _WARN,
        )


def _draw_last_step(log, norm: str = "RMS"):
    last = log.last_step
    vbd = log.vbd
    vbd = vbd[vbd["step"] == last]
    solves = log.newton_solves
    solves = solves[solves["step"] == last]
    subproblems = log.newton_subproblems
    subproblems = subproblems[subproblems["step"] == last]
    n_substeps = max(vbd.shape[0], solves.shape[0])
    imgui.Text(f"Time step {last}")
    shown = range(n_substeps)
    if n_substeps > 1:
        _draw_substeps_summary(vbd, solves, n_substeps)
        k = min(_substep.get("picked", n_substeps - 1), n_substeps - 1)
        changed, k = imgui.SliderInt("Substep##last_step", k, 0, n_substeps - 1)
        _tooltip("Which substep of the time step to show.")
        if changed:
            _substep["picked"] = k
        shown = (k,)
    for sub in shown:
        newton = subproblems[subproblems["substep"] == sub]
        with _Rows(f"last_step_{sub}") as rows:
            for r in vbd[vbd["substep"] == sub]:
                _draw_vbd_solve(r, rows, norm)
            for s in solves[solves["substep"] == sub]:
                _draw_newton_solve(s, newton, rows)
        if solves[solves["substep"] == sub].shape[0]:
            _draw_subproblem_browser(newton, sub)


def _draw_window(log):
    _, _window["steps"] = imgui.SliderInt("Window (steps)", _window["steps"], 1, 1000)
    _tooltip("How many of the most recent time steps the summary below covers.")
    summary = log.summary(_window["steps"])
    imgui.Text(f"Time steps: {summary['steps']}")
    if "vbd" in summary:
        v = summary["vbd"]
        r = v["reasons"]
        budget = r["GRADIENT_UNMET"] + r["FEASIBILITY_UNMET"] + r["BOTH_UNMET"]
        imgui.Separator()
        imgui.Text("VBD family")
        with _Rows("window_vbd") as rows:
            rows.row("Solves", f"{v['solves']}", "Solves in the window (one per time step or substep).")
            rows.row(
                "Converged",
                f"{r['CONVERGED']}",
                "Solves that met the gradient test and feasibility within the budget.",
                _GOOD if r["CONVERGED"] else None,
            )
            rows.row(
                "Budget hit",
                f"{budget}",
                f"Solves whose outer iterations ran out:\n"
                f"  gradient unmet: {r['GRADIENT_UNMET']}\n"
                f"  feasibility unmet: {r['FEASIBILITY_UNMET']}\n"
                f"  both unmet: {r['BOTH_UNMET']}",
                _WARN if budget else None,
            )
            rows.row("Outer iterations (mean)", f"{v['outer_mean']:.2f}", "Mean outer iterations per solve.")
            rows.row(
                "Outer iterations (max)",
                f"{v['outer_max']} / {v['outer_budget']}",
                "Most outer iterations of one solve / budget.",
            )
            rows.row("Sweeps (mean)", f"{v['sweeps_mean']:.1f}", "Mean VBD sweeps per solve.")
            rows.row(
                "Sweeps (max)",
                f"{v['sweeps_max']} / {v['sweeps_budget']}",
                "Most VBD sweeps of one solve / budget.",
            )
            ratio = v["test_ratio_max"]
            rows.row("Worst stationarity / tol", _ratio(ratio), _STATIONARITY_TIP, None if ratio <= 1 else _WARN)
            ratio = v["feasibility_ratio_max"]
            rows.row("Worst penetration / tol", _ratio(ratio), _PENETRATION_TIP, None if ratio <= 1 else _WARN)
            if v["lost"]:
                rows.row(
                    "Lost solves",
                    f"{v['lost']}",
                    "Solves whose status rows were overwritten before the host read them (all steps so far).",
                    _WARN,
                )
    if "newton" in summary:
        n = summary["newton"]
        st = n["statuses"]
        imgui.Separator()
        imgui.Text("Newton")
        with _Rows("window_newton") as rows:
            rows.row("Solves", f"{n['solves']}", "Solves in the window (one per time step or substep).")
            rows.row(
                "Outer test passed",
                f"{n['converged']}",
                "Solves whose outer test (Lagrangian gradient and feasibility) passed within the budget.",
            )
            rows.row(
                "Outer iterations (max)",
                f"{n['outer_max']} / {n['outer_budget']}",
                f"Most outer iterations of one solve / budget (mean {n['outer_mean']:.2f}).",
            )
            rows.row("Subproblems", f"{n['subproblems']}", "Subproblems solved in the window.")
            for status in _SUBPROBLEM:  # converged, budget hit, non-descent, line-search failed
                label, tip = _SUBPROBLEM[status]
                count = st[status.name]
                bad = status in (SubproblemStatus.NON_DESCENT, SubproblemStatus.LINE_SEARCH_FAILED)
                rows.row(f"  {label}", f"{count}", tip, _WARN if bad and count else None)
            rows.row(
                "Newton steps (mean)",
                f"{n['newton_steps_mean']:.2f}",
                "Mean Newton steps per subproblem.",
            )
            rows.row(
                "Newton steps (max)",
                f"{n['newton_steps_max']} / {n['newton_steps_budget']}",
                "Most Newton steps of one subproblem / budget.",
            )
            rows.row(
                "CG iterations (max)",
                f"{n['cg_iters_max']} / {n['cg_cap']}",
                "Most CG iterations of one Newton step / cap.",
            )
            rows.row(
                "CG cap hits",
                f"{n['cg_capped']}",
                "Subproblems where a CG solve stopped at its iteration cap (inexact Newton direction).",
                _WARN if n["cg_capped"] else None,
            )
            rows.row(
                "Line-search backtracks",
                f"{n['backtracks']}",
                "Step-size reductions over all line searches in the window.",
            )
            _ratio_rows(
                rows,
                "Worst ",
                n["test_ratio_max"],
                n["feasibility_ratio_max"],
                "at the end of the window's worst solve",
            )


def draw_status_panel(state, ui_state):
    """Draw the "Solve status" tree node from ``state.status_log`` (host memory only).

    Args:
        state: The `state.SimulationState`.
        ui_state: The `ui.UIState` (unused).
    """
    if not imgui.TreeNode("Solve status"):
        return
    imgui.PushID("solve_status")
    check = getattr(state, "convergence", None)
    # The node norm of the stationarity test (labels the ratio rows; rows recorded under another norm aren't marked)
    norm = check.params.node_norm.name if check is not None else "RMS"
    if check is not None:
        astol = float(check.params.astol)
        imgui.Text(f"Absolute allowance: {astol:.3g} |f_ext,i| / m_i per node")
        _tooltip(
            "Each free node's absolute allowance alpha_i = max(astol * |f_ext,i| / m_i, 1e-6 m/s^2), astol times\n"
            "its own external acceleration: astol * |g| on every node under a uniform gravity g, only the floor on\n"
            "a node without a load. It's the residual acceleration the stationarity test tolerates where a node's\n"
            "forces are small (dynamic problems), independent of the time step. A step left at x^t under its load\n"
            "has q_i = 1 / astol at every loaded node, so it can't pass."
        )
    log = state.status_log
    if log.last_step is None:
        imgui.Text("No solves recorded yet")
    else:
        if imgui.TreeNodeEx("Last step", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            _draw_last_step(log, norm)
            imgui.TreePop()
        if imgui.TreeNode(f"Last {_window['steps']} steps###window"):  # closed by default (maintainer)
            _draw_window(log)
            imgui.TreePop()
    imgui.PopID()
    imgui.TreePop()


import unittest


class TestStatusPanel(unittest.TestCase):
    def test_draw_balanced(self):
        from types import SimpleNamespace
        from unittest import mock

        import numpy as np

        from ..gpu.newton.solver import NewtonSolveRecord, NewtonSubproblemRecord
        from ..status import StatusLog

        rows = np.zeros(
            2,
            dtype=[("solve", np.int32), ("outer", np.int32), ("sweeps", np.int32), ("reason", np.int32),
                   ("gradient_ratio", np.float32), ("feasibility_ratio", np.float32)],
        )
        rows["solve"] = [0, 1]
        rows["reason"] = [int(SolveEnd.CONVERGED), int(SolveEnd.BOTH_UNMET)]
        rows["gradient_ratio"] = [0.5, 3.0]
        log = StatusLog()
        log.add_vbd(0, 0, rows, 0, 5, 500)
        sub = NewtonSubproblemRecord(SubproblemStatus.LINE_SEARCH_FAILED, 2, 500, True, 9)
        log.add_newton(1, [NewtonSolveRecord(1, False, (sub,), ((40.0, float("nan")), (2.5, 0.5)))], 5, 10, 500)
        fake = mock.MagicMock()
        fake.TreeNode.return_value = True
        fake.TreeNodeEx.return_value = True
        fake.IsItemHovered.return_value = True
        fake.SliderInt.side_effect = lambda label, v, lo, hi: (False, v)
        state = SimpleNamespace(status_log=log)
        with mock.patch(f"{__name__}.imgui", fake):
            draw_status_panel(state, None)  # last step: Newton
            log.add_vbd(2, 0, rows, 0, 5, 500)
            draw_status_panel(state, None)  # last step: VBD, two substeps
            draw_status_panel(SimpleNamespace(status_log=StatusLog()), None)  # empty
        self.assertEqual(fake.PushID.call_count, fake.PopID.call_count)
        self.assertEqual(fake.TreeNode.call_count + fake.TreeNodeEx.call_count, fake.TreePop.call_count)
        columns = [c.args[0] for c in fake.Columns.call_args_list]
        self.assertEqual(columns.count(2), columns.count(1))
        # Column ids are unique within one draw (here: the last of the three draws that drew rows)
        ids = [c.args[1] for c in fake.Columns.call_args_list if c.args[0] == 2]
        self.assertEqual(len(ids[-4:]), len(set(ids[-4:])))
        shown = [str(c.args[-1]) for c in fake.TextColored.call_args_list] + [
            str(c.args[0]) for c in fake.Text.call_args_list
        ]
        self.assertIn("Line-search failed", shown)
        self.assertIn("Budget: both unmet", shown)
        self.assertIn("Converged", shown)
        self.assertIn("0 / 5", shown)
        # Newton's outer-test ratios: initial, after the subproblem and at the end
        self.assertIn("Initial Stationarity / tol", shown)
        self.assertIn("40", shown)
        self.assertIn("n/a", shown)
        # The solve's result and after subproblem 0 (first draw), the window's worst (first two draws)
        self.assertEqual(shown.count("2.5"), 4)
        # No comma-delimited lists in the visible values
        self.assertFalse(any("," in t for t in shown if not t.startswith("Time steps")))
        tips = " ".join(str(c.args[0]) for c in fake.SetTooltip.call_args_list)
        self.assertIn("ftol * max(|dmin|", tips)
        self.assertNotIn("%", tips)

    def test_stationarity_ratio(self):
        """The solve's and the window's ratio is the stationarity test's: a failed floor clause shows above 1,
        warned, even with RMS_M(q) below 1."""
        from types import SimpleNamespace
        from unittest import mock

        from ..status import StatusLog

        rows = np.zeros(
            1,
            dtype=[("solve", np.int32), ("outer", np.int32), ("sweeps", np.int32), ("reason", np.int32),
                   ("gradient_ratio", np.float32), ("feasibility_ratio", np.float32), ("max_ratio", np.float32),
                   ("no_floor_ratio", np.float32), ("floor_clause_ratio", np.float32)],
        )
        rows["reason"] = int(SolveEnd.GRADIENT_UNMET)
        rows["gradient_ratio"], rows["max_ratio"] = 0.25, 3.0
        rows["no_floor_ratio"], rows["floor_clause_ratio"] = 1.5, 4.5
        log = StatusLog()
        log.add_vbd(0, 0, rows, 0, 5, 500)
        fake = mock.MagicMock()
        fake.TreeNode.return_value = True
        fake.TreeNodeEx.return_value = True
        fake.SliderInt.side_effect = lambda label, v, lo, hi: (False, v)
        with mock.patch(f"{__name__}.imgui", fake):
            draw_status_panel(SimpleNamespace(status_log=log), None)
        labels = [str(c.args[0]) for c in fake.Text.call_args_list]
        warned = [str(c.args[-1]) for c in fake.TextColored.call_args_list]
        self.assertIn("Stationarity / tol", labels)
        self.assertIn("Worst stationarity / tol", labels)
        self.assertEqual(warned.count("1.5"), 2)  # the solve's and the window's worst, both warned
        self.assertIn("0.25", labels)  # RMS node / tol, information
        self.assertIn("RMS node / tol", labels)
        # With the max norm the node row is labeled so, and the decisive ratio is the same helper's
        from ..gpu.vbd.convergence import NodeNorm

        check = SimpleNamespace(params=SimpleNamespace(node_norm=NodeNorm.MAX, astol=0.1))
        fake.reset_mock()
        with mock.patch(f"{__name__}.imgui", fake):
            draw_status_panel(SimpleNamespace(status_log=log, convergence=check), None)
        labels = [str(c.args[0]) for c in fake.Text.call_args_list]
        self.assertIn("MAX node / tol", labels)
        self.assertNotIn("RMS node / tol", labels)
        self.assertIn("Absolute allowance: 0.1 |f_ext,i| / m_i per node", labels)
        tips = " ".join(str(c.args[0]) for c in fake.SetTooltip.call_args_list)
        self.assertIn("node_norm", tips)

    def test_subproblem_browser(self):
        """Newton's summary counts its subproblems; the browser shows one at a time, the last by default."""
        from types import SimpleNamespace
        from unittest import mock

        from ..gpu.newton.solver import NewtonSolveRecord, NewtonSubproblemRecord
        from ..status import StatusLog

        subs = (
            NewtonSubproblemRecord(SubproblemStatus.STEPPED, 2, 500, True, 0),
            NewtonSubproblemRecord(SubproblemStatus.CONVERGED, 1, 300, False, 1),
            NewtonSubproblemRecord(SubproblemStatus.LINE_SEARCH_FAILED, 2, 500, False, 9),
        )
        log = StatusLog()
        log.add_newton(0, [NewtonSolveRecord(3, False, subs, ((9.0, 0.1), (5.0, 0.1), (3.0, 0.1), (7.5, 0.1)))], 5, 2, 500)
        fake = mock.MagicMock()
        fake.TreeNode.return_value = True
        fake.TreeNodeEx.return_value = True
        fake.SliderInt.side_effect = lambda label, v, lo, hi: (False, v)
        _subproblem.clear()
        with mock.patch(f"{__name__}.imgui", fake):
            draw_status_panel(SimpleNamespace(status_log=log), None)
        shown = [str(c.args[-1]) for c in fake.TextColored.call_args_list] + [
            str(c.args[0]) for c in fake.Text.call_args_list
        ]
        for value in ("1 / 3", "Line-search failed", "Subproblem 2"):  # converged, cap hit, rejected: 1 / 3 each
            self.assertIn(value, shown)
        self.assertEqual(shown.count("1 / 3"), 3)
        self.assertNotIn("Subproblem 0", shown)  # only the picked (last) subproblem
        (slider,) = [c for c in fake.SliderInt.call_args_list if c.args[0].startswith("Subproblem##")]
        self.assertEqual(slider.args[1:], (2, 0, 2))

    def test_substep_browser(self):
        """With substeps, a summary of all of them, then one substep at a time, the last by default."""
        from types import SimpleNamespace
        from unittest import mock

        from ..status import StatusLog

        rows = np.zeros(
            3,
            dtype=[("solve", np.int32), ("outer", np.int32), ("sweeps", np.int32), ("reason", np.int32),
                   ("gradient_ratio", np.float32), ("feasibility_ratio", np.float32)],
        )
        rows["solve"] = [0, 1, 2]
        rows["sweeps"] = [11, 22, 33]
        rows["reason"] = [int(SolveEnd.CONVERGED), int(SolveEnd.BOTH_UNMET), int(SolveEnd.CONVERGED)]
        rows["gradient_ratio"] = [0.5, 7.25, 0.75]
        log = StatusLog()
        log.add_vbd(0, 0, rows, 0, 5, 500)
        fake = mock.MagicMock()
        fake.TreeNode.return_value = True
        fake.TreeNodeEx.return_value = True
        fake.SliderInt.side_effect = lambda label, v, lo, hi: (False, v)
        _substep.clear()
        with mock.patch(f"{__name__}.imgui", fake):
            draw_status_panel(SimpleNamespace(status_log=log), None)
        shown = [str(c.args[-1]) for c in fake.TextColored.call_args_list] + [
            str(c.args[0]) for c in fake.Text.call_args_list
        ]
        self.assertIn("2 / 3", shown)  # substeps that passed
        self.assertIn("7.25", shown)  # the worst residual ratio, from substep 1
        self.assertIn("33 / 500", shown)  # the picked (last) substep's sweeps
        self.assertNotIn("11 / 500", shown)
        (slider,) = [c for c in fake.SliderInt.call_args_list if c.args[0].startswith("Substep##")]
        self.assertEqual(slider.args[1:], (2, 0, 2))
        self.assertEqual(fake.PushID.call_count, fake.PopID.call_count)


if __name__ == "__main__":
    unittest.main()

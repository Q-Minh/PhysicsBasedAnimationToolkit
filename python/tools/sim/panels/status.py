"""
GUI panel for per-solve statuses (see status.py), for interactive tuning.

Two tree nodes, read top to bottom:

- **Last step**: per substep, how the solve ended and what it used, one fact per row. VBD family: result,
  outer iterations and sweeps against their budgets, and the final ratios of the gradient residual and
  the penetration to their tolerances (at most 1 passes). Newton: the outer test with the same two ratios
  (at the end, at the first test, and after each subproblem), and per subproblem its status, Newton steps,
  CG iterations and line-search backtracks.
- **Last N steps**: the same counts summarized over a rolling window (``StatusLog.summary``).

Visible text stays short; tooltips explain each row. ``SimulationState.step`` records the statuses with
data it already synchronizes on, so the panel reads host memory only.
"""

import numpy as np
import polyscope.imgui as imgui

from ..gpu.newton.solver import SubproblemStatus
from ..gpu.vbd.convergence import SolveEnd

_window = {"steps": 100}
_GOOD = (0.4, 0.85, 0.4, 1.0)
_WARN = (1.0, 0.6, 0.2, 1.0)

_RESULT = {
    SolveEnd.CONVERGED: ("Converged", "The gradient test and feasibility were met within the outer-iteration budget."),
    SolveEnd.GRADIENT_UNMET: (
        "Budget: gradient unmet",
        "The outer iterations ran out with the gradient test unmet (the step is feasible).\n"
        "More sweeps or outer iterations, or a looser Solver.Convergence.rtol, would help.",
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

_RESIDUAL_TIP = (
    "Gradient residual over its tolerance at the end of the solve; at most 1 passes.\n"
    "Relative test (Solver.Convergence.rtol > 0): sqrt(sum |g_i|^2 / m_i) / (rtol * RMS displacement scale),\n"
    "with g the gradient of the step's Lagrangian. Absolute test: |g| / gtol."
)
_PENETRATION_TIP = (
    "Largest penetration into the contact margin over its tolerance ftol * max(|dmin|, 1e-6 m),\n"
    "at the end of the solve; at most 1 is feasible."
)
_NEWTON_RESIDUAL_TIP = (
    "Outer test's gradient residual over its tolerance, {when}; at most 1 passes.\n"
    "Relative test (Solver.Convergence.rtol > 0, dynamic problems), as the VBD family's:\n"
    "sqrt(sum |g_i|^2 / m_i) / (rtol * RMS displacement scale), with g the gradient of the step's Lagrangian.\n"
    "Absolute test (rtol = 0 or static problems): |g| / sqrt(gtol2), g the augmented-Lagrangian gradient."
)
_NEWTON_PENETRATION_TIP = (
    "Largest penetration into the contact margin over its tolerance ftol * max(|dmin|, 1e-6 m),\n"
    "{when}; at most 1 is feasible. Part of the relative outer test only (the absolute test, as C++'s,\n"
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
    """Residual and penetration rows of one Newton outer test, ``when`` naming it in the tooltips."""
    rows.row(
        f"{prefix}Residual / tol",
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


def _draw_vbd_solve(r, rows: _Rows):
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
    ratio = float(r["gradient_ratio"])
    rows.row("Residual / tol", _ratio(ratio), _RESIDUAL_TIP, None if ratio <= 1 else _WARN)
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
    for p in subproblems:
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


def _draw_last_step(log):
    last = log.last_step
    vbd = log.vbd
    vbd = vbd[vbd["step"] == last]
    solves = log.newton_solves
    solves = solves[solves["step"] == last]
    subproblems = log.newton_subproblems
    subproblems = subproblems[subproblems["step"] == last]
    n_substeps = max(vbd.shape[0], solves.shape[0])
    imgui.Text(f"Time step {last}")
    for sub in range(n_substeps):
        if n_substeps > 1:
            imgui.TextDisabled(f"Substep {sub}")
        with _Rows(f"last_step_{sub}") as rows:
            for r in vbd[vbd["substep"] == sub]:
                _draw_vbd_solve(r, rows)
            for s in solves[solves["substep"] == sub]:
                _draw_newton_solve(s, subproblems[subproblems["substep"] == sub], rows)


def _draw_window(log):
    _, _window["steps"] = imgui.SliderInt("Window (steps)", _window["steps"], 10, 1000)
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
            ratio = v["gradient_ratio_max"]
            rows.row("Worst residual / tol", _ratio(ratio), _RESIDUAL_TIP, None if ratio <= 1 else _WARN)
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
                n["gradient_ratio_max"],
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
    log = state.status_log
    if log.last_step is None:
        imgui.Text("No solves recorded yet")
    else:
        if imgui.TreeNodeEx("Last step", imgui.ImGuiTreeNodeFlags_DefaultOpen):
            _draw_last_step(log)
            imgui.TreePop()
        if imgui.TreeNodeEx(f"Last {_window['steps']} steps###window", imgui.ImGuiTreeNodeFlags_DefaultOpen):
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
        self.assertIn("Initial Residual / tol", shown)
        self.assertIn("40", shown)
        self.assertIn("n/a", shown)
        # The solve's result and after subproblem 0 (first draw), the window's worst (first two draws)
        self.assertEqual(shown.count("2.5"), 4)
        # No comma-delimited lists in the visible values
        self.assertFalse(any("," in t for t in shown if not t.startswith("Time steps")))
        tips = " ".join(str(c.args[0]) for c in fake.SetTooltip.call_args_list)
        self.assertIn("ftol * max(|dmin|", tips)
        self.assertNotIn("%", tips)


if __name__ == "__main__":
    unittest.main()

"""
GUI panel for per-solve statuses (see status.py), for interactive tuning.

Shows the last time step, one line per substep: for the VBD family, outer iterations and sweeps used
against their budgets, how the solve ended (converged early, or the budget ran out with the gradient
test, feasibility or both unmet) and the final ratios to the tolerances (<= 1 passes); for Newton, each
subproblem's status, Newton steps, CG iterations and line-search backtracks. Below, a summary over a
rolling window of time steps. ``SimulationState.step`` records the statuses with data it already
synchronizes on, so the panel reads host memory only.
"""

import polyscope.imgui as imgui

from ..gpu.newton.solver import SubproblemStatus
from ..gpu.vbd.convergence import SolveEnd

_window = {"steps": 100}
_GOOD = (0.4, 0.85, 0.4, 1.0)
_BAD = (1.0, 0.6, 0.2, 1.0)


def _draw_vbd_step(rows):
    for r in rows:
        reason = SolveEnd(int(r["reason"]))
        color = _GOOD if reason == SolveEnd.CONVERGED else _BAD
        imgui.TextColored(color, f"Substep {int(r['substep'])}: {reason.name.lower().replace('_', ' ')}")
        imgui.Text(
            f"  outer {int(r['outer'])}/{int(r['max_outer'])}, sweeps {int(r['sweeps'])}/{int(r['max_sweeps'])}"
        )
        imgui.Text(
            f"  gradient/tol {float(r['gradient_ratio']):.3g}, penetration/tol {float(r['feasibility_ratio']):.3g}"
        )


def _draw_newton_step(solves, subproblems):
    for s in solves:
        sub = int(s["substep"])
        color = _GOOD if bool(s["converged"]) else _BAD
        imgui.TextColored(
            color,
            f"Substep {sub}: outer {int(s['outer'])}/{int(s['max_outer'])}, "
            f"outer test {'passed' if bool(s['converged']) else 'not passed'}",
        )
        for p in subproblems[subproblems["substep"] == sub]:
            status = SubproblemStatus(int(p["status"]))
            ok = status in (SubproblemStatus.CONVERGED, SubproblemStatus.STEPPED)
            imgui.TextColored(
                _GOOD if ok else _BAD,
                f"  subproblem {int(p['outer'])}: {status.name.lower().replace('_', ' ')}, "
                f"Newton steps {int(p['newton_steps'])}/{int(p['max_newton_steps'])}",
            )
            cap = " (cap hit)" if bool(p["cg_capped"]) else ""
            imgui.Text(
                f"    CG max {int(p['max_cg_iters'])}/{int(p['cg_cap'])}{cap}, backtracks {int(p['backtracks'])}"
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
    last = log.last_step
    if last is None:
        imgui.Text("No solves recorded yet")
    else:
        imgui.Text(f"Last time step: {last}")
        vbd = log.vbd
        vbd = vbd[vbd["step"] == last]
        solves = log.newton_solves
        solves = solves[solves["step"] == last]
        if vbd.shape[0]:
            _draw_vbd_step(vbd)
        if solves.shape[0]:
            subproblems = log.newton_subproblems
            _draw_newton_step(solves, subproblems[subproblems["step"] == last])
        imgui.Separator()
        _, _window["steps"] = imgui.SliderInt("Window (steps)", _window["steps"], 10, 1000)
        for line in log.format_summary(_window["steps"]):
            imgui.TextWrapped(line)
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
        log = StatusLog()
        log.add_vbd(0, 0, rows, 0, 5, 500)
        sub = NewtonSubproblemRecord(SubproblemStatus.LINE_SEARCH_FAILED, 2, 500, True, 9)
        log.add_newton(1, [NewtonSolveRecord(1, False, (sub,))], 5, 10, 500)
        fake = mock.MagicMock()
        fake.TreeNode.return_value = True
        fake.SliderInt.side_effect = lambda label, v, lo, hi: (False, v)
        state = SimpleNamespace(status_log=log)
        with mock.patch(f"{__name__}.imgui", fake):
            draw_status_panel(state, None)  # last step: Newton
            log.add_vbd(2, 0, rows, 0, 5, 500)
            draw_status_panel(state, None)  # last step: VBD
            draw_status_panel(SimpleNamespace(status_log=StatusLog()), None)  # empty
        self.assertEqual(fake.PushID.call_count, fake.PopID.call_count)
        self.assertEqual(fake.TreeNode.call_count, fake.TreePop.call_count)
        texts = " ".join(str(c.args[-1]) for c in fake.TextColored.call_args_list)
        self.assertIn("line search failed", texts)
        self.assertIn("both unmet", texts)


if __name__ == "__main__":
    unittest.main()

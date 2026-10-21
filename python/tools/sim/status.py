"""
Per-solve statuses of a simulation, for interactive tuning (``panels/status.py``) and headless runs
(``simulate.py``).

Each solve is one time step, or one substep with ``SimulationState.substeps > 1``.

- VBD family: rows come from the convergence kernels' device ring
  (:class:`gpu.vbd.convergence.SolveStatus`). ``SimulationState.step`` queues an async copy of it
  before its existing synchronization, so recording adds no synchronization and no graph nodes.
- Newton: its loop is host-driven and already synchronizes per Newton step, so it records
  :class:`gpu.newton.solver.NewtonSolveRecord` on the host.

:class:`StatusLog` keeps every solve (a few dozen bytes each), summarizes a rolling window and writes
ranges of time steps to HDF5 (``sim/<frame>/solve_status`` in ``simulate.py`` outputs: the steps since
the previous frame).
"""

import numpy as np

from .gpu.newton.solver import NewtonSolveRecord, SubproblemStatus
from .gpu.vbd.convergence import SolveEnd

VBD_DTYPE = np.dtype(
    [
        ("step", np.int32),
        ("substep", np.int32),
        ("solver", np.int32),  # state.SolverType value
        ("outer", np.int32),
        ("sweeps", np.int32),
        ("reason", np.int32),  # SolveEnd
        ("gradient_ratio", np.float32),
        ("feasibility_ratio", np.float32),
        ("max_outer", np.int32),
        ("max_sweeps", np.int32),  # n_max_iters * n_subproblem_max_iters
    ]
)
"""One VBD-family solve."""

NEWTON_SOLVE_DTYPE = np.dtype(
    [
        ("step", np.int32),
        ("substep", np.int32),
        ("outer", np.int32),
        ("converged", np.bool_),
        ("max_outer", np.int32),
        # Ratios to their tolerances (at most 1 passes) of the first outer test and of the last one (the result)
        ("initial_gradient_ratio", np.float32),
        ("initial_feasibility_ratio", np.float32),
        ("gradient_ratio", np.float32),
        ("feasibility_ratio", np.float32),
    ]
)
"""One Newton solve."""

NEWTON_SUBPROBLEM_DTYPE = np.dtype(
    [
        ("step", np.int32),
        ("substep", np.int32),
        ("outer", np.int32),  # outer iteration of the subproblem, from 0
        ("status", np.int32),  # SubproblemStatus
        ("newton_steps", np.int32),
        ("max_cg_iters", np.int32),
        ("cg_capped", np.bool_),
        ("backtracks", np.int32),
        ("max_newton_steps", np.int32),  # n_subproblem_max_iters
        ("cg_cap", np.int32),  # LinearSolverParams.n_max_iters
        # Ratios to their tolerances of the outer test after the subproblem (at most 1 passes)
        ("gradient_ratio", np.float32),
        ("feasibility_ratio", np.float32),
    ]
)
"""One Newton subproblem."""


class _Table:
    """A structured array that grows by doubling, so appending a step stays O(rows appended)."""

    def __init__(self, dtype: np.dtype):
        self._data = np.zeros(64, dtype=dtype)
        self._n = 0

    def append(self, rows: np.ndarray) -> None:
        n = self._n + rows.shape[0]
        if n > self._data.shape[0]:
            grown = np.zeros(max(n, 2 * self._data.shape[0]), dtype=self._data.dtype)
            grown[: self._n] = self._data[: self._n]
            self._data = grown
        self._data[self._n : n] = rows
        self._n = n

    @property
    def rows(self) -> np.ndarray:
        return self._data[: self._n]


class StatusLog:
    """Every solve's status, appended by ``SimulationState.step``."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self._vbd = _Table(VBD_DTYPE)
        self._newton_solves = _Table(NEWTON_SOLVE_DTYPE)
        self._newton_subproblems = _Table(NEWTON_SUBPROBLEM_DTYPE)
        self.lost = 0  # VBD-family solves whose rows the device ring overwrote before a read

    def add_vbd(
        self, step: int, solver: int, rows: np.ndarray, lost: int, max_outer: int, max_sweeps: int
    ) -> None:
        """Append one step's VBD-family solves.

        Args:
            step (int): Time step index (``SimulationState.t`` before the step).
            solver (int): ``SolverType`` value.
            rows (np.ndarray): :class:`gpu.vbd.convergence.SolveStatus` rows of the step's solves, by
                solve index (one per substep).
            lost (int): Solves lost to the ring wrapping.
            max_outer (int): Outer-iteration budget.
            max_sweeps (int): Sweep budget of a solve, outer iterations included.
        """
        self.lost += int(lost)
        if rows.shape[0] == 0:
            return
        out = np.zeros(rows.shape[0], dtype=VBD_DTYPE)
        out["step"] = step
        out["substep"] = np.arange(rows.shape[0])
        out["solver"] = solver
        for f in ("outer", "sweeps", "reason", "gradient_ratio", "feasibility_ratio"):
            out[f] = rows[f]
        out["max_outer"] = max_outer
        out["max_sweeps"] = max_sweeps
        self._vbd.append(out)

    def add_newton(
        self,
        step: int,
        records: list[NewtonSolveRecord],
        max_outer: int,
        max_newton_steps: int,
        cg_cap: int,
    ) -> None:
        """Append one step's Newton solves (one per substep)."""
        if not records:
            return
        solves = np.zeros(len(records), dtype=NEWTON_SOLVE_DTYPE)
        subs = []
        nan = (np.nan, np.nan)
        for s, r in enumerate(records):
            tests = r.outer_tests or (nan,)
            solves[s] = (step, s, r.outer, r.converged, max_outer, *tests[0], *tests[-1])
            for k, sp in enumerate(r.subproblems):
                after = tests[k + 1] if k + 1 < len(tests) else nan
                subs.append(
                    (step, s, k, int(sp.status), sp.newton_steps, sp.max_cg_iters, sp.cg_capped,
                     sp.backtracks, max_newton_steps, cg_cap, *after)
                )
        self._newton_solves.append(solves)
        if subs:
            self._newton_subproblems.append(np.array(subs, dtype=NEWTON_SUBPROBLEM_DTYPE))

    @property
    def vbd(self) -> np.ndarray:
        """All VBD-family solves (:data:`VBD_DTYPE`), a view."""
        return self._vbd.rows

    @property
    def newton_solves(self) -> np.ndarray:
        """All Newton solves (:data:`NEWTON_SOLVE_DTYPE`), a view."""
        return self._newton_solves.rows

    @property
    def newton_subproblems(self) -> np.ndarray:
        """All Newton subproblems (:data:`NEWTON_SUBPROBLEM_DTYPE`), a view."""
        return self._newton_subproblems.rows

    @staticmethod
    def _since(rows: np.ndarray, first: int) -> np.ndarray:
        # Rows are appended in step order, so the window is a suffix
        return rows[np.searchsorted(rows["step"], first, side="left") :]

    @staticmethod
    def _steps(rows: np.ndarray, steps: tuple[int, int] | None) -> np.ndarray:
        if steps is None:
            return rows
        lo, hi = np.searchsorted(rows["step"], steps, side="left")
        return rows[lo:hi]

    @property
    def last_step(self) -> int | None:
        """The last recorded time step, or None."""
        steps = [a["step"][-1] for a in (self.vbd, self.newton_solves) if a.shape[0]]
        return int(max(steps)) if steps else None

    def summary(self, window: int | None = None) -> dict:
        """Counts and means over the last ``window`` time steps (all if None).

        Returns:
            dict: ``steps``, and ``vbd`` / ``newton`` sub-dicts for the solver families present.
        """
        last = self.last_step
        if last is None:
            return {"steps": 0}
        first = -1 if window is None else last - int(window) + 1
        out: dict = {"steps": last - max(first, 0) + 1}
        v = self._since(self.vbd, first)
        if v.shape[0]:
            out["vbd"] = {
                "solves": int(v.shape[0]),
                "reasons": {e.name: int((v["reason"] == e).sum()) for e in SolveEnd if e != SolveEnd.RUNNING},
                "outer_mean": float(v["outer"].mean()),
                "outer_max": int(v["outer"].max()),
                "outer_budget": int(v["max_outer"][-1]),
                "sweeps_mean": float(v["sweeps"].mean()),
                "sweeps_max": int(v["sweeps"].max()),
                "sweeps_budget": int(v["max_sweeps"][-1]),
                "gradient_ratio_max": float(v["gradient_ratio"].max()),
                "feasibility_ratio_max": float(v["feasibility_ratio"].max()),
                "lost": int(self.lost),
            }
        n = self._since(self.newton_solves, first)
        if n.shape[0]:
            sp = self._since(self.newton_subproblems, first)
            out["newton"] = {
                "solves": int(n.shape[0]),
                "converged": int(n["converged"].sum()),
                "outer_mean": float(n["outer"].mean()),
                "outer_max": int(n["outer"].max()),
                "outer_budget": int(n["max_outer"][-1]),
                "subproblems": int(sp.shape[0]),
                "statuses": {e.name: int((sp["status"] == e).sum()) for e in SubproblemStatus},
                "newton_steps_mean": float(sp["newton_steps"].mean()) if sp.shape[0] else 0.0,
                "newton_steps_max": int(sp["newton_steps"].max()) if sp.shape[0] else 0,
                "newton_steps_budget": int(sp["max_newton_steps"][-1]) if sp.shape[0] else 0,
                "cg_iters_max": int(sp["max_cg_iters"].max()) if sp.shape[0] else 0,
                "cg_cap": int(sp["cg_cap"][-1]) if sp.shape[0] else 0,
                "cg_capped": int(sp["cg_capped"].sum()),
                "backtracks": int(sp["backtracks"].sum()),
                "gradient_ratio_max": float(np.nanmax(n["gradient_ratio"], initial=0.0)),
                "feasibility_ratio_max": float(np.nanmax(n["feasibility_ratio"], initial=0.0)),
            }
        return out

    def format_summary(self, window: int | None = None) -> list[str]:
        """:meth:`summary` as text lines (the GUI panel and ``simulate.py`` print them)."""
        s = self.summary(window)
        if s["steps"] == 0:
            return ["No solves recorded"]
        lines = [f"{s['steps']} time steps"]
        if "vbd" in s:
            v = s["vbd"]
            r = v["reasons"]
            budget = r["GRADIENT_UNMET"] + r["FEASIBILITY_UNMET"] + r["BOTH_UNMET"]
            lines += [
                f"VBD-family solves: {v['solves']}: {r['CONVERGED']} converged early, {budget} hit the "
                f"budget (gradient unmet {r['GRADIENT_UNMET']}, feasibility unmet "
                f"{r['FEASIBILITY_UNMET']}, both {r['BOTH_UNMET']})",
                f"  outer iterations: mean {v['outer_mean']:.2f}, max {v['outer_max']} / {v['outer_budget']}",
                f"  sweeps: mean {v['sweeps_mean']:.1f}, max {v['sweeps_max']} / {v['sweeps_budget']}",
                f"  worst ratios to tolerance: gradient {v['gradient_ratio_max']:.3g}, feasibility "
                f"{v['feasibility_ratio_max']:.3g}",
            ]
            if v["lost"]:
                lines.append(f"  {v['lost']} solves lost (status ring overwritten between reads)")
        if "newton" in s:
            n = s["newton"]
            st = n["statuses"]
            lines += [
                f"Newton solves: {n['solves']}: {n['converged']} converged, outer iterations mean "
                f"{n['outer_mean']:.2f}, max {n['outer_max']} / {n['outer_budget']}",
                f"  subproblems: {n['subproblems']}: "
                + ", ".join(f"{k.lower()} {c}" for k, c in st.items()),
                f"  Newton steps per subproblem: mean {n['newton_steps_mean']:.2f}, max "
                f"{n['newton_steps_max']} / {n['newton_steps_budget']}",
                f"  CG: max {n['cg_iters_max']} / {n['cg_cap']} iterations, {n['cg_capped']} subproblems "
                f"hit the cap; line-search backtracks: {n['backtracks']}",
                f"  worst ratios to tolerance: gradient {n['gradient_ratio_max']:.3g}, feasibility "
                f"{n['feasibility_ratio_max']:.3g}",
            ]
        return lines

    def write_h5(self, group, steps: tuple[int, int] | None = None) -> None:
        """Write the log under an h5py group: datasets ``vbd``, ``newton_solves`` and
        ``newton_subproblems`` (structured, empty ones omitted), with enum names as attributes.

        Args:
            group: The h5py group.
            steps (tuple[int, int] | None): Time steps ``[first, last)`` to write; all if None.
        """
        for name, a in (
            ("vbd", self._steps(self.vbd, steps)),
            ("newton_solves", self._steps(self.newton_solves, steps)),
            ("newton_subproblems", self._steps(self.newton_subproblems, steps)),
        ):
            if a.shape[0]:
                if name in group:
                    del group[name]
                group.create_dataset(name, data=np.ascontiguousarray(a))
        group.attrs["solve_end"] = [e.name for e in SolveEnd]
        group.attrs["subproblem_status"] = [e.name for e in SubproblemStatus]
        group.attrs["lost"] = self.lost


import unittest


class TestStatusLog(unittest.TestCase):
    @staticmethod
    def _rows(solves, reasons=None):
        # numpy layout of gpu.vbd.convergence.SolveStatus
        dt = np.dtype(
            [("solve", np.int32), ("outer", np.int32), ("sweeps", np.int32), ("reason", np.int32),
             ("gradient_ratio", np.float32), ("feasibility_ratio", np.float32)]
        )
        r = np.zeros(len(solves), dtype=dt)
        r["solve"] = solves
        r["reason"] = reasons if reasons is not None else int(SolveEnd.CONVERGED)
        r["outer"] = 1
        r["sweeps"] = 20
        return r

    def test_select_new_rows(self):
        from .gpu.vbd.convergence import select_new_status_rows

        # Ring of 4 after solves 0..5: rows hold 4, 5, 2, 3 (by slot); unwritten rows are RUNNING
        ring = self._rows([4, 5, 2, 3])
        new, nxt, lost = select_new_status_rows(ring, 2)
        self.assertEqual(new["solve"].tolist(), [2, 3, 4, 5])
        self.assertEqual((nxt, lost), (6, 0))
        new, nxt, lost = select_new_status_rows(ring, 6)
        self.assertEqual((new.shape[0], nxt, lost), (0, 6, 0))
        # Two solves lost: the ring wrapped past 0 and 1
        new, nxt, lost = select_new_status_rows(ring, 0)
        self.assertEqual((new["solve"].tolist(), nxt, lost), ([2, 3, 4, 5], 6, 2))
        # Fresh ring: zero rows are RUNNING, so nothing is new
        fresh = self._rows([0, 0], reasons=int(SolveEnd.RUNNING))
        self.assertEqual(select_new_status_rows(fresh, 0)[0].shape[0], 0)

    def test_summary_and_h5(self):
        import tempfile, os, h5py
        from .gpu.newton.solver import NewtonSubproblemRecord

        log = StatusLog()
        log.add_vbd(0, 1, self._rows([0, 1], [int(SolveEnd.CONVERGED), int(SolveEnd.GRADIENT_UNMET)]), 0, 5, 400)
        log.add_vbd(1, 1, self._rows([2, 3]), 1, 5, 400)
        sub = NewtonSubproblemRecord(SubproblemStatus.LINE_SEARCH_FAILED, 3, 300, True, 12)
        log.add_newton(2, [NewtonSolveRecord(1, False, (sub,), ((40.0, 3.0), (2.5, 0.5)))], 5, 10, 300)
        s = log.summary()
        n = log.newton_solves[0]
        self.assertEqual((n["initial_gradient_ratio"], n["initial_feasibility_ratio"]), (40.0, 3.0))
        self.assertEqual((n["gradient_ratio"], n["feasibility_ratio"]), (2.5, 0.5))
        self.assertEqual(log.newton_subproblems[0]["gradient_ratio"], 2.5)
        self.assertEqual((s["newton"]["gradient_ratio_max"], s["newton"]["feasibility_ratio_max"]), (2.5, 0.5))
        self.assertEqual(s["steps"], 3)
        self.assertEqual(s["vbd"]["reasons"]["CONVERGED"], 3)
        self.assertEqual(s["vbd"]["reasons"]["GRADIENT_UNMET"], 1)
        self.assertEqual(s["vbd"]["lost"], 1)
        self.assertEqual(s["newton"]["statuses"]["LINE_SEARCH_FAILED"], 1)
        self.assertEqual((s["newton"]["cg_capped"], s["newton"]["backtracks"]), (1, 12))
        self.assertEqual(log.vbd["substep"].tolist(), [0, 1, 0, 1])
        # Window: only the last time step (Newton)
        self.assertNotIn("vbd", log.summary(window=1))
        self.assertTrue(any("hit the budget" in line for line in log.format_summary()))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            with h5py.File(path, "w") as f:
                log.write_h5(f.require_group("sim/status"))
            with h5py.File(path, "r") as f:
                g = f["sim/status"]
                np.testing.assert_array_equal(g["vbd"][()], log.vbd)
                self.assertEqual(g["newton_subproblems"][()]["backtracks"].tolist(), [12])
                self.assertEqual(int(g.attrs["lost"]), 1)
            # Ranges of steps, as simulate.py writes them per frame
            with h5py.File(path, "w") as f:
                log.write_h5(f.require_group("sim/00000001/solve_status"), steps=(0, 2))
                log.write_h5(f.require_group("sim/00000002/solve_status"), steps=(2, 3))
                log.write_h5(f.require_group("sim/00000003/solve_status"), steps=(3, 3))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["sim/00000001/solve_status/vbd"][()]["step"].tolist(), [0, 0, 1, 1])
                self.assertNotIn("newton_solves", f["sim/00000001/solve_status"])
                self.assertEqual(f["sim/00000002/solve_status/newton_solves"][()]["step"].tolist(), [2])
                self.assertNotIn("vbd", f["sim/00000002/solve_status"])
                self.assertEqual(len(f["sim/00000003/solve_status"]), 0)
        for t in range(3, 103):  # grows past the initial capacity
            log.add_vbd(t, 1, self._rows([2 * t, 2 * t + 1]), 0, 5, 400)
        self.assertEqual(log.vbd.shape[0], 204)
        self.assertEqual(log.summary(window=10)["vbd"]["solves"], 20)

    def test_device_ring_readback(self):
        import warp as wp
        from .gpu.vbd import convergence

        wp.init()

        @wp.kernel
        def write(status: wp.array[convergence.SolveStatus], solves: wp.array[wp.int32], n: wp.int32):
            # As the start/end kernels: one row per solve at solve % capacity
            for k in range(n):
                s = solves[0]
                solves[0] = s + 1
                row = convergence.SolveStatus()
                row.solve = s
                row.outer = 2
                row.sweeps = 7 * s
                row.reason = convergence.SOLVE_CONVERGED
                status[s % status.shape[0]] = row

        class _Ring:
            # The readback half of ConvergenceCheck, without a FEM
            enqueue_status_readback = convergence.ConvergenceCheck.enqueue_status_readback
            read_status = convergence.ConvergenceCheck.read_status

            def __init__(self, capacity):
                self.status = wp.zeros(capacity, dtype=convergence.SolveStatus)
                self.solves = wp.zeros(1, dtype=wp.int32)
                self._status_host = wp.zeros(
                    capacity, dtype=convergence.SolveStatus, device="cpu", pinned=True
                )
                self._next_solve = 0

        ring = _Ring(3)
        wp.launch(write, dim=1, inputs=[ring.status, ring.solves, 2])
        ring.enqueue_status_readback()
        wp.synchronize()
        rows, lost = ring.read_status()
        self.assertEqual((rows["solve"].tolist(), rows["sweeps"].tolist(), lost), ([0, 1], [0, 7], 0))
        wp.launch(write, dim=1, inputs=[ring.status, ring.solves, 5])  # 5 solves into 3 rows
        ring.enqueue_status_readback()
        wp.synchronize()
        rows, lost = ring.read_status()
        self.assertEqual((rows["solve"].tolist(), lost), ([4, 5, 6], 2))


if __name__ == "__main__":
    unittest.main()

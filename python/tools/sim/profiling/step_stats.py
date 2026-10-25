"""
Per-step statistics of a profiling result, for the report's distributions and percentile slices.

Per step: the GPU time of each phase as a union over streams (``union_ms``: work running concurrently on
several streams counts once), the step's union, its solve outcome (``status``, from ``simulate.py``) and its
peak node speed (``peak_speed``, from ``profiling/trajectory.py``). Steps are then grouped into slices by
percentiles of one per-step quantity, e.g. the 10% slowest steps, and each slice summarized.

Like ``results.py``, this module must not import the GPU stack or h5py.
"""

import numpy as np

from . import results

# Scopes whose entries count one solver iteration, for results without solve statuses.
ITERATION_SCOPES = ("vbd.sweep", "aaavbd.sweep", "newton.linear_solve")
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
QUANTILE_NAMES = ("min", "P10", "Q1", "median", "Q3", "P90", "max")


def phase_union_ms(step: dict) -> dict[str, float]:
    """A step's GPU time per phase (innermost scope name), as the union over streams of each scope path's
    own activities, summed over paths with the same innermost name (busy time for results without unions)."""
    per_path = step.get("union_ms") or step.get("busy_ms", {})
    out: dict[str, float] = {}
    for path, ms in per_path.items():
        key = results.leaf(path)
        out[key] = out.get(key, 0.0) + ms
    return out


def has_unions(result: dict) -> bool:
    """Whether every step has per-phase unions (otherwise phases fall back to busy time)."""
    return bool(result["steps"]) and all("union_ms" in s for s in result["steps"])


def step_time_ms(step: dict) -> float:
    """The step's GPU time: the union over streams of all its activities, or its production timeline."""
    for key in ("step_union_ms", "step_timeline_ms"):
        if key in step:
            return float(step[key])
    return float("nan")


def solve_outcome(step: dict) -> dict:
    """The step's solves, summed over substeps.

    Returns:
        dict: ``solves`` and ``converged`` (counts; ``converged`` is the early exits, solves that passed
        their tests within the budget), ``iterations`` (VBD-family sweeps or Newton steps), ``outer``
        (outer iterations), ``unit`` of the iterations, and the worst ``gradient_ratio`` and
        ``feasibility_ratio`` to their tolerances (NaN if unknown). Without a ``status``, only
        ``iterations`` is known, counted from the iteration scopes' calls.
    """
    nan = float("nan")
    status = step.get("status") or {}

    def worst(rows, key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return max(values) if values else nan

    if "vbd" in status:
        rows = status["vbd"]
        return {
            "solves": len(rows),
            "converged": sum(r["reason"] == "CONVERGED" for r in rows),
            "iterations": sum(r["sweeps"] for r in rows),
            "outer": sum(r["outer"] for r in rows),
            "unit": "sweeps",
            "gradient_ratio": worst(rows, "gradient_ratio"),
            "feasibility_ratio": worst(rows, "feasibility_ratio"),
        }
    if "newton" in status:
        rows = status["newton"]
        return {
            "solves": len(rows),
            "converged": sum(bool(r["converged"]) for r in rows),
            "iterations": sum(r["newton_steps"] for r in rows),
            "outer": sum(r["outer"] for r in rows),
            "unit": "Newton steps",
            "gradient_ratio": worst(rows, "gradient_ratio"),
            "feasibility_ratio": worst(rows, "feasibility_ratio"),
        }
    calls = step.get("calls", {})
    for name in ITERATION_SCOPES:
        n = sum(c for p, c in calls.items() if results.leaf(p) == name)
        if n > 0:
            unit = "Newton steps" if name.startswith("newton") else "sweeps"
            break
    else:
        n, unit = nan, ""
    return {"solves": 0, "converged": 0, "iterations": n, "outer": nan, "unit": unit,
            "gradient_ratio": nan, "feasibility_ratio": nan}


def per_step(result: dict) -> dict:
    """Per-step arrays of a result, in step order.

    Returns:
        dict: ``t``, ``time_ms`` (:func:`step_time_ms`), ``span_ms``, ``peak_speed``, ``rms_speed``, the
        :func:`solve_outcome` fields ``solves``, ``converged``, ``iterations``, ``outer``,
        ``gradient_ratio``, ``feasibility_ratio`` (NaN where unknown), ``unit`` (a string) and
        ``phases`` (phase name -> per-step ms, 0 where a step has no work in it).
    """
    steps = result["steps"]
    outcomes = [solve_outcome(s) for s in steps]
    phases_per_step = [phase_union_ms(s) for s in steps]
    names = sorted({p for ph in phases_per_step for p in ph})
    nan = float("nan")
    return {
        "t": np.array([s["t"] for s in steps], dtype=np.int64),
        "time_ms": np.array([step_time_ms(s) for s in steps]),
        "span_ms": np.array([s.get("step_span_ms", nan) for s in steps], dtype=float),
        "peak_speed": np.array([s.get("peak_speed", nan) for s in steps], dtype=float),
        "rms_speed": np.array([s.get("rms_speed", nan) for s in steps], dtype=float),
        **{k: np.array([o[k] for o in outcomes], dtype=float)
           for k in ("solves", "converged", "iterations", "outer", "gradient_ratio", "feasibility_ratio")},
        "unit": next((o["unit"] for o in outcomes if o["unit"]), ""),
        "phases": {p: np.array([ph.get(p, 0.0) for ph in phases_per_step]) for p in names},
    }


def on_step_grid(table: dict) -> dict:
    """:func:`per_step` arrays on every step from the first to the last, NaN at the steps the result lacks (the
    warm-up steps at the seams of merged legs, ``profiling/legs.py``), so that plots break there rather than
    interpolate; ``measured`` marks the steps the result has."""
    t = table["t"]
    if t.size == 0 or t[-1] - t[0] + 1 == t.size:
        return {**table, "measured": np.ones(t.size, dtype=bool)}
    index = t - t[0]

    def grid(values):
        out = np.full(int(t[-1] - t[0]) + 1, np.nan)
        out[index] = values
        return out

    out = {k: grid(v) if isinstance(v, np.ndarray) else v for k, v in table.items()}
    out["t"] = np.arange(t[0], t[-1] + 1)
    out["phases"] = {p: grid(v) for p, v in table["phases"].items()}
    out["measured"] = ~np.isnan(grid(np.zeros(t.size)))
    return out


def describe(values: np.ndarray) -> dict | None:
    """Count, mean and :data:`QUANTILES` (named by :data:`QUANTILE_NAMES`) of the finite values, or None."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    q = np.quantile(v, QUANTILES)
    return {"n": int(v.size), "mean": float(v.mean()), **dict(zip(QUANTILE_NAMES, map(float, q)))}


def percentile_slices(values: np.ndarray, edges: tuple[float, ...] = (10.0, 90.0)) -> list[dict]:
    """Groups of steps between consecutive percentiles of ``values`` (NaN values in none).

    The first slice holds values up to the first percentile, the next ones values above the previous
    percentile and up to the next, the last values above the last percentile. Ties at a percentile fall
    into the lower slice, so slices of a discrete quantity can be uneven or empty.

    Returns:
        list[dict]: Per slice, ``label`` (e.g. ``"P10-P90"``), the percentile range ``lo``/``hi`` (0-100),
        the value range ``vmin``/``vmax`` of its members (None if empty) and ``mask``.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return []
    cuts = np.percentile(v[finite], edges)
    bounds = [-np.inf, *cuts, np.inf]
    pct = [0.0, *edges, 100.0]
    out = []
    for i in range(len(bounds) - 1):
        lo_ok = v >= bounds[i] if i == 0 else v > bounds[i]
        mask = finite & lo_ok & (v <= bounds[i + 1])
        members = v[mask]
        out.append({
            "label": f"P{pct[i]:g}-P{pct[i + 1]:g}",
            "lo": pct[i],
            "hi": pct[i + 1],
            "vmin": float(members.min()) if members.size else None,
            "vmax": float(members.max()) if members.size else None,
            "mask": mask,
        })
    return out


def summarize_slice(table: dict, mask: np.ndarray) -> dict:
    """A slice of :func:`per_step` arrays: step count, mean phase times (ms/step), medians of the step
    time, iterations, time per iteration (``ms_per_iteration``, separating more iterations from more
    expensive ones), outer iterations and peak speed, the early-exit fraction of its solves, and the
    worst and median per-step gradient ratios and the worst feasibility ratio."""

    def median(key):
        d = describe(table[key][mask])
        return d["median"] if d else None

    solves = float(np.nansum(table["solves"][mask]))
    grad = describe(table["gradient_ratio"][mask])
    feas = describe(table["feasibility_ratio"][mask])
    n = int(mask.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        per_iteration = describe(np.where(table["iterations"] > 0, table["time_ms"] / table["iterations"], np.nan)[mask])
    return {
        "steps": n,
        "phases": {p: float(v[mask].mean()) if n else 0.0 for p, v in table["phases"].items()},
        "time_ms": median("time_ms"),
        "iterations": median("iterations"),
        "ms_per_iteration": per_iteration["median"] if per_iteration else None,
        "outer": median("outer"),
        "peak_speed": median("peak_speed"),
        "solves": int(solves),
        "early_exit": float(np.nansum(table["converged"][mask])) / solves if solves > 0 else None,
        "gradient_ratio_median": grad["median"] if grad else None,
        "gradient_ratio_max": grad["max"] if grad else None,
        "feasibility_ratio_max": feas["max"] if feas else None,
    }


def union_overlap(table: dict) -> float | None:
    """How far the phases' unions add up beyond the steps' unions, as a fraction of the total step
    union (0 when phases never run concurrently with each other), or None without step unions."""
    time = table["time_ms"]
    ok = np.isfinite(time)
    if not ok.any() or time[ok].sum() <= 0.0:
        return None
    phases = sum(v[ok] for v in table["phases"].values()) if table["phases"] else np.zeros(ok.sum())
    return float(max(phases.sum() - time[ok].sum(), 0.0) / time[ok].sum())


import unittest


class TestStepStats(unittest.TestCase):

    @staticmethod
    def _result():
        steps = []
        for t in range(10):
            reason = "CONVERGED" if t < 7 else "GRADIENT_UNMET"
            steps.append({
                "t": t,
                "busy_ms": {"solve/aaavbd.sweep": 2.0 + t, "solve/contact.detect": 1.5},
                "union_ms": {"solve/aaavbd.sweep": 2.0 + t, "solve/contact.detect": 1.0,
                             "init/contact.detect": 0.5},
                "calls": {"solve/aaavbd.sweep": 40},
                "step_union_ms": 3.5 + t,
                "step_span_ms": 5.0 + t,
                "status": {"vbd": [
                    {"substep": 0, "outer": 1, "sweeps": 20, "reason": "CONVERGED",
                     "gradient_ratio": 0.5, "feasibility_ratio": 0.1, "max_outer": 5, "max_sweeps": 100},
                    {"substep": 1, "outer": 5, "sweeps": 20 + 10 * t, "reason": reason,
                     "gradient_ratio": 0.1 * t, "feasibility_ratio": 0.2, "max_outer": 5, "max_sweeps": 100},
                ]},
                "peak_speed": float(t % 5),
            })
        return {"steps": steps}

    def test_per_step(self):
        table = per_step(self._result())
        self.assertEqual(table["unit"], "sweeps")
        # Same-named phases under different parents add up (disjoint in time)
        self.assertEqual(table["phases"]["contact.detect"].tolist(), [1.5] * 10)
        self.assertEqual(table["iterations"][3], 20 + 50)
        self.assertEqual(table["outer"][0], 6)
        self.assertAlmostEqual(table["gradient_ratio"][9], 0.9)
        self.assertEqual(union_overlap(table), 0.0)

    def test_without_status_counts_iteration_scopes(self):
        result = self._result()
        for s in result["steps"]:
            del s["status"]
        o = solve_outcome(result["steps"][0])
        self.assertEqual((o["iterations"], o["unit"], o["solves"]), (40, "sweeps", 0))
        self.assertTrue(np.isnan(o["gradient_ratio"]))

    def test_slices(self):
        table = per_step(self._result())
        slices = percentile_slices(table["time_ms"], (10.0, 90.0))
        self.assertEqual([s["label"] for s in slices], ["P0-P10", "P10-P90", "P90-P100"])
        self.assertEqual([int(s["mask"].sum()) for s in slices], [1, 8, 1])
        self.assertEqual((slices[2]["vmin"], slices[2]["vmax"]), (12.5, 12.5))
        top = summarize_slice(table, slices[2]["mask"])
        self.assertEqual((top["steps"], top["solves"], top["early_exit"]), (1, 2, 0.5))
        self.assertEqual(top["phases"]["aaavbd.sweep"], 11.0)
        fast = summarize_slice(table, slices[0]["mask"])
        self.assertEqual((fast["early_exit"], fast["iterations"]), (1.0, 40.0))
        self.assertAlmostEqual(fast["ms_per_iteration"], 3.5 / 40)
        # Ties fall into the lower slice: speeds 0..4 twice each
        speed = percentile_slices(table["peak_speed"], (50.0,))
        self.assertEqual([int(s["mask"].sum()) for s in speed], [6, 4])
        self.assertEqual(percentile_slices(np.full(3, np.nan)), [])

    def test_describe_and_overlap(self):
        d = describe(np.array([1.0, 2.0, 3.0, np.nan]))
        self.assertEqual((d["n"], d["median"], d["max"]), (3, 2.0, 3.0))
        self.assertIsNone(describe(np.array([np.nan])))
        table = {"time_ms": np.array([2.0, 2.0]), "phases": {"a": np.array([1.5, 1.5]), "b": np.array([1.0, 1.0])}}
        self.assertAlmostEqual(union_overlap(table), 0.25)


if __name__ == "__main__":
    unittest.main()

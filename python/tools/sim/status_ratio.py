"""
The stationarity test's decisive ratio, from a solve status row's recorded ratios (``status.VBD_DTYPE``, or a
profiling result's ``status`` entry, ``status.StatusLog.step_entry``).

The VBD family's stationarity test (``gpu/vbd/convergence.py``, ``_is_converged``) passes iff

    RMS_M(q) <= 1   and   (RMS_M(q0) <= 1   or   RMS_M(r) <= kg RMS_M(phi)),

and the status records RMS_M(q) as ``gradient_ratio``, RMS_M(q0) as ``no_floor_ratio`` and
RMS_M(r) / (kg RMS_M(phi)) as ``floor_clause_ratio``. With ``Solver.Convergence.node_norm`` MAX every RMS_M is the
largest value over free nodes instead, in the test and in the recorded ratios alike, so the same formula decides.
With ``Solver.Convergence.smoothing_passes`` > 0 the unsmoothed and the smoothed test must both pass and the row
records the stricter one's ratios, so it still decides. So ``max(gradient_ratio, min(no_floor_ratio,
floor_clause_ratio))`` is at most 1 exactly when the test passes, whereas ``gradient_ratio`` alone can be below 1 on
a step that failed the second clause.

Kept apart from ``status.py``, which imports the GPU stack: reports and paper figures read results without it
(``profiling/results.py``, ``TestReportImports``).
"""

import numpy as np


def stationarity_test_ratio(row):
    """The stationarity test's decisive ratio of one status row: ``max(gradient_ratio, min(no_floor_ratio,
    floor_clause_ratio))``, at most 1 exactly when the test passed (feasibility isn't included).

    Rows without the two other ratios (recorded before they existed, 2026-11-06; GPU Newton's rows, whose
    ``gradient_ratio`` is the decisive ratio itself, ``ConvergenceCheck.decisive_ratio``, RMS_M(q0) before Newton had
    the float32 floor, 2026-11-07) give ``gradient_ratio``. The absolute test records them as 0, which also gives
    ``gradient_ratio``.

    Args:
        row: A mapping (a profiling result's status entry, None for unknown values) or a numpy structured
            record or array (:data:`status.VBD_DTYPE`, vectorized).

    Returns:
        The ratio (an array for a structured array), or None if ``gradient_ratio`` is None.
    """
    fields = row.dtype.names if hasattr(row, "dtype") else row
    gradient = row["gradient_ratio"]
    if gradient is None or "no_floor_ratio" not in fields or "floor_clause_ratio" not in fields:
        return gradient
    no_floor, floor_clause = row["no_floor_ratio"], row["floor_clause_ratio"]
    if no_floor is None or floor_clause is None:
        return gradient
    out = np.maximum(gradient, np.minimum(no_floor, floor_clause))
    return float(out) if np.ndim(out) == 0 else out


import unittest


class TestStationarityTestRatio(unittest.TestCase):
    def test_decides_like_the_test(self):
        # (gradient, no floor, floor clause) -> passes: RMS(q) <= 1 and (RMS(q0) <= 1 or floor clause <= 1)
        cases = [((0.2, 1.1, 4.6), False), ((0.2, 0.9, 4.6), True), ((0.2, 37.8, 0.85), True),
                 ((1.2, 0.5, 0.5), False), ((0.9, 2.0, 1.5), False), ((1.0, 1.0, 9.0), True)]
        for (g, nf, fc), passes in cases:
            ratio = stationarity_test_ratio({"gradient_ratio": g, "no_floor_ratio": nf, "floor_clause_ratio": fc})
            self.assertEqual(ratio <= 1.0, passes, (g, nf, fc))
        self.assertEqual(stationarity_test_ratio({"gradient_ratio": 0.2, "no_floor_ratio": 1.1,
                                                  "floor_clause_ratio": 4.6}), 1.1)

    def test_fallbacks(self):
        self.assertEqual(stationarity_test_ratio({"gradient_ratio": 0.5}), 0.5)  # older runs, GPU Newton
        self.assertEqual(stationarity_test_ratio({"gradient_ratio": 0.5, "no_floor_ratio": None,
                                                  "floor_clause_ratio": 2.0}), 0.5)
        self.assertIsNone(stationarity_test_ratio({"gradient_ratio": None, "no_floor_ratio": 1.0,
                                                   "floor_clause_ratio": 1.0}))
        # The absolute test records the other ratios as 0
        self.assertEqual(stationarity_test_ratio({"gradient_ratio": 3.0, "no_floor_ratio": 0.0,
                                                  "floor_clause_ratio": 0.0}), 3.0)

    def test_structured_rows(self):
        dt = np.dtype([("gradient_ratio", np.float32), ("no_floor_ratio", np.float32),
                       ("floor_clause_ratio", np.float32)])
        rows = np.array([(0.2, 1.1, 4.5), (0.2, 37.5, 0.75), (2.0, 0.5, 0.5)], dtype=dt)
        np.testing.assert_allclose(stationarity_test_ratio(rows), [1.1, 0.75, 2.0], rtol=1e-6)
        self.assertAlmostEqual(stationarity_test_ratio(rows[1]), 0.75)
        old = np.array([(0.25,)], dtype=[("gradient_ratio", np.float32)])
        np.testing.assert_array_equal(stationarity_test_ratio(old), [0.25])


if __name__ == "__main__":
    unittest.main()

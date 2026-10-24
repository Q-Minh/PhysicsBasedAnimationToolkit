"""
Per-step quantities from a ``simulate.py`` trajectory, added to the profiling result of the same run.

They come from the saved frames, so they need neither a GPU nor changes to the solvers or the step loop:

- ``peak_speed`` (m/s): the largest node displacement between two consecutive frames divided by the time
  between them, given to every time step of that frame interval. Peaks shorter than a frame interval are
  smoothed out; run ``simulate.py`` with ``--fps`` at the step rate (1 / dt) for one value per step. Steps
  after the last frame get none.
- ``rms_speed`` (m/s): over the same frame interval, the mass-weighted RMS node speed sqrt(sum_i m_i |v_i|^2 /
  sum_i m_i) = sqrt(2 K / M), the overall motion (tied to the kinetic energy K, comparable across scenes since it
  doesn't grow with the node count). Over the free nodes only (``FemElastoDynamics/dmask`` 0, masses
  ``FemElastoDynamics/m``): Dirichlet nodes, static or kinematically driven, are left out, so static colliders
  don't dilute it, but anchored deformable bodies' free nodes count. Without masses in the output, unweighted.

Usage (from the repo root; adds the keys to the result in place, or writes ``-o``):

    python -m python.tools.sim.profiling.trajectory profile.json out.h5 [-o annotated.json]
"""

import argparse

import h5py
import numpy as np

from . import results


def frame_steps(h5f: h5py.File) -> np.ndarray:
    """The time step count at each saved frame, in frame order: the frames' ``t`` attribute, or, for
    outputs written before it existed, the frame schedule of ``state.should_save``."""
    frames = sorted(k for k in h5f["sim"].keys() if k.isdigit())
    if all("t" in h5f["sim"][k].attrs for k in frames):
        return np.array([int(h5f["sim"][k].attrs["t"]) for k in frames], dtype=np.int64)
    dt = float(h5f["params/Integration"].attrs["dt"])
    fps = float(h5f["params"].attrs["fps"])
    steps, t = [0], 0
    while len(steps) < len(frames):
        t += 1
        if int(t * dt * fps) > int((t - 1) * dt * fps):  # state.should_save
            steps.append(t)
    return np.array(steps, dtype=np.int64)


def frame_speeds(h5f: h5py.File) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Node speeds per frame interval: the peak and the mass-weighted RMS (see the module docstring).

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]: The time step count at each frame (F,), and ``peak_speed`` and
        ``rms_speed`` (m/s) between consecutive frames, each (F - 1,).
    """
    steps = frame_steps(h5f)
    dt = float(h5f["params/Integration"].attrs["dt"])
    frames = sorted(k for k in h5f["sim"].keys() if k.isdigit())
    fem = h5f.get("FemElastoDynamics")
    n_nodes = h5f["sim"][frames[0]]["x"].shape[0] if frames else 0
    m = np.asarray(fem["m"], dtype=np.float64) if fem is not None and "m" in fem else np.ones(n_nodes)
    free = np.asarray(fem["dmask"]) == 0 if fem is not None and "dmask" in fem else np.ones(n_nodes, dtype=bool)
    w = np.where(free, m, 0.0)
    w = w / max(float(w.sum()), 1e-30)
    out = {key: np.zeros(max(len(frames) - 1, 0)) for key in ("peak_speed", "rms_speed")}
    x0 = h5f["sim"][frames[0]]["x"][()].astype(np.float64) if frames else None
    for f in range(1, len(frames)):
        x1 = h5f["sim"][frames[f]]["x"][()].astype(np.float64)
        elapsed = (steps[f] - steps[f - 1]) * dt
        if elapsed > 0:
            speed2 = np.sum((x1 - x0) ** 2, axis=1) / elapsed**2
            out["peak_speed"][f - 1] = np.sqrt(speed2.max())
            out["rms_speed"][f - 1] = np.sqrt(w @ speed2)
        x0 = x1
    return steps, out


def annotate(result: dict, steps: np.ndarray, speeds: dict[str, np.ndarray]) -> int:
    """Set each step's speeds (:func:`frame_speeds`) from its frame interval: time step t (from count t to t + 1)
    lies in the interval of frames f - 1 and f with ``steps[f - 1] <= t < steps[f]``.

    Returns:
        int: The number of steps annotated.
    """
    n = 0
    for step in result["steps"]:
        f = int(np.searchsorted(steps, step["t"], side="right"))
        if 1 <= f < len(steps):
            for key, values in speeds.items():
                step[key] = float(values[f - 1])
            n += 1
        else:
            for key in speeds:
                step.pop(key, None)
    return n


def main():
    """Command line: annotate a profiling result with per-step quantities from its trajectory."""
    parser = argparse.ArgumentParser(description="Add per-step trajectory quantities to a profiling result")
    parser.add_argument("result", help="Profiling result .json (simulate.py --profile-out).")
    parser.add_argument("trajectory", help="The same run's simulate.py output .h5.")
    parser.add_argument("-o", "--out", default=None, help="Output .json (default: update the result).")
    args = parser.parse_args()
    result = results.load(args.result)
    with h5py.File(args.trajectory, "r") as h5f:
        dt = float(h5f["params/Integration"].attrs["dt"])
        if "dt" in result["metadata"] and not np.isclose(result["metadata"]["dt"], dt):
            raise ValueError(f"time steps differ: result dt {result['metadata']['dt']}, trajectory dt {dt}")
        steps, speeds = frame_speeds(h5f)
    n = annotate(result, steps, speeds)
    out = args.out or args.result
    results.save(result, out)
    print(f"Annotated {n} of {len(result['steps'])} steps with {', '.join(speeds)}, wrote {out}")


import unittest


class TestTrajectory(unittest.TestCase):

    def _write(self, path, with_t: bool):
        dt, fps = 0.01, 50.0  # a frame every 2 steps
        with h5py.File(path, "w") as f:
            f.create_group("params").attrs["fps"] = fps
            f["params"].create_group("Integration").attrs["dt"] = dt
            f.create_dataset("FemElastoDynamics/m", data=np.array([1.0, 3.0, 5.0]))
            f.create_dataset("FemElastoDynamics/dmask", data=np.array([0, 0, 1]))  # node 2 held
            for frame, t in enumerate([0, 2, 4, 6]):
                x = np.zeros((3, 3))
                x[1, 0] = 0.1 * frame**2  # node 1 moves 0.1, 0.3, 0.5 m per frame interval
                g = f.create_group(f"sim/{frame:08d}")
                g.create_dataset("x", data=x)
                if with_t:
                    g.attrs["t"] = t

    def test_peak_speed_per_step(self):
        import os
        import tempfile

        for with_t in (True, False):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "out.h5")
                self._write(path, with_t)
                with h5py.File(path, "r") as f:
                    steps, speeds = frame_speeds(f)
            self.assertEqual(steps.tolist(), [0, 2, 4, 6])
            np.testing.assert_allclose(speeds["peak_speed"], [5.0, 15.0, 25.0])
            # Node 1 (mass 3 of the free 4) moves, node 0 doesn't, node 2 is held: sqrt(3/4) of node 1's speed
            np.testing.assert_allclose(speeds["rms_speed"], np.sqrt(0.75) * np.array([5.0, 15.0, 25.0]))
            result = {"steps": [{"t": t} for t in range(8)]}
            self.assertEqual(annotate(result, steps, speeds), 6)
            np.testing.assert_allclose([s["peak_speed"] for s in result["steps"][:6]],
                                       [5.0, 5.0, 15.0, 15.0, 25.0, 25.0])
            np.testing.assert_allclose(result["steps"][2]["rms_speed"], np.sqrt(0.75) * 15.0)
            self.assertNotIn("peak_speed", result["steps"][6])
            self.assertNotIn("rms_speed", result["steps"][7])


if __name__ == "__main__":
    main()

"""
Per-step quantities from a ``simulate.py`` trajectory, added to the profiling result of the same run.

They come from the saved frames, so they need neither a GPU nor changes to the solvers or the step loop:

- ``peak_speed`` (m/s): the largest node displacement between two consecutive frames divided by the time
  between them, given to every time step of that frame interval. Peaks shorter than a frame interval are
  smoothed out; run ``simulate.py`` with ``--fps`` at the step rate (1 / dt) for one value per step. Steps
  after the last frame get none.

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


def peak_speeds(h5f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Peak node speed per frame interval.

    Returns:
        tuple[np.ndarray, np.ndarray]: The time step count at each frame (F,) and the largest node speed
        (m/s) between consecutive frames (F - 1,).
    """
    steps = frame_steps(h5f)
    dt = float(h5f["params/Integration"].attrs["dt"])
    frames = sorted(k for k in h5f["sim"].keys() if k.isdigit())
    speeds = np.zeros(max(len(frames) - 1, 0))
    x0 = h5f["sim"][frames[0]]["x"][()] if frames else None
    for f in range(1, len(frames)):
        x1 = h5f["sim"][frames[f]]["x"][()]
        elapsed = (steps[f] - steps[f - 1]) * dt
        speeds[f - 1] = np.linalg.norm(x1 - x0, axis=1).max() / elapsed if elapsed > 0 else 0.0
        x0 = x1
    return steps, speeds


def annotate(result: dict, steps: np.ndarray, speeds: np.ndarray) -> int:
    """Set each step's ``peak_speed`` from its frame interval: time step t (from count t to t + 1) lies
    in the interval of frames f - 1 and f with ``steps[f - 1] <= t < steps[f]``.

    Returns:
        int: The number of steps annotated.
    """
    n = 0
    for step in result["steps"]:
        f = int(np.searchsorted(steps, step["t"], side="right"))
        if 1 <= f < len(steps):
            step["peak_speed"] = float(speeds[f - 1])
            n += 1
        else:
            step.pop("peak_speed", None)
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
        steps, speeds = peak_speeds(h5f)
    n = annotate(result, steps, speeds)
    out = args.out or args.result
    results.save(result, out)
    print(f"Annotated {n} of {len(result['steps'])} steps with peak_speed, wrote {out}")


import unittest


class TestTrajectory(unittest.TestCase):

    def _write(self, path, with_t: bool):
        dt, fps = 0.01, 50.0  # a frame every 2 steps
        with h5py.File(path, "w") as f:
            f.create_group("params").attrs["fps"] = fps
            f["params"].create_group("Integration").attrs["dt"] = dt
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
                    steps, speeds = peak_speeds(f)
            self.assertEqual(steps.tolist(), [0, 2, 4, 6])
            np.testing.assert_allclose(speeds, [5.0, 15.0, 25.0])
            result = {"steps": [{"t": t} for t in range(8)]}
            self.assertEqual(annotate(result, steps, speeds), 6)
            np.testing.assert_allclose([s["peak_speed"] for s in result["steps"][:6]],
                                       [5.0, 5.0, 15.0, 15.0, 25.0, 25.0])
            self.assertNotIn("peak_speed", result["steps"][6])
            self.assertNotIn("peak_speed", result["steps"][7])


if __name__ == "__main__":
    main()

"""
Frames of benchmark problems for the paper's bar figures (``benchmark_plot.py --frames``): per benchmark, an example
problem's initial iterate x(0) and reference solution x* (the rest shape X for static problems, ``x_ref`` for dynamic
ones), rendered with Polyscope in the style of the profiling snapshots (``profiling/snapshots.py``): white background,
no ground plane, smooth shading without edges, SSAA 4, one camera, every image cropped to one box.

Two pairs of frames per benchmark, from the same sample (default 0) of two cases:

- ``error``: a homogeneous case (default ``Y1e7``), colored by the per-vertex error |x_i - x*_i| on 0 .. max_i
  |x_i(0) - x*_i| (viridis), as ``benchmark_viewer.py``'s default coloring; the reference is uniformly 0.
- ``material``: the heterogeneous case (default ``het``), colored per boundary face by its tetrahedron's first Lame
  parameter lambda, on the CPU editor's log10(lambda + 1) scale and colormap (``vbd/ui/material.py``). The range
  starts the materials' log span below the softest, so that it isn't drawn near-white on the white background.

Colorbars: Polyscope's (matcap) materials change the colormap's shades, so a plain colormap wouldn't match the
frames. Each pair carries the rendered shades instead: a flat horizontal strip through the camera's target, colored
by a ramp over the pair's range and rendered from the frames' camera, shaded as the body's undeformed top face, read
back along its length. Its points are at one depth (the camera looks along the y-z plane), so screen x is linear in
the value.

Writes per benchmark ``<prefix>_<benchmark>_{error,material}_{initial,reference}.png`` and ``<prefix>_<benchmark>.json``
(the cases, sample, color ranges, colorbar shades and image names), which ``benchmark_plot.py --frames`` reads.
Polyscope has no headless backend on Windows: rendering opens a window briefly.

Usage (from the repo root; ``--dataset`` when the results file's ``dataset`` attribute doesn't resolve, e.g. from a
worktree):

    python -m python.tools.sim.benchmark_frames path/to/results.h5 -o out/figures/trampolines_frames \
        [--dataset path/to/PhysicsBasedAnimationToolkit/data/benchmarks/trampolines.h5] [--benchmarks static ...] \
        [--error-case Y1e7] [--material-case het] [--sample 0] [--material clay] [--elev 40] [--distance 1.4]         [--size 1600 1200] [--max-width 900]

Tests: ``python -m unittest python.tools.sim.benchmark_frames``.
"""

import argparse
import json
import os

import h5py
import numpy as np

from . import benchmark_results as results
from .benchmark_viewer import FEM_GROUP, MESH_GROUP, boundary_triangles, problem_fem_group
from .profiling.snapshots import crop_box
from ..vbd.ui import material

ERROR_CMAP = "viridis"  # benchmark_viewer.py's error coloring
RAMP_SAMPLES = 256  # colorbar shades per pair
EDGE_TRIM = 3  # strip columns dropped at each end (antialiased against the background)


def material_range(shown: np.ndarray) -> tuple[float, float]:
    """The material coloring's range: from the values' span below their minimum, so the softest material is the
    colormap's middle shade rather than its lightest (near-white on white), to their maximum."""
    lo, hi = float(np.min(shown)), float(np.max(shown))
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return lo - span, hi


def strip_shades(image: np.ndarray, n: int = RAMP_SAMPLES, trim: int = EDGE_TRIM, background: float = 1.0) -> np.ndarray:
    """(n, 3) RGB in [0, 1]: a rendered ramp strip's colors from its left to its right end, evenly spaced. Per image
    column of the strip (non-background pixels), the mean over its middle rows; ``trim`` columns are dropped at each
    end."""
    rgb = image[..., :3]
    mask = np.any(np.abs(rgb - background) > 1e-3, axis=2)
    cols = np.nonzero(mask.any(axis=0))[0]
    if cols.size <= 2 * trim + 1:
        raise ValueError("the ramp strip isn't visible in its render")
    cols = cols[trim:cols.size - trim]
    shades = []
    for c in cols:
        rows = np.nonzero(mask[:, c])[0]
        middle = rows[rows.size // 4:max(rows.size // 4 + 1, 3 * rows.size // 4)]
        shades.append(rgb[middle, c].mean(axis=0))
    shades = np.array(shades)
    t = np.linspace(0.0, 1.0, cols.size)
    return np.stack([np.interp(np.linspace(0.0, 1.0, n), t, shades[:, k]) for k in range(3)], axis=1)


def downsample(image: np.ndarray, max_width: int) -> np.ndarray:
    """``image`` reduced by the smallest integer factor that makes it at most ``max_width`` wide (block means)."""
    k = max(1, -(-image.shape[1] // max_width))
    h, w = image.shape[0] // k * k, image.shape[1] // k * k
    return image[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def load_problem(dataset: h5py.File, set_name: str, sample: int) -> dict:
    """The problem's boundary mesh, initial iterate, reference and per-face lambda."""
    fem = dataset[problem_fem_group(dataset, set_name)][FEM_GROUP]
    X = fem[f"{MESH_GROUP}/X"][()].T.astype(np.float64)
    T = fem[f"{MESH_GROUP}/E"][()].T
    faces, elements = boundary_triangles(X, T, return_elements=True)
    samples = dataset[set_name]["samples"]
    static = str(dataset[set_name].attrs["problem"]) == "static"
    lam = material.fem_parameters(fem["lamegU"][()], fem["egU"][()], len(T), fem["m"][()])[material.LAME_LAMBDA]
    return dict(X=X, faces=faces, lam=lam[elements], x0=samples["x"][sample].astype(np.float64),
                reference=X if static else samples["x_ref"][sample].astype(np.float64))


def main():
    parser = argparse.ArgumentParser(description="Polyscope frames of benchmark problems for benchmark_plot.py --frames.")
    parser.add_argument("results", help="Results file (.h5) from python -m python.tools.sim.benchmark.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output prefix: writes <prefix>_<benchmark>.json and its PNGs.")
    parser.add_argument("--dataset", default=None,
                        help="The dataset file the results were run on (default: the results file's dataset attribute).")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Benchmarks (default: all in the file).")
    parser.add_argument("--error-case", default="Y1e7", dest="error_case",
                        help="Case colored by the error (default: Y1e7).")
    parser.add_argument("--material-case", default="het", dest="material_case",
                        help="Case colored by Lame lambda (default: het).")
    parser.add_argument("--sample", type=int, default=0, help="Problem index in each case's set (default: 0).")
    parser.add_argument("--elev", type=float, default=40.0, help="Camera elevation above the body's plane (degrees).")
    parser.add_argument("--material", default="clay",
                        help="Polyscope material (default: clay, the profiling snapshots' Polyscope default).")
    parser.add_argument("--distance", type=float, default=1.4,
                        help="Camera distance in body widths along x (default: 1.05).")
    parser.add_argument("--size", type=int, nargs=2, default=(1600, 1200), metavar=("W", "H"),
                        help="Window size in pixels before cropping.")
    parser.add_argument("--max-width", type=int, default=900, dest="max_width",
                        help="Saved images at most this many pixels wide (integer downsampling of the crop).")
    args = parser.parse_args()

    import polyscope as ps

    with h5py.File(args.results, "r") as f:
        results.check_format(f, args.results)
        dataset = args.dataset or str(f.attrs["dataset"])
        benchmarks = args.benchmarks or [str(b) for b in f.attrs["benchmarks"]]
        sets = {(b, c): str(f[b][c].attrs["set"]) for b in benchmarks for c in (args.error_case, args.material_case)}
    if not os.path.isfile(dataset):
        raise SystemExit(f"Dataset {dataset!r} not found; pass its path with --dataset")

    ps.set_program_name("Benchmark frames")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("none")
    ps.set_SSAA_factor(4)
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_window_size(*args.size)
    ps.init()
    ps.set_window_size(*args.size)
    ps.set_build_default_gui_panels(False)

    def shot() -> np.ndarray:
        ps.frame_tick()
        return ps.screenshot_to_buffer(transparent_bg=False, include_UI=False).astype(np.float64) / 255

    renders, specs = {}, {}
    body = None
    with h5py.File(dataset, "r") as d:
        for b in benchmarks:
            error = load_problem(d, sets[b, args.error_case], args.sample)
            het = load_problem(d, sets[b, args.material_case], args.sample)
            if body is None:
                body = ps.register_surface_mesh("body", error["x0"], error["faces"], smooth_shade=True, edge_width=0.0,
                                                material=args.material)
                body.set_back_face_policy("identical")
                X = error["X"]
                lo, hi = X.min(axis=0), X.max(axis=0)
                center = (lo + hi) / 2
                top = float(hi[2])
                el = np.radians(args.elev)
                distance = args.distance * float(hi[0] - lo[0])
                ps.look_at(center + distance * np.array([0.0, -np.cos(el), np.sin(el)]), center)
            vmax = max(float(np.linalg.norm(error["x0"] - error["reference"], axis=1).max()), 1e-12)
            for state in ("initial", "reference"):
                x = error["x0"] if state == "initial" else error["reference"]
                body.update_vertex_positions(x)
                body.add_scalar_quantity("color", np.linalg.norm(x - error["reference"], axis=1), vminmax=(0.0, vmax),
                                         cmap=ERROR_CMAP, enabled=True)
                renders[b, "error", state] = shot()
            shown, options = material.display(material.LAME_LAMBDA, het["lam"])
            vminmax = material_range(shown)
            for state in ("initial", "reference"):
                body.update_vertex_positions(het["x0"] if state == "initial" else het["reference"])
                body.add_scalar_quantity("color", shown, defined_on="faces", vminmax=vminmax, cmap=options["cmap"],
                                         enabled=True)
                renders[b, "material", state] = shot()
            specs[b] = {
                "error": dict(case=args.error_case, sample=args.sample, quantity="error", vmin=0.0, vmax=vmax,
                              cmap=ERROR_CMAP),
                "material": dict(case=args.material_case, sample=args.sample, quantity="lambda", scale="log10(x + 1)",
                                 vmin=vminmax[0], vmax=vminmax[1], cmap=options["cmap"],
                                 values=sorted({float(f"{v:.4g}") for v in het["lam"]})),
            }
        # Colorbar shades: a flat strip along x through the camera's target, at the body's top face
        body.set_enabled(False)
        width = float(hi[0] - lo[0])
        u = np.linspace(-0.45 * width, 0.45 * width, RAMP_SAMPLES)
        strip_x = np.concatenate([np.stack([center[0] + u, np.full_like(u, center[1] + s * 0.02 * width),
                                            np.full_like(u, top)], axis=1) for s in (-1, 1)])
        n = RAMP_SAMPLES
        quads = np.array([[i, i + 1, n + i + 1, n + i] for i in range(n - 1)])
        strip = ps.register_surface_mesh("strip", strip_x, quads, smooth_shade=True, edge_width=0.0,
                                         material=args.material)
        strip.set_back_face_policy("identical")
        ramp = np.concatenate([np.linspace(0.0, 1.0, n)] * 2)
        for kind, cmap in (("error", ERROR_CMAP), ("material", material.lame_parameters_cmap())):
            strip.add_scalar_quantity("ramp", ramp, vminmax=(0.0, 1.0), cmap=cmap, enabled=True)
            shades = strip_shades(shot())
            for b in benchmarks:
                specs[b][kind]["shades"] = np.round(shades, 4).tolist()
    r0, r1, c0, c1 = crop_box(list(renders.values()))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    import matplotlib.pyplot as plt

    for (b, kind, state), image in renders.items():
        name = f"{os.path.basename(args.output)}_{b}_{kind}_{state}.png"
        plt.imsave(os.path.join(os.path.dirname(os.path.abspath(args.output)), name),
                   np.clip(downsample(image[r0:r1, c0:c1, :3], args.max_width), 0, 1))
        specs[b][kind][state] = name
    for b, spec in specs.items():
        path = f"{args.output}_{b}.json"
        with open(path, "w", encoding="utf-8") as out:
            json.dump(spec, out, indent=1)
        print(f"Wrote {path} and its PNGs")


import unittest


class TestBenchmarkFrames(unittest.TestCase):

    def test_material_range(self):
        lo, hi = material_range(np.array([6.0, 8.0, 8.0]))
        self.assertEqual((lo, hi), (4.0, 8.0))
        self.assertEqual(material_range(np.array([2.0, 2.0])), (0.0, 2.0))

    def test_strip_shades(self):
        image = np.ones((20, 40, 3))
        ramp = np.linspace(0.0, 1.0, 30)
        image[8:12, 5:35, 0] = ramp  # red rises left to right, green and blue at 0
        image[8:12, 5:35, 1:] = 0.0
        shades = strip_shades(image, n=5, trim=0)
        np.testing.assert_allclose(shades[:, 0], np.linspace(0.0, 1.0, 5), atol=1e-12)
        np.testing.assert_allclose(shades[:, 1:], 0.0)
        with self.assertRaises(ValueError):
            strip_shades(np.ones((5, 5, 3)))

    def test_downsample(self):
        image = np.arange(4 * 6 * 3, dtype=float).reshape(4, 6, 3)
        self.assertEqual(downsample(image, 6).shape, (4, 6, 3))
        small = downsample(image, 3)
        self.assertEqual(small.shape, (2, 3, 3))
        np.testing.assert_allclose(small[0, 0], image[:2, :2].mean(axis=(0, 1)))


if __name__ == "__main__":
    main()

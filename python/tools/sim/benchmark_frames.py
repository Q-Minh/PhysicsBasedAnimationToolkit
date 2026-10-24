"""
Frames of benchmark problems for the paper's bar figures (``benchmark_plot.py --frames``): per benchmark, an example
problem's initial iterate x(0) and reference solution x* (the rest shape X for static problems, ``x_ref`` for dynamic
ones), rendered with Polyscope in the style of the profiling snapshots (``profiling/snapshots.py``): white background,
no ground plane, smooth shading without edges, SSAA 4 (Polyscope's highest), one camera, every image cropped to one
box and reduced by an integer factor with block means to at most ``--max-width`` pixels (more anti-aliasing; the
default window and width give about 900 px per frame, some 750 dpi at the bar figures' frame width). The camera looks
at the center of the box around every rendered shape, from ``--elev`` above the front, far enough that the box's
bounding sphere fits the view (``--zoom`` scales that), so strongly deformed problems (e.g. a dataset sampled at a
higher energy) don't clip; a render that still touches the window's border is an error. The crop is the union of
every render's content, so it covers the most deformed one.

Two pairs of frames per benchmark, each from one sample of a case:

- ``error``: a homogeneous case (``--error-case``, default ``Y1e5`` for the static benchmark and ``Y1e7`` for the
  others), colored by the per-vertex error |x_i - x*_i| on 0 .. max_i |x_i(0) - x*_i| (viridis), as
  ``benchmark_viewer.py``'s default coloring; the reference is uniformly 0.
- ``material``: the heterogeneous case (``--material-case``, default ``het``), colored per boundary face by its
  tetrahedron's first Lame
  parameter lambda, on the CPU editor's log10(lambda + 1) scale and colormap (``vbd/ui/material.py``). The range
  starts the materials' log span below the softest, so that it isn't drawn near-white on the white background.

The options ``--error-case``, ``--material-case`` and ``--sample`` take ``[BENCHMARK=]VALUE`` and repeat: a benchmark's
own entry wins over the entries without one, later entries over earlier ones. ``--sample`` is a problem index in the
case's set or ``auto`` (default: index 0 for the static benchmark, ``auto`` for the others): for a dynamic set, the
sample whose start velocity reads best as arrows, the largest kinetic energy 1/2 v^T M v times the velocity's
coherence |sum_i m_i v_i| / sum_i m_i |v_i| (1 when every node moves the same way); a static set has no velocity,
so it needs an index.

Velocity arrows (dynamic benchmarks, initial iterates only): the start velocity v (the set's ``samples/v``) at nodes of
the body's upward-facing surface, one per cell of a grid of ``ARROW_SPACING`` body widths over the rest shape, drawn
from the node as v times a scale shared by the figure's two pairs, so that its longest arrow is ``ARROW_LENGTH`` body
widths: an arrow's length is |v| times the JSON's ``arrow_seconds`` (the distance covered at that velocity in that
time).

Colorbars: Polyscope's (matcap) materials change the colormap's shades, so a plain colormap wouldn't match the
frames. Each pair carries the rendered shades instead: a flat horizontal strip through the camera's target, colored
by a ramp over the pair's range and rendered from the frames' camera, shaded as the body's undeformed top face, read
back along its length. Its points are at one depth (the camera looks along the y-z plane), so screen x is linear in
the value.

Writes per benchmark ``<prefix>_<benchmark>_{error,material}_{initial,reference}.png`` and ``<prefix>_<benchmark>.json``
(the cases, samples, color ranges, arrow scale, colorbar shades and image names), which ``benchmark_plot.py
--frames`` reads.
Nothing here is specific to the trampolines: cases, sizes and ranges come from the results file and the dataset.
Polyscope has no headless backend on Windows: rendering opens a window briefly.

After a benchmark re-run (e.g. a dataset sampled at another energy), regenerate the frames from the new results file,
then the bar figures with ``--frames`` (``benchmark_plot.py``): the color ranges, colorbar ticks, camera and crop follow
the new data.

Usage (from the repo root; ``--dataset`` when the results file's ``dataset`` attribute doesn't resolve, e.g. from a
worktree):

    python -m python.tools.sim.benchmark_frames path/to/results.h5 -o out/figures/trampolines_frames \
        [--dataset path/to/PhysicsBasedAnimationToolkit/data/benchmarks/trampolines.h5] [--benchmarks static ...] \
        [--error-case static=Y1e5 --error-case Y1e7] [--material-case het] [--sample static=0 --sample auto] \
        [--material clay] [--elev 55] [--zoom 1] [--size 2400 1400] [--max-width 1000]

The window must fit the screen (Polyscope's window, hence its screenshots, is clamped to it).

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
ARROW_LENGTH = 0.3  # body widths: a figure's longest velocity arrow
ARROW_SPACING = 0.11  # body widths: the grid over the rest shape that picks the arrows' nodes, one per cell
ARROW_RADIUS = 0.006  # Polyscope's relative radius of the arrows
ARROW_COLOR = (0.2, 0.2, 0.2)  # dark gray, neutral against viridis and the blues


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


def camera_distance(lo: np.ndarray, hi: np.ndarray, aspect: float, fov: float = 45.0, zoom: float = 1.0) -> float:
    """The distance from the box lo..hi's center at which its bounding sphere fits a perspective view of vertical
    field of view ``fov`` (degrees, Polyscope's default) and width / height ``aspect``, divided by ``zoom``."""
    half = np.radians(fov) / 2
    half = min(half, np.arctan(np.tan(half) * aspect))
    return float(0.5 * np.linalg.norm(np.asarray(hi) - np.asarray(lo)) / np.sin(half) / zoom)


def touches_border(image: np.ndarray, background: float = 1.0) -> bool:
    """Whether a render has content on its outermost pixels (the shape may be clipped)."""
    content = np.any(np.abs(image[..., :3] - background) > 1e-3, axis=2)
    return bool(content[0].any() or content[-1].any() or content[:, 0].any() or content[:, -1].any())


def downsample(image: np.ndarray, max_width: int) -> np.ndarray:
    """``image`` reduced by the smallest integer factor that makes it at most ``max_width`` wide (block means)."""
    k = max(1, -(-image.shape[1] // max_width))
    h, w = image.shape[0] // k * k, image.shape[1] // k * k
    return image[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def per_benchmark(values: list[str], benchmark: str) -> str:
    """A repeatable ``[BENCHMARK=]VALUE`` option's value for ``benchmark``: its last own entry, else the last entry
    without a benchmark."""
    own = [v.partition("=")[2] for v in values if v.partition("=")[0] == benchmark and "=" in v]
    default = [v for v in values if "=" not in v]
    if not own and not default:
        raise SystemExit(f"No value for {benchmark} in {values}")
    return (own or default)[-1]


def pick_sample(m: np.ndarray, v: np.ndarray) -> int:
    """The sample whose velocity reads best as arrows: the largest kinetic energy 1/2 v^T M v times the velocity's
    coherence |sum_i m_i v_i| / sum_i m_i |v_i|. ``m``: (N,) lumped masses, ``v``: (S, N, 3) velocities."""
    speed = np.linalg.norm(v, axis=2)
    kinetic = 0.5 * np.einsum("i,si->s", m, speed**2)
    coherence = np.linalg.norm(np.einsum("i,sij->sj", m, v), axis=1) / np.maximum(np.einsum("i,si->s", m, speed), 1e-30)
    return int(np.argmax(kinetic * coherence))


def arrow_nodes(X: np.ndarray, faces: np.ndarray, spacing: float) -> np.ndarray:
    """Nodes for velocity arrows: of the boundary faces facing up (+z) at rest, one node per cell of a grid of
    ``spacing`` over the rest positions' x and y (the node nearest the cell's center)."""
    a, b, c = (X[faces[:, i]] for i in range(3))
    up = np.cross(b - a, c - a)[:, 2] > 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    nodes = np.unique(faces[up])
    cells = np.floor(X[nodes, :2] / spacing)
    offset = np.linalg.norm(X[nodes, :2] - (cells + 0.5) * spacing, axis=1)
    order = np.lexsort((offset, cells[:, 1], cells[:, 0]))
    _, first = np.unique(cells[order], axis=0, return_index=True)
    return nodes[order[first]]


def load_problem(dataset: h5py.File, set_name: str, sample: str) -> dict:
    """The problem (``sample``: an index or ``auto``, see the module docstring): its boundary mesh, initial iterate,
    reference, per-face lambda and, for dynamic problems, start velocity."""
    fem = dataset[problem_fem_group(dataset, set_name)][FEM_GROUP]
    X = fem[f"{MESH_GROUP}/X"][()].T.astype(np.float64)
    T = fem[f"{MESH_GROUP}/E"][()].T
    faces, elements = boundary_triangles(X, T, return_elements=True)
    samples = dataset[set_name]["samples"]
    static = str(dataset[set_name].attrs["problem"]) == "static"
    if sample == "auto":
        if static:
            raise SystemExit(f"{set_name}: --sample auto picks by velocity; a static set needs an index")
        index = pick_sample(fem["m"][()].ravel().astype(np.float64), samples["v"][()].astype(np.float64))
    else:
        index = int(sample)
    lam = material.fem_parameters(fem["lamegU"][()], fem["egU"][()], len(T), fem["m"][()])[material.LAME_LAMBDA]
    return dict(X=X, faces=faces, lam=lam[elements], sample=index, x0=samples["x"][index].astype(np.float64),
                reference=X if static else samples["x_ref"][index].astype(np.float64),
                v=None if static else samples["v"][index].astype(np.float64))


def main():
    parser = argparse.ArgumentParser(description="Polyscope frames of benchmark problems for benchmark_plot.py --frames.")
    parser.add_argument("results", help="Results file (.h5) from python -m python.tools.sim.benchmark.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output prefix: writes <prefix>_<benchmark>.json and its PNGs.")
    parser.add_argument("--dataset", default=None,
                        help="The dataset file the results were run on (default: the results file's dataset attribute).")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Benchmarks (default: all in the file).")
    parser.add_argument("--error-case", action="append", default=["static=Y1e5", "Y1e7"], dest="error_case",
                        help="[BENCHMARK=]CASE colored by the error, repeatable (default: Y1e5 for static, else Y1e7).")
    parser.add_argument("--material-case", action="append", default=["het"], dest="material_case",
                        help="[BENCHMARK=]CASE colored by Lame lambda, repeatable (default: het).")
    parser.add_argument("--sample", action="append", default=["static=0", "auto"],
                        help="[BENCHMARK=]INDEX or auto, the problem of each case's set, repeatable (default: 0 for "
                        "static, else auto; see the module docstring).")
    parser.add_argument("--elev", type=float, default=55.0, help="Camera elevation above the body's plane (degrees).")
    parser.add_argument("--material", default="clay",
                        help="Polyscope material (default: clay, the profiling snapshots' Polyscope default).")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Camera zoom: 1 fits the bounding sphere of every rendered shape (default: 1).")
    parser.add_argument("--size", type=int, nargs=2, default=(2400, 1400), metavar=("W", "H"),
                        help="Window size in pixels before cropping (at most the screen's).")
    parser.add_argument("--max-width", type=int, default=1000, dest="max_width",
                        help="Saved images at most this many pixels wide (integer downsampling of the crop).")
    args = parser.parse_args()

    import polyscope as ps

    with h5py.File(args.results, "r") as f:
        results.check_format(f, args.results)
        dataset = args.dataset or str(f.attrs["dataset"])
        benchmarks = args.benchmarks or [str(b) for b in f.attrs["benchmarks"]]
        cases = {b: (per_benchmark(args.error_case, b), per_benchmark(args.material_case, b)) for b in benchmarks}
        sets = {(b, c): str(f[b][c].attrs["set"]) for b in benchmarks for c in cases[b]}
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

    def shot(check: bool = True) -> np.ndarray:
        ps.frame_tick()
        image = ps.screenshot_to_buffer(transparent_bg=False, include_UI=False).astype(np.float64) / 255
        if check and touches_border(image):
            raise SystemExit("A frame touches the window's border (clipped): lower --zoom")
        return image

    renders, specs = {}, {}
    with h5py.File(dataset, "r") as d:
        problems = {b: tuple(load_problem(d, sets[b, c], per_benchmark(args.sample, b)) for c in cases[b])
                    for b in benchmarks}
    # Velocity arrows: per dynamic benchmark, one scale for both pairs, the longest arrow ARROW_LENGTH body widths
    arrows = {}
    for b, pair in problems.items():
        if pair[0]["v"] is None:
            continue
        width = float(np.ptp(pair[0]["X"][:, 0]))
        nodes = [arrow_nodes(p["X"], p["faces"], ARROW_SPACING * width) for p in pair]
        top = max(float(np.linalg.norm(p["v"][n], axis=1).max()) for p, n in zip(pair, nodes))
        arrows[b] = (nodes, ARROW_LENGTH * width / max(top, 1e-12))
    # One camera for every frame: at the center of the box around every rendered shape (arrows included), its
    # bounding sphere in view
    shapes = [p[k] for pair in problems.values() for p in pair for k in ("x0", "reference")]
    shapes += [p["x0"][n] + seconds * p["v"][n] for b, (nodes, seconds) in arrows.items()
               for p, n in zip(problems[b], nodes)]
    lo = np.min([x.min(axis=0) for x in shapes], axis=0)
    hi = np.max([x.max(axis=0) for x in shapes], axis=0)
    center = (lo + hi) / 2
    el = np.radians(args.elev)
    distance = camera_distance(lo, hi, args.size[0] / args.size[1], zoom=args.zoom)
    ps.look_at(center + distance * np.array([0.0, -np.cos(el), np.sin(el)]), center)
    first = problems[benchmarks[0]][0]
    body = ps.register_surface_mesh("body", first["x0"], first["faces"], smooth_shade=True, edge_width=0.0,
                                    material=args.material)
    body.set_back_face_policy("identical")

    def show_arrows(b: str, column: int, state: str):
        """The velocity arrows of a dynamic benchmark's initial iterate, else none."""
        if b not in arrows or state != "initial":
            ps.remove_point_cloud("arrows", error_if_absent=False)
            return
        nodes, seconds = arrows[b]
        problem = problems[b][column]
        n = nodes[column]
        cloud = ps.register_point_cloud("arrows", problem["x0"][n], radius=1e-6, color=ARROW_COLOR)
        cloud.add_vector_quantity("v", seconds * problem["v"][n], vectortype="ambient", radius=ARROW_RADIUS,
                                  color=ARROW_COLOR, enabled=True)

    for b in benchmarks:
        error, het = problems[b]
        vmax = max(float(np.linalg.norm(error["x0"] - error["reference"], axis=1).max()), 1e-12)
        for state in ("initial", "reference"):
            x = error["x0"] if state == "initial" else error["reference"]
            body.update_vertex_positions(x)
            body.add_scalar_quantity("color", np.linalg.norm(x - error["reference"], axis=1), vminmax=(0.0, vmax),
                                     cmap=ERROR_CMAP, enabled=True)
            show_arrows(b, 0, state)
            renders[b, "error", state] = shot()
        shown, options = material.display(material.LAME_LAMBDA, het["lam"])
        vminmax = material_range(shown)
        for state in ("initial", "reference"):
            body.update_vertex_positions(het["x0"] if state == "initial" else het["reference"])
            body.add_scalar_quantity("color", shown, defined_on="faces", vminmax=vminmax, cmap=options["cmap"],
                                     enabled=True)
            show_arrows(b, 1, state)
            renders[b, "material", state] = shot()
        specs[b] = {
            "error": dict(case=cases[b][0], sample=error["sample"], quantity="error", vmin=0.0, vmax=vmax,
                          cmap=ERROR_CMAP),
            "material": dict(case=cases[b][1], sample=het["sample"], quantity="lambda", scale="log10(x + 1)",
                             vmin=vminmax[0], vmax=vminmax[1], cmap=options["cmap"],
                             values=sorted({float(f"{v:.4g}") for v in het["lam"]})),
        }
        if b in arrows:
            for pair in specs[b].values():
                pair["arrow_seconds"] = arrows[b][1]
    ps.remove_point_cloud("arrows", error_if_absent=False)
    # Colorbar shades: a flat horizontal strip along x through the camera's target (so at one depth), facing up
    # as the body's undeformed top face
    body.set_enabled(False)
    width = float(hi[0] - lo[0])
    u = np.linspace(-0.45 * width, 0.45 * width, RAMP_SAMPLES)
    strip_x = np.concatenate([np.stack([center[0] + u, np.full_like(u, center[1] + s * 0.02 * width),
                                        np.full_like(u, center[2])], axis=1) for s in (-1, 1)])
    n = RAMP_SAMPLES
    quads = np.array([[i, i + 1, n + i + 1, n + i] for i in range(n - 1)])
    strip = ps.register_surface_mesh("strip", strip_x, quads, smooth_shade=True, edge_width=0.0,
                                     material=args.material)
    strip.set_back_face_policy("identical")
    ramp = np.concatenate([np.linspace(0.0, 1.0, n)] * 2)
    for kind, cmap in (("error", ERROR_CMAP), ("material", material.lame_parameters_cmap())):
        strip.add_scalar_quantity("ramp", ramp, vminmax=(0.0, 1.0), cmap=cmap, enabled=True)
        shades = strip_shades(shot(check=False))
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

    def test_camera(self):
        lo, hi = np.array([-2.0, -2.0, -0.05]), np.array([2.0, 2.0, 0.05])
        d = camera_distance(lo, hi, aspect=2.0)
        radius = 0.5 * np.linalg.norm(hi - lo)
        self.assertAlmostEqual(d, radius / np.sin(np.radians(22.5)))  # the vertical field of view is the narrower
        self.assertGreater(camera_distance(lo, hi, aspect=0.5), d)  # a tall window: the horizontal one
        self.assertAlmostEqual(camera_distance(2 * lo, 2 * hi, aspect=2.0), 2 * d)  # larger shapes, farther
        image = np.ones((6, 8, 3))
        image[2:4, 3:5] = 0.2
        self.assertFalse(touches_border(image))
        image[0, 4] = 0.2
        self.assertTrue(touches_border(image))

    def test_options_and_picks(self):
        values = ["static=Y1e5", "Y1e7", "dynamic_h0.01=Y1e6"]
        self.assertEqual([per_benchmark(values, b) for b in ("static", "dynamic_h0.01", "dynamic_h0.0001")],
                         ["Y1e5", "Y1e6", "Y1e7"])
        self.assertEqual(per_benchmark(values + ["Y1e8"], "dynamic_h0.0001"), "Y1e8")  # later entries win
        with self.assertRaises(SystemExit):
            per_benchmark(["static=0"], "dynamic_h0.01")
        m = np.ones(3)
        v = np.zeros((3, 3, 3))
        v[0, :, 2] = [1.0, -1.0, 1.0]  # fast but incoherent
        v[1, :, 2] = 0.9  # a little slower, coherent: K 1.215 x 1 beats 1.5 x 1/3
        v[2, :, 2] = 0.1  # coherent, slow
        self.assertEqual(pick_sample(m, v), 1)
        # A flat 2 x 1 slab: its top face's nodes, one per cell
        X = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0], [0, 0, 1], [2, 0, 1], [0, 1, 1], [2, 1, 1.0]])
        top = np.array([[4, 5, 7], [4, 7, 6]])  # counter-clockwise from above
        bottom = np.array([[0, 3, 1], [0, 2, 3]])
        self.assertEqual(sorted(arrow_nodes(X, np.concatenate([top, bottom]), 10.0)), [7])  # one cell, centered at (5, 5)
        self.assertEqual(sorted(arrow_nodes(X, np.concatenate([top, bottom]), 1.5)), [6, 7])  # two cells along x

    def test_downsample(self):
        image = np.arange(4 * 6 * 3, dtype=float).reshape(4, 6, 3)
        self.assertEqual(downsample(image, 6).shape, (4, 6, 3))
        small = downsample(image, 3)
        self.assertEqual(small.shape, (2, 3, 3))
        np.testing.assert_allclose(small[0, 0], image[:2, :2].mean(axis=(0, 1)))


if __name__ == "__main__":
    main()

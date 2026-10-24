"""
Records the benchmark viewer's side-by-side solver comparisons (``benchmark_viewer.py``) to videos, one MP4 per
benchmark: every trajectory sample of every case in turn, each played over all its frames. With
``--samples-per-case N``, a case's N samples with the largest initial error (its largest vertex error |x(0) - x*|,
the color scale's; read from the first frames only), in sample order; the choice is printed with the errors.

Each clip is what the viewer shows with its defaults: the solvers' copies side by side, labeled with their names
and e/e0, colored by the per-vertex error |x - x*| from 0 to the problem's initial error, or a fraction of it
per problem (``--error-fraction``, as the viewer's: values above it saturate, so a stalled solve stays bright);
plus a title (benchmark, the case's material as in the paper figures, e.g. "E = 10⁵ Pa", sample, solver time
over T_N and the slowdown) and the color scale with its fraction.
Each clip opens on the start state, held for ``--hold-seconds`` (default 1 s; 0: no hold) at t = 0 with the
caption "Initial", so viewers see the problem before the solvers change the shape; the caption goes when the
solve starts. Frames are evenly spaced in solver time over the problem's budget and play at
``F / video_seconds`` frames per second (24 fps for the full run's 120 frames in 5 s), so every clip shows its
solve slowed down by its ``playback_factor``. The viewer's translucent overlays are off: near convergence a
reference overlay coincides with the solver meshes and flickers (z-fighting), and the colors already show the
distance to the reference.

Text: the titles, labels and captions are in Linux Biolinum, the paper figures' sans serif, drawn with Pillow onto
each captured frame (sizes in video pixels, for 1080p and scaled with the height); the font comes from
``--font-dir``, else ``PBAT_FONT_DIR``, else the installed fonts (``benchmark_plot.register_paper_fonts``).
Without it, a warning, and imgui's default font draws them in the window.

Polyscope has no headless backend on Windows: the recorder opens a window and drives it itself, so leave it open
and unminimized until it finishes. Frames go straight from the window to ffmpeg's stdin (no image files). ffmpeg
is found from ``--ffmpeg``, then the ``FFMPEG`` environment variable, then the ``imageio-ffmpeg`` package's
binary, then ``PATH``.

Usage:
    python -m python.tools.sim.benchmark_video path/to/results.h5 -o out/videos/trampolines \
        [--dataset path/to/trampolines.h5] [--benchmarks static ...] [--cases Y1e6 ...] [--samples 0 4 ...] \
        [--samples-per-case 3] \
        [--width 1920 --height 1080] [--crf 18] [--ffmpeg path/to/ffmpeg] [--hold-seconds 1] \
        [--font-dir path/to/fonts] [--error-fraction static/Y1e5=0.2 ...]

Writes ``<prefix>_<benchmark>.mp4`` per benchmark.

Tests: ``python -m unittest python.tools.sim.benchmark_video``.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import time
import warnings

import numpy as np

from .benchmark_plot import benchmark_title, case_label, register_paper_fonts
from .benchmark_viewer import (COLOR_MODES, FEM_GROUP, MESH_GROUP, ErrorFractions, Viewer, _rgb,
                               add_error_fraction_argument, boundary_triangles, check_error_fractions, display_name,
                               problem_fem_group, slowdown_text, solver_color, solver_order)

TEXT_SCALE = 2.0  # the labels and titles, over imgui's default font, for a 1080p video (without the paper font)
TEXT_PIXELS = 34  # the paper font's size in pixels at a 1080-pixel video height, scaled with it
TEXT_COLOR = (0.1, 0.1, 0.1, 1.0)
HOLD_SECONDS = 1.0  # each clip's start state, held before the solve


SUPERSCRIPTS = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")


def plain_label(label: str, unicode: bool = True) -> str:
    """A paper figure's label (math as Matplotlib mathtext, e.g. ``$E \\in \\{10^{6}, 10^{8}\\}$ Pa``) as plain text:
    ``E ∈ {10⁶, 10⁸} Pa``, or without ``unicode`` (imgui's default font), ``E in {10^6, 10^8} Pa``. Lines join
    with commas."""
    text = label.replace("$", "").replace("\n", ", ").replace("\\{", "{").replace("\\}", "}")
    text = text.replace("\\in", "∈" if unicode else "in").replace("\\times", "×" if unicode else "x")
    return re.sub(r"\^\{([^}]*)\}", lambda m: m.group(1).translate(SUPERSCRIPTS) if unicode else f"^{m.group(1)}",
                  text)


def paper_font_file(font_dir: str | None = None) -> str | None:
    """Linux Biolinum's file (``benchmark_plot.register_paper_fonts``), or None with a warning."""
    from matplotlib import font_manager

    sans = register_paper_fonts(font_dir)[0]
    if sans is None:
        warnings.warn("Linux Biolinum not found (--font-dir, PBAT_FONT_DIR or an installed font): the video text "
                      "uses imgui's default font")
        return None
    return font_manager.findfont(font_manager.FontProperties(family=sans), fallback_to_default=False)


def initial_errors(path: str, benchmark: str, case: str, samples: list[int],
                   dataset: str | None = None) -> dict[int, float]:
    """The samples' initial errors, as the viewer's "Initial error" color scale (``Viewer.initial_error``): the
    largest |x_i(0) - x*_i| over the boundary vertices, x(0) being the first frame of the first solver with the
    sample's trajectory (every solver starts there). Reads only those frames and, for dynamic problems, x_ref."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f[benchmark][case]
        dataset = dataset or str(f.attrs["dataset"])
        set_name, problem = str(g.attrs["set"]), str(g.attrs["problem"])
        solvers = solver_order(f, g)
        x0 = {}
        for sample in samples:
            key = f"trajectories/{sample:06d}/x"
            x0[sample] = g[next(solver for solver in solvers if key in g[solver])][key][0]
    with h5py.File(dataset, "r") as d:
        mesh = d[problem_fem_group(d, set_name)][FEM_GROUP][MESH_GROUP]
        X = mesh["X"][()].T.astype(np.float64)
        vertices = np.unique(boundary_triangles(X, mesh["E"][()].T))
        references = None if problem == "static" else d[set_name]["samples"]["x_ref"]
        errors = {}
        for sample, x in x0.items():
            reference = X if references is None else references[sample].astype(np.float64)
            errors[sample] = float(np.linalg.norm(x[vertices].astype(np.float64) - reference[vertices], axis=1).max())
    return errors


def top_samples(errors: dict[int, float], n: int) -> list[int]:
    """The ``n`` samples of ``errors`` with the largest initial errors (ties: the lower sample), in sample order."""
    return sorted(sorted(errors, key=lambda sample: (-errors[sample], sample))[:n])


def recorded_samples(path: str, cases: list, samples: list[int] | None = None, samples_per_case: int | None = None,
                     dataset: str | None = None) -> list[list[int]]:
    """Per case of ``cases`` (``benchmark_viewer.list_cases``), the samples to record: those in ``samples`` (all
    without it), then of them the ``samples_per_case`` with the largest initial errors (:func:`top_samples`), each
    case's choice printed with the errors."""
    out = []
    for benchmark, case, _, case_samples in cases:
        candidates = [s for s in case_samples if samples is None or s in samples]
        if samples_per_case is None or not candidates:
            out.append(candidates)
            continue
        errors = initial_errors(path, benchmark, case, candidates, dataset)
        chosen = top_samples(errors, samples_per_case)
        print(f"{benchmark}/{case}: samples " + ", ".join(f"{s} ({errors[s]:.4g} m)" for s in chosen) +
              f", the {len(chosen)} of {len(candidates)} with the largest initial errors", flush=True)
        out.append(chosen)
    return out


def find_ffmpeg(path: str | None) -> str:
    """ffmpeg: ``path``, else the FFMPEG environment variable, else imageio-ffmpeg's binary, else PATH."""
    candidates = [path, os.environ.get("FFMPEG")]
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        pass
    candidates.append(shutil.which("ffmpeg"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise SystemExit("ffmpeg not found: pass --ffmpeg, set FFMPEG, or pip install imageio-ffmpeg")


class Recorder:
    """Drives a :class:`Viewer` frame by frame and draws the labels and titles a video needs, without the panel:
    composited onto each captured frame in the font of ``font_file`` (Pillow), or without one, drawn by imgui."""

    def __init__(self, viewer: Viewer, hold_seconds: float = HOLD_SECONDS, font_file: str | None = None):
        self.viewer = viewer
        self.hold_seconds = hold_seconds
        self.font_file = font_file
        self.fonts = {}  # Pillow fonts by pixel size
        self.texts = []  # with font_file, this frame's (x, y, text, color, center) in window coordinates
        self.labels = {}  # the cases' titles (plain_label of benchmark_plot.case_label), by (benchmark, case)
        self.initial = False  # the hold: the start state, captioned "Initial"

    def font(self, height: int):
        """The Pillow font for a video ``height`` pixels tall."""
        from PIL import ImageFont

        size = max(round(TEXT_PIXELS * height / 1080), 1)
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_file, size)
        return self.fonts[size]

    def case_label(self) -> str:
        """The selected case's material as in the paper figures (``benchmark_plot.case_label``), as plain text."""
        import h5py

        v, s = self.viewer, self.viewer.selection
        key = (s.benchmark, s.case)
        if key not in self.labels:
            with h5py.File(v.path, "r") as f:
                label = case_label(f[s.benchmark][s.case], s.case, v.dataset)
            self.labels[key] = plain_label(label, unicode=self.font_file is not None)
        return self.labels[key]

    def line_height(self) -> float:
        """The titles' and labels' line spacing in window coordinates."""
        import polyscope as ps
        import polyscope.imgui as imgui

        if self.font_file is None:
            return imgui.GetFontSize() * 1.25
        return 1.25 * TEXT_PIXELS * ps.get_window_size()[1] / 1080

    def hold_frames(self) -> int:
        """The hold's video frames at the selected sample's frame rate."""
        s = self.viewer.selection
        return round(self.hold_seconds * s.n_frames / s.video_seconds)

    def draw_text(self, draw, x: float, y: float, text: str, color=TEXT_COLOR, center: bool = False):
        import polyscope.imgui as imgui

        if self.font_file is not None:  # composited after the capture
            self.texts.append((x, y, text, color, center))
            return
        width = imgui.CalcTextSize(text)[0]  # imgui's global font scale is TEXT_SCALE (the callback sets it)
        draw.AddText((x - width / 2 if center else x, y), imgui.GetColorU32(color), text)

    def composite(self, image: np.ndarray) -> np.ndarray:
        """The captured RGB ``image`` with this frame's texts drawn on it in the paper font."""
        import polyscope as ps
        from PIL import Image, ImageDraw

        picture = Image.fromarray(image)
        draw = ImageDraw.Draw(picture)
        font = self.font(picture.height)
        scale = picture.width / ps.get_window_size()[0]  # window to framebuffer pixels
        for x, y, text, color, center in self.texts:
            x, y = x * scale, y * scale
            if center:
                x -= font.getlength(text) / 2
            draw.text((x, y), text, font=font, fill=tuple(round(255 * c) for c in color[:3]))
        self.texts = []
        return np.asarray(picture)

    def capture(self) -> np.ndarray:
        """The window's RGB image after a frame tick, with the texts in the paper font."""
        import polyscope as ps

        image = np.ascontiguousarray(ps.screenshot_to_buffer(transparent_bg=False, include_UI=True)[..., :3])
        return image if self.font_file is None else self.composite(image)

    def callback(self):
        """Per frame: the meshes, then the titles and each copy's label (name and e/e0) on the background."""
        import polyscope as ps
        import polyscope.imgui as imgui

        v = self.viewer
        s = v.selection
        v.update_meshes()
        imgui.GetIO().FontGlobalScale = TEXT_SCALE
        draw = imgui.GetForegroundDrawList()
        self.texts = []
        line = self.line_height()
        self.draw_text(draw, 24, 20, f"{benchmark_title(s.benchmark)}: {self.case_label()}, sample {s.sample}")
        self.draw_text(draw, 24, 20 + line, f"Solver time {v.t / s.T_N:.3f} T_N  (T_N = {s.T_N:.4g} s, "
                                            f"{slowdown_text(s.playback_factor)})")
        self.draw_text(draw, 24, 20 + 2 * line, f"Color: {COLOR_MODES[v.color_mode]}, {v.scale_text()}")
        if self.initial:
            self.draw_text(draw, ps.get_window_size()[0] / 2, 20, "Initial", center=True)
        params = ps.get_view_camera_parameters()
        V = np.asarray(params.get_view_mat(), dtype=np.float64)
        f = 1.0 / math.tan(math.radians(params.get_fov_vertical_deg()) / 2)
        aspect = params.get_aspect()
        width, height = ps.get_window_size()
        lo, hi = s.X.min(axis=0), s.X.max(axis=0)
        below = np.array([(lo[0] + hi[0]) / 2, lo[1], lo[2]])
        names = [tr.solver for tr in s.trajectories]
        for tr in s.trajectories:
            p = V @ np.append(below + v.offsets[tr.solver], 1.0)
            if p[2] >= 0:
                continue
            x = (f / aspect * p[0] / -p[2] + 1) / 2 * width
            y = (1 - f * p[1] / -p[2]) / 2 * height + 12
            self.draw_text(draw, x, y, display_name(tr.solver), (*_rgb(solver_color(tr.solver, names)), 1.0), center=True)
            frame = v.shown.get(tr.solver, 0)
            self.draw_text(draw, x, y + line, f"e/e0 {tr.relative_error[frame]:.1e}", center=True)

    def frames(self, case_index: int, sample_index: int):
        """Selects a sample and yields its frames' RGB images: the hold (one captured image, repeated), then one per
        viewer frame."""
        import polyscope as ps

        v = self.viewer
        v.select(case_index, sample_index)
        if v.error:
            raise SystemExit(v.error)
        s = v.selection
        for overlay in s.overlays:  # off: a converged solve's overlay z-fights with its mesh
            v.show_overlay[overlay.name] = False
        v.dirty = True
        hold = self.hold_frames()
        if hold > 0:
            v.frame, v.t = 0, s.frame_time(0)
            self.initial = True
            ps.frame_tick()
            image = self.capture()
            self.initial = False
            for _ in range(hold):
                yield image
        for k in range(s.n_frames + 1):
            v.frame, v.t = k, s.frame_time(k)
            ps.frame_tick()  # runs the callback (meshes, text) and draws
            yield self.capture()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Videos of the benchmark viewer's side-by-side solver comparisons.")
    parser.add_argument("results", help="Results file (.h5) written by benchmark.py --snapshots.")
    parser.add_argument("-o", "--output", required=True, help="Output prefix: writes <prefix>_<benchmark>.mp4.")
    parser.add_argument("--dataset", default=None,
                        help="The dataset file the results were run on (default: the results file's dataset attribute).")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Benchmarks to record (default: all).")
    parser.add_argument("--cases", nargs="+", default=None, help="Cases (scenes) to record (default: all).")
    parser.add_argument("--samples", nargs="+", type=int, default=None,
                        help="Samples to record (default: every sample with a trajectory).")
    parser.add_argument("--samples-per-case", type=int, default=None, dest="samples_per_case", metavar="N",
                        help="Per case, record only the N samples (of --samples, if given) with the largest initial "
                        "error, the largest |x(0) - x*| at a vertex (ties: the lower sample), in sample order "
                        "(default: every sample).")
    parser.add_argument("--width", type=int, default=1920, help="Video width (default: %(default)s).")
    parser.add_argument("--height", type=int, default=1080, help="Video height (default: %(default)s).")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality, lower is better (default: %(default)s).")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable (default: FFMPEG, imageio-ffmpeg, PATH).")
    add_error_fraction_argument(parser)
    parser.add_argument("--hold-seconds", type=float, default=HOLD_SECONDS, dest="hold_seconds", metavar="S",
                        help="Seconds of video each clip holds the start state, captioned \"Initial\", before the "
                        "solve; 0: no hold (default: %(default)g).")
    parser.add_argument("--font-dir", default=None, dest="font_dir",
                        help="Directory with the Linux Libertine OpenType files (default: PBAT_FONT_DIR, then the "
                        "installed fonts; without them, imgui's default font).")
    args = parser.parse_args(argv)
    if args.hold_seconds < 0:
        parser.error(f"--hold-seconds must be at least 0, got {args.hold_seconds:g}")
    if args.samples_per_case is not None and args.samples_per_case < 1:
        parser.error(f"--samples-per-case must be at least 1, got {args.samples_per_case}")
    try:
        args.error_fractions = ErrorFractions(args.error_fractions)
    except ValueError as e:
        parser.error(str(e))
    return args


def main():
    args = parse_args()
    ffmpeg = find_ffmpeg(args.ffmpeg)
    font_file = paper_font_file(args.font_dir)

    import polyscope as ps

    ps.set_program_name("Benchmark video")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("none")
    ps.set_SSAA_factor(4)
    ps.set_window_size(args.width, args.height)
    ps.set_build_default_gui_panels(False)
    ps.set_open_imgui_window_for_user_callback(False)
    ps.init()
    ps.set_window_size(args.width, args.height)
    viewer = Viewer(args.dataset, args.error_fractions)
    viewer.load(args.results)
    if viewer.error:
        raise SystemExit(viewer.error)
    check_error_fractions(args.error_fractions, viewer.cases)
    recorder = Recorder(viewer, args.hold_seconds, font_file)
    ps.set_user_callback(recorder.callback)

    indices = [i for i, (b, c, _, _) in enumerate(viewer.cases)
               if (args.benchmarks is None or b in args.benchmarks) and (args.cases is None or c in args.cases)]
    chosen = recorded_samples(args.results, [viewer.cases[i] for i in indices], args.samples, args.samples_per_case,
                              args.dataset)
    cases = [(i, viewer.cases[i][0], viewer.cases[i][1], viewer.cases[i][3], recorded)
             for i, recorded in zip(indices, chosen)]
    if args.samples_per_case is not None:
        # Entries naming only samples of the recorded cases left out by --samples or the pick (all entries were
        # checked against every sample above)
        elsewhere = args.error_fractions.unmatched([(b, c, [], s) for _, b, c, s, _ in cases])
        unrecorded = [spec for spec in args.error_fractions.unmatched([(b, c, [], r) for _, b, c, _, r in cases])
                      if spec not in elsewhere]
        if unrecorded:
            print(f"Note: --error-fraction {', '.join(unrecorded)}: not among the recorded samples", flush=True)
    benchmarks = list(dict.fromkeys(b for _, b, _, _, _ in cases))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    start = time.time()
    for benchmark in benchmarks:
        path = f"{args.output}_{benchmark}.mp4"
        process = None
        n = 0
        for i, b, c, samples, recorded in cases:
            if b != benchmark:
                continue
            for sample in recorded:
                for image in recorder.frames(i, samples.index(sample)):
                    if process is None:
                        h, w = image.shape[:2]
                        fps = viewer.selection.n_frames / viewer.selection.video_seconds
                        process = subprocess.Popen(
                            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf",
                             str(args.crf), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", path],
                            stdin=subprocess.PIPE)
                    process.stdin.write(image.tobytes())
                    n += 1
                print(f"{benchmark}/{c} sample {sample}: {n} frames so far ({time.time() - start:.0f} s)", flush=True)
        if process is not None:
            process.stdin.close()
            process.wait()
            print(f"Wrote {path} ({n} frames)", flush=True)


import unittest


class TestBenchmarkVideo(unittest.TestCase):
    @staticmethod
    def stubs():
        """Stubbed polyscope and imgui modules, and the texts the stubbed imgui draws."""
        from unittest import mock

        texts = []
        imgui = mock.MagicMock()
        imgui.CalcTextSize.return_value = (20.0, 10.0)
        imgui.GetFontSize.return_value = 26.0
        imgui.GetForegroundDrawList.return_value.AddText.side_effect = lambda pos, color, text: texts.append(text)
        imgui.GetBackgroundDrawList.return_value.AddText.side_effect = lambda pos, color, text: texts.append(text)
        ps = mock.MagicMock()
        ps.imgui = imgui
        ps.register_surface_mesh.side_effect = lambda *a, **k: mock.MagicMock()
        camera = ps.get_view_camera_parameters.return_value
        camera.get_view_mat.return_value = np.eye(4)
        camera.get_fov_vertical_deg.return_value = 45.0
        camera.get_aspect.return_value = 1.6
        ps.get_window_size.return_value = (64, 48)
        ps.screenshot_to_buffer.side_effect = lambda **k: np.zeros((48, 64, 4), dtype=np.uint8)
        return ps, imgui, texts

    def test_error_fraction(self):
        """--error-fraction entries reach the viewer's color scale per problem, and the caption gives the clamped
        max and fraction."""
        import sys
        import tempfile
        from unittest import mock

        from .benchmark_viewer import TestBenchmarkViewer

        for bad in ("0", "S:x=0.1", "static/=0.1"):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                parse_args(["r.h5", "-o", "out", "--error-fraction", bad])
        args = parse_args(["r.h5", "-o", "out", "--error-fraction", "0.5", "--error-fraction", "dynamic_h0.01/S=0.05"])
        ps, imgui, texts = self.stubs()
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"polyscope": ps,
                                                                                  "polyscope.imgui": imgui}):
            dataset, path, _ = TestBenchmarkViewer.write_files(root)
            viewer = Viewer(dataset, args.error_fractions)
            viewer.load(path, "dynamic_h0.01", "S", TestBenchmarkViewer.SAMPLE)
            self.assertIsNone(viewer.error)
            recorder = Recorder(viewer)
            texts.clear()
            recorder.callback()
        self.assertAlmostEqual(viewer.vmax, 0.05 * viewer.initial_error(), places=6)
        caption = f"Color: {COLOR_MODES[0]}, 0 to {viewer.vmax:.4g} m (5% of the initial error)"
        self.assertIn(caption, texts)
        self.assertEqual(args.error_fractions("static", "S", TestBenchmarkViewer.SAMPLE), 0.5)
        with self.assertRaises(SystemExit):  # an entry matching no problem
            check_error_fractions(ErrorFractions(["static/T=0.1"]), viewer.cases)

    def test_samples_per_case(self):
        """--samples-per-case: per case, the samples with the largest initial errors (ties: the lower sample), in
        sample order, after --samples; the errors are the viewer's color scale's."""
        import sys
        import tempfile
        from unittest import mock

        import h5py

        from . import benchmark_results as results
        from .benchmark_viewer import TestBenchmarkViewer, list_cases

        self.assertIsNone(parse_args(["r.h5", "-o", "out"]).samples_per_case)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parse_args(["r.h5", "-o", "out", "--samples-per-case", "0"])
        self.assertEqual(top_samples({4: 0.2, 1: 0.5, 3: 0.5, 0: 0.3}, 2), [1, 3])
        self.assertEqual(top_samples({3: 0.5, 1: 0.5}, 1), [1])  # a tie: the lower sample
        self.assertEqual(top_samples({2: 0.1, 0: 0.3}, 5), [0, 2])
        X, _ = TestBenchmarkViewer.cube()
        # Static samples: one boundary vertex displaced by d from X = x*; samples 1 and 3 tie
        displacements = {0: 0.3, 1: 0.5, 2: 0.1, 3: 0.5, 4: 0.2}
        ps, imgui, _ = self.stubs()
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"polyscope": ps,
                                                                                  "polyscope.imgui": imgui}):
            dataset, _, shapes = TestBenchmarkViewer.write_files(root)
            path = os.path.join(root, "samples.h5")
            starts = {"static": {}, "dynamic_h0.01": {s: shapes["x"][s] for s in range(2)}}
            for sample, d in displacements.items():
                starts["static"][sample] = X.astype(np.float32)
                starts["static"][sample][5, 2] += d
            levels = (0.5,)
            with h5py.File(path, "w") as f:
                f.attrs.update(format=results.FORMAT, version=results.VERSION, dataset=dataset,
                               solvers=["VBD", "Newton"])
                for benchmark, problem in (("static", "static"), ("dynamic_h0.01", "dynamic")):
                    case = results.require_case(f, benchmark, "S", {"set": f"S/{benchmark}/E1", "problem": problem},
                                                levels=levels)
                    for solver in ("VBD", "Newton"):
                        summaries, logs, trajectories = [], [], []
                        for sample, x0 in starts[benchmark].items():
                            c = results.level_crossings(np.array([1.0, 0.4]), np.array([0.0, 1.0]), levels)
                            summaries.append(dict(sample=sample, e0=1.0, iterations=1, sweeps=1, solver_time=1.0,
                                                  final_relative_error=0.4, termination="target",
                                                  deformation_energy=0.0, crossings=c, budget=1.0,
                                                  normalization=1.0))
                            logs.append((np.array([1.0, 0.4]), np.array([0.0, 1.0]), np.array([0, 1])))
                            # VBD has no trajectory of sample 0: its start is Newton's first frame
                            trajectories.append(None if solver == "VBD" and sample == 0 else dict(
                                iteration=np.array([0, 1]), t=np.array([0.0, 1.0]), t_target=np.array([0.0, 1.0]),
                                level_frames=np.array([1]), level_exact=np.array([True]),
                                x=np.stack([x0, X.astype(np.float32)]), requested_frames=1, ring_slots=2, stride=1,
                                device_bytes=0, video_seconds=2.0, playback_factor=2.0))
                        results.write_solver(case, solver, levels, summaries, logs, trajectories)
            cases = list_cases(path)
            for benchmark, case, _, samples in cases:  # the viewer's initial errors
                errors = initial_errors(path, benchmark, case, samples)
                viewer = Viewer(dataset)
                for sample in samples:
                    viewer.load(path, benchmark, case, sample)
                    self.assertIsNone(viewer.error)
                    self.assertEqual(errors[sample], viewer.initial_error())
                if benchmark == "static":
                    for sample, d in displacements.items():
                        self.assertAlmostEqual(errors[sample], d, places=6)
            with mock.patch("builtins.print") as printed:
                self.assertEqual(recorded_samples(path, cases, samples_per_case=3), [[0, 1, 3], [0, 1]])
                self.assertEqual(recorded_samples(path, cases, [2, 3, 4], 2), [[3, 4], []])
                self.assertEqual(recorded_samples(path, cases, [2, 4]), [[2, 4], []])  # without a pick: --samples
                self.assertEqual(recorded_samples(path, cases), [[0, 1, 2, 3, 4], [0, 1]])
            self.assertIn("static/S: samples 3 (0.5 m), 4 (0.2 m), the 2 of 3 with the largest initial errors",
                          [call.args[0] for call in printed.call_args_list])
            # The typo guard checks every sample, recorded or not
            check_error_fractions(ErrorFractions(["static/S:2=0.1"]), cases)

    def test_plain_label(self):
        """The paper figures' case labels as plain text, with and without Unicode; the title uses them."""
        self.assertEqual(plain_label("$E = 10^{5}$ Pa"), "E = 10⁵ Pa")
        self.assertEqual(plain_label(r"$E \in \{10^{6}, 10^{8}\}$ Pa"), "E ∈ {10⁶, 10⁸} Pa")
        self.assertEqual(plain_label(r"$E \in \{10^{6}, 10^{8}\}$ Pa", unicode=False), "E in {10^6, 10^8} Pa")
        self.assertEqual(plain_label("$E = 2.5 \\times 10^{-3}$ Pa\nE = 500 J"), "E = 2.5 × 10⁻³ Pa, E = 500 J")
        self.assertEqual(plain_label("Heterogeneous"), "Heterogeneous")

    def test_paper_font(self):
        """With a font file, the texts are composited onto the captured frames (none drawn by imgui); the paper
        font's file, when found, is Linux Biolinum."""
        import sys
        import tempfile
        from unittest import mock

        import matplotlib

        from .benchmark_viewer import TestBenchmarkViewer

        found = paper_font_file()
        if found is not None:
            self.assertIn("biolinum", os.path.basename(found).lower())
        font_file = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")  # any TrueType font
        ps, imgui, texts = self.stubs()
        ps.get_window_size.return_value = (640, 360)
        ps.screenshot_to_buffer.side_effect = lambda **k: np.full((360, 640, 4), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"polyscope": ps,
                                                                                  "polyscope.imgui": imgui}):
            dataset, path, _ = TestBenchmarkViewer.write_files(root)
            viewer = Viewer(dataset)
            viewer.load(path, "dynamic_h0.01", "S", TestBenchmarkViewer.SAMPLE)
            recorder = Recorder(viewer, hold_seconds=0.0, font_file=font_file)
            ps.frame_tick.side_effect = lambda: recorder.callback()
            texts.clear()
            images = list(recorder.frames(0, 0))
        self.assertEqual(texts, [])  # nothing drawn by imgui
        self.assertEqual(recorder.texts, [])
        for image in images:
            self.assertEqual(image.shape, (360, 640, 3))
            self.assertTrue((image[:60, :300] < 128).any())  # the dark titles, top left
        self.assertEqual(recorder.font(360).size, round(TEXT_PIXELS * 360 / 1080))
        # The title's material: the cube's two materials, E = mu (3 lambda + 2 mu) / (lambda + mu) = 2.8 mu
        self.assertEqual(recorder.labels[viewer.selection.benchmark, "S"], "E ∈ {2.8 × 10⁵, 2.8 × 10⁷} Pa")

    def test_hold(self):
        """The hold: round(S x fps) copies of one captured start state, captioned "Initial" at t = 0, then the
        solve's frames without the caption; none with --hold-seconds 0."""
        import sys
        import tempfile
        from unittest import mock

        from .benchmark_viewer import TestBenchmarkViewer

        self.assertEqual(parse_args(["r.h5", "-o", "out"]).hold_seconds, HOLD_SECONDS)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parse_args(["r.h5", "-o", "out", "--hold-seconds", "-1"])
        ps, imgui, texts = self.stubs()
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"polyscope": ps,
                                                                                  "polyscope.imgui": imgui}):
            dataset, path, _ = TestBenchmarkViewer.write_files(root)
            viewer = Viewer(dataset)
            viewer.load(path, "dynamic_h0.01", "S", TestBenchmarkViewer.SAMPLE)
            s = viewer.selection
            fps = s.n_frames / s.video_seconds  # 0.5 fps: 1 frame over 2 s
            recorder = Recorder(viewer, hold_seconds=6.0)
            ps.frame_tick.side_effect = lambda: recorder.callback()
            captions = []  # per captured image: the shown time and whether it was captioned "Initial"

            def capture(**kwargs):
                captions.append((viewer.t, "Initial" in texts))
                texts.clear()
                return np.zeros((48, 64, 4), dtype=np.uint8)

            ps.screenshot_to_buffer.side_effect = capture
            case = next(i for i, (b, _, _, _) in enumerate(viewer.cases) if b == "dynamic_h0.01")
            images = list(recorder.frames(case, 0))
            hold = round(6.0 * fps)
            self.assertEqual(hold, 3)
            self.assertEqual(len(images), hold + s.n_frames + 1)
            self.assertTrue(all(image is images[0] for image in images[:hold]))  # captured once
            self.assertEqual(captions, [(0.0, True)] + [(s.frame_time(k), False) for k in range(s.n_frames + 1)])
            recorder.hold_seconds = 0.0
            self.assertEqual(len(list(recorder.frames(case, 0))), s.n_frames + 1)


if __name__ == "__main__":
    main()

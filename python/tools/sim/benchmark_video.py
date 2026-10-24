"""
Records the benchmark viewer's side-by-side solver comparisons (``benchmark_viewer.py``) to videos, one MP4 per
benchmark: every trajectory sample of every case in turn, each played over all its frames.

Each clip is what the viewer shows with its defaults: the solvers' copies side by side, labeled with their names
and e/e0, colored by the per-vertex error |x - x*| from 0 to a fraction of the problem's initial error (the
viewer's default, or ``--error-fraction``; values above it saturate, so a stalled solve stays bright); plus a
title (benchmark, scene, sample, solver time over T_N and the slowdown) and the color scale with its fraction.
Each clip opens on the start state, held for ``--hold-seconds`` (default 1 s; 0: no hold) at t = 0 with the
caption "Initial", so viewers see the problem before the solvers change the shape; the caption goes when the
solve starts. Frames are evenly spaced in solver time over the problem's budget and play at
``F / video_seconds`` frames per second (24 fps for the full run's 120 frames in 5 s), so every clip shows its
solve slowed down by its ``playback_factor``. The viewer's translucent overlays are off: near convergence a
reference overlay coincides with the solver meshes and flickers (z-fighting), and the colors already show the
distance to the reference.

Polyscope has no headless backend on Windows: the recorder opens a window and drives it itself, so leave it open
and unminimized until it finishes. Frames go straight from the window to ffmpeg's stdin (no image files). ffmpeg
is found from ``--ffmpeg``, then the ``FFMPEG`` environment variable, then the ``imageio-ffmpeg`` package's
binary, then ``PATH``.

Usage:
    python -m python.tools.sim.benchmark_video path/to/results.h5 -o out/videos/trampolines \
        [--dataset path/to/trampolines.h5] [--benchmarks static ...] [--cases Y1e6 ...] [--samples 0 4 ...] \
        [--width 1920 --height 1080] [--crf 18] [--ffmpeg path/to/ffmpeg] [--error-fraction 0.03]         [--hold-seconds 1]

Writes ``<prefix>_<benchmark>.mp4`` per benchmark.

Tests: ``python -m unittest python.tools.sim.benchmark_video``.
"""

import argparse
import math
import os
import shutil
import subprocess
import time

import numpy as np

from .benchmark_plot import benchmark_title
from .benchmark_viewer import (COLOR_MODES, ERROR_FRACTION, ERROR_FRACTION_RANGE, Viewer, _rgb, display_name,
                               slowdown_text, solver_color)

TEXT_SCALE = 2.0  # the labels and titles, over imgui's default font, for a 1080p video
TEXT_COLOR = (0.1, 0.1, 0.1, 1.0)
HOLD_SECONDS = 1.0  # each clip's start state, held before the solve


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
    """Drives a :class:`Viewer` frame by frame and draws the labels and titles a video needs, without the panel."""

    def __init__(self, viewer: Viewer, hold_seconds: float = HOLD_SECONDS):
        self.viewer = viewer
        self.hold_seconds = hold_seconds
        self.initial = False  # the hold: the start state, captioned "Initial"

    def hold_frames(self) -> int:
        """The hold's video frames at the selected sample's frame rate."""
        s = self.viewer.selection
        return round(self.hold_seconds * s.n_frames / s.video_seconds)

    def draw_text(self, draw, x: float, y: float, text: str, color=TEXT_COLOR, center: bool = False):
        import polyscope.imgui as imgui

        width = imgui.CalcTextSize(text)[0]  # imgui's global font scale is TEXT_SCALE (the callback sets it)
        draw.AddText((x - width / 2 if center else x, y), imgui.GetColorU32(color), text)

    def callback(self):
        """Per frame: the meshes, then the titles and each copy's label (name and e/e0) on the background."""
        import polyscope as ps
        import polyscope.imgui as imgui

        v = self.viewer
        s = v.selection
        v.update_meshes()
        imgui.GetIO().FontGlobalScale = TEXT_SCALE
        draw = imgui.GetForegroundDrawList()
        line = imgui.GetFontSize() * 1.25
        self.draw_text(draw, 24, 20, f"{benchmark_title(s.benchmark)}: {s.label}, sample {s.sample}")
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
            image = np.ascontiguousarray(ps.screenshot_to_buffer(transparent_bg=False, include_UI=True)[..., :3])
            self.initial = False
            for _ in range(hold):
                yield image
        for k in range(s.n_frames + 1):
            v.frame, v.t = k, s.frame_time(k)
            ps.frame_tick()  # runs the callback (meshes, text) and draws
            image = ps.screenshot_to_buffer(transparent_bg=False, include_UI=True)
            yield np.ascontiguousarray(image[..., :3])


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
    parser.add_argument("--width", type=int, default=1920, help="Video width (default: %(default)s).")
    parser.add_argument("--height", type=int, default=1080, help="Video height (default: %(default)s).")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality, lower is better (default: %(default)s).")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable (default: FFMPEG, imageio-ffmpeg, PATH).")
    parser.add_argument("--error-fraction", type=float, default=ERROR_FRACTION, dest="error_fraction", metavar="F",
                        help="The error colors' max as a fraction F of each problem's initial error, "
                        f"{ERROR_FRACTION_RANGE[0]:g} to {ERROR_FRACTION_RANGE[1]:g} (default: the viewer's, "
                        "%(default)g).")
    parser.add_argument("--hold-seconds", type=float, default=HOLD_SECONDS, dest="hold_seconds", metavar="S",
                        help="Seconds of video each clip holds the start state, captioned \"Initial\", before the "
                        "solve; 0: no hold (default: %(default)g).")
    args = parser.parse_args(argv)
    if args.hold_seconds < 0:
        parser.error(f"--hold-seconds must be at least 0, got {args.hold_seconds:g}")
    lo, hi = ERROR_FRACTION_RANGE
    if not lo <= args.error_fraction <= hi:
        parser.error(f"--error-fraction must be in [{lo:g}, {hi:g}], got {args.error_fraction:g}")
    return args


def main():
    args = parse_args()
    ffmpeg = find_ffmpeg(args.ffmpeg)

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
    viewer = Viewer(args.dataset, args.error_fraction)
    viewer.load(args.results)
    if viewer.error:
        raise SystemExit(viewer.error)
    recorder = Recorder(viewer, args.hold_seconds)
    ps.set_user_callback(recorder.callback)

    cases = [(i, b, c, samples) for i, (b, c, _, samples) in enumerate(viewer.cases)
             if (args.benchmarks is None or b in args.benchmarks) and (args.cases is None or c in args.cases)]
    benchmarks = list(dict.fromkeys(b for _, b, _, _ in cases))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    start = time.time()
    for benchmark in benchmarks:
        path = f"{args.output}_{benchmark}.mp4"
        process = None
        n = 0
        for i, b, c, samples in cases:
            if b != benchmark:
                continue
            for j, sample in enumerate(samples):
                if args.samples is not None and sample not in args.samples:
                    continue
                for image in recorder.frames(i, j):
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
        """--error-fraction reaches the viewer's color scale, and the caption gives the clamped max and fraction."""
        import sys
        import tempfile
        from unittest import mock

        from .benchmark_viewer import TestBenchmarkViewer

        self.assertEqual(parse_args(["r.h5", "-o", "out"]).error_fraction, ERROR_FRACTION)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parse_args(["r.h5", "-o", "out", "--error-fraction", "0"])
        args = parse_args(["r.h5", "-o", "out", "--error-fraction", "0.05"])
        ps, imgui, texts = self.stubs()
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"polyscope": ps,
                                                                                  "polyscope.imgui": imgui}):
            dataset, path, _ = TestBenchmarkViewer.write_files(root)
            viewer = Viewer(dataset, args.error_fraction)
            viewer.load(path, "dynamic_h0.01", "S", TestBenchmarkViewer.SAMPLE)
            self.assertIsNone(viewer.error)
            recorder = Recorder(viewer)
            texts.clear()
            recorder.callback()
        self.assertAlmostEqual(viewer.vmax, 0.05 * viewer.initial_error(), places=6)
        caption = f"Color: {COLOR_MODES[0]}, 0 to {viewer.vmax:.4g} m (5% of the initial error)"
        self.assertIn(caption, texts)

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

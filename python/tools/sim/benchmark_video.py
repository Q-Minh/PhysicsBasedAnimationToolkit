"""
Records the benchmark viewer's side-by-side solver comparisons (``benchmark_viewer.py``) to videos, one MP4 per
benchmark: every trajectory sample of every case in turn, each played over all its frames.

Each clip is what the viewer shows with its defaults: the solvers' copies side by side, labeled with their names
and e/e0, colored by the per-vertex error |x - x*| from 0 to the problem's initial error, or a fraction of it
per problem (``--error-fraction``, as the viewer's: values above it saturate, so a stalled solve stays bright);
plus a title (benchmark, scene, sample, solver time over T_N and the slowdown) and the color scale with its
fraction.
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
        [--width 1920 --height 1080] [--crf 18] [--ffmpeg path/to/ffmpeg] [--hold-seconds 1] \
        [--font-dir path/to/fonts] [--error-fraction static/Y1e5=0.2 ...]

Writes ``<prefix>_<benchmark>.mp4`` per benchmark.

Tests: ``python -m unittest python.tools.sim.benchmark_video``.
"""

import argparse
import math
import os
import shutil
import subprocess
import time
import warnings

import numpy as np

from .benchmark_plot import benchmark_title, register_paper_fonts
from .benchmark_viewer import (COLOR_MODES, ErrorFractions, Viewer, _rgb, add_error_fraction_argument,
                               check_error_fractions, display_name, slowdown_text, solver_color)

TEXT_SCALE = 2.0  # the labels and titles, over imgui's default font, for a 1080p video (without the paper font)
TEXT_PIXELS = 34  # the paper font's size in pixels at a 1080-pixel video height, scaled with it
TEXT_COLOR = (0.1, 0.1, 0.1, 1.0)
HOLD_SECONDS = 1.0  # each clip's start state, held before the solve


def paper_font_file(font_dir: str | None = None) -> str | None:
    """Linux Biolinum's file (``benchmark_plot.register_paper_fonts``), or None with a warning."""
    from matplotlib import font_manager

    sans = register_paper_fonts(font_dir)[0]
    if sans is None:
        warnings.warn("Linux Biolinum not found (--font-dir, PBAT_FONT_DIR or an installed font): the video text "
                      "uses imgui's default font")
        return None
    return font_manager.findfont(font_manager.FontProperties(family=sans), fallback_to_default=False)


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
        self.initial = False  # the hold: the start state, captioned "Initial"

    def font(self, height: int):
        """The Pillow font for a video ``height`` pixels tall."""
        from PIL import ImageFont

        size = max(round(TEXT_PIXELS * height / 1080), 1)
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_file, size)
        return self.fonts[size]

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

"""
Records the benchmark viewer's side-by-side solver comparisons (``benchmark_viewer.py``) to videos, one MP4 per
benchmark: every trajectory sample of every case in turn, each played over all its frames.

Each clip is what the viewer shows with its defaults: the solvers' copies side by side, labeled with their names
and e/e0, colored by the per-vertex error |x - x*| from 0 to the problem's initial error; plus a title (benchmark, scene, sample, solver time over T_N and
the slowdown) and the color scale. Frames are evenly spaced in solver time over the problem's budget and play at
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
        [--width 1920 --height 1080] [--crf 18] [--ffmpeg path/to/ffmpeg]

Writes ``<prefix>_<benchmark>.mp4`` per benchmark.
"""

import argparse
import math
import os
import shutil
import subprocess
import time

import numpy as np

from .benchmark_plot import benchmark_title
from .benchmark_viewer import COLOR_MODES, Viewer, _rgb, display_name, slowdown_text, solver_color

TEXT_SCALE = 2.0  # the labels and titles, over imgui's default font, for a 1080p video
TEXT_COLOR = (0.1, 0.1, 0.1, 1.0)


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

    def __init__(self, viewer: Viewer):
        self.viewer = viewer

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
        self.draw_text(draw, 24, 20 + 2 * line, f"Color: {COLOR_MODES[v.color_mode]} from 0 to {1e3 * v.vmax:.3g} mm")
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
        """Selects a sample and yields its frames' RGB images, one per viewer frame."""
        import polyscope as ps

        v = self.viewer
        v.select(case_index, sample_index)
        if v.error:
            raise SystemExit(v.error)
        s = v.selection
        for overlay in s.overlays:  # off: a converged solve's overlay z-fights with its mesh
            v.show_overlay[overlay.name] = False
        v.dirty = True
        for k in range(s.n_frames + 1):
            v.frame, v.t = k, s.frame_time(k)
            ps.frame_tick()  # runs the callback (meshes, text) and draws
            image = ps.screenshot_to_buffer(transparent_bg=False, include_UI=True)
            yield np.ascontiguousarray(image[..., :3])


def main():
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
    args = parser.parse_args()
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
    viewer = Viewer(args.dataset)
    viewer.load(args.results)
    if viewer.error:
        raise SystemExit(viewer.error)
    recorder = Recorder(viewer)
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


if __name__ == "__main__":
    main()

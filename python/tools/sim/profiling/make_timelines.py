"""
Make the paper's timeline figures from a timeline kit, with snapshots from your own renders.

A kit is a folder with everything the timelines need besides the code: the production and kernels run summaries
(``.json``), our placeholder snapshots (``Placeholders/<Scene>.Frame<N>.png``) and ``manifest.json``, which lists per
scene and per set (``EqualSweep``, ``EqualTime``) the run files and the figure's options. The snapshot frames come
from ``snapshot_frames.toml`` next to this file (frame N at t = N / rate s), the single source of truth for them.

For each scene and set, the snapshot of frame N is ``<renders>/<Scene>.Frame<N>.png`` (or ``.jpg``), else the kit's
placeholder, else none; renders of any size and aspect work (one row height, each image as wide as its aspect
needs). The figures are ``<out>/<Scene>.Timeline.pdf`` and ``<out>/<Scene>.EqualTime.Timeline.pdf``, drawn by
``paper_figures.plot_timeline`` exactly as the paper's. It needs only NumPy and Matplotlib (``requirements.txt``
next to this file) and Python 3.11 or newer, from a PBAT checkout's root:

    python -m python.tools.sim.profiling.make_timelines --kit path/to/TimelineKit --renders path/to/renders

``--out`` (default ``Timelines``) is where the PDFs go; ``--scenes`` and ``--sets`` restrict them. The kit's
``Fonts/`` (the paper's Linux Biolinum) are used if present, else ``--font-dir``, ``PBAT_FONT_DIR`` or the installed
fonts. At the end it lists the snapshots that fell back to a placeholder and those it found nowhere.
"""

import argparse
import json
import os
import sys
import tomllib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import paper_figures, results  # noqa: E402
from ..figure_style import use_paper_fonts  # noqa: E402

FRAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot_frames.toml")
SETS = {"EqualSweep": "Timeline", "EqualTime": "EqualTime.Timeline"}  # set -> output name suffix
RENDER_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_frames(path: str = FRAMES) -> dict[str, dict]:
    """The snapshot frames per scene (``snapshot_frames.toml``): ``rate`` (frames per second), ``frames`` (0-based
    frame indices) and ``notes`` (what each shows)."""
    with open(path, "rb") as f:
        frames = tomllib.load(f)
    for scene, entry in frames.items():
        if len(entry.get("notes", entry["frames"])) != len(entry["frames"]):
            raise ValueError(f"{path}: [{scene}] has {len(entry['frames'])} frames but {len(entry['notes'])} notes")
    return frames


def snapshot_image(scene: str, frame: int, renders: str | None, kit: str) -> tuple[str | None, str]:
    """The image of a scene's frame and where it came from: ``"render"``, ``"placeholder"`` or ``"missing"``
    (path None)."""
    if renders:
        for extension in RENDER_EXTENSIONS:
            path = os.path.join(renders, f"{scene}.Frame{frame}{extension}")
            if os.path.isfile(path):
                return path, "render"
    path = os.path.join(kit, "Placeholders", f"{scene}.Frame{frame}.png")
    return (path, "placeholder") if os.path.isfile(path) else (None, "missing")


def make_timeline(kit: str, scene: str, entry: dict, frames: dict | None, renders: str | None, out: str) -> dict:
    """One timeline (``entry``: a manifest set, its ``kernels``, ``production`` and ``options``) with the scene's
    snapshots; returns where each snapshot came from, ``{frame: "render" | "placeholder" | "missing"}``, and the
    figure's path under ``"pdf"``."""
    kernels = [results.load(os.path.join(kit, entry["kernels"]))]
    production_runs = [results.load(os.path.join(kit, p)) for p in entry["production"]]
    production = paper_figures.pair_production(kernels, production_runs)
    snapshots, sources = [], {}
    for frame in (frames or {}).get("frames", []):
        path, source = snapshot_image(scene, frame, renders, kit)
        sources[frame] = source
        if path is not None:
            snapshots.append((frame / float(frames["rate"]), path))
    options = entry.get("options", {})
    labels = entry.get("labels", {})
    fig = paper_figures.plot_timeline(
        kernels, production, [labels.get(paper_figures.scene_key(k), scene) for k in kernels],
        width=options.get("width", 7.0), bin_n=options.get("bin", 1), snapshots=snapshots,
        others=paper_figures.other_production(kernels, production_runs),
        smooth=options.get("smooth", paper_figures.SMOOTH_STEPS), rms_speed=options.get("rms_speed", "paired"),
        vbd_ratio=options.get("vbd_ratio", False))
    return {**sources, "pdf": fig}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make the paper's timelines from a timeline kit and your renders.")
    parser.add_argument("--kit", required=True, help="The timeline kit folder (with manifest.json).")
    parser.add_argument("--renders", default=None,
                        help="Folder of your renders, <Scene>.Frame<N>.png (default: the kit's placeholders only).")
    parser.add_argument("--out", default="Timelines", help="Output folder for the PDFs (default: Timelines).")
    parser.add_argument("--scenes", nargs="*", default=None, help="Only these scenes (default: every scene).")
    parser.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS),
                        help="Only these sets (default: EqualSweep and EqualTime).")
    parser.add_argument("--font-dir", default=None, dest="font_dir",
                        help="Folder with the Linux Libertine fonts (default: the kit's Fonts/, else PBAT_FONT_DIR, "
                        "else the installed fonts).")
    args = parser.parse_args(argv)
    with open(os.path.join(args.kit, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    frames = load_frames()
    fonts = args.font_dir or (os.path.join(args.kit, "Fonts") if os.path.isdir(os.path.join(args.kit, "Fonts"))
                              else None)
    use_paper_fonts(fonts)
    os.makedirs(args.out, exist_ok=True)
    fell_back, missing, written = [], [], []
    for scene, spec in manifest["scenes"].items():
        if args.scenes and scene not in args.scenes:
            continue
        if scene in frames and spec.get("frames") not in (None, frames[scene]["frames"]):
            warnings.warn(f"{scene}: the kit's frames {spec['frames']} differ from {FRAMES}'s "
                          f"{frames[scene]['frames']}; using the latter")
        for name in args.sets:
            if name not in spec["sets"]:
                continue
            made = make_timeline(args.kit, scene, spec["sets"][name], frames.get(scene), args.renders, args.out)
            fig = made.pop("pdf")
            path = os.path.join(args.out, f"{scene}.{SETS[name]}.pdf")
            fig.savefig(path, bbox_inches="tight", pad_inches=paper_figures.PAD)
            plt.close(fig)
            written.append(path)
            fell_back += [f"{scene}.Frame{n}" for n, s in made.items() if s == "placeholder"]
            missing += [f"{scene}.Frame{n}" for n, s in made.items() if s == "missing"]
    for path in written:
        print("Wrote", path)
    if args.renders is not None and fell_back:
        print(f"No render for {len(set(fell_back))} snapshots, used the placeholders:",
              ", ".join(sorted(set(fell_back))))
    if missing:
        print(f"No render or placeholder for {len(set(missing))} snapshots, left out:", ", ".join(sorted(set(missing))))
    return 0


import unittest  # noqa: E402


class TestMakeTimelines(unittest.TestCase):

    def test_frames_file(self):
        frames = load_frames()
        for scene, entry in frames.items():
            self.assertEqual(len(entry["frames"]), 5, scene)
            self.assertEqual(entry["frames"], sorted(entry["frames"]), scene)
            self.assertIn(entry["rate"], (30, 60), scene)

    def test_kit(self):
        import tempfile

        import numpy as np

        def run(source, solver):
            steps = [{"t": t, "step_timeline_ms": 10.0 + t % 3, "step_union_ms": 9.0, "union_ms": {"solve": 9.0},
                      "rms_speed": 0.1, "status": {"vbd": [{"substep": 0, "outer": 1, "sweeps": 20, "reason":
                      "CONVERGED", "gradient_ratio": 0.5, "feasibility_ratio": 0.0, "smoothed_ratio": 0.4,
                      "max_outer": 1, "max_sweeps": 20}]}} for t in range(10, 40)]
            return {"schema": results.SCHEMA, "source": source, "steps": steps, "kernels": [],
                    "metadata": {"scene": "path/to/thing.h5", "solver": solver, "dt": 1 / 60, "substeps": 1}}

        with tempfile.TemporaryDirectory() as d:
            kit, renders, out = (os.path.join(d, p) for p in ("Kit", "Renders", "Out"))
            for folder in (kit, os.path.join(kit, "Placeholders"), renders):
                os.makedirs(folder)
            results.save(run("kernels", "AAAVBD"), os.path.join(kit, "Thing.AAAVBD.Kernels.json"))
            for solver in ("AAAVBD", "VBD"):
                results.save(run("production", solver), os.path.join(kit, f"Thing.{solver}.json"))
            entry = {"kernels": "Thing.AAAVBD.Kernels.json", "production": ["Thing.AAAVBD.json", "Thing.VBD.json"]}
            with open(os.path.join(kit, "manifest.json"), "w") as f:
                json.dump({"scenes": {"Thing": {"frames": [12, 24, 30], "sets": {"EqualSweep": entry,
                                                                                  "EqualTime": entry}}}}, f)
            # A wide render of frame 12, the kit's placeholder for 24, nothing for 30
            plt.imsave(os.path.join(renders, "Thing.Frame12.png"), np.random.default_rng(0).random((50, 200)))
            plt.imsave(os.path.join(kit, "Placeholders", "Thing.Frame24.png"), np.zeros((80, 80)))
            frames = {"Thing": {"rate": 60, "frames": [12, 24, 30], "notes": ["a", "b", "c"]}}
            global load_frames
            original = load_frames
            load_frames = lambda path=FRAMES: frames  # noqa: E731
            try:
                from contextlib import redirect_stdout
                from io import StringIO

                with redirect_stdout(StringIO()) as printed:
                    main(["--kit", kit, "--renders", renders, "--out", out])
            finally:
                load_frames = original
            self.assertTrue(os.path.isfile(os.path.join(out, "Thing.Timeline.pdf")))
            self.assertTrue(os.path.isfile(os.path.join(out, "Thing.EqualTime.Timeline.pdf")))
            self.assertIn("used the placeholders: Thing.Frame24", printed.getvalue())
            self.assertIn("left out: Thing.Frame30", printed.getvalue())
            # The sources and times per frame
            sources = make_timeline(kit, "Thing", entry, frames["Thing"], renders, out)
            plt.close(sources.pop("pdf"))
            self.assertEqual(sources, {12: "render", 24: "placeholder", 30: "missing"})


if __name__ == "__main__":
    sys.exit(main())

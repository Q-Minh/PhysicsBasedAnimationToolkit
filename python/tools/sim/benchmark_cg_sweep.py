"""
Newton's CG limits per case from a CG sweep: Newton-only benchmark runs (``benchmark.py --solvers Newton``), one
results file per (rtol, cap) setting, e.g. ``--set Solver.Newton.LinearSolver.rtol=1e-2 --set
Solver.Newton.LinearSolver.n_max_iters=4000``. Each file's setting is read from its cases' ``newton_cg_rtol``
and ``newton_cg_n_max_iters`` attributes, so file names don't matter, and files of one setting are pooled (e.g.
a re-time pass's).

Per case, the winner is the setting with the smallest mean time to the benchmark's highest level over all its
solves; a setting that misses that level on any solve is out. A reference setting within ``--tie`` (default
10%) of the winner's time wins instead: the reference is the case's scene's setting at ``--reference-benchmark``
in the ``--reference`` run file (default: trampolines.toml at h = 1e-2, the tuned table). If no setting reaches
the level on every solve, the case keeps the reference and is flagged.

It prints a table per case (each setting's mean time to the highest level, its per-solve times, Newton steps,
CG iterations per step and terminations) and the run file's ``[case]`` tables, and with ``--write RUN.toml``
replaces that run file's tables in place: everything from its ``# Newton's CG per case`` comment to its end,
which must hold only Newton's linear-solver case tables. The tables keep the run file's order, and its cases
the sweep didn't run keep their settings, marked as such.

Usage, from the repo root:
    python -m python.tools.sim.benchmark_cg_sweep data/benchmarks/results/cg-sweep/LargeTimeSteps.*.h5 \
        [--write python/tools/sim/configs/benchmark/trampolines.LargeTimeSteps.toml]

Tests: ``python -m unittest python.tools.sim.benchmark_cg_sweep``.
"""

import argparse
import glob
import os
import unittest
from dataclasses import dataclass, field

import h5py
import numpy as np

from . import benchmark_results as results
from . import benchmark_run

REFERENCE = "python/tools/sim/configs/benchmark/trampolines.toml"
REFERENCE_BENCHMARK = "dynamic_h0.01"
TABLES_MARKER = "# Newton's CG per case"


@dataclass
class Timing:
    """One setting's Newton solves of one case, pooled over files."""

    times: list = field(default_factory=list)  # time to the highest level per solve (NaN: missed)
    steps: list = field(default_factory=list)  # Newton steps per solve
    cg_per_step: list = field(default_factory=list)  # mean CG iterations per Newton step, per solve
    terminations: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.times) and bool(np.all(np.isfinite(self.times)))

    @property
    def mean(self) -> float:
        return float(np.mean(self.times)) if self.complete else np.inf


@dataclass
class Case:
    scene: str
    level: float  # the benchmark's highest level
    settings: dict = field(default_factory=dict)  # (rtol, cap) -> Timing


def format_rtol(rtol: float) -> str:
    """1e-2, not 0.01, as the run files write it."""
    mantissa, exponent = f"{rtol:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def setting_key(rtol: float, cap: int) -> tuple:
    return (float(f"{rtol:.3g}"), int(cap))


def setting_name(key: tuple) -> str:
    return f"rtol {format_rtol(key[0])} cap {key[1]}"


def input_files(paths: list[str]) -> list[str]:
    """The results files: each path a file or a directory (its ``*.h5``)."""
    out = []
    for path in paths:
        out += sorted(glob.glob(os.path.join(path, "*.h5"))) if os.path.isdir(path) else [path]
    return out


def read_sweep(paths: list[str]) -> dict:
    """``{(benchmark, case): Case}`` over the results files, in the first file's order."""
    cases = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            results.check_format(f, path)
            for benchmark, name, g in results.iter_cases(f):
                if "Newton" not in g:
                    continue
                n = g["Newton"]
                level = float(results.benchmark_levels(f, benchmark)[-1])
                case = cases.setdefault((benchmark, name), Case(scene=str(g.attrs.get("scene", name)), level=level))
                if level != case.level:
                    raise ValueError(f"{path}: {benchmark}/{name}'s highest level {level} != {case.level} in other files")
                key = setting_key(g.attrs["newton_cg_rtol"], g.attrs["newton_cg_n_max_iters"])
                t = case.settings.setdefault(key, Timing())
                t.times += [float(x) for x in n["crossing_time"][:, -1]]
                t.steps += [int(x) for x in n["iterations"][()]]
                t.terminations += [str(x) for x in n["termination"].asstr()[()]]
                for sample in n["sample"][()]:
                    cg = n[f"logs/{int(sample):06d}/newton/cg_iterations"][()]
                    t.cg_per_step.append(float(np.mean(cg)) if cg.size else np.nan)
    return cases


def pick(settings: dict, reference: tuple | None, tie: float) -> tuple[tuple, str]:
    """The winning setting and why: the fastest complete setting, or the reference within ``tie`` of it."""
    complete = {k: t.mean for k, t in settings.items() if t.complete}
    if not complete:
        return reference, "no setting reached the highest level on every solve: kept the reference"
    best = min(complete, key=complete.get)
    if reference is None or best == reference:
        return best, "fastest"
    if reference in complete and complete[reference] <= (1.0 + tie) * complete[best]:
        return reference, (f"reference, {complete[reference] / complete[best] - 1:+.0%} on the fastest "
                           f"({setting_name(best)}): a tie")
    return best, "fastest"


def reference_setting(run_file, benchmark: str, scene: str) -> tuple | None:
    lin = benchmark_run.case_parameters(run_file, benchmark, scene).get("Solver", {}).get("Newton", {}).get(
        "LinearSolver", {})
    return setting_key(lin["rtol"], lin["n_max_iters"]) if {"rtol", "n_max_iters"} <= set(lin) else None


def report(key, case: Case, winner: tuple, why: str, reference: tuple | None) -> str:
    lines = [f"{key[0]}/{key[1]}: {setting_name(winner) if winner else '-'} ({why})"]
    for k in sorted(case.settings, key=lambda k: (case.settings[k].mean, k)):
        t = case.settings[k]
        mark = " <- winner" if k == winner else ""
        mark += " (reference)" if k == reference else ""
        mean = f"{t.mean:7.3f} s" if t.complete else "   miss  "
        lines.append(f"  {setting_name(k):>20}: {mean}  per solve [{', '.join(f'{x:.3f}' for x in t.times)}], "
                     f"steps {np.mean(t.steps):.1f}, CG/step {np.nanmean(t.cg_per_step):.0f}, "
                     f"{','.join(sorted(set(t.terminations)))}{mark}")
    return "\n".join(lines)


def format_level(level: float) -> str:
    return f"{100 * level:.4f}".rstrip("0").rstrip(".") + "%"


def table(name: str, setting: tuple, comment: str) -> list[str]:
    return ["", f"# {comment}", f'[case."{name}".Solver.Newton.LinearSolver]', f"n_max_iters = {setting[1]}",
            f"rtol = {format_rtol(setting[0])}"]


def case_tables(cases: dict, winners: dict, tie: float, reference_label: str, kept: dict | None = None) -> str:
    """The run file's tail: the comment saying how the settings were picked, then one table per case, in the
    order of ``kept`` (``{"BENCHMARK/CASE": (rtol, cap)}``, the run file's tables), whose cases the sweep didn't
    run keep their setting, then the other swept cases."""
    kept = kept or {}
    settings = sorted({k for c in cases.values() for k in c.settings}, key=lambda k: (-k[0], k[1]))
    solves = sorted({len(t.times) for c in cases.values() for t in c.settings.values()})
    levels = sorted({c.level for c in cases.values()})
    out = [
        f"{TABLES_MARKER}: the winners of a Newton-only CG sweep on these problems (benchmark_cg_sweep.py; "
        f"settings {', '.join(setting_name(k) for k in settings)}; {'/'.join(map(str, solves))} "
        f"solve{'' if solves == [1] else 's'} per case and setting): the smallest mean time to "
        f"{'/'.join(map(format_level, levels))}, a setting missing it on any solve out; {reference_label}'s "
        f"setting when within {tie:.0%} of it.",
    ]
    swept = {f"{b}/{c}": (b, c) for b, c in winners}
    for name in list(kept) + [n for n in swept if n not in kept]:
        if name not in swept:
            out += table(name, kept[name], "not in the sweep: unchanged")
            continue
        winner, why = winners[swept[name]]
        t = cases[swept[name]].settings.get(winner)
        out += table(name, winner, f"{t.mean:.3f} s mean; {why}" if t is not None and t.complete else why)
    return wrap_comment(out[0]) + "\n" + "\n".join(out[1:]) + "\n"


def run_file_settings(path: str) -> dict:
    """``{"BENCHMARK/CASE": (rtol, cap)}``: the run file's Newton CG case tables, in its order."""
    out = {}
    for name, cfg in benchmark_run.read_run_file(path).cases.items():
        lin = cfg.get("Solver", {}).get("Newton", {}).get("LinearSolver", {})
        if {"rtol", "n_max_iters"} <= set(lin):
            out[name] = setting_key(lin["rtol"], lin["n_max_iters"])
    return out


def wrap_comment(text: str, width: int = 116) -> str:
    lines, line = [], "#"
    for word in text[2:].split():
        if len(line) + 1 + len(word) > width:
            lines.append(line)
            line = "#"
        line += " " + word
    return "\n".join(lines + [line])


def replace_tables(text: str, tables: str) -> str:
    """A run file's text with everything from ``TABLES_MARKER`` on replaced by ``tables``."""
    start = text.find(TABLES_MARKER)
    if start < 0:
        raise ValueError(f"no '{TABLES_MARKER}' comment to replace from")
    tail = [line.strip() for line in text[start:].splitlines()]
    allowed = ("#", "[case.", "n_max_iters", "rtol", "atol")
    other = [line for line in tail if line and not line.startswith(allowed)]
    if other or any(line.startswith("[") and not line.endswith(".Solver.Newton.LinearSolver]") for line in tail):
        raise ValueError(f"the run file's tail holds more than Newton's CG case tables: {other or tail}")
    return text[:start] + tables


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="results files, or directories of them (*.h5)")
    parser.add_argument("--reference", default=REFERENCE, help="run file holding the reference settings")
    parser.add_argument("--reference-benchmark", default=REFERENCE_BENCHMARK,
                        help="the reference run file's benchmark whose per-scene settings are the references")
    parser.add_argument("--tie", type=float, default=0.1, help="relative time difference under which the reference wins")
    parser.add_argument("--write", metavar="RUN.toml", help="replace this run file's Newton CG case tables")
    args = parser.parse_args()

    cases = read_sweep(input_files(args.inputs))
    if not cases:
        raise SystemExit("no Newton solves in the inputs")
    run_file = benchmark_run.read_run_file(args.reference)
    winners = {}
    for key, case in cases.items():
        reference = reference_setting(run_file, args.reference_benchmark, case.scene)
        winners[key] = pick(case.settings, reference, args.tie)
        print(report(key, case, *winners[key], reference) + "\n")
    label = f"{os.path.basename(args.reference)} {args.reference_benchmark}"
    tables = case_tables(cases, winners, args.tie, label, run_file_settings(args.write) if args.write else None)
    print(tables)
    if args.write:
        with open(args.write, encoding="utf-8", newline="") as f:
            text = f.read()
        newline = "\r\n" if "\r\n" in text else "\n"  # the file's own
        with open(args.write, "w", encoding="utf-8", newline=newline) as f:
            f.write(replace_tables(text.replace("\r\n", "\n"), tables))
        print(f"wrote {args.write}")


class TestPick(unittest.TestCase):
    @staticmethod
    def _timing(*times):
        return Timing(times=list(times), steps=[1] * len(times), cg_per_step=[1.0] * len(times),
                      terminations=["target"] * len(times))

    def test_pick(self):
        a, b, c = (1e-2, 4000), (1e-4, 1000), (1e-6, 500)
        settings = {a: self._timing(1.05, 1.07), b: self._timing(1.0, 0.98), c: self._timing(0.5, np.nan)}
        self.assertEqual(pick(settings, a, 0.1)[0], a)  # within 10% of b: the reference
        self.assertEqual(pick(settings, a, 0.05)[0], b)
        self.assertEqual(pick(settings, None, 0.1)[0], b)  # c misses a solve: out
        self.assertEqual(pick(settings, c, 0.1)[0], b)
        winner, why = pick({c: self._timing(np.nan)}, a, 0.1)
        self.assertEqual(winner, a)
        self.assertIn("kept the reference", why)

    def test_format(self):
        self.assertEqual([format_rtol(x) for x in (0.01, 1e-4, 1e-6, 2.5e-3)], ["1e-2", "1e-4", "1e-6", "2.5e-3"])
        self.assertEqual(setting_key(np.float32(1e-2), np.int64(4000)), (0.01, 4000))

    def test_tables(self):
        import tomllib

        cases = {("dynamic_h0.0333333", "Y1e8"): Case("Y1e8", 0.9999, {(1e-6, 500): self._timing(0.4, 0.5)})}
        winners = {("dynamic_h0.0333333", "Y1e8"): ((1e-6, 500), "fastest")}
        kept = {"dynamic_h0.0166667/Y1e8": (1e-6, 500), "dynamic_h0.0333333/Y1e8": (1e-2, 4000)}
        tables = case_tables(cases, winners, 0.1, "trampolines.toml dynamic_h0.01", kept)
        self.assertIn("time to 99.99%,", tables.replace("\n# ", " "))
        text = replace_tables('extends = ["shared.toml"]\n\n# Newton\'s CG per case: old\n\n'
                              '[case."dynamic_h0.0333333/Y1e8".Solver.Newton.LinearSolver]\nn_max_iters = 4000\n'
                              'rtol = 1e-2\n', tables)
        self.assertTrue(all(len(line) <= 120 for line in text.splitlines()))
        lin = tomllib.loads(text)["case"]["dynamic_h0.0333333/Y1e8"]["Solver"]["Newton"]["LinearSolver"]
        self.assertEqual(lin, {"n_max_iters": 500, "rtol": 1e-6})
        # The run file's order; its case the sweep didn't run unchanged
        tables_ = tomllib.loads(text)["case"]
        self.assertEqual(list(tables_), ["dynamic_h0.0166667/Y1e8", "dynamic_h0.0333333/Y1e8"])
        self.assertEqual(tables_["dynamic_h0.0166667/Y1e8"]["Solver"]["Newton"]["LinearSolver"]["rtol"], 1e-6)
        with self.assertRaisesRegex(ValueError, "more than"):
            replace_tables("# Newton's CG per case\n[run]\nsets = []\n", tables)


if __name__ == "__main__":
    main()

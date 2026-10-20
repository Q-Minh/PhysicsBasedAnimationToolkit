"""
Nsight Compute metrics for one kernel of the GPU simulator.

Runs ``simulate.py`` under Nsight Compute, profiles ``--count`` consecutive launches of the kernels
matching ``--kernel`` after skipping ``--skip`` launches (e.g. the 9 color launches of one VBD sweep),
and summarizes them in JSON: duration, occupancy, throughput, instruction counts and the warp stall
breakdown from PC sampling. Nsight Compute locks clocks to base and flushes caches between passes by
default, so durations are longer than in normal runs; compare kernels with each other, not with
Nsight Systems timings.

With ``--lineinfo``, kernels are compiled with line information, and the summary adds source hotspots:
warp stall samples, stall reasons and executed instructions per line of the Warp (Python) source,
including inlined helper functions, and per source file.

``ncu`` is located by :func:`nsight.find_tool`: ``--ncu path/to/ncu``, else the ``NCU`` environment
variable, else ``PATH``, else the standard install locations.

Usage (from the repo root; Warp kernel names start with an underscore):

    python -m python.tools.sim.profiling.ncu run -o out/aaavbd_ncu --kernel ^_accelerated_vertex_solve \\
        --skip 900 --count 9 [--lineinfo] -- --fem-elasto-dynamics scene.h5:fem --solver AAAVBD \\
        --duration 0.05 -o x.h5
"""

import argparse
import ast
import csv
import io
import json
import os
import statistics
import subprocess
import sys

from .nsight import find_tool

# (label, section, metric) from the "details" page.
_DETAILS = [
    ("duration_us", "GPU Speed Of Light Throughput", "Duration"),
    ("elapsed_cycles", "GPU Speed Of Light Throughput", "Elapsed Cycles"),
    ("sm_throughput_pct", "GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
    ("memory_throughput_pct", "GPU Speed Of Light Throughput", "Memory Throughput"),
    ("dram_throughput_pct", "GPU Speed Of Light Throughput", "DRAM Throughput"),
    ("ipc_active", "Compute Workload Analysis", "Executed Ipc Active"),
    ("l1_hit_pct", "Memory Workload Analysis", "L1/TEX Hit Rate"),
    ("l2_hit_pct", "Memory Workload Analysis", "L2 Hit Rate"),
    ("no_eligible_pct", "Scheduler Statistics", "No Eligible"),
    ("warp_cycles_per_instruction", "Warp State Statistics", "Warp Cycles Per Issued Instruction"),
    ("active_threads_per_warp", "Warp State Statistics", "Avg. Active Threads Per Warp"),
    ("instructions", "Instruction Statistics", "Executed Instructions"),
    ("registers", "Launch Statistics", "Registers Per Thread"),
    ("grid", "Launch Statistics", "Grid Size"),
    ("block", "Launch Statistics", "Block Size"),
    ("waves_per_sm", "Launch Statistics", "Waves Per SM"),
    ("theoretical_occupancy_pct", "Occupancy", "Theoretical Occupancy"),
    ("achieved_occupancy_pct", "Occupancy", "Achieved Occupancy"),
    ("branch_efficiency_pct", "Source Counters", "Branch Efficiency"),
]
_STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _csv(ncu: str, report: str, page: str, *extra: str) -> list[list[str]]:
    """Export one page of an Nsight Compute report as CSV rows."""
    out = subprocess.run(
        [ncu, "--import", report, "--csv", "--page", page, *extra],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout
    return list(csv.reader(io.StringIO(out)))


def summarize(report: str, ncu: str | None = None) -> dict:
    """Mean metrics over the profiled launches of an ``.ncu-rep`` report. ``ncu`` is the executable's
    path, or None to locate it (:func:`nsight.find_tool`)."""
    ncu = find_tool("ncu", ncu)
    rows = _csv(ncu, report, "details")
    header = rows[0]
    col = {name: i for i, name in enumerate(header)}
    per_launch: dict[str, dict[str, float]] = {}
    names: dict[str, str] = {}
    wanted = {(section, metric): label for label, section, metric in _DETAILS}
    for row in rows[1:]:
        key = (row[col["Section Name"]], row[col["Metric Name"]])
        if key in wanted:
            value = _number(row[col["Metric Value"]])
            if value is not None:
                per_launch.setdefault(row[col["ID"]], {})[wanted[key]] = value
                names[row[col["ID"]]] = row[col["Kernel Name"]]
    raw = _csv(ncu, report, "raw")
    raw_header, stalls = raw[0], {}
    for row in raw[2:]:
        for i, name in enumerate(raw_header):
            if name.startswith(_STALL_PREFIX) and not name.endswith("not_issued"):
                value = _number(row[i])
                if value is not None:
                    reason = name[len(_STALL_PREFIX) :]
                    stalls[reason] = stalls.get(reason, 0.0) + value
    total = sum(stalls.values()) or 1.0
    launches = list(per_launch.values())
    metrics = {
        label: statistics.mean(l[label] for l in launches if label in l)
        for label, _, _ in _DETAILS
        if any(label in l for l in launches)
    }
    return {
        "schema": "pbat-sim-ncu/1",
        "kernel": next(iter(names.values()), ""),
        "launches": len(launches),
        "metrics": metrics,
        "per_launch_duration_us": [l.get("duration_us") for l in launches],
        "stall_share": {k: v / total for k, v in sorted(stalls.items(), key=lambda kv: -kv[1])},
    }


_REPO_MARKER = "python/tools/sim/"
_STALL_COLUMNS = ("stall_long_sb", "stall_wait", "stall_short_sb", "stall_lg", "stall_mio", "stall_math")


def _short_path(path: str) -> str:
    path = path.replace("\\", "/")
    if _REPO_MARKER in path:
        return path[path.index(_REPO_MARKER) + len(_REPO_MARKER) :]
    return path.rsplit("/", 1)[-1]


_function_cache: dict[str, list[tuple[int, int, str]]] = {}


def _enclosing_function(path: str, line: int) -> str:
    """Name of the innermost Python function containing ``line`` in the source file ``path``."""
    if path not in _function_cache:
        spans = []
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    spans.append((node.lineno, node.end_lineno or node.lineno, node.name))
        except (OSError, SyntaxError):
            pass
        _function_cache[path] = sorted(spans, key=lambda s: s[1] - s[0])
    return next((name for lo, hi, name in _function_cache[path] if lo <= line <= hi), "(module)")


def source_hotspots(report: str, ncu: str | None = None, top: int = 30) -> dict:
    """Stall samples, stall reasons and executed instructions by source location, from the source page
    of a report recorded with line information. Values are summed over launches.

    ``lines`` and ``files`` use the innermost location, which for inlined Warp builtins is a native
    header. ``python_lines`` and ``functions`` instead fold each native-header instruction into the
    nearest preceding instruction (by address) that maps to a simulator source line, its likely
    caller, since inlined code is laid out next to the code that calls it. That folding is a heuristic.
    ``ncu`` is the executable's path, or None to locate it (:func:`nsight.find_tool`).
    """
    ncu = find_tool("ncu", ncu)
    rows = _csv(ncu, report, "source", "--print-source", "cuda,sass")
    keys = ("samples", "instructions") + _STALL_COLUMNS
    col: dict[str, int] = {}
    path, current = "", None
    lines: dict[tuple[str, int], dict] = {}
    instructions: list[tuple[int, tuple[str, int, str], list[float]]] = []

    def values(row):
        return [
            (_number(row[col[k]]) or 0.0) if k in col and col[k] < len(row) else 0.0 for k in keys
        ]

    for row in rows:
        if not row:
            continue
        if row[0] == "File Path":
            path = row[1].replace("\\", "/")
        elif row[0] == "Line No":
            col = {}
            for i, name in enumerate(row):
                col.setdefault(name, i)  # first occurrence: the issued-sample stall columns
            col["samples"] = col["Warp Stall Sampling (All Samples)"]
            col["instructions"] = col["Instructions Executed"]
        elif row[0].isdigit() and col:
            current = (path, int(row[0]), row[1].strip())
            entry = lines.setdefault(current[:2], {"location": current, **{k: 0.0 for k in keys}})
            for k, v in zip(keys, values(row)):
                entry[k] += v
        elif row[0] == "" and len(row) > 2 and row[2].startswith("0x") and current and col:
            instructions.append((int(row[2], 16), current, values(row)))

    folded: dict[tuple[str, int], dict] = {}
    owner = None
    for _, location, vals in sorted(instructions, key=lambda i: i[0]):
        if _REPO_MARKER in location[0]:
            owner = location
        target = owner or location
        entry = folded.setdefault(target[:2], {"location": target, **{k: 0.0 for k in keys}})
        for k, v in zip(keys, vals):
            entry[k] += v

    def finish(entries, top_n=None):
        total_s = sum(e["samples"] for e in entries.values()) or 1.0
        total_i = sum(e["instructions"] for e in entries.values()) or 1.0
        out = []
        for e in sorted(entries.values(), key=lambda e: -e["samples"]):
            path_, line, source = e["location"]
            item = {"file": _short_path(path_), "line": line, "source": source,
                    **{k: e[k] for k in keys},
                    "sample_share": e["samples"] / total_s, "instruction_share": e["instructions"] / total_i}
            out.append(item)
        return out[:top_n] if top_n else out

    def group(entries, by):
        groups: dict[str, dict] = {}
        for e in entries.values():
            name = by(e["location"])
            g = groups.setdefault(name, {"location": (name, 0, ""), **{k: 0.0 for k in keys}})
            for k in keys:
                g[k] += e[k]
        return groups

    files = finish(group(lines, lambda loc: loc[0]))
    functions = finish(group(folded, lambda loc: (
        f"{_short_path(loc[0])}::{_enclosing_function(loc[0], loc[1])}" if _REPO_MARKER in loc[0]
        else f"{_short_path(loc[0])} (no simulator caller)")))
    for f in files:
        f.pop("line"), f.pop("source")
    for f in functions:
        f["function"] = f.pop("file")
        f.pop("line"), f.pop("source")
    return {
        "lines": finish(lines, top),
        "files": files,
        "python_lines": finish(folded, top),
        "functions": functions,
    }


def run(
    out_prefix: str,
    kernel: str,
    skip: int,
    count: int,
    simulate_args: list[str],
    ncu: str | None = None,
    lineinfo: bool = False,
) -> dict:
    """Profile matching kernel launches of simulate.py under Nsight Compute and summarize them, with
    source hotspots if ``lineinfo``. ``ncu`` is the executable's path, or None to locate it
    (:func:`nsight.find_tool`)."""
    ncu = find_tool("ncu", ncu)
    venvrun = os.path.join(os.path.dirname(__file__), "venvrun.py")
    subprocess.run(
        [
            ncu,
            "--kernel-name",
            f"regex:{kernel}",
            "--launch-skip",
            str(skip),
            "--launch-count",
            str(count),
            "--set",
            "full",
            *(["--import-source", "yes"] if lineinfo else []),
            "-f",
            "-o",
            out_prefix,
            os.path.join(sys.base_prefix, "python.exe"),
            venvrun,
            sys.prefix,
            "python.tools.sim.simulate",
            *simulate_args,
        ],
        check=True,
        env=dict(os.environ, PBAT_WARP_LINEINFO="1") if lineinfo else None,
    )
    summary = summarize(out_prefix + ".ncu-rep", ncu)
    if lineinfo:
        summary["hotspots"] = source_hotspots(out_prefix + ".ncu-rep", ncu)
    return summary


_NCU_HELP = "Path to the ncu executable (default: $NCU, PATH, then the standard install locations)."


def main():
    """Command line: run (profile and summarize) or summarize (an existing report)."""
    parser = argparse.ArgumentParser(description="Nsight Compute metrics for one simulator kernel")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Profile kernels under ncu and summarize them.")
    p_run.add_argument("-o", "--out", required=True, help="Output prefix.", dest="out")
    p_run.add_argument("--kernel", required=True, help="Regex matching the kernel name.")
    p_run.add_argument("--skip", type=int, default=0, help="Matching launches to skip first.")
    p_run.add_argument("--count", type=int, default=1, help="Matching launches to profile.")
    p_run.add_argument("--ncu", default=None, help=_NCU_HELP)
    p_run.add_argument("--lineinfo", action="store_true", help="Add per-source-line hotspots.")
    p_run.add_argument("simulate_args", nargs=argparse.REMAINDER, help="-- simulate.py arguments")
    p_sum = sub.add_parser("summarize", help="Summarize an existing .ncu-rep report.")
    p_sum.add_argument("report")
    p_sum.add_argument("-o", "--out", required=True, help="Output .json.", dest="out")
    p_sum.add_argument("--ncu", default=None, help=_NCU_HELP)
    p_sum.add_argument("--lineinfo", action="store_true", help="Add hotspots (report recorded with line info).")
    args = parser.parse_args()
    if args.command == "run":
        simulate_args = args.simulate_args
        if simulate_args and simulate_args[0] == "--":
            simulate_args = simulate_args[1:]
        summary = run(
            args.out, args.kernel, args.skip, args.count, simulate_args, args.ncu, args.lineinfo
        )
        out = args.out + ".ncu.json"
    else:
        summary = summarize(args.report, args.ncu)
        if args.lineinfo:
            summary["hotspots"] = source_hotspots(args.report, args.ncu)
        out = args.out
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    m = summary["metrics"]
    print(f"{summary['kernel']}: {summary['launches']} launches, mean {m.get('duration_us', 0):.2f} us")
    for k, v in m.items():
        print(f"  {k:30s} {v:12.2f}")
    print("  stalls:", ", ".join(f"{k} {100 * v:.0f}%" for k, v in list(summary["stall_share"].items())[:6]))
    if "hotspots" in summary:
        print("  hottest source lines (share of stall samples, of instructions):")
        for h in summary["hotspots"]["lines"][:12]:
            print(f"    {100 * h['sample_share']:5.1f}%  {100 * h['instruction_share']:5.1f}%  "
                  f"{h['file']}:{h['line']}  {h['source'][:70]}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

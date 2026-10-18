"""
Nsight Compute metrics for one kernel of the GPU simulator.

Runs ``simulate.py`` under Nsight Compute, profiles ``--count`` consecutive launches of the kernels
matching ``--kernel`` after skipping ``--skip`` launches (e.g. the 9 color launches of one VBD sweep),
and summarizes them in JSON: duration, occupancy, throughput, instruction counts and the warp stall
breakdown from PC sampling. Nsight Compute locks clocks to base and flushes caches between passes by
default, so durations are longer than in normal runs; compare kernels with each other, not with
Nsight Systems timings.

Usage (from the repo root):

    python -m python.tools.sim.profiling.ncu run -o out/aaavbd_ncu --kernel accelerated_vertex_solve \\
        --skip 900 --count 9 -- --fem-elasto-dynamics scene.h5:fem --solver AAAVBD --duration 0.05 -o x.h5
"""

import argparse
import csv
import io
import json
import os
import statistics
import subprocess
import sys

DEFAULT_NCU = (
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe"
)

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


def _csv(ncu: str, report: str, page: str) -> list[list[str]]:
    out = subprocess.run(
        [ncu, "--import", report, "--csv", "--page", page], check=True, capture_output=True, text=True
    ).stdout
    return list(csv.reader(io.StringIO(out)))


def summarize(report: str, ncu: str = DEFAULT_NCU) -> dict:
    """Mean metrics over the profiled launches of an ``.ncu-rep`` report."""
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


def run(out_prefix: str, kernel: str, skip: int, count: int, simulate_args: list[str], ncu: str) -> dict:
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
    )
    return summarize(out_prefix + ".ncu-rep", ncu)


def main():
    parser = argparse.ArgumentParser(description="Nsight Compute metrics for one simulator kernel")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Profile kernels under ncu and summarize them.")
    p_run.add_argument("-o", "--out", required=True, help="Output prefix.", dest="out")
    p_run.add_argument("--kernel", required=True, help="Regex matching the kernel name.")
    p_run.add_argument("--skip", type=int, default=0, help="Matching launches to skip first.")
    p_run.add_argument("--count", type=int, default=1, help="Matching launches to profile.")
    p_run.add_argument("--ncu", default=DEFAULT_NCU, help="Path to ncu.exe.")
    p_run.add_argument("simulate_args", nargs=argparse.REMAINDER, help="-- simulate.py arguments")
    p_sum = sub.add_parser("summarize", help="Summarize an existing .ncu-rep report.")
    p_sum.add_argument("report")
    p_sum.add_argument("-o", "--out", required=True, help="Output .json.", dest="out")
    p_sum.add_argument("--ncu", default=DEFAULT_NCU, help="Path to ncu.exe.")
    args = parser.parse_args()
    if args.command == "run":
        simulate_args = args.simulate_args
        if simulate_args and simulate_args[0] == "--":
            simulate_args = simulate_args[1:]
        summary = run(args.out, args.kernel, args.skip, args.count, simulate_args, args.ncu)
        out = args.out + ".ncu.json"
    else:
        summary = summarize(args.report, args.ncu)
        out = args.out
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    m = summary["metrics"]
    print(f"{summary['kernel']}: {summary['launches']} launches, mean {m.get('duration_us', 0):.2f} us")
    for k, v in m.items():
        print(f"  {k:30s} {v:12.2f}")
    print("  stalls:", ", ".join(f"{k} {100 * v:.0f}%" for k, v in list(summary["stall_share"].items())[:6]))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

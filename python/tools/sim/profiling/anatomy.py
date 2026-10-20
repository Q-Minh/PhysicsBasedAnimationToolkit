"""
Sweep anatomy: how one VBD sweep runs inside the production CUDA graph.

Profiles production steps (graph replay, profiling off) under Nsight Systems with per-node graph
tracing, then splits each step's graph-node activities into sweeps. A sweep window runs from the end
of one run of vertex-solve kernels (one per color) to the end of the next. Windows whose sequence of
activities is the most common one are the steady-state sweeps; the others also contain subproblem
boundary work (dual updates, contact detection, constraint sets). For the steady state, the anatomy
lists every node in launch order with its mean duration and the mean idle gap before it, so the gap
time between nodes, which graph replay adds on top of kernel work, is attributed to specific nodes.

Usage (from the repo root):

    python -m python.tools.sim.profiling.anatomy run -o out/aaavbd_graph [--phases out/aaavbd.json] -- \\
        --fem-elasto-dynamics scene.h5:fem --solver AAAVBD --duration 0.3 --profile-warmup 10 -o x.h5

``--phases`` takes an eager profile of the same solver (nsys or kernels backend) to label each node
with its profiling phase.
"""

import argparse
import collections
import re
import sqlite3
import statistics

from . import nsys, results
from ..gpu.common.profiling import interval_union_ns

_VERTEX_SOLVE = re.compile(r"vertex_solve")
_WARP_SUFFIX = re.compile(r"_[0-9a-f]{8}_cuda_kernel_forward$")


def label(name: str) -> str:
    """Readable activity name: Warp kernels lose their hash suffix and leading underscore."""
    return _WARP_SUFFIX.sub("", name).lstrip("_")


def _graph_activities(path: str) -> dict[int, list[tuple[int, int, str]]]:
    """Graph-node activities per graph launch (correlation id): [(start, end, name)] sorted by start."""
    db = sqlite3.connect(path)
    strings = nsys._strings(db)
    tables = nsys._tables(db)
    launches: dict[int, list] = collections.defaultdict(list)
    queries = [
        ("CUPTI_ACTIVITY_KIND_KERNEL", "start, end, correlationId, shortName, graphNodeId", True),
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "start, end, correlationId, copyKind, graphNodeId", False),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "start, end, correlationId, 0, graphNodeId", False),
    ]
    memcpy_kinds = {}
    if "ENUM_CUDA_MEMCPY_OPER" in tables:
        memcpy_kinds = dict(db.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"))
    for table, columns, is_kernel in queries:
        if table not in tables:
            continue
        for start, end, cid, name, node in db.execute(f"SELECT {columns} FROM {table}"):
            if not node:
                continue  # not launched from a graph
            if is_kernel:
                name = label(strings.get(name, "?"))
            elif table.endswith("MEMCPY"):
                name = f"memcpy {memcpy_kinds.get(name, name)}"
            else:
                name = "memset"
            launches[cid].append((start, end, name))
    return {cid: sorted(acts) for cid, acts in launches.items()}


def _sweep_windows(acts: list[tuple[int, int, str]]) -> list[list[tuple[int, int, str]]]:
    """Split one step's activities into windows, each ending with a run of vertex-solve kernels."""
    windows, current, in_run = [], [], False
    for act in acts:
        solve = bool(_VERTEX_SOLVE.search(act[2]))
        if in_run and not solve:
            windows.append(current)
            current = []
        current.append(act)
        in_run = solve
    if in_run:
        windows.append(current)
    return windows


def _phase_map(phases_result: dict | None) -> dict[str, str]:
    """Activity label -> profiling phase, from an eager profile (most frequent phase per label)."""
    if phases_result is None:
        return {}
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for k in phases_result["kernels"]:
        votes[label(k["name"])][results.leaf(k["path"])] += k["count"]
    return {name: counter.most_common(1)[0][0] for name, counter in votes.items()}


def analyze(path: str, metadata_result: dict | None = None, phases_result: dict | None = None) -> dict:
    """Sweep anatomy of a graph-mode Nsight Systems SQLite export."""
    phase_of = _phase_map(phases_result)
    steps, windows = [], []
    for cid, acts in _graph_activities(path).items():
        if len(acts) < 2:
            continue
        intervals = [(s, e) for s, e, _ in acts]
        busy = sum(e - s for s, e in intervals)
        steps.append(
            {
                "span_ms": (acts[-1][1] - acts[0][0]) * 1e-6,
                "busy_ms": busy * 1e-6,
                "union_ms": interval_union_ns(intervals) * 1e-6,
                "activities": len(acts),
            }
        )
        windows.extend(_sweep_windows(acts)[1:])  # the first window also holds the step's setup
    # Concurrent nodes (e.g. dual updates on 4 streams) start in varying order, so steady windows are
    # those with the most common multiset of nodes; the node-by-node anatomy uses the most common order.
    content = lambda w: tuple(sorted(name for _, _, name in w))
    steady_content, n_steady = (
        collections.Counter(content(w) for w in windows).most_common(1)[0] if windows else ((), 0)
    )
    steady = [w for w in windows if content(w) == steady_content]
    boundary = [w for w in windows if content(w) != steady_content]
    order = collections.Counter(tuple(name for _, _, name in w) for w in steady)
    steady_signature = order.most_common(1)[0][0] if order else ()
    ordered = [w for w in steady if tuple(name for _, _, name in w) == steady_signature]
    nodes = []
    for i, name in enumerate(steady_signature):
        durations, gaps, starts = [], [], []
        for w in ordered:
            start, end, _ = w[i]
            frontier = max((e for _, e, _ in w[:i]), default=start)
            durations.append((end - start) * 1e-3)
            gaps.append(max(0, start - frontier) * 1e-3)
            starts.append((start - w[0][0]) * 1e-3)
        nodes.append(
            {
                "name": name,
                "phase": phase_of.get(name, ""),
                "start_us": statistics.mean(starts),
                "duration_us": statistics.mean(durations),
                "gap_before_us": statistics.mean(gaps),
            }
        )
    span = lambda w: (w[-1][1] - w[0][0]) * 1e-3
    union = lambda w: interval_union_ns([(s, e) for s, e, _ in w]) * 1e-3
    n_steps = max(len(steps), 1)
    budget = {
        "sweep_kernels_ms": sum(union(w) for w in steady) * 1e-3 / n_steps,
        "sweep_gaps_ms": sum(span(w) - union(w) for w in steady) * 1e-3 / n_steps,
        "boundary_kernels_ms": sum(union(w) for w in boundary) * 1e-3 / n_steps,
        "boundary_gaps_ms": sum(span(w) - union(w) for w in boundary) * 1e-3 / n_steps,
    }
    mean_span = statistics.mean(st["span_ms"] for st in steps) if steps else 0.0
    budget["other_ms"] = max(0.0, mean_span - sum(budget.values()))
    # Per boundary window and phase: kernel time, and idle gaps before the phase's nodes.
    phases: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    for w in boundary:
        frontier = w[0][0]
        for start, end, name in w:
            acc = phases[phase_of.get(name, name)]
            acc[0] += (end - start) * 1e-3
            acc[1] += max(0, start - frontier) * 1e-3
            acc[2] += 1
            frontier = max(frontier, end)
    n_boundary = max(len(boundary), 1)
    return {
        "schema": "pbat-sim-anatomy/1",
        "metadata": dict(metadata_result["metadata"]) if metadata_result else {},
        "steps": steps,
        "budget": budget,
        "sweep": {
            "windows": len(windows),
            "steady_windows": n_steady,
            "span_us": statistics.mean(span(w) for w in steady) if steady else 0.0,
            "busy_us": sum(n["duration_us"] for n in nodes),
            "gap_us": sum(n["gap_before_us"] for n in nodes),
            "nodes": nodes,
        },
        "boundary_windows": {
            "count": len(boundary),
            "per_step": len(boundary) / n_steps,
            "mean_span_us": statistics.mean(span(w) for w in boundary) if boundary else 0.0,
            "mean_activities": statistics.mean(len(w) for w in boundary) if boundary else 0.0,
            "phases": [
                {"phase": p, "kernel_us": v[0] / n_boundary, "gap_before_us": v[1] / n_boundary,
                 "nodes": v[2] / n_boundary}
                for p, v in sorted(phases.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
            ],
        },
    }


def format_summary(anatomy: dict) -> str:
    """Text summary of a sweep anatomy: step budget, boundary phases and the steady sweep node by
    node."""
    sweep = anatomy["sweep"]
    meta = anatomy["metadata"]
    steps = anatomy["steps"]
    lines = [
        f"{meta.get('solver', '?')}: {len(steps)} graph replays, "
        f"span {statistics.mean(s['span_ms'] for s in steps):.3f} ms/step, kernel union "
        f"{statistics.mean(s['union_ms'] for s in steps):.3f} ms/step"
        if steps
        else "no graph replays found",
        f"Steady sweep ({sweep['steady_windows']} of {sweep['windows']} windows): span "
        f"{sweep['span_us']:.1f} us = {sweep['busy_us']:.1f} us busy + {sweep['gap_us']:.1f} us gaps",
    ]
    budget = anatomy["budget"]
    lines.append("Per step: " + ", ".join(f"{k[:-3].replace('_', ' ')} {v:.2f} ms" for k, v in budget.items()))
    bw = anatomy["boundary_windows"]
    lines.append(f"Boundary windows: {bw['per_step']:.1f}/step, {bw['mean_span_us']:.1f} us each")
    for ph in bw["phases"][:6]:
        lines.append(f"  {ph['phase']:28s} kernels {ph['kernel_us']:8.1f} us  gaps {ph['gap_before_us']:8.1f} us  "
                     f"nodes {ph['nodes']:6.1f}")
    for n in sweep["nodes"]:
        lines.append(
            f"  gap {n['gap_before_us']:6.2f} us  run {n['duration_us']:7.2f} us  {n['name']}"
            + (f"  [{n['phase']}]" if n["phase"] else "")
        )
    return "\n".join(lines)


def main():
    """Command line: run (profile and analyze) or analyze (an existing SQLite export)."""
    parser = argparse.ArgumentParser(description="Sweep anatomy of the production CUDA graph")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Profile production steps under nsys and analyze them.")
    p_run.add_argument("-o", "--out", required=True, help="Output prefix.", dest="out")
    p_run.add_argument("--nsys", default=None, help="Path to the nsys executable (default: $NSYS, PATH, then the standard install locations).")
    p_run.add_argument("--phases", default=None, help="Eager profile .json to label phases.")
    p_run.add_argument("simulate_args", nargs=argparse.REMAINDER, help="-- simulate.py arguments")
    p_analyze = sub.add_parser("analyze", help="Analyze a graph-mode nsys SQLite export.")
    p_analyze.add_argument("sqlite")
    p_analyze.add_argument("--metadata", default=None, help="JSON written by simulate.py.")
    p_analyze.add_argument("--phases", default=None, help="Eager profile .json to label phases.")
    p_analyze.add_argument("-o", "--out", required=True, help="Output .json.", dest="out")
    args = parser.parse_args()
    phases = results.load(args.phases) if args.phases else None
    if args.command == "run":
        simulate_args = args.simulate_args
        if simulate_args and simulate_args[0] == "--":
            simulate_args = simulate_args[1:]
        sqlite, metadata_json = nsys.profile(args.out, simulate_args, args.nsys, graph=True)
        anatomy = analyze(sqlite, results.load(metadata_json), phases)
        out = args.out + ".anatomy.json"
    else:
        metadata = results.load(args.metadata) if args.metadata else None
        anatomy = analyze(args.sqlite, metadata, phases)
        out = args.out
    results.save(anatomy, out)
    print(format_summary(anatomy))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

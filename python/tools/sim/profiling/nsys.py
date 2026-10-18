"""
Nsight Systems profiling of the offline GPU simulator.

``run`` profiles ``simulate.py --profile nvtx`` under Nsight Systems, exports the report to SQLite and
imports it. ``import`` converts an existing SQLite export. Both write the result schema of
``profiling/results.py`` (``source: "nsys"``), so reports treat them like the in-process ``kernels``
backend. Every GPU activity (Warp, ``cuda.compute``/CUB and cupy kernels, memsets, memcpys) is timed by
CUPTI from its actual start and end on the GPU, and attributed to the innermost ``pbat`` NVTX range that
was open on the launching thread when the launch API call was made.

Per step and scope path, the result holds:
- ``busy_ms``: sum of activity durations (comparable with the ``kernels`` backend),
- ``union_ms``: duration of the union of activity intervals, which does not double-count work running
  concurrently on several streams,
plus ``step_union_ms`` (union over the whole step) and ``step_span_ms`` (first activity start to last
activity end).

Usage (from the repo root):

    python -m python.tools.sim.profiling.nsys run -o out/aaavbd -- \\
        --fem-elasto-dynamics scene.h5:fem --solver AAAVBD --duration 0.3 -o out/aaavbd.h5
    python -m python.tools.sim.profiling.nsys import out/aaavbd.sqlite --metadata out/aaavbd.json \\
        -o out/aaavbd.json
"""

import argparse
import os
import sqlite3
import subprocess
import sys

from . import results
from ..gpu.common.profiling import interval_union_ns

DEFAULT_NSYS = (
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe"
)
_NVTX_PUSH_POP_RANGE = 59
_NVTX_DOMAIN_CREATE = 75
_DOMAIN = "pbat"
_STEP = "step"


def _tables(db: sqlite3.Connection) -> set[str]:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _strings(db: sqlite3.Connection) -> dict[int, str]:
    return dict(db.execute("SELECT id, value FROM StringIds"))


def _pbat_ranges(db: sqlite3.Connection, strings: dict[int, str]) -> dict[int, list]:
    """Push/pop ranges of the pbat NVTX domain, per thread: [(start, end, name)] sorted by start."""
    domain_ids = [
        domain_id
        for domain_id, text, text_id in db.execute(
            "SELECT domainId, text, textId FROM NVTX_EVENTS WHERE eventType = ?",
            (_NVTX_DOMAIN_CREATE,),
        )
        if (text if text is not None else strings.get(text_id)) == _DOMAIN
    ]
    if not domain_ids:
        # The domain is created at its first range, which may precede the capture range: identify it
        # by its step ranges instead.
        domain_ids = [
            domain_id
            for domain_id, text, text_id in db.execute(
                "SELECT DISTINCT domainId, text, textId FROM NVTX_EVENTS WHERE eventType = ?",
                (_NVTX_PUSH_POP_RANGE,),
            )
            if (text if text is not None else strings.get(text_id)) == _STEP
        ]
    if not domain_ids:
        raise ValueError(f"No NVTX ranges in the '{_DOMAIN}' domain. Was --profile nvtx used?")
    ranges: dict[int, list] = {}
    query = (
        "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS "
        f"WHERE eventType = ? AND domainId IN ({','.join('?' * len(domain_ids))}) "
        "AND end IS NOT NULL ORDER BY start"
    )
    for start, end, tid, text, text_id in db.execute(query, (_NVTX_PUSH_POP_RANGE, *domain_ids)):
        name = text if text is not None else strings.get(text_id, "?")
        ranges.setdefault(tid, []).append((start, end, name))
    return ranges


def _launches(db: sqlite3.Connection) -> dict[int, tuple[int, int]]:
    """correlationId -> (API call start, globalTid) for every CUDA runtime/driver API call."""
    launches = {}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in _tables(db):
            for correlation_id, start, tid in db.execute(
                f"SELECT correlationId, start, globalTid FROM {table}"
            ):
                launches[correlation_id] = (start, tid)
    return launches


def _activities(db: sqlite3.Connection, strings: dict[int, str]) -> list[tuple]:
    """GPU activities: (start, end, correlationId, name)."""
    tables = _tables(db)
    out = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        for start, end, cid, name_id in db.execute(
            "SELECT start, end, correlationId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            out.append((start, end, cid, strings.get(name_id, "?")))
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for start, end, cid in db.execute(
            "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET"
        ):
            out.append((start, end, cid, "memset"))
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        kinds = {}
        if "ENUM_CUDA_MEMCPY_OPER" in tables:
            kinds = dict(db.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"))
        for start, end, cid, kind in db.execute(
            "SELECT start, end, correlationId, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ):
            out.append((start, end, cid, f"memcpy {kinds.get(kind, kind)}"))
    return out


class _ThreadScopes:
    """Innermost-range attribution on one thread, for properly nested push/pop ranges."""

    def __init__(self, ranges: list[tuple[int, int, str]]):
        self.ranges = ranges
        # Per range: the index of its enclosing top-level step range (-1 outside any step), and its
        # scope path below that step.
        self.steps: list[int] = []
        self.paths: list[str] = []
        stack: list[int] = []
        n_steps = 0
        for i, (start, _, name) in enumerate(ranges):
            while stack and ranges[stack[-1]][1] <= start:
                stack.pop()
            if not stack:
                self.steps.append(n_steps if name == _STEP else -1)
                n_steps += name == _STEP
                self.paths.append("" if name == _STEP else name)
            else:
                self.steps.append(self.steps[stack[0]])
                parent = self.paths[stack[-1]]
                self.paths.append(f"{parent}/{name}" if parent else name)
            stack.append(i)

    def calls(self) -> dict[int, dict[str, int]]:
        """Number of entries per scope path, per step index."""
        out: dict[int, dict[str, int]] = {}
        for step, path in zip(self.steps, self.paths):
            if step >= 0 and path:
                per_path = out.setdefault(step, {})
                per_path[path] = per_path.get(path, 0) + 1
        return out

    def attribute(self, queries: list[tuple[int, int]]) -> dict[int, tuple[int, str]]:
        """Map (time, key) queries to (step index, scope path) of the innermost range open at that
        time, skipping times outside any step."""
        out = {}
        stack: list[int] = []
        ri = 0
        for t, key in sorted(queries):
            while ri < len(self.ranges) and self.ranges[ri][0] <= t:
                while stack and self.ranges[stack[-1]][1] <= self.ranges[ri][0]:
                    stack.pop()
                stack.append(ri)
                ri += 1
            while stack and self.ranges[stack[-1]][1] <= t:
                stack.pop()
            if stack and self.steps[stack[-1]] >= 0:
                out[key] = (self.steps[stack[-1]], self.paths[stack[-1]])
        return out


def import_sqlite(path: str, metadata_result: dict | None = None) -> dict:
    """Convert an Nsight Systems SQLite export to the profiling result schema.

    Args:
        path (str): The SQLite export of a ``simulate.py --profile nvtx`` run.
        metadata_result (dict | None): The JSON that run wrote with ``--profile-out``, whose metadata
            (and warm-up step count) the result reuses.
    """
    db = sqlite3.connect(path)
    strings = _strings(db)
    scopes = {tid: _ThreadScopes(r) for tid, r in _pbat_ranges(db, strings).items()}
    activities = _activities(db, strings)
    launches = _launches(db)
    queries: dict[int, list[tuple[int, int]]] = {}
    for _, _, cid, _ in activities:
        launch = launches.get(cid)
        if launch is not None and launch[1] in scopes:
            queries.setdefault(launch[1], []).append((launch[0], cid))
    attributed: dict[int, tuple[int, str]] = {}
    for tid, thread_queries in queries.items():
        attributed.update(scopes[tid].attribute(thread_queries))
    calls: dict[int, dict[str, int]] = {}
    for thread_scopes in scopes.values():
        for step, per_path in thread_scopes.calls().items():
            for path, n in per_path.items():
                calls.setdefault(step, {})[path] = calls.get(step, {}).get(path, 0) + n
    per_step: dict[int, dict[str, list]] = {}
    kernels: dict[tuple[str, str], list] = {}
    for start, end, cid, name in activities:
        if cid not in attributed:
            continue
        step, scope_path = attributed[cid]
        per_step.setdefault(step, {}).setdefault(scope_path, []).append((start, end))
        acc = kernels.setdefault((scope_path, name), [0.0, 0])
        acc[0] += (end - start) * 1e-6
        acc[1] += 1
    metadata = dict(metadata_result["metadata"]) if metadata_result else {}
    warmup = metadata.get("warmup_steps", 0)
    steps = []
    for step in sorted(per_step):
        by_path = per_step[step]
        everything = [interval for intervals in by_path.values() for interval in intervals]
        steps.append(
            {
                "t": warmup + step,
                "busy_ms": {p: sum(e - s for s, e in iv) * 1e-6 for p, iv in by_path.items()},
                "launches": {p: len(iv) for p, iv in by_path.items()},
                "calls": calls.get(step, {}),
                "union_ms": {p: interval_union_ns(iv) * 1e-6 for p, iv in by_path.items()},
                "step_union_ms": interval_union_ns(everything) * 1e-6,
                "step_span_ms": (max(e for _, e in everything) - min(s for s, _ in everything))
                * 1e-6,
            }
        )
    kernel_list = [
        {"path": p, "name": n, "busy_ms": ms, "count": count}
        for (p, n), (ms, count) in kernels.items()
    ]
    kernel_list.sort(key=lambda k: -k["busy_ms"])
    return {
        "schema": results.SCHEMA,
        "source": "nsys",
        "metadata": metadata,
        "steps": steps,
        "kernels": kernel_list,
    }


def _base_python() -> str:
    """The base interpreter of the running venv (Nsight Systems can't launch the venv stub)."""
    return os.path.join(sys.base_prefix, "python.exe")


def run(out_prefix: str, simulate_args: list[str], nsys: str) -> dict:
    """Profile ``simulate.py`` under Nsight Systems and import the result.

    Args:
        out_prefix (str): Prefix for the outputs ``.nsys-rep``, ``.sqlite`` and ``.json``.
        simulate_args (list[str]): Arguments for ``python.tools.sim.simulate``.
        nsys (str): Path to ``nsys.exe``.
    """
    out_json = out_prefix + ".json"
    profile_args = ["--profile", "nvtx", "--profile-out", out_json]
    venvrun = os.path.join(os.path.dirname(__file__), "venvrun.py")
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "-o",
        out_prefix,
        _base_python(),
        venvrun,
        sys.prefix,
        "python.tools.sim.simulate",
        *simulate_args,
        *profile_args,
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            nsys,
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "-o",
            out_prefix + ".sqlite",
            out_prefix + ".nsys-rep",
        ],
        check=True,
    )
    result = import_sqlite(out_prefix + ".sqlite", results.load(out_json))
    results.save(result, out_json)
    return result


def main():
    parser = argparse.ArgumentParser(description="Nsight Systems profiling of the GPU simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Profile simulate.py under nsys and import the result.")
    p_run.add_argument("-o", "--out", required=True, help="Output prefix.", dest="out")
    p_run.add_argument("--nsys", default=DEFAULT_NSYS, help="Path to nsys.exe.")
    p_run.add_argument("simulate_args", nargs=argparse.REMAINDER, help="-- simulate.py arguments")
    p_import = sub.add_parser("import", help="Import an nsys SQLite export.")
    p_import.add_argument("sqlite")
    p_import.add_argument(
        "--metadata", default=None, help="JSON written by simulate.py --profile-out."
    )
    p_import.add_argument("-o", "--out", required=True, help="Output .json.", dest="out")
    args = parser.parse_args()
    if args.command == "run":
        simulate_args = args.simulate_args
        if simulate_args and simulate_args[0] == "--":
            simulate_args = simulate_args[1:]
        result = run(args.out, simulate_args, args.nsys)
        out = args.out + ".json"
    else:
        metadata = results.load(args.metadata) if args.metadata else None
        result = import_sqlite(args.sqlite, metadata)
        results.save(result, args.out)
        out = args.out
    print(results.format_summary(result))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

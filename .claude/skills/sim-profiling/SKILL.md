---
name: sim-profiling
description: Profile the GPU simulator (python/tools/sim) — choose between production timing, the in-process CUPTI `kernels` backend, NVTX + Nsight Systems, the graph sweep anatomy and Nsight Compute; run them through simulate.py and the profiling/ tools; and produce JSON results or HTML reports. Use when asked where a solver's time goes, how fast a step is, why a phase is slow, to compare before/after a change, or for a profiling report.
---

# Profiling the GPU simulator

Profiling has three layers:
- **Scopes:** solver phases are wrapped in `profiling.scope("vbd.sweep")` (`gpu/common/profiling.py`), which nest into paths such as `solve/aaavbd.subproblem/aaavbd.sweep`.
- **Backend:** one per process, decides what a scope does.
- **Tools:** `python/tools/sim/profiling/` runs the simulator and writes JSON (schema `pbat-sim-profile/1`, `profiling/results.py`) and HTML reports.

Context and the findings so far are in `python/tools/sim/CLAUDE.md` ("Profiling") and `plans/02-profiling.md`.

Run from the repo root with `/c/git/python-3.12/Scripts/python.exe` (Bash).

## Choose the measurement

| Question | Method | Execution | Pros | Cons |
|---|---|---|---|---|
| How long does a step really take? Before/after a change? | **Production timing**: `--profile off --profile-out f.json` | CUDA graph replay | The real cost: includes graph scheduling gaps. Cheap | One number per step, no phases |
| Where does kernel time go, by phase? | **`kernels` backend** (CUPTI, in-process) | Eager | Per-scope busy time, launches, calls, union. Within 0.6% (sweep) / 2.9% (step) of Nsight. Quick, no external tools | Eager, so no graph gaps (a lower bound on cost). Sub-µs kernels read ~0.2 µs long, inflating tiny-kernel contact phases 27–40% |
| Reference per-phase timings, or a timeline to inspect | **`nvtx` + Nsight Systems** (`profiling.nsys run`) | Eager | Ground truth for kernel durations. Viewable in the Nsight Systems GUI | Needs nsys. Slower turnaround. Still eager |
| Why is production slower than kernel work? Where are the idle gaps? | **Sweep anatomy** (`profiling.anatomy run`) | Graph replay under nsys | Node-by-node graph timeline: sweep vs boundary, kernels vs gaps, labelled by phase | Needs nsys and an eager profile for labels |
| Why is one kernel slow (occupancy, stalls, memory, hot lines)? | **Nsight Compute** (`profiling.ncu run`, `--lineinfo`) | Eager | Stall reasons, per-Python-line hotspots. Insensitive to other GPU work | Locked base clocks and cache flushes: compare kernels with each other, not with nsys times |

Rules of thumb:
- Quote wall-clock cost from production timing.
- Quote phase breakdowns from `kernels` or nsys.
- Explain the difference between the two with the anatomy.
- `kernels` and nsys can't run in one process: use `--profile nvtx` under nsys.

## Commands

```bash
# Production step time (graph replay); 10 warm-up steps excluded
python -m python.tools.sim.simulate --fem-elasto-dynamics data/scenarios/trampoline.Dump.h5:fem -o out/x.h5 \
    --solver AAAVBD --duration 0.3 --profile-warmup 10 --profile-out out/aaavbd_prod.json

# Per-phase breakdown in-process (CUPTI); prints a summary and writes JSON
python -m python.tools.sim.simulate ... --profile kernels --profile-warmup 10 --profile-out out/aaavbd.json

# Nsight Systems reference (runs simulate.py --profile nvtx, imports into the same JSON schema)
python -m python.tools.sim.profiling.nsys run -o out/aaavbd -- \
    --fem-elasto-dynamics data/scenarios/trampoline.Dump.h5:fem --solver AAAVBD --duration 0.3 -o out/x.h5
python -m python.tools.sim.profiling.nsys import out/aaavbd.sqlite --metadata out/aaavbd.json -o out/aaavbd.json

# Sweep anatomy of the production graph (label phases with an eager profile)
python -m python.tools.sim.profiling.anatomy run -o out/aaavbd_graph --phases out/aaavbd.json -- \
    --fem-elasto-dynamics data/scenarios/trampoline.Dump.h5:fem --solver AAAVBD --duration 0.3 --profile-warmup 10 -o out/x.h5

# Nsight Compute on 9 launches (one sweep's colors) of the AAAVBD vertex solve, with per-line hotspots
python -m python.tools.sim.profiling.ncu run -o out/aaavbd_ncu --kernel ^_accelerated_vertex_solve --skip 900 --count 9 --lineinfo -- \
    --fem-elasto-dynamics data/scenarios/trampoline.Dump.h5:fem --solver AAAVBD --duration 0.05 -o out/x.h5
```

The `simulate.py` arguments after `--` are passed through unchanged, so `--sim-params`, `--set`, `--solver` and so on apply as usual (see the `sim-run` skill).

## Outputs: JSON and HTML

- **JSON:** every runner writes `pbat-sim-profile/1` results, from the `kernels` backend, nsys import, production runs or the GUI export.
  - `steps[]` hold per-path `busy_ms` (exclusive of nested scopes), `union_ms`, `launches` and `calls`, plus `step_union_ms` and `step_span_ms`. Production runs record `step_timeline_ms` instead.
  - `kernels[]` hold the busy time and count of each kernel.
- **HTML, per-phase comparison:** pass one result per solver or variant.

  ```bash
  python -m python.tools.sim.profiling.report out/vbd.json out/aaavbd.json -o out/report.html --note "trampoline, Dump params"
  ```

- **HTML, overheads:** a production time budget, the anatomy timeline, kernel comparisons and before/after timings.

  ```bash
  python -m python.tools.sim.profiling.overheads --anatomy AAAVBD=out/aaavbd_graph.anatomy.json \
      --ncu AAAVBD=out/aaavbd_ncu.ncu.json --production AAAVBD=out/aaavbd_prod.json --findings findings.txt -o out/overheads.html
  ```

- **Publishing:** to share a report, publish the HTML as an Artifact.

When the user asks for "HTML output", run the matching measurement (above) to JSON first, then `report` (phases across solvers) or `overheads` (production overheads). "JSON only" stops at `--profile-out`.

## GUI

`python -m python.tools.sim.ui ...` → **Profiling** tree node. Enabling it switches on the CUPTI backend, so steps run eagerly. It shows mean GPU busy time per step over a rolling window, by phase or by full path. **Export** writes JSON for `report`.

## Reading results correctly

- **Stopping rule:** with early exit (`gtol > 0`), ms/step is the time to pass the stopping test at that run's settings, **not** an equal-accuracy solver comparison. Always report sweeps/step (`calls` of `*.sweep`), outer iterations/step (`calls` of `solve.prepare_subproblem`), ms/sweep and whether steps converged or hit the budget, and name the stopping rule. For speed comparisons, set fixed budgets (`--set Solver.<S>.gtol=0`) or use the benchmark tooling (`sim-benchmarks` skill).
- **Exclusive times:** a parent scope excludes its children, and paths sum to the step. `busy_ms` double-counts streams that run concurrently; `union_ms` doesn't.
- **Eager vs graph:** eager kernel durations match graph mode within a few percent, but eager runs hide graph gaps.

## Before measuring

- **The GPU must be quiet.** Other processes (other sessions, the GUI) distort timings badly: an unchanged baseline drifted from 27.6 to 47 ms. While parallel sessions run, reserve a window via `plans/MASTER.md` (`GPU run START`/`END`). `nvidia-smi` shows other users.
- **Warm-up:** use `--profile-warmup` ≥ 5 so module loading and the first capture are excluded.
- **Deterministic scenes for before/after:** the beam and the trampoline are bit-reproducible, while contact scenes vary run to run.
- **nsys on Windows** needs the base interpreter. The runners handle that through `profiling/venvrun.py`. The default nsys path is in `profiling/nsys.py` (`--nsys` overrides it).

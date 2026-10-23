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

Run from the repo root with the dev venv's Python (Python <= 3.12; its location is machine-specific, see `CLAUDE.local.md`). The `data/...` paths below exist only in the main checkout; from a worktree, point to them as `path/to/PhysicsBasedAnimationToolkit/data/scenarios/trampoline.h5` (with `--scene`).

## Choose the measurement

| Question | Method | Execution | Pros | Cons |
|---|---|---|---|---|
| How long does a step really take? Before/after a change? | **Production timing**: `--profile off --profile-out f.json` | CUDA graph replay | The real cost: includes graph scheduling gaps. Cheap | One number per step, no phases. The CUDA events bracket all of `state.step()`, i.e. replay plus post-step host work (contact browser updates, normally < 1 ms). For clean before/after numbers, time `wp.capture_launch` only |
| Where does kernel time go, by phase? | **`kernels` backend** (CUPTI, in-process) | Eager | Per-scope busy time, launches, calls, union. Within 0.6% (sweep) / 2.9% (step) of Nsight. Quick, no external tools | Eager, so no graph gaps (a lower bound on cost). Sub-µs kernels read ~0.2 µs long, inflating tiny-kernel contact phases 27–40% |
| Reference per-phase timings, or a timeline to inspect | **`nvtx` + Nsight Systems** (`profiling.nsys run`) | Eager | Ground truth for kernel durations. Viewable in the Nsight Systems GUI | Needs nsys. Slower turnaround. Still eager |
| Why is production slower than kernel work? Where are the idle gaps? | **Sweep anatomy** (`profiling.anatomy run`) | Graph replay under nsys | Node-by-node graph timeline: sweep vs boundary, kernels vs gaps, labelled by phase | Needs nsys and an eager profile for labels |
| Why is one kernel slow (occupancy, stalls, memory, hot lines)? | **Nsight Compute** (`profiling.ncu run`, `--lineinfo`) | Either (profiles individual kernel launches) | Stall reasons, per-Python-line hotspots. Much less sensitive to other GPU work (replays each kernel with locked clocks) | Locked base clocks and cache flushes: compare kernels with each other, not with nsys times |

Rules of thumb:
- Quote wall-clock cost from production timing.
- Quote phase breakdowns from `kernels` or nsys.
- Explain the difference between the two with the anatomy.
- `kernels` and nsys can't run in one process: use `--profile nvtx` under nsys.

## Commands

```bash
# Production step time (graph replay); 10 warm-up steps excluded
python -m python.tools.sim.simulate --scene data/scenarios/trampoline.h5 -o out/x.h5 \
    --solver AAAVBD --duration 0.3 --profile-warmup 10 --profile-out out/aaavbd_prod.json

# Per-phase breakdown in-process (CUPTI); prints a summary and writes JSON
python -m python.tools.sim.simulate ... --profile kernels --profile-warmup 10 --profile-out out/aaavbd.json

# Nsight Systems reference (runs simulate.py --profile nvtx, imports into the same JSON schema)
python -m python.tools.sim.profiling.nsys run -o out/aaavbd -- \
    --scene data/scenarios/trampoline.h5 --solver AAAVBD --duration 0.3 -o out/x.h5
python -m python.tools.sim.profiling.nsys import out/aaavbd.sqlite --metadata out/aaavbd.json -o out/aaavbd.json

# Sweep anatomy of the production graph. --phases takes an eager profile from the same code version;
# kernels missing from it (e.g. newly fused ones) keep their raw names in the phase column
python -m python.tools.sim.profiling.anatomy run -o out/aaavbd_graph --phases out/aaavbd.json -- \
    --scene data/scenarios/trampoline.h5 --solver AAAVBD --duration 0.3 --profile-warmup 10 -o out/x.h5

# Nsight Compute on 9 launches (one sweep's colors) of the AAAVBD vertex solve, with per-line hotspots.
# --kernel is a regex on the Warp kernel name, which starts with an underscore; without it nothing matches
python -m python.tools.sim.profiling.ncu run -o out/aaavbd_ncu --kernel ^_accelerated_vertex_solve --skip 900 --count 9 --lineinfo -- \
    --scene data/scenarios/trampoline.h5 --solver AAAVBD --duration 0.05 -o out/x.h5
```

The `simulate.py` arguments after `--` are passed through unchanged, so `--sim-params`, `--set`, `--solver` and so on apply as usual (see the `sim-run` skill).

## Outputs: JSON and HTML

- **JSON:** every runner writes `pbat-sim-profile/1` results, from the `kernels` backend, nsys import, production runs or the GUI export.
  - `steps[]` hold per-path `busy_ms` (exclusive of nested scopes), `union_ms`, `launches` and `calls`, plus `step_union_ms` and `step_span_ms`. Production runs record `step_timeline_ms` instead.
  - `kernels[]` hold the busy time and count of each kernel.
  - `union_incl_ms` holds the inclusive union per path and ancestor path, computed from raw intervals; it can't be summed.
  - Inclusive (Total) busy time and launches aren't stored: `results.inclusive()` derives them from path prefixes, and `results.scope_tree()` gives the tree rows.
- **HTML, per-phase comparison:** pass one result per solver or variant. The report also has a collapsible "Scope tree" per solver: Total/Self busy, total share, total union, total/self launches and calls.

  ```bash
  python -m python.tools.sim.profiling.report out/vbd.json out/aaavbd.json -o out/report.html --note "trampoline, Dump params"
  ```

- **Per-step statistics** (in the same report, `profiling/step_stats.py`): quantiles of step and phase times, the phases over the run, and the steps grouped by percentiles of their GPU time and of their peak node speed (`--slices 10,90`), each group with a phase donut, early exits, iterations, ms/iteration and residual ratios. Every profile entry already carries the step's solve statuses (`simulate.py` writes them). The peak speeds come from the run's saved frames, added in a separate step before `report`; run `simulate.py` with `--fps` at the step rate (1 / dt) for one value per step:

  ```bash
  python -m python.tools.sim.profiling.trajectory out/aaavbd.json out/aaavbd_traj.h5   # adds peak_speed per step
  ```

- **HTML, overheads:** a production time budget, the anatomy timeline, kernel comparisons and before/after timings, plus "stall hotspots by function" when given `ncu.py --lineinfo` summaries.

  ```bash
  python -m python.tools.sim.profiling.overheads --anatomy AAAVBD=out/aaavbd_graph.anatomy.json \
      --ncu AAAVBD=out/aaavbd_ncu.ncu.json --production AAAVBD=out/aaavbd_prod.json --findings findings.txt -o out/overheads.html
  ```

- **Paper figures** (`profiling/paper_figures.py`, PDFs in the benchmark figures' style: Biolinum, printed `--width`, one legend, no titles): `timeline` (phase groups over simulated time, groups under 3% folded into Other, with every production run's step time; a strip of rolling-median residual ratios over each run's peak speed, one legend entry per speed; and a row of numbered snapshots from `profiling/snapshots.py`), `solvers` (per solver of a scene: phase shares and ms per sweep from fixed-budget kernels runs, without the shared negligible groups, then per-step residual ratios and step times from production runs) and `scenes` (the paper's LaTeX `table*` of production step times, iterations and early exits, per scene and solver); `slices` writes the fastest/middle/slowest steps' medians for the text. Kernels and production results pair by scene and solver; phase groups are its `PHASE_GROUPS`. See `python/tools/sim/CLAUDE.md` ("Paper figures").
- **Publishing:** to share a report, publish the HTML as an Artifact.

When the user asks for "HTML output", run the matching measurement (above) to JSON first, then `report` (phases across solvers) or `overheads` (production overheads). "JSON only" stops at `--profile-out`.

## GUI

`python -m python.tools.sim.ui ...` → **Profiling** tree node. Enabling it switches on the CUPTI backend, so steps run eagerly. It shows mean GPU busy time per step over a rolling window in three views: By phase, By scope path, and Tree (Total, Self, share of step, total union, calls). **Export** writes JSON for `report`. The backend needs `nvidia-cuda-cupti-cu12` installed; without it the panel shows the error.

## Reading results correctly

- **Stopping rule:** with early exit (`gtol > 0`), ms/step is the time to pass the stopping test at that run's settings, **not** an equal-accuracy solver comparison. Always report sweeps/step (`calls` of `*.sweep`), outer iterations/step (`calls` of `solve.prepare_subproblem`), ms/sweep and whether steps converged or hit the budget, and name the stopping rule. For speed comparisons, set fixed budgets (`--set Solver.<S>.gtol=0`) or use the benchmark tooling (`sim-benchmarks` skill).
- **Exclusive times:** a parent scope excludes its children, and paths sum to the step. `busy_ms` double-counts streams that run concurrently; `union_ms` doesn't. Inclusive (Total) busy time and launches are derived from the paths, while the inclusive union `union_incl_ms` is stored separately.
- **Eager vs graph:** eager kernel durations match graph mode within a few percent, but eager runs hide graph gaps.

## Before measuring

- **The GPU must be quiet.** Other processes (other sessions, the GUI) distort timings badly: an unchanged baseline drifted from 27.6 to 47 ms. While parallel sessions run, reserve a window via `plans/MASTER.md` (`GPU run START`/`END`). `nvidia-smi` shows other users.
- **Warm-up:** use `--profile-warmup` ≥ 5 so module loading and the first capture are excluded.
- **Deterministic scenes for before/after:** the beam and the trampoline are bit-reproducible for VBD, AAAVBD and Chebyshev only. Anderson and Newton vary run to run (~2e-4 on the trampoline), and contact scenes vary for every solver, so compare those within noise.
- **nsys on Windows** needs the base interpreter. The runners handle that through `profiling/venvrun.py`.
- **Finding nsys and ncu:** `profiling/nsight.py::find_tool` resolves them at run time: `--nsys`/`--ncu path/to/tool`, else the `NSYS`/`NCU` environment variable, else `PATH`, else the standard install locations, newest version first (`%ProgramFiles%/NVIDIA Corporation/Nsight Systems <version>/...` and `Nsight Compute <version>/...` on Windows, `/opt/nvidia/nsight-*/<version>/` and the CUDA `bin` on Linux). If neither is found, it raises an error saying how to point at the tool. Never hard-code an install path.

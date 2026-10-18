# Task 2: Solver profiling and cost breakdowns

Status: **not started** · Branch `feature/gpu-profiling` · Worktree `C:\git\pbat-profiling`
Read first: `MASTER.md` (rules, ownership, GPU-sharing rule), `python/tools/sim/CLAUDE.md`.

## Goal

Measure where a time step's GPU time goes, by algorithm phase, for every GPU solver (Newton, VBD, AAAVBD,
Chebyshev, Anderson). Produce per-solver pie charts and cross-solver comparisons that hold up in a paper.

## Facts that shape the design

- **Warp activity timing** (`wp.timing_begin`/`wp.timing_end`, or `wp.ScopedTimer(cuda_filter=...)`)
  records the GPU duration of each *eager* kernel launch, memset and memcpy, in launch order. A captured CUDA
  graph is recorded as a single `TIMING_GRAPH` entry, with no per-kernel detail.
- **`wp.ScopedCapture` itself provides no timings.** It only records the graph.
- **Production runs use graph capture** (`ui.py` and `simulate.py` capture `substeps × (setup → solve → step)`),
  which removes host launch overhead. Eager timings include gaps where the GPU waits for the host, so eager
  *wall* time overstates the real cost.
- Graph-safe instrumentation: CUDA events recorded inside a capture become graph nodes only when recorded
  as external events. Check whether `wp.record_event` does this during capture before relying on it.
- `nvtx` is already pinned in `python/tools/sim/requirements.txt`, and `ui.py` contains a commented-out
  `wp.ScopedTimer(..., use_nvtx=True)`. Nsight Systems with `--cuda-graph-trace=node` shows per-kernel
  times even inside graphs, which makes it the reference to validate against.
- Newton isn't graph-capturable (`supports_graph_capture == False`), so it always runs eagerly. Comparisons
  must account for that, for example by reporting Newton's eager GPU busy time next to the others' graph times.

## Proposed design

**Scope API** (the stub lands in Phase 0, in `python/tools/sim/gpu/common/profiling.py`):
`with profiling.scope("anderson.gram"):` with hierarchical dotted names, a no-op when profiling is off, so
production graphs are unaffected. Suggested phase taxonomy (adjust as needed):
`step.setup`, `contact.detect`, `contact.filter`, `contact.constraint_set`, `contact.dual`,
`contact.penalty`, `vbd.sweep`, `aaavbd.sweep`, `anderson.gram`, `anderson.solve`, `anderson.mix`,
`chebyshev.mix`, `newton.assemble`, `newton.linear_solve`, `newton.line_search`, `step.finalize`.

**Backends**, selected at run time:
- `off` (default).
- `phases`: CUDA events at scope boundaries, eager mode. Gives per-phase GPU timeline durations, including idle gaps.
- `kernels`: Warp activity timing, eager mode. Each kernel's busy time is attributed to the innermost open
  scope by tracking launch order. This is the most accurate breakdown of GPU work.
- `nvtx`: NVTX ranges per scope, for Nsight Systems; works in graph mode too.

**Profiling disables graph capture (maintainer decision).** When any backend other than `off` is active,
`SimulationState.step` runs eagerly even for capturable solvers. All solvers, Newton included, are then
measured the same way.

**Measurement protocol per solver and scene:**
1. Warm up (compile), excluding the first steps.
2. Eager mode with the `kernels` backend: per-phase GPU busy time over N steps. This gives the pie charts
   and cross-solver comparisons.
3. Optionally, as context only: an unprofiled graph-mode run timed with events around `capture_launch`,
   to show how much host launch overhead graph capture removes in production.

Record GPU model, driver, Warp version and solver parameters with every result.

**Outputs:**
- `simulate.py --profile {off,phases,kernels,nvtx} --profile-out results.json`: per-step, per-phase times.
- A report script: pie chart per solver, stacked bars across solvers, phase time per sweep. The maintainer
  prefers rendered pages for results, so publish comparisons as an HTML artifact.
- A live table in a `panels/profiling.py` GUI panel.

## Validation

- Per step, the sum of phase busy times plus unattributed time equals the activity-timing total.
- On one scene, compare against Nsight Systems: per-phase totals within a few percent.
- With profiling off, captured graphs and step times are unchanged (no overhead).

## Files

Own: `gpu/common/profiling.py`, `panels/profiling.py`, report scripts, `--profile` flags. Add `scope(...)`
lines in solver and contact files, but don't restructure those files. Tasks 1 and 3 instrument their own new code.

## Open questions for the maintainer

1. Which comparisons matter most for the paper: per-step cost at equal accuracy, or time-to-tolerance? The latter needs task 3's convergence metric.
2. Is installing Nsight Systems acceptable as the reference profiler?
3. Should the GPU breakdown also include a host-side view (Python/launch overhead), or only GPU work?

## Progress

(update as you go)

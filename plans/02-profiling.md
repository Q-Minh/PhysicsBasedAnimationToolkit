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

1. ~~Which comparisons matter most?~~ **Answered (2026-10-17): cost breakdown.** Per-step GPU cost by phase at
   the production parameters, plus normalized cost per VBD sweep / Newton iteration. Time-to-tolerance stays
   in task 3's graph-mode benchmark.
2. ~~Nsight Systems as reference?~~ **Answered (2026-10-17):** it is already installed (2026.1.3). **Warp + nsys
   import:** the in-process `kernels` backend (Warp activity timing plus per-scope event brackets, with the
   remainder reported as "untimed") serves the GUI and quick runs. **Paper figures come from nsys:** NVTX
   scopes, then a script attributes every kernel in the nsys sqlite export to its innermost scope. That covers
   CUB/cupy kernels and concurrent streams exactly. No new dependencies.
3. ~~Host-side view?~~ **Answered (2026-10-17): GPU only.** Time things as if the code were written directly in
   C++/CUDA: report GPU busy time per phase, excluding host launch overhead, Python time and host-sync gaps
   (relevant to Newton). Drop protocol step 3 (eager vs graph replay wall time) as a reported result.
4. ~~Lock GPU clocks?~~ **Answered (2026-10-17): no lock, validate.** Keep default clocks. Compare per-kernel
   averages from an eager run against an nsys graph-mode trace (`--cuda-graph-trace=node`) once, and report the
   difference.
5. ~~Scenes?~~ **Answered (2026-10-17): mostly trampoline** (`data/scenarios/trampoline.Dump.h5`), but scene
   choice is deferred: "for now, I just want to see profiling work." Develop on trampoline (MediumLongBendingBeam
   for quick checks).
6. ~~Contact detection?~~ **Answered (2026-10-17): VertexSdf only.**
7. ~~Contact cost inside the fused vertex-solve kernel?~~ **Answered (2026-10-17): one fused slice,**
   "vertex solve (elastic + contact)". Dual updates, detection, filtering and constraint-set updates get their own
   slices. No ablation, no split kernels.
8. ~~Solver parameters?~~ **Answered (2026-10-17): equal sweep budget.** VBD, AAAVBD, Chebyshev and Anderson
   run the same total number of VBD sweeps per step. Newton keeps its defaults. Default budget: the Dump's
   `n_max_iters × n_subproblem_max_iters`, with Anderson's `n_subproblem_max_iters` set to budget / `p`.
9. ~~First milestone?~~ **Answered (2026-10-17): CLI + report first.** Scope backend, scopes in all solvers,
   `simulate.py --profile`, the nsys importer, and a published HTML report (pies plus stacked bars) on
   trampoline. The GUI panel (`panels/profiling.py`) comes in milestone 2.
10. ~~Backend set?~~ **Answered (2026-10-17): `off` / `kernels` / `nvtx`.** `phases` is dropped. Per-scope event
    brackets are folded into `kernels` as a diagnostic for spotting untimed work.

## Progress

- 2026-10-17 · Investigation findings:
  - **Warp activity timing** (`warp/native/cuda_util.h:359`) brackets each *Warp* launch, memset and memcpy
    with a CUDA event pair and skips capturing streams. It returns durations without timestamps.
    **Not covered:** `cuda.compute` (CCCL) sort, lower_bound, scan and reduce in `contact/mesh/pairs.py`,
    `contact/dynamics.py` and `contact/mesh/ogc.py`, plus cupy copies.
  - **First nsys probe** (MediumLongBendingBeam, AAAVBD, VertexSdf, 30 steps, graph mode with node trace):
    1.70 s of kernel time. `_accelerated_vertex_solve_kernel` is 85.8% (130,500 launches, avg 11.2 µs,
    ~4,350 per step). Non-Warp kernels (CCCL radix sort and binary search, cupy copies) are ~4%. Contact dual
    updates are ~5%. Detection is ~1.5%.
  - **Concurrent streams:** contact work runs on 4×6 `wp.Stream`s (`contact/constraints.py:105`) and Newton
    triplet assembly on 6 streams, so per-kernel durations summed per scope overcount the timeline.
  - **Newton** syncs to the host every iteration (`.numpy()` on the gradient norm, slope and energy;
    `warp.optim.linear.cg` with `use_cuda_graph=False`). Its kernels are all Warp (`warp.sparse`, `warp.optim`).
  - **In-graph events:** `Stream.record_event(..., external=True)` (`warp/_src/context.py:5335`) records
    external event nodes during capture. Relayed to task 3 via the coordinator.
  - **nsys on Windows:** the venv's `python.exe` launcher stub hangs under nsys. Launch the base interpreter
    with a wrapper that sets `sys.prefix` and calls `site.addsitedir` on the venv. `PYTHONPATH` alone breaks
    `cuda.compute` (cuda.pathfinder can't find nvrtc). The wrapper will be committed with the profiling scripts.
- 2026-10-17 · All open questions answered (batches 1 to 3, see above). Waiting for Phase 0, then rebase onto
  `dev` and implement milestone 1.
- 2026-10-17 · Coordinator confirmed the Phase 0 stub contract: `profiling.scope(name: str)` is the only public
  API, a cheap no-op when off (no allocations, no device work, capture-safe). The coordinator will message when
  Phase 0 is done.
- 2026-10-18 · **Milestone 1 implemented** on `feature/gpu-profiling` (14 commits, rebased on dev `800e87ef`):
  - **Backend (`gpu/common/profiling.py`):** `scope`, the `scoped` decorator, `push`/`pop`, `set_backend`,
    `begin_step`/`end_step`, with backends OFF/KERNELS/NVTX. Unit tests are in the module.
  - **Scopes:** in the contact, VBD-family and Newton files.
  - **Eager profiling:** `state.step` runs eagerly while profiling. The substep loop moved into
    `SimulationState._advance`.
  - **`simulate.py`:** `--profile`, `--profile-out` and `--profile-warmup`. Warm-up runs unprofiled in graph
    mode. With profiling off, `--profile-out` records production (graph-replay) step times.
  - **Tools:** `profiling/nsys.py` (runner plus importer), `profiling/venvrun.py`, `profiling/report.py`, and docs
    in `python/tools/sim/CLAUDE.md`.
- 2026-10-18 · Validation:
  - **Kernels backend vs nsys** (beam, AAAVBD, same eager steps): Warp activity timing gives 208 ms/step, nsys
    54 ms/step. The sweep alone is 110 vs 48 ms, the contact phases 15–30× too high. The cause is launch
    latency: event pairs start on an idle GPU before the kernel is submitted. Eager nsys sweep time (47.8 ms)
    matches graph-mode nsys (49 ms), so eager kernel durations are representative. **Quote only nsys numbers.**
  - **Profiling off leaves production unchanged:** captured graphs have identical node counts and types on the
    merge base and the branch, for VBD, AAAVBD, Chebyshev and Anderson (11,978–14,898 nodes on the beam), and
    trajectories are bit-identical. Replay timings agree within run-to-run noise (±5 ms).
  - **Side finding:** every captured step contains 21 MEM_ALLOC/MEM_FREE node pairs (allocations during capture)
    and ~2,500 memcpy nodes.
  - Not done yet: the "per-phase totals within a few percent of nsys" check for the kernels backend. It fails
    by construction (see above).
- 2026-10-18 · **Trampoline report** (5 solvers, nsys, steps 10–14, Dump parameters = 500 sweeps/step):
  https://claude.ai/artifact/LHMcGJs6Z6N89iruhYLxYz
  - **VBD family:** 65–72 ms/step of kernel work vs ~100 ms/step graph replay, i.e. ~30 ms/step of graph
    scheduling overhead. The sweep is 78–88% of the work.
  - **Contact:** VertexSdf detection costs ~6 ms/step even with zero contacts (trampoline has none).
  - **Anderson:** the Gram matrix plus solve costs 3.4 ms/step.
  - **Newton:** converged after one outer iteration (13.6 ms/step), so it isn't at equal accuracy.
- ~~Open: should the inaccurate `kernels` backend stay?~~ **Answered (2026-10-18): switch the in-process
  backend to CUPTI.** `cupti-python` is Linux-only, so I'll write a small ctypes binding to the CUPTI activity
  API: kernel, memcpy and memset records with GPU start/end times, attributed to scopes through external
  correlation IDs. **DLL source (maintainer): pip `nvidia-cuda-cupti-cu12==12.9.79`**, which matches the venv's
  CUDA 12.9. Add it to `requirements.txt`; the coordinator installs it in the shared venv. This is milestone 2
  work, together with the GUI panel.
- 2026-10-18 · **Maintainer direction:** iterate on the breakdowns and visuals so the maintainer can suggest data
  points. The goal is to find AAAVBD's performance overheads and make it faster, since AAAVBD is the technique
  to publish. **Order: milestone 2 first** (CUPTI backend plus GUI panel), then **milestone 3: AAAVBD analysis**
  with two new data points, chosen by the maintainer:
  - **Sweep anatomy:** import an nsys graph-mode trace with node-level timing and show one production sweep as
    kernel time vs gaps, by node type.
  - **Nsight Compute** on the VBD and AAAVBD vertex-solve kernels: occupancy, stall reasons, memory traffic.
  First observations from the trampoline traces (to re-measure, since task 1's runs overlapped):
  - **Launches:** 9 colors of ~667 vertices. Each color launch is one partial wave (667 blocks × 32 threads,
    128 registers/thread) and takes ~12.7 µs (VBD) vs ~14.0 µs (AAAVBD), +12%.
  - **Graph replay:** 206 µs/sweep vs 144 µs/sweep of kernel work, so ~60 µs/sweep goes to gaps.
  - **Fixed nodes per sweep:** 4 dual-update kernels (even with no contacts), an `xb` memcpy, and memsets.
- 2026-10-18 · Milestone 1 merged into dev (`1d52c8ff`). **Milestone 2 implemented** on `feature/gpu-profiling`
  (4 commits after `7c485e35`):
  - **CUPTI binding:** `gpu/common/cupti.py`, a ctypes binding to the activity API with external correlation.
    The DLL is resolved through `importlib.metadata` from `nvidia-cuda-cupti-cu12==12.9.79`, which is pinned.
  - **Backend switch:** the `kernels` backend now runs on CUPTI and reports union/span like nsys.
  - **GUI panel:** `panels/profiling.py` (live rolling-window table, JSON export), plus 3 small edits in
    `ui.py`.
  - **Validation vs nsys** (beam, AAAVBD, idle GPU):
    - Activity counts are identical (11,946 per step).
    - The sweep is within 0.6% (48.06 vs 47.76 ms/step) and the total within 2.9% (55.7 vs 54.1).
    - Sub-µs activities read ~0.2 µs longer in CUPTI, which makes the tiny-kernel contact phases 27–40% higher.
      Use nsys for those.
  - **Tested:** the GUI panel's step hook and export were tested headlessly. Drawing it needs the
    interactive GUI (maintainer smoke test).
- 2026-10-18 · Milestone 2 merged into dev. **Milestone 3 (AAAVBD analysis)** on `feature/gpu-profiling`,
  7 commits after `92791e16`:
  - **Tools:** `profiling/anatomy.py` (nsys graph-mode sweep anatomy: per-step budget, boundary phases, steady
    sweep node by node), `profiling/ncu.py` (Nsight Compute summary), `profiling/overheads.py` (report), and a
    capture range for production runs in `simulate.py`.
  - **Report:** https://claude.ai/artifact/NngLUsxKfkhZz4Fzmx9XuL
  - **Findings (trampoline, AAAVBD, 107 ms production step):**
    - **Step budget:** sweep kernels 65.8 ms, sweep gaps 1.1, boundary kernels 8.1, boundary gaps 29.6,
      other 2.2.
    - **Boundary gaps:** almost all of the boundary gap time sits before the constraint-set update's memcpy and fill
      nodes (~10 µs each).
    - **Root cause:** `ConstraintSet.update_constraint_set` repeats its copies and fills 4× (loop over all 4
      stream lists).
    - **Measured fix:** a 2-line change (one stream list) is bit-identical and gives AAAVBD 108.1 → 83.5 ms
      (−23%), VBD 98.7 → 74.9 (−24%). **Not committed** (not my file). Patch in scratchpad.
    - **Remaining:** ~7.6 ms/step of boundary gaps would mostly go with a fused copy/fill kernel.
    - **Steady sweeps:** nearly gap-free (0.17 µs between color launches).
    - **AAAVBD vs VBD kernel (ncu, base clocks):** +7% duration, +13% instructions, ~+28% DRAM traffic.
      Both are latency-bound: 73–74% long-scoreboard stalls, 83% of cycles with no eligible warp.
    - **Occupancy:** limited by the grid (667 one-warp blocks per color, 0.87 waves, 24% achieved occupancy),
      not by registers.
  - **Possible contact bug on dev** (logged for the coordinator): CubeFallingOnCube + VertexSdf detects 0
    contacts, and the top cube falls through.
- 2026-10-18 · **Maintainer decisions:** commit the constraint-set fix. The coordinator landed it on dev as
  `a9abadc1`, using one stream per field. The cube fall-through is expected (coincident surfaces, zero
  penetration), not a bug. Milestone 3 tools were merged (`7ad6a983`). Next directions: kernel source hotspots
  and the `capture_while` anatomy.
- 2026-10-18 · **Kernel source hotspots** (3 commits after `7ad6a983`):
  - **Tooling:** `ncu.py --lineinfo` (venvrun's `PBAT_WARP_LINEINFO` hook) reports stall samples, stall reasons and
    instructions per Python line and function, with native builtins folded into their callers by address.
  - **Finding:** on contact-free trampoline, contact gathering (`local_contact_derivatives` plus half-edge
    helpers) is ~60% of AAAVBD's kernel stall samples (VBD ~72%) and ~half of its instructions.
  - **Skip-gather experiment** (bit-identical, contact-free only): AAAVBD 83.8 → 40.6 ms/step (−52%), VBD
    75.2 → 46.3 (−38%).
  - **Proposed fix (not implemented, task 3 owns the solver kernels):** a per-vertex contact count per
    constraint-set update, with an early-out in the vertex solve.
  - **To recheck:** AAAVBD is faster than VBD with the gather skipped.
  - **Report:** v2 of https://claude.ai/artifact/NngLUsxKfkhZz4Fzmx9XuL
  - **Pending:** the `capture_while` anatomy waits for task 3's merge.
- 2026-10-18 · **Contact-gather early-out implemented** (maintainer: task 2 implements it), on dev after task 3's merge:
  - **Commits:** `f9f689a3`, `2e4498e9`. `MeshDynamicsData.in_contact` is flagged per constraint-set update from the
    gather's own prefix ranges, plus one guarded condition per vertex solve.
  - **Validation:**
    - VBD/AAAVBD/Chebyshev are bit-identical on trampoline and the beam. Anderson is within run-to-run noise.
    - On BeamFallingOnBeam, flags exactly match the gathered sets.
  - **Production timings (dev, capture_while):**
    - Trampoline: AAAVBD 58.7 → 28.4 ms, VBD 75.1 → 30.5, Chebyshev 47.8 → 22.2, Anderson 9.1 → 5.0.
    - BeamFallingOnBeam: −4…−5%.
  - **`capture_while` anatomy** (dev graph, AAAVBD trampoline): step 18.3 ms traced, gaps 1.2 ms (7%),
    ~237 sweeps per step (convergence test).
  - **ncu after the early-out:** AAAVBD's vertex solve takes 10.0 µs vs VBD's 7.6 µs (+32%, +36% instructions). The
    acceleration functions hold ~25% of the stalls.
  - **Ideas, not implemented:** hoist AAAVBD's per-vertex state loads, pre-filter surface neighbours, early-out for
    the dual updates.
  - **Report:** v3 of the same link.

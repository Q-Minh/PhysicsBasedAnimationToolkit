# Master plan: GPU simulator roadmap (4 parallel tasks)

Status: **worktrees created; answering open questions (no implementation yet)** · Created 2026-10-17 · Coordinator: the original Claude session in the main checkout

## How each session starts (read this first)

1. **Questions first, no code.** Read this file and your task plan (absolute paths below). Ask the
   maintainer your plan's open questions, plus anything else you need decided, and record the answers in
   your plan's "Open questions" and "Progress" sections. Investigating the code and prototyping in a
   scratch directory are fine; editing tracked files is not.
2. **Wait for Phase 0.** Don't change tracked files until the status board shows Phase 0 as **done**. The
   maintainer wants all open questions answered across all tasks before any implementation starts.
3. **Rebase, then implement.** Once Phase 0 is done, run `git rebase dev` in your worktree so you have
   `state.py`, the profiling stub and the panel modules. Then follow your plan.
4. **Keep the board current.** Update your status-board row and append to the coordination log when you
   reach milestones or need something from another task.

Plan files: `C:\git\PhysicsBasedAnimationToolkit\plans\MASTER.md` and `...\plans\0N-*.md`.

## Decisions

- 2026-10-17: Plans are **tracked in git on `dev`** for the duration of the implementation effort and
  deleted once everything is merged. Every worktree therefore has a read-only snapshot under `plans/`.
  **Edit only the canonical copy in the main checkout** (absolute path above), and never commit `plans/` on
  a feature branch; the coordinator commits plan updates to `dev`.
- 2026-10-17: Branch names use the full prefix `feature/...`.
- 2026-10-17: Phase 0 and all implementation start only after every task's open questions are answered.
- 2026-10-17: Task-level decisions are recorded in each task plan (static: no added regularization;
  profiling: graph capture off while profiling; kinematics: fix the drift in both simulators and fix the
  release bug).

The canonical copies of these plan files live in the main checkout at
`C:\git\PhysicsBasedAnimationToolkit\plans\`. Sessions working in other git worktrees read and update
them **through that absolute path**, never a copy inside their own worktree.

## Tasks

| # | Task | Plan | Branch / worktree | Depends on |
|---|---|---|---|---|
| 1 | Kinematic (procedural) Dirichlet constraints on the GPU | [01-kinematic-constraints.md](01-kinematic-constraints.md) | `feature/gpu-kinematic-constraints` · `C:\git\pbat-kinematics` | Phase 0 |
| 2 | Solver profiling and cost breakdowns | [02-profiling.md](02-profiling.md) | `feature/gpu-profiling` · `C:\git\pbat-profiling` | Phase 0 |
| 3 | Static (equilibrium) solves and convergence benchmark | [03-static-elasticity.md](03-static-elasticity.md) | `feature/gpu-static-solves` · `C:\git\pbat-static` | Phase 0; task 4's sample files (benchmark inputs); task 2's scope API |
| 4 | MCMC sampling of quadratic-manifold shapes | [04-mcmc-quadratic-manifold.md](04-mcmc-quadratic-manifold.md) | `feature/rom-mcmc` · `C:\git\pbat-mcmc` | none |

## Phase 0 (coordinator, before the parallel work starts)

These changes remove the main merge-conflict hotspots. Each is small; they land on `dev` before the
worktrees are created.

1. **Extract the simulation state out of `ui.py`.** Move `SimulationState`, `SolverType`, `CDType`, the
   `load_*` helpers and the parameter (de)serialization into `python/tools/sim/state.py`. `ui.py` and
   `simulate.py` import from it. Tasks 1, 2 and 3 all need to extend the state, the CLI and the GUI; without
   this, all three edit the same 850-line file.
2. **Add the profiling scope API as a no-op stub** in `python/tools/sim/gpu/common/profiling.py`:
   `with profiling.scope("vbd.sweep"): ...`. Tasks 1 and 3 can then instrument their new code right away,
   and task 2 implements the backend without editing their files.
3. **Give GUI features their own panel modules.** `ui.py` calls one `draw_<feature>_panel(state, ui_state)`
   per feature, from `python/tools/sim/panels/<feature>.py`. Each task adds its own panel file and a single
   call line in `ui.py`.
4. ~~Create the four branches and worktrees.~~ Done early, at the maintainer's request. They were created
   from `dev` before Phase 0, so each session rebases onto `dev` once Phase 0 lands.

## File ownership

Edit freely what you own. For anything else, keep changes minimal and note them in the coordination log below.

| Area | Owner | Others may touch |
|---|---|---|
| `gpu/elasticity/fem.py`: Dirichlet handling (`dmask`, `dbc`, `ndbc`, release logic) | 1 | 3 (read-only use) |
| `gpu/elasticity/fem.py`: objective / inertia / time-integration setup | 3 | 1 (only the Dirichlet lines of the setup kernel) |
| New `gpu/kinematics/` package, `panels/kinematics.py` | 1 | none |
| `python/tools/vbd/` (closed-form transform fix, Dump format) | 1 | none |
| `gpu/common/profiling.py`, `panels/profiling.py`, profiling scripts | 2 | none |
| `profiling.scope(...)` lines in solver and contact files | 2 | 1 and 3 add scopes around their own new code |
| Solver kernels (`gpu/vbd/*`, `gpu/newton/*`) | 3 (static mode) | 2 (scope lines only) |
| `gpu/vbd/solver.py::check_convergence` (currently a stub) | 3 | none |
| `state.py`, `simulate.py` CLI | shared: append-only (new fields, new flags) | all |
| `python/examples/modalrom.py` and new ROM/MCMC modules | 4 | none |

Hotspots to watch: the Dirichlet lines in `fem.py`'s `_setup_time_integration_optimization_kernel` (tasks 1
and 3), and the per-vertex inertia term in `gpu/vbd/kernels.py::add_inertia_derivatives` (task 3, which task 2
will instrument).

## Rules for every session

- **Shared Python environment.** All worktrees share the venv `C:\git\python-3.12`, which has exactly one
  installed `pbatoolkit`. **Do not reinstall pbatoolkit from a worktree.** If a task needs C++ or binding
  changes, describe them in the coordination log; the coordinator builds and installs them.
- **Shared GPU.** There is one RTX 5070. Long GPU runs from several sessions distort each other's timings.
  Task 2's measurements (and task 3's timing curves) must be taken when no other session is using the GPU;
  announce them in the coordination log first.
- **Scene data is not in worktrees.** `data/` is git-ignored, so use absolute paths such as
  `C:\git\PhysicsBasedAnimationToolkit\data\scenarios\MediumLongBendingBeam.Dump.h5:fem`. Absolute
  Windows paths work since the `parse_archive_path` fix.
- **Permissions.** `.claude/settings.local.json` is git-ignored. The coordinator copied it into each
  worktree when creating it, so every session has the same local allowlist (the venv Python and `dumpbin`).
- **Context.** `CLAUDE.md`, `python/tools/sim/CLAUDE.md` and the skills (`pbat-build`, `pbat-python-api`,
  `warp-port`) are tracked and available in every worktree. The `cpu-gpu-parity` agent is useful for tasks 1 and 3.
- **Commits.** Small commits on your own branch, ending with the Co-Authored-By line. Rebase on `dev` before
  asking for a merge. Never push to `dev` or `master` yourself; the coordinator merges.
- **Asking the maintainer.** Each plan lists open questions. Ask them in your own session before
  implementing the affected part, and record the answers in your plan.

## Shared technical facts (verified)

- **Timing inside CUDA graphs:** Warp 1.17's `Stream.record_event(event, external=True)` records an
  external event node during capture (`warp/_src/context.py`, `Stream.record_event`). Per-sweep timestamps
  inside a captured graph are possible, not only events bracketing the replay. (Reported by task 2.)
- **Device-side control flow in graphs:** Warp 1.17 has `wp.capture_while` / `wp.capture_if`. (Reported by task 3.)
- **Nsight Systems** 2026.1.3 is installed, not on PATH:
  `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe`. Under nsys,
  the venv's `python.exe` launcher stub hangs. Launch the base interpreter through a wrapper that sets
  `sys.prefix` and calls `site.addsitedir` on the venv; plain `PYTHONPATH` breaks `cuda.compute`. Task 2 is
  committing a working wrapper with its profiling scripts. (Reported by task 2.)
- **Profiling and benchmarks are separate:** task 3's convergence benchmark runs in graph mode with
  profiling off. `profiling.scope` is a no-op when off, so captured graphs are unaffected. Task 3 needs
  task 2's scopes only for an optional per-phase breakdown, which is an eager profiling run.

## Contract: MCMC sample files (task 4 produces, task 3 consumes)

**Agreed 2026-10-17** (maintainer-approved via task 4; task 3 to confirm it can consume it):
- **HDF5 file**, one per batch, with **one target energy E\* per file**. Root attributes: source mesh / Dump
  path, sampler kind and parameters, E\*, the energy definition, random seed, git commit, and chain metadata
  (weight kind, burn-in, thinning, acceptance rate, ESS).
- **Energy definition:** E\* is the **deformation energy in absolute joules**, Ψ(X + u) − Ψ(X). Stable
  Neo-Hookean has a nonzero rest energy, 4.96e-4 J on MediumLongBendingBeam, hence the subtraction.
- **`X`**: **(N, 3)** rest positions, float32. Row i is node i, i.e. `X = fem.X.T` for pbat's (3, N) `fem.X`;
  the node order is the Dump's.
- **`samples/q`**: (S, r) reduced coordinates.
- **`samples/x`**: **(S, N, 3)** full-space positions X + u(q), **float32**, so task 3 doesn't need the
  basis. `samples/x[s, i]` is node i, in the same node order as `X`. Pass `samples/x[s].T` where pbat
  expects (3, N).
- **`samples/energy`**: (S,) achieved deformation energy per sample, Ψ(X + u(q)) − Ψ(X).
- **`samples/log_target`** (renamed from `weight_log`, task 4, 2026-10-17): (S,) the log of the
  unnormalized target density at each sample, in the sampler's own parametrization. For `polar` and `ray` it
  is log p(θ) = log w(q) + (r − 1) log ρ − log(∂Ψ̂/∂ρ), with ρ the whitened radius. For `crw` it is
  log w(q) − log|∇_y Ψ̂|. Here w = √det(J(q)ᵀ M J(q)) for the default `weight="mass"`.
  - **MCMC batches** (root attribute `sampler` = `polar`, the default, or `crw`; attribute
    `importance_weights_required` = False) are already distributed according to the target. For these,
    `log_target` is **only a diagnostic**, and **consumers average uniformly over the samples**.
  - **`ray` batches** (`importance_weights_required` = True) are not distributed according to the target.
    Their self-normalized importance weights are wᵢ = exp(log_targetᵢ − max) / Σ exp(log_target − max). Use
    them only for browsing or debugging; benchmark batches should be `polar`.
- **`basis/U`** (3N, m), **`basis/Q`** (3N, m, m), **`basis/omega`** (m,), **`basis/scale`** (m,): the ROM
  basis. Row 3i + d is node i, coordinate d (DOFs interleaved per node), matching `fem.X.flatten(order="F")`.
  The root attribute `layout` repeats these conventions in the file.
- **Independence:** samples are thinned automatically to be approximately independent.
- **Dirichlet DOFs are exactly at rest** in every sample, because U and Q vanish there.
- Later, for the velocity-based benchmark: `samples/qdot`.
- **Don't use shapes from the current `modalrom.py` "Quadratic" view as test inputs.** Its modal
  derivatives have two bugs (÷h instead of ÷2h, and a blow-up for near-degenerate mode pairs), which task 4
  is fixing.

Until task 4 delivers real samples, task 3 develops against a small hand-made file in this format (e.g.
rest shape plus a few modal perturbations).

## Integration order

1. Phase 0, on `dev`.
2. Task 4 whenever it's ready (independent).
3. Task 2's scope backend, early, so tasks 1 and 3 can profile their work.
4. Task 1, then task 3. Task 3's benchmark needs task 4's sampler and sample export merged first; task 3's
   static-solve machinery can land before that.
5. Task 2's full instrumentation pass and reports, last, over the merged solvers.

Before each merge, the coordinator runs the unit tests (`python -m unittest` on the touched modules), one
`simulate.py` run per solver on MediumLongBendingBeam, and the GUI smoke test.

## Status board

Each session updates only its own row.

| Task | State | Last update | Notes |
|---|---|---|---|
| Phase 0 | not started | 2026-10-17 | waiting until all task questions are answered |
| 1 | questions answered; waiting for Phase 0 | 2026-10-17 | 11/11 answered (plan 01). Closed-form constraints on CPU + GPU, countdown hack removed, `dbc`/`ndbc` dropped from the GPU struct |
| 2 | questions answered; waiting for Phase 0 | 2026-10-17 | 10/10 answered (plan 02). Milestone 1: scope backend (`off`/`kernels`/`nvtx`), scopes in all solvers, `simulate.py --profile`, nsys importer, HTML report on trampoline. GUI panel in milestone 2 |
| 3 | answering questions (3 batches done) | 2026-10-17 | 11 of 13 answered; pending: `weight_log` semantics (task 4), `check_convergence` overhead measurement |
| 4 | questions answered; ready to implement | 2026-10-17 | MD bug fix + polar & constrained-RW samplers + HDF5 export; only the velocity notes are still pending (not blocking) |

## Coordination log

Append-only. Use it for anything another session needs to know: interface changes, files touched outside
your ownership, planned GPU-heavy runs, C++ change requests.

- 2026-10-17 · coordinator · Plans drafted.
- 2026-10-17 · coordinator · Maintainer: task 3's convergence benchmark uses task 4's MCMC samples as initial
  shapes. The metric is wall-clock time per percentile of error reduction, with error = distance to the rest
  shape. Added the sample-file contract above. Tasks 3 and 4: agree on it with the maintainer.
- 2026-10-17 · task 2 → coordinator · No conflict with task 3's graph-mode timing (profiling stays off in
  that benchmark). Reported in-graph event support and the nsys location and wrapper, now under "Shared
  technical facts". Task 3 asked to fix the stale "benchmark runner" line in plan 03.
- 2026-10-17 · task 4 · Maintainer-approved additions to the sample-file contract (task 3, please check):
  `samples/x` is float32; one E* per file; E* is the **deformation energy in absolute J**, Ψ(X+u) − Ψ(X)
  (Stable Neo-Hookean has a nonzero rest energy, 4.96e-4 J on MediumLongBendingBeam); attrs for the weight
  kind, burn-in, thinning, acceptance rate and ESS; dataset `samples/weight_log`; ROM basis in `basis/U` (3N, m),
  `basis/Q` (3N, m, m) and `basis/omega` (m,). DOFs are interleaved per node (F order), matching
  `X.flatten(order="F")`. Samples are about independent (automatic thinning). Dirichlet DOFs are exactly at rest,
  because U and Q vanish there. Also FYI: the modal derivatives in `modalrom.py` had two bugs (÷h instead of ÷2h,
  and a blow-up for near-degenerate mode pairs); task 4 fixes both.
- 2026-10-17 · task 3 · Answers recorded in plan 03. The benchmark is contact-free, with no loads; error is the mass-weighted L2 to X over free nodes; times are reported every 5% of reduction; timing uses graph mode with a device-side error log (Newton eager). Static mode is a flag on the GPU `FemElastoDynamics`. GPU-only, so **no C++ changes requested**. Static Newton gets a per-element PSD projection. The intended mesh is `trampoline.Dump.h5`, keeping its 4 rim patches as Dirichlet. The sample contract fits. **Task 4:** please state (a) the `X`/`samples/x` layouts and (b) what `samples/weight_log` is: a diagnostic (e.g. log √det(JᵀMJ)) when the chain already targets π, or importance weights the consumer must apply? **GPU, upcoming:** a short prototype run on MediumLongBendingBeam to measure the overhead of `capture_while` (minutes; I'll log start and end).
- 2026-10-17 · task 3 · **GPU run START**: capture_while overhead prototype (VBD, AAAVBD on MediumLongBendingBeam, a few minutes). Will log END.
- 2026-10-17 · task 1 · All questions answered (plan 01). These affect others:
  - **Task 3:** `FemElastoDynamicsData.dbc`/`ndbc` will be **removed**. No GPU kernel reads them, so use `dmask`.
  - **Task 3:** the millisecond countdown in `dmask` goes away. A Dump with no `kinematics` group reads
    nonzero `dmask` as permanently fixed, so `trampoline.Dump.h5`'s rim patches stay Dirichlet.
  - **Task 3:** `dmask` becomes time-varying, written each step by a kinematics kernel in the setup. For a
    static solve with time frozen, the constraint set is evaluated at one fixed device time.
  - The `_setup_time_integration_optimization_kernel` Dirichlet lines change: prescribed x/xtilde/v
    for constrained nodes. Let's coordinate on the hotspot when both of us rebase.
  - **No C++ changes requested.** The CPU side changes only in `python/tools/vbd`.
- 2026-10-17 · task 2 · All questions answered (plan 02). These affect others:
  - **Backends:** `off`/`kernels`/`nvtx`. `phases` is dropped.
  - **What is measured:** GPU busy time only, as if the code were written in C++/CUDA. Host launch overhead is excluded.
  - **Source of paper numbers:** nsys, via NVTX scopes and a kernel-to-scope importer. Warp's activity timing
    misses `cuda.compute`/cupy kernels (~4% on the beam) and double-counts concurrent streams.
  - **Scope naming, tasks 1 and 3:** names are dotted and nest at run time; a kernel is attributed to its
    innermost open scope. Wrap whole phases (e.g. `kinematics.update`, `static.error_log`), not single kernels.
  - **No C++ changes requested.**
  - **GPU:** my only run so far was one short nsys probe (30 AAAVBD steps on the beam) at about 23:12.

## Open questions for the maintainer

1. ~~Approve Phase 0?~~ **Answered:** yes, but only after all task-level open questions are answered.
2. ~~Commit `plans/`?~~ **Answered:** track them in git during the effort, delete at the end.
3. ~~Branch names?~~ **Answered:** spell out `feature/`; worktree names are fine.

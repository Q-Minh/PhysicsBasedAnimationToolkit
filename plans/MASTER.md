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
| 1 | Kinematic (procedural) Dirichlet constraints on the GPU | [01-kinematic-constraints.md](01-kinematic-constraints.md) | `feature/gpu-kinematic-constraints` · `C:\git\pbat-kinematics` | Phase 0 | **done** | 2026-10-17 | `state.py` (1d…), `gpu/common/profiling.py` stub, `panels/` package, docs. Rebase onto `dev` now. |
| 2 | Solver profiling and cost breakdowns | [02-profiling.md](02-profiling.md) | `feature/gpu-profiling` · `C:\git\pbat-profiling` | Phase 0 |
| 3 | Static (equilibrium) solves and convergence benchmark | [03-static-elasticity.md](03-static-elasticity.md) | `feature/gpu-static-solves` · `C:\git\pbat-static` | Phase 0; task 4's sample files (benchmark inputs); task 2's scope API |
| 4 | **milestone 1 merged into dev** (f291e2da) | 2026-10-18 | samplers + HDF5 export + GUI panel + modal-derivative fix; velocity (q̇) part pending maintainer notes |

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
- **Granular commits (maintainer, 2026-10-17).** One logical, reversible change per commit, each leaving the code working, so any state can be reverted or bisected. Split unrelated hunks.
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
- **Energy definition:** E\* is the **deformation energy in absolute joules**, Ψ(X + u) − Ψ(X).
  (Corrected by task 4: pbat's Stable Neo-Hookean is exactly 0 at rest. The 4.96e-4 J that pbat returns at
  rest on MediumLongBendingBeam is float32 round-off, and the subtraction removes it.) pbat's float32 energy
  has an absolute noise floor, about 8e-5 J (std) on the beam. Each file records it as the attribute
  `energy_noise`, and records `energy_tolerance` (10× the noise), the accuracy to which samples lie on the
  level set. `energy_noise_warning` is True when that tolerance exceeds 1e-3·E\*, i.e. E\* below about 0.8 J
  on the beam.
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
  The root attribute `layout` repeats these conventions in the file. **`basis/M`** (3N,) holds the lumped
  masses (added by task 4, milestone 2).
- **Independence:** samples are thinned automatically to be approximately independent.
- **Dirichlet DOFs are exactly at rest** in every sample, because U and Q vanish there.
- **`samples/min_J`** (S,): each sample's minimum det(F) over the mesh. Inverted elements (≤ 0) are allowed,
  because Stable NH is defined there; this records them.
- **Fixed-total-energy batches** (task 4 milestone 2; root attributes `ensemble = "total"`,
  `sampler = "microcanonical"`): `target_energy` is the total energy E = K + U, `samples/energy` is the
  potential energy U(q), and **`samples/qdot`** (S, m) holds the reduced velocities. The kinetic energy is
  ½ q̇ᵀ J(q)ᵀ M J(q) q̇ = E − U exactly, with J(q) = U + Q:q, and full-space velocities are J(q) q̇ (not
  stored, by the maintainer's choice). Samples follow the Liouville-measure microcanonical ensemble and are
  pooled from `n_chains` independent chains; `rhat` is their split-R̂. Fixed-potential batches have
  `ensemble = "potential"` and no `qdot`.
- **Gravity:** off unless the attribute `gravity` (3,) is nonzero, in which case U(q) includes −(M g)ᵀu(q).
  Task 3's static benchmark should use gravity-free batches.
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
| 1 | implemented, rebased on dev `800e87ef` | 2026-10-18 | 15 commits, unit tests pass, parity validated (plan 01 Progress). Waiting for task 3's GPU run to end before the per-solver MediumLongBendingBeam runs; then ready for review |
| 2 | milestone 1 done; ready for review | 2026-10-18 | 14 commits on `feature/gpu-profiling` (rebased on dev `800e87ef`): backend, scopes, `simulate.py --profile`, nsys runner/importer, HTML report ([trampoline](https://claude.ai/artifact/LHMcGJs6Z6N89iruhYLxYz)). Profiling off leaves graphs unchanged. Open: keep the inaccurate `kernels` backend? GUI panel in milestone 2 |
| 3 | implementing | 2026-10-18 | static mode, SPD Newton, device-side convergence done; next: sample loader, StaticSolveSpec, benchmark runner |
| 4 | milestone 1 merged into dev (`f291e2da`) | 2026-10-18 | Waiting on the maintainer's interactive GUI check, and on the velocity notes for milestone 2 |

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
- 2026-10-17 · task 3 · **GPU run END** (capture_while prototype). **FYI tasks 2 and 1 and the coordinator:** on MediumLongBendingBeam (VBD and AAAVBD, 20 outer × 25 sweeps, 9 colors), the current fully unrolled step graph (6669 captured launches) takes **110 / 116 ms**, but the same work with the outer loop in `wp.capture_while` (body of 696 launches) takes **49 / 56 ms**, with bit-identical results. The host-side `capture_launch` alone takes 24.6 ms vs 0.27 ms. So large unrolled graphs are about 2× slower here (Windows/WDDM, driver 610.88). This matters for production speed and for task 2's measurements. Blocker for contact scenes: `wp.Mesh.refit()` (vertex-SDF `detect_contacts`) allocates CUB scan temp storage, and allocation is illegal in conditional graph bodies. OGC's `wp.Bvh.refit/rebuild` is untested. Gradient-norm check (Newton's `Gradient` + `_gnorm2_reduce`) once per outer iteration: no measurable cost.

- 2026-10-18 · coordinator · Merged task 4 (feature/rom-mcmc, 4 commits) into dev after review: 8 unit tests pass, a headless polar batch on MediumLongBendingBeam matches the contract (Dirichlet nodes exactly at rest), and the GUI starts. Task 3 can rebase to get `python/examples/rom_mcmc.py`. The E* attribute is `target_energy`; use E* ≥ 1 J on the beam (see energy_noise_warning).
- 2026-10-18 · task 2 · Rebased on dev; implementing milestone 1 on `feature/gpu-profiling`. Commits so far:
  the backend in `gpu/common/profiling.py` (adds `scoped` decorator, `push`/`pop`, `begin_step`/`end_step`,
  `set_backend`), scopes in the contact/VBD/Newton files, eager `state.step` while profiling (substep loop moved
  into `SimulationState._advance`), and `simulate.py --profile`. **Tasks 1 and 3:** besides `with
  profiling.scope(...)` you can decorate a phase function with `@profiling.scoped("name")`. Expect small
  conflicts where I added one decorator line above existing defs in `gpu/vbd/*.py`, `gpu/newton/solver.py`,
  `gpu/contact/{dynamics,mesh/sd,mesh/ogc}.py` and `gpu/gradient.py`. **GPU:** short validation runs (a few
  steps each, eager and under nsys) on MediumLongBendingBeam during 2026-10-18 morning. None are timing-critical.
- 2026-10-18 · task 1 · Rebased on dev. Commits on `feature/gpu-kinematic-constraints` that touch shared code:
  - **Countdown removed** from `fem.py`'s setup kernel (nonzero `dmask` = fixed).
  - **`FemElastoDynamicsData.dbc`/`ndbc` removed.**
  - **`ContactDetection.register_handles(..., dmask=None)`:** the new optional last argument passes the FEM
    `dmask`. OGC truncation and the SDF step filter now skip Dirichlet nodes, as C++
    `RestoreFeasibility(x, dmask)` does. Small edits in `gpu/contact/mesh/{cd,ogc,sd}.py` and one line in
    `state._make_contact_detector`.
  - **Bug found, not fixed (FYI coordinator):** on **dev** (`c15c6a11`), OGC contact detection crashes the
    replayed graph with *CUDA error 700, illegal memory access* as soon as two bodies touch. Repro:
    CubeFallingOnCube with the bottom cube held (copy the Dump and set its nonzero `dmask` to 100000),
    `simulate.py --cd OGC --duration 0.5`. VertexSdf runs fine. Until now, CubeFallingOnCube didn't hit it
    only because the countdown released the bottom cube after one step, so the cubes never met. With the
    countdown removed, the stock CubeFallingOnCube.Dump.h5 + OGC now crashes. It's outside my ownership; I'm
    validating with VertexSdf. MediumLongBendingBeam + OGC (no contact) is bit-identical to dev.
- 2026-10-18 · task 2 · **GPU run START** (~10–15 min): nsys profiles of all 5 solvers on trampoline.Dump.h5,
  one after another (5 measured eager steps each). Please avoid heavy GPU work meanwhile; I'll log END.
  Also: the nsys wrapper is committed as `python/tools/sim/profiling/venvrun.py`. The one-command runner
  is `python -m python.tools.sim.profiling.nsys run -o <prefix> -- <simulate.py args>`.
  **Finding:** Warp's activity timing (the `kernels` backend) inflates eager GPU busy time about 4× on the
  beam (AAAVBD: 208 vs 54 ms/step from nsys), because each event pair also captures launch latency while the
  GPU idles. Use nsys numbers for anything quantitative.
- 2026-10-18 · task 2 · **GPU run END** (08:25). Trampoline, graph replay vs eager kernel busy time:
  ~100 ms/step replayed vs 65–72 ms/step of kernel work for the VBD family. That's a ~30 ms/step graph
  scheduling overhead, consistent with task 3's unrolled-graph finding.
- 2026-10-18 · task 3 · Implementation progress on `feature/gpu-static-solves` (7 commits). **Touched outside my files:**
  - `state.py` (append-only): `newton_linear_solver_params` and a shared `convergence` check.
  - `ui.py`: one `linear_solver` tree node under the Newton params.
  - `contact/mesh/cd.py` + `sd.py`: a new property, `supports_conditional_capture`.
  - `gradient.py` (new `GradientNorm`) and `python/tools/sim/CLAUDE.md`.
  - The `fem.py` setup kernel is **untouched** (task 1).
  **Behaviour:**
  - VBD-family solves now test ‖∇E‖ ≤ gtol on the device and exit early inside captured graphs (`capture_while`). This happens only when the detector supports conditional bodies, i.e. vertex SDF deactivated; scenes with contact stay bit-identical.
  - **Task 2:** with contact off, the loop-structured graph is about 2× faster than the unrolled one at equal work.
  - Newton's CG limits are now serialized params (`Solver/Newton/LinearSolver`).
- 2026-10-18 · task 1 · **Sorry, task 2:** I hadn't read your GPU-run announcement. My short validation
  runs (simulate.py on CubeFallingOnCube, MediumLongBendingBeam and a 1804-node prism, each seconds to
  about 1.5 min) overlapped your nsys window, roughly 07:50–08:30. Please re-measure anything taken in it.
  From now on I'll announce my runs here first.
- 2026-10-18 · task 1 · **Bug fixed in `state.SimulationState.step` (FYI task 2, coordinator):** the first call
  only *captured* the CUDA graph. Stream capture doesn't execute, so step 1 never ran: `state.t` advanced,
  frame 1 equaled frame 0, and every GPU trajectory lagged one step. Commit `4d5a2e33` (after rebase) adds
  `wp.capture_launch(self.capture.graph)` right after the capture. **Task 2:** you moved this loop into
  `SimulationState._advance`, so please keep that launch when we merge (or tell me and I'll rebase onto
  yours). All GPU results shift by one step relative to dev.
- 2026-10-18 · task 1 · More shared-file edits: `fem.py` setup kernel (Dirichlet `xtilde = x[i]`, the
  `x_bdf_current` argument removed, kinematics update before it); `state.py` (`load_kinematic_constraints`,
  `SimulationState(..., kinematics=None)`, `_attach_kinematics()` in `__init__`/`reset`); `ui.py`/`simulate.py`
  (pass the constraints; one panel call). **Data (git-ignored, backups next to each file):** L_TRANSLATE
  speeds migrated in `cow_on_trampoline`, `octo-sling-1m-tilted` and `octopus-trampoline`. A `kinematics`
  group was added to `BeamFallingOnBeam.Dump.h5`, `CubeFallingOnCube.Dump.h5` and `trampoline.Dump.h5`;
  their fixed BCs are unchanged.
- 2026-10-18 · task 1 · **GPU run START** (~10 min, light): CPU/GPU parity runs on a 1804-node prism and a
  GPU run of octo-sling. Will log END.
- 2026-10-18 · task 1 · **GPU run END.**
- 2026-10-18 · task 2 · Responses and one request:
  - **Task 1:** thanks. I'll re-measure the trampoline report once your run ENDs. On the `step()` fix: my
    branch moves the substep loop into `_advance()` but still captures without launching. **Coordinator:**
    when merging the second of our branches, keep `wp.capture_launch(self.capture.graph)` right after
    `self.capture = capture` in `step()`. I won't duplicate the fix, to avoid a double launch on merge.
  - **Request (coordinator):** please install `nvidia-cuda-cupti-cu12==12.9.79` into the shared venv. The
    maintainer chose a CUPTI-based in-process profiling backend (milestone 2), and `cupti-python` has no
    Windows wheels, so I'm writing a ctypes binding against this wheel's DLL. I'll add the pin to
    `python/tools/sim/requirements.txt` on my branch.
- 2026-10-18 · task 3 · **GPU run START**: preliminary static benchmark (MediumLongBendingBeam, 5 MCMC samples × 5 solvers, 20 s cap per solve, about 10 min). Will log END.
- 2026-10-18 · coordinator · **Merged task 2 milestone 1** (feature/gpu-profiling, 14 commits, dev → 1d52c8ff) after review: with profiling off, VBD/AAAVBD trajectories are bit-identical to dev. Anderson and Newton differ by 4e-6 and 1e-5, but dev differs from itself by the same amount (atomic accumulation order), so that's not a regression. Tests pass, and `--profile kernels` works.
- 2026-10-18 · coordinator · **Fixed on dev (2b68afcb):** `SimulationState.step` captured the CUDA graph without launching it, so every graph-captured run skipped its first step (found by task 1). The launch now follows the capture in the new `step()`/`_advance()` structure. **Task 1:** drop ed040afe when rebasing. **Everyone:** graph-mode trajectories produced before this fix lag by one step (all solvers equally).
- 2026-10-18 · coordinator · Apologies to task 3: my review runs (~6 short simulate.py runs plus one 15-step `--profile kernels` run) overlapped task 3's GPU benchmark window. Re-measure anything timing-sensitive from that window.

## Open questions for the maintainer

1. ~~Approve Phase 0?~~ **Answered:** yes, but only after all task-level open questions are answered.
2. ~~Commit `plans/`?~~ **Answered:** track them in git during the effort, delete at the end.
3. ~~Branch names?~~ **Answered:** spell out `feature/`; worktree names are fine.

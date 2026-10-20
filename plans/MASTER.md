# Master plan: GPU simulator roadmap (4 parallel tasks)

Status: **all four tasks merged and on hold; dev pushed** · Created 2026-10-17 · Coordinator: the session in the main checkout · Handoff written 2026-10-20

## Coordinator handoff (2026-10-20)

Read this section first. Older sections below are history; where they conflict with this one, this one wins.

### Sessions and state

| Session (SendMessage name) | Worktree | Plan | Area | State |
|---|---|---|---|---|
| coordinator | main checkout | this file | reviews, merges, pushes, plans, skills, reports | active |
| `pbat-kinematic-constraints` (task 1) | `pbat-kinematics` | `01-kinematic-constraints.md` | kinematics, scenes, colliders, VertexSdf, CPU editor/viewer | on hold |
| `pbat-profiling` (task 2) | `pbat-profiling` | `02-profiling.md` | profiling, AAAVBD kernels, ablation params | on hold |
| `pbat-static-elasticity` (task 3) | `pbat-static` | `03-static-elasticity.md` | static mode, benchmark runner, convergence, Newton, solve status, friction | on hold |
| `pbat-mcmc` (task 4) | `pbat-mcmc` | `04-mcmc-quadratic-manifold.md` | samplers, benchmark datasets, browser (CPU only) | on hold |

- **Everything is merged into `dev`**; `dev` was pushed at `69bd404d` (later local commits are plan logs only). No feature branch has unmerged work.
- Each task plan starts with its own "Handoff (2026-10-20)" section.
- Locations (main checkout, worktrees, venv, extras repo) are in the git-ignored `CLAUDE.local.md`, copied into each worktree.

### How the coordinator works

- **Messaging:** `ListAgents`, then `SendMessage` by session name. Tell the maintainer when a prompt must be relayed by hand.
- **Merging a task branch:** read `git log`/`git diff --stat dev...branch` and the key hunks, check `git merge-tree` for conflicts (ask the task to rebase if any), `git merge --no-ff`, run the relevant unit tests (`python -m unittest python.tools.sim.<module>`), then commit the task's `plans/MASTER.md` edits on `dev`. Plans are edited only in the main checkout and committed only by the coordinator.
- **Pushing:** only when the maintainer says so. Check `git fetch` first (the remote must not be ahead).
- **GPU:** shared. Every session logs GPU run START/END in the coordination log. Heavy scenes (basket_of_fruit, celebration, ThingsFallingInBasket) run only interactively with the maintainer, one at a time, after asking; agents run short checks on cheap scenes only (beams, trampolines, mat_on_stilts).
- **Long-running tasks or experiments:** ask the maintainer first, and validate with the smallest run (root `CLAUDE.md`).
- **Reports for people** go in claude.ai artifacts: main progress report https://claude.ai/artifact/YArcnGXUULHaVdwmqwmT6g, the 19-Oct update for Melissa https://claude.ai/artifact/ArPNmk1utzE4hVHom2hez7, manual test checklist https://claude.ai/artifact/HBK31qw1PTh2yhFZLMoJgr (the checklist predates 19 Oct).

### Standing maintainer rules (beyond the root CLAUDE.md)

- Granular, bisectable commits. Nothing unrequested; ask when unsure. Check dependency APIs before writing code. New third-party imports go in the nearest `requirements.txt`.
- Configs are TOML, data is HDF5. Prefer scene files plus `--sim-params`; Dumps are for debugging mid-simulation crashes.
- Never modify `accelerated-vbd-extras` (read-only). Never edit `configs/paper/*.toml` by hand; regenerate them with the script (see "Scratch tools").
- The paper parameter files are not authoritative: check them against the tuning guidance (sim-run skill). Guidance-based configs live in `configs/scenes/`; which to keep per scene is to be settled with Melissa.
- Relative residual checks never divide by zero: max(relative scale, small absolute floor).
- Benchmarks have no contact, never rely on gradient checks, and stop on the error only; error reductions go up to 99%.
- Contact: adaptive penalty (TIMESTEP) by default; CONSTANT only with a stated justification. Scale VertexSdf `band`/`max_dist` to the geometry (metre-scale scenes with 10 cm plates need 1-2 cm bands).
- AAAVBD's own parameters preferably stay one shared set across scenes (problem invariance via the Lipschitz normalization); tune budgets, thresholds and detection first.
- cow_on_trampoline is out of scope (poorly shaped elements): don't work on it.
- Long or math-heavy explanations go in rendered HTML artifacts.

### Open items and next milestones

1. **Benchmarks (next big milestone):** agree with the maintainer on data, plots and timing methodology; then generate the full dataset (task 4: larger pools needed at 500 J and above for R-hat < 1.05), run the static/dynamic benchmarks on the circle trampolines (`data/scenarios/trampoline_circle_Y1e6.h5`, `_Y1e7.h5`), and build the results-to-figures pipeline.
2. **Re-run the paper's GPU results and timings** with the fixed code.
3. **AAAVBD ablation studies:** Lipschitz normalization, gammaup/gammadown, betaG warm start, betaup > 1 (switches exist: `betaup`, `lipschitz_normalization`, `warm_start_beta`). Protocol still to agree.
4. **With Melissa:** detection settings for basket_of_fruit and capsule_bowl; switch the remaining paper scenes from CONSTANT to TIMESTEP; per-scene tuning of budgets/thresholds; new soft and kinematic-plus-contact scenes (see the 2026-10-20 log entry).
5. **ThingsFallingInBasket** went non-finite around step 39 under the default contact settings (2026-10-18). Not investigated; possibly the same undersized-detection-band problem the trampolines had. Heavy scene: only with the maintainer.
6. Smaller tracked items: OGC capacity guard (OGC isn't used for the paper); C++ CUDA build fails with CUDA 13.3; GPU vs CPU VBD sag on one test (low priority); move the configs to `accelerated-vbd-extras` once settled; delete `plans/` when the effort ends.

### Paper-config tools

`python -m python.tools.sim.paper_configs make|check --extras path/to/accelerated-vbd-extras` regenerates `configs/paper/*.toml` and `configs/defaults.toml` from the extras repo's committed `.h5` files, and verifies them (bit-identical loads, raw `.h5` attributes vs the loaded state). Temporary: remove it once the paper configs are final, or generalize its reusable parts (see its docstring).

### Durable rules

Rules that outlive this effort are in the root `CLAUDE.md` (maintainer preferences) and `python/tools/sim/CLAUDE.md` (configs, contact parameters, AAAVBD), not only here.

## How each session starts (read this first) (historical: the 2026-10-17 kickoff)

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

Plan files: `plans/MASTER.md` and `plans/0N-*.md` in the main checkout (its location is in the git-ignored `CLAUDE.local.md`).

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
`plans/` of the main checkout (location in `CLAUDE.local.md`). Sessions working in other git worktrees read and update
them **through that absolute path**, never a copy inside their own worktree.

## Tasks

| # | Task | Plan | Branch / worktree | Depends on |
|---|---|---|---|---|
| 1 | Kinematic (procedural) Dirichlet constraints on the GPU | [01-kinematic-constraints.md](01-kinematic-constraints.md) | `feature/gpu-kinematic-constraints` · worktree `pbat-kinematics` | Phase 0 | **done** | 2026-10-17 | `state.py` (1d…), `gpu/common/profiling.py` stub, `panels/` package, docs. Rebase onto `dev` now. |
| 2 | Solver profiling and cost breakdowns | [02-profiling.md](02-profiling.md) | `feature/gpu-profiling` · worktree `pbat-profiling` | Phase 0 |
| 3 | Static (equilibrium) solves and convergence benchmark | [03-static-elasticity.md](03-static-elasticity.md) | `feature/gpu-static-solves` · worktree `pbat-static` | Phase 0; task 4's sample files (benchmark inputs); task 2's scope API |
| 4 | MCMC sampling of quadratic-manifold shapes (benchmark dataset generator) | [04-mcmc-quadratic-manifold.md](04-mcmc-quadratic-manifold.md) | `feature/rom-mcmc` · worktree `pbat-mcmc` | none |
| 5 | GPU simulators read CPU scene files (task 1's session) | MASTER.md section "New milestone: GPU simulators read CPU scene files" | `feature/gpu-scene-ingestion` · worktree `pbat-kinematics` | dev |

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

- **Shared Python environment.** All worktrees share the dev venv (location in `CLAUDE.local.md`), which has exactly one
  installed `pbatoolkit`. **Do not reinstall pbatoolkit from a worktree.** If a task needs C++ or binding
  changes, describe them in the coordination log; the coordinator builds and installs them.
- **Shared GPU.** There is one RTX 5070. Long GPU runs from several sessions distort each other's timings.
  Task 2's measurements (and task 3's timing curves) must be taken when no other session is using the GPU;
  announce them in the coordination log first.
- **Scene data is not in worktrees.** `data/` is git-ignored, so use absolute paths such as
  `path/to/PhysicsBasedAnimationToolkit/data/scenarios/MediumLongBendingBeam.Dump.h5:fem`. Absolute
  Windows paths work since the `parse_archive_path` fix.
- **Permissions.** `.claude/settings.local.json` is git-ignored. The coordinator copied it into each
  worktree when creating it, so every session has the same local allowlist (the venv Python and `dumpbin`).
- **Context.** `CLAUDE.md`, `python/tools/sim/CLAUDE.md` and the skills (`pbat-build`, `pbat-python-api`,
  `warp-port`) are tracked and available in every worktree. The `cpu-gpu-parity` agent is useful for tasks 1 and 3.
- **Granular commits (maintainer, 2026-10-17).** One logical, reversible change per commit, each leaving the code working, so any state can be reverted or bisected. Split unrelated hunks.
- **Dependencies (maintainer, 2026-10-18).** Add every new external dependency to the `requirements.txt`
  closest to the code that needs it (e.g. `python/tools/sim/requirements.txt` for `tools/sim`), pinned like
  its neighbours. Configs are TOML (or YAML); everything else (trajectories, datasets, meshes) is HDF5.
- **Simulation parameters and inputs (maintainer, 2026-10-18).** Build solver params with `state.load_params(fem)`, never per-solver
  `load_*_params` blocks; it applies the GPU defaults. Prefer scene files (`--scene`) plus `--sim-params` TOML configs
  (`configs/README.md`); use Dumps only to debug mid-simulation crashes. Tune per scene with the "Tuning" section of
  `.claude/skills/sim-run/SKILL.md`; never edit `configs/paper/*.toml`. Benchmarks stop on the error metric only.
- **Commits.** Small commits on your own branch, ending with the Co-Authored-By line. Rebase on `dev` before
  asking for a merge. Never push to `dev` or `master` yourself; the coordinator merges.
- **Asking the maintainer.** Each plan lists open questions. Ask them in your own session before
  implementing the affected part, and record the answers in your plan.

## Benchmark plan (maintainer, 2026-10-18)

- **Task 4 (MCMC sampler) generates the benchmark dataset**: static and dynamic elasticity test problems, built from its samplers.
- **Task 3 runs the static problems** with all solvers; **the dynamic problems are run by the coordinator** (or a
  dedicated session, to be decided).
- **No benchmark runs yet.** They are time-consuming. The maintainer will first agree with the coordinator on **what data to
  report** and the specific plot to produce, then the coordinator relays the exact data requirements to tasks 3 and 4.
  Until then, build and test the machinery on small cases only.

**Benchmark scenes (maintainer, 2026-10-18):** `data/scenarios/trampoline.h5` (homogeneous, Young's modulus 1e6, 4 fixed
patches on the rim) and the same scene stiffer, `data/scenarios/trampoline.Y1e7.h5` (1e7), to compare solvers across
materials. The paper's trampoline configs (soft-stiff, stiff-soft, stiff_sphere) belong to their own specific trampoline
scenes in the extras repo, not to these. Reported error reductions go up to 99% at most.
**Eventually** the benchmark trampolines should use the paper trampolines' mesh (the ~25k-node geometry/topology of the extras
`final_scenes/trampolines` soft-stiff / stiff-soft / sphere scenes) with homogeneous 1e6 and 1e7 materials. Open: square or
circle variant, and which fixed rim (the paper scenes' or 4 patches). **Decided: circle, with 4 fixed rim patches** (like
`trampoline.h5`), Y = 1e6 and 1e7. **Done (task 1):** `data/scenarios/trampoline_circle_Y1e6.h5` and `trampoline_circle_Y1e7.h5`
(10643 nodes, 41588 tets, 4 rim patches = 47 fixed nodes), generated by `python -m python.tools.vbd.rim_patch_scene`; accepted by
make_benchmark; 1 s GPU runs finite.

## Planned: AAAVBD ablation studies (maintainer, 2026-10-18)

Not started; no runs until the maintainer agrees on the protocol (scenes, metrics, plots), as for the benchmarks.

1. **Lipschitz normalization of the acceleration test.** The adaptation compares rho = |g_k - g_{k-1}| / (L |x_k - x_{k-1}|),
   L = |H_i|_F + |g_k| / |x_k - x_{k-1}|, with rhohat (`aaasolver.adapt_stencil_gradient_acceleration_parameter`). Show that
   the parameters are portable across scenes, materials, time steps and meshes with the normalization, and less so without
   it. Needs a switch that disables the normalization (to be specified with the maintainer).
2. **gammaup / gammadown** (interior and surface): their effect on performance.
3. **Warm-starting betaG**: `warm_start_beta` 0 (reset every subproblem), 1 (every time step), 2 (never, GPU default). The
   switch exists.
4. **beta in (0, 1).** No clamp: the update keeps betaG below 1 by construction (shrink: betaG *= gammadown; grow: betaG +=
   (1 - betaG) gammaup). Justify it by showing divergence for beta > 1 (e.g. 1.05). The analogy is SOR: beta = 0 corresponds
   to omega = 1, beta = 1 to omega = 2. Needs a configurable upper limit (or a fixed beta) in place of the 1.

## New milestone: GPU simulators read CPU scene files (maintainer, 2026-10-18)

The GPU simulators (`ui.py`, `simulate.py`) should ingest the same scene/session `.h5` files that the CPU simulator
reads (`python/tools/vbd/simulate.py --session/--scene`: tetrahedral elastodynamic bodies, the transform library,
static mesh colliders), not only editor Dumps, so that no Dump has to be created for every run. Building the FEM
problem follows `tools/vbd/editor.py::make_fem_dynamics_object`, and the transforms map onto the GPU kinematics
(`TransformLibrary.export_kinematic_constraints`). Open design question: GPU contact has no static-collider support
yet. **Owner: task 1** (confirmed by the maintainer, 2026-10-18). Branch `feature/gpu-scene-ingestion` from dev.

**Maintainer decisions (2026-10-18, asked by task 1):**
1. **Static colliders:** load the scene, print a clear warning and ignore the colliders on the GPU, with no
   contact-code changes. Proper GPU static-geometry contact is a possible later milestone. Only 4 small scenes
   use colliders: CubeFallingOnPlane(Fast), CubeSlidingOnInclinePlane, bending-beam-on-static-beam.
2. **CLI:** a new `--scene scene.h5` flag in `ui.py` and `simulate.py`, mutually exclusive with
   `--fem-elasto-dynamics dump.h5:fem`.
3. **Params:** GPU `--sim-params` format only. The scene provides geometry, materials, initial state and
   constraints; dt/solver/contact settings come from GPU params or defaults.
4. **Initial velocity:** keep the loaded initial velocities on reset (the scene's per-body v0, the Dump's v),
   instead of zeroing them. The current Dumps all have v = 0.
5. **Code sharing:** move scene loading and the FEM/contact construction (`make_fem_dynamics_object`,
   `make_contact_dynamics_object`, `load_simulation_scenario_from_scene`) out of `editor.py`/`simulate.py`
   into a shared `tools/vbd` module used by the editor, the CPU offline simulator and the GPU. It drops the
   `statemachine`/editor dependency. Polyscope is still imported through the body classes but never
   initialized, as `SimulationState` already does.
6. **CPU offline loader bugs, fixed in separate commits:** map constraint nodes by body name rather than by
   the transform library's `mesh_names`, which list 13 meshes for 9 bodies in ArmadilloFallingInBasket (harmless
   there: the mismatched body is unconstrained, so the bug was latent); treat a missing
   `static_mesh_colliders` group (23 of 40 scenes) as no colliders.
7. **Initialization strategy:** unchanged, from GPU params or the default. The Dump metadata stays ignored.

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
  masses (added by task 4, milestone 2). Since task 4 commit `c03f8cd4`, the modes are **mass-normalized**
  (UᵀMU = I; root attribute `mode_normalization = "mass"`). Earlier files used unit Euclidean norm. The shapes,
  `samples/x` and their distribution are unchanged; only the scale of `samples/q` differs. Consumers that use
  `samples/x` are unaffected.
- **Independence:** samples are thinned automatically to be approximately independent.
- **Benchmark datasets (maintainer, 2026-10-18; task 4 writes, `benchmark.py`, the GUI static panel and the
  browser read).** One HDF5 file per benchmark (`make_benchmark.py -o dataset.h5`), replacing the per-set
  directories (no compatibility code):
  - `fem/`: the input FemElastoDynamics (pbat Archive). `benchmark.py --dataset` solves on it (no `--scene`);
    solver parameters still come only from `--sim-params`/`--set`.
  - `X`, `basis/`: stored once. Every batch group's `X` and `basis` are HDF5 hard links to them.
  - `pools/{pool,pool_gravity}/E{E}`, `static/E{E}`, `dynamic_h{h}/E{E}`: batch groups, each following the
    contract above as a file root does. Problem sets carry `pool` (the pool group) instead of `pool_file`.
  - Root attributes: `dataset_format = "pbat-benchmark-dataset"`, `dataset_version = 1`, `sets` (problem set
    groups), `pools`, `input`, `git_commit` and the generator's settings.
  - A batch is addressed as `dataset.h5:static/E50` (`static.load_sample_batch`); a `modalrom.py` sample file is
    still a batch at its root.
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
- **Superseded (2026-10-19) for the file layout** by "Benchmark datasets" above: one file, with pools and sets as
  groups and `pool` instead of `pool_file`. The per-sample datasets listed here still apply inside each dynamic
  batch group, plus `samples/ref_error_mnorm`. Old text: **Benchmark dataset** (task 4,
  `python/examples/make_benchmark.py`; draft, the reported data is still to be fixed). Under one output directory, for each total energy E there are `pool/E{E}.h5` (all pooled
  microcanonical samples), `static/E{E}.h5` (an evenly spaced subset of S problems, today's format with
  `problem = "static"`) and `dynamic_h{h}/E{E}.h5` (the **same** S samples, `problem = "dynamic"`). Problem
  files carry `pool_file` and `pool_indices`. Dynamic files add:
  - root attributes `h`, `bdf_order = 1` and `aext` (3,), the scene gravity;
  - **`samples/v`** (S, N, 3) float32, the full-space start velocities J(q) q̇;
  - **`samples/xtilde`** (S, N, 3) float32, the inertial targets x + h v + h² aext (aext applied to free DOFs
    only);
  - **`samples/x_ref`** (S, N, 3) float32, the solutions of min 1/(2h²)‖x − x̃‖²_M + Ψ(x) with Dirichlet DOFs at
    rest, from full-order CPU Newton converged to the float32 noise floor;
  - `samples/ref_newton_iterations`, `samples/ref_newton_decrement`, `samples/ref_newton_correction` (S,) (the
    last correction's largest entry in m, an accuracy estimate), and `samples/ref_min_J`.

  `samples/x` is the start state x_n. A small test set (E = 50 and 5000 J, 3 problems, pools of 40, so its rhat
  is not meaningful) is at `data/benchmarks/trampoline-test/` (now `trampoline-test.h5`).
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
| 1 | on hold (all merged) | 2026-10-18 | Merged: kinematics, scene ingestion, VertexSdf slot/barycentric/stream-order fixes, VF capacity, static and plane colliders (finite planes with thickness), CPU `-o` fix, Dump-to-scene converter + beam/TwistPull scenes. Worktree rebased on dev. Waiting for the maintainer |
| 2 | on hold | 2026-10-19 | feature/sd-detect-scope merged (15340ab4). Waiting for the next task |
| 3 | on hold | 2026-10-20 | All work merged; handoff written at the top of plan 03 |
| 4 | benchmark generator merged into dev (`4029072e`); full dataset on hold | 2026-10-18 | Test set in `data/benchmarks/trampoline-test` passes task 3's loader. Full generation waits for the coordinator's data, plot and measurement requirements |

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
- 2026-10-18 · coordinator · **Maintainer decision on GPU OGC:** not in use. The paper's GPU results use **VertexSdf** contact detection. The GPU OGC crash (CUDA error 700 once bodies come within contact range; repro above) is **low priority**: the coordinator may investigate it, together with other robustness gaps vs the CPU OGC (penetration-free via per-vertex step truncation), when nothing else is pending. No task should block on OGC; validate contact with VertexSdf.
- 2026-10-18 · task 3 · **GPU run END**: preliminary static benchmark (re-run on an idle GPU after the overlap). The GPU is free.
- 2026-10-18 · task 1 · **GPU run START** (~5 min, not timing): GPU unit tests, plus 5 solvers × 0.2 s on
  MediumLongBendingBeam (VertexSdf). Will log END.
- 2026-10-18 · task 1 · **GPU run END** (my runs are done; I'm not using the GPU now). **Task 2:** your run
  below started while mine were still going, so re-check its first minutes.
- 2026-10-18 · task 1 · **Ready for review:** `feature/gpu-kinematic-constraints`, 14 commits on dev `7c485e35`.
  GPU unit tests pass. All 5 solvers run 0.2 s on MediumLongBendingBeam (VertexSdf) with zero Dirichlet
  drift, and VBD/AAAVBD are bit-identical to current dev there. Validation details are in plan 01's Progress.
- 2026-10-18 · task 2 · **GPU run START** (~20–30 min): CUPTI backend validation on MediumLongBendingBeam, then the re-measured trampoline report (5 solvers under nsys plus graph-replay timings). Will log END.
- 2026-10-18 · task 3 · Rebased on dev c947de87. Conflicts were only next to task 2's `@profiling.scoped` decorators; both kept. Static benchmark machinery is complete (12 commits). The next GPU-heavy item is the full run (30 samples × 5 solvers × 60 s ≈ 2.5 h), which waits for the maintainer's params file and will be announced here.
- 2026-10-18 · task 2 · **GPU run END** (09:24). The CUPTI backend matches nsys on the beam: the same 11,946 activities per step, the sweep within 0.6%, the total within 3%. Re-measured trampoline (idle GPU): VBD 69.0, AAAVBD 75.4, Chebyshev 68.8, Anderson 70.4 ms/step of kernel work vs 102–112 ms/step graph replay. Milestone 2 (CUPTI backend + GUI panel) is on `feature/gpu-profiling`, 4 commits after `7c485e35`, ready for review.
- 2026-10-18 · coordinator · **GPU run START** (~10 min): review of task 1's kinematics branch (GPU unit tests + one short simulate.py run per solver on branch and dev). Will log END.
- 2026-10-18 · task 2 · **GPU run START** (short, intermittent over the next hour): milestone 3 exploration. An Nsight Compute permission check, then single-kernel ncu captures and nsys graph-mode traces on trampoline. Will log END.
- 2026-10-18 · task 3 · Maintainer: the full static benchmark is **on hold** until task 4 generates the benchmark dataset of problems. Tasks 3 and 4 coordinate on its contents.
- 2026-10-18 · coordinator · **GPU run END.** Merged **task 1** (c41bc1ad, merge commit): 17 tests pass, VBD/AAAVBD/Chebyshev bit-identical to dev, Anderson/Newton within nondeterminism, zero Dirichlet drift. Merged **task 2 milestone 2** (CUPTI backend + profiling panel; ui.py conflict resolved by keeping both panels): 14 tests pass, AAAVBD 56 ms/step on an idle GPU. Tasks 2 and 3: rebase onto dev. The maintainer still needs to check the GUI panels (kinematics, profiling).
- 2026-10-18 · coordinator · Maintainer refined the benchmark plan (new section "Benchmark plan" above): task 4 builds the static+dynamic dataset, task 3 runs static, the coordinator runs dynamic. **Hold all full benchmark runs** until the maintainer and coordinator fix the reported data and plot.
- 2026-10-18 · task 3 · **GPU run START** (~2 min, correctness only: unit tests + 2 small static solves after the rebase onto dev 91ad5a90). Task 2: ignore any timing taken in this window. Will log END.
- 2026-10-18 · task 3 · **GPU run END** (correctness checks after the rebase).
- 2026-10-18 · coordinator · **GPU run START** (~2 min): headless check of a kinematics test scene (TwistPullBeam), then the maintainer's interactive GUI test of it. Will log END.
- 2026-10-18 · task 2 · **GPU run END** (milestone 3 exploration). Two findings for everyone:
  - **Big, solver-independent overhead (FYI coordinator):** `ConstraintSet.update_constraint_set` (`gpu/contact/constraints.py`) loops over all 4 of its stream lists, so each contact type copies its 6 fields and fills its 4 fields 4× per call. In the captured graph, each of the resulting 96 memcpy + 64 fill nodes per subproblem boundary waits ~10 µs: ~30 ms of a ~107 ms trampoline step. A 2-line fix (`self._streams[:1]`, patch kept in my scratchpad) is bit-identical and makes AAAVBD 108.1 → 83.5 ms/step, VBD 98.7 → 74.9. **Not committed** (not my file); asking the maintainer who lands it.
  - **Possible contact bug on dev (not mine to fix):** CubeFallingOnCube.Dump + VertexSdf (default) with AAAVBD: the top cube falls straight through the fixed bottom cube (z ≈ −43 after 3 s), and 0 contacts are detected at every step. Not caused by the profiling decorators (same result without them) nor by `31ff9a75`'s filter change (it doesn't touch detection). Repro: `simulate.py --fem-elasto-dynamics data/CubeFallingOnCube.Dump.h5:fem --solver AAAVBD --duration 3`.
  Report: AAAVBD overheads https://claude.ai/artifact/NngLUsxKfkhZz4Fzmx9XuL
- 2026-10-18 · task 3 · **GPU run START** (~3 min, small machinery test: 1 trampoline-test static problem × VBD/AAAVBD/Newton, 5 s cap). Will log END.
- 2026-10-18 · task 3 · **GPU run END** (trampoline-test machinery check).
- 2026-10-18 · task 2 · **GPU run START** (~1 h, intermittent): verify the constraint-set fix (maintainer: commit it on `feature/gpu-profiling`), then ncu source hotspots of the AAAVBD vertex solve, and the anatomy of a `capture_while` graph. Will log END.
- 2026-10-18 · coordinator · Maintainer notes:
  - **Measurement methodology:** how solver execution is measured will be revisited with the maintainer before the benchmarks (no runs until then).
  - **Convergence check frequency:** VBD-family solves should not test convergence every iteration. Make the interval a parameter (e.g. every 20 iterations). → task 3.
  - **Mesh refit and capture:** the maintainer recalls capturing the whole step with VertexSdf. (It is captured today; the limitation is only inside *conditional* bodies, `capture_while`/`capture_if`, where allocations are forbidden.) → task 3 to re-check whether refit can avoid allocating.
  - **CubeFallingOnCube + VertexSdf passing through is expected:** the top cube's vertices coincide with the bottom cube's surface, so the penetration is always 0 and no contact force is generated. Not a bug.
  - New milestone, CPU scene files: see the section above.
- 2026-10-18 · coordinator · Landed task 2's finding on dev (a9abadc1): ConstraintSet copies/fills each field once (was 4×). Bit-identical; AAAVBD trampoline graph step ~106 → ~83 ms.
- 2026-10-18 · task 3 · Maintainer approved: convergence tested every N sweeps (default 20) with C++'s full-step (Lagrangian) criterion, and a loop restructure (unrolled outer, device-replayed sweep blocks, refit/detection outside conditional bodies) so contact scenes get early exit and a small graph. Refit evidence is in plan 03, question 17. Implementing on `feature/gpu-static-solves`.
- 2026-10-18 · coordinator · Maintainer: add a **feasibility residual norm check between outer iterations**. The outer loop continues with the next subproblem until both optimality (gradient norm) and feasibility pass, subject to n_max_iters and the check interval. → task 3 to propose the exact residual definition to the maintainer.
- 2026-10-18 · coordinator · **GPU run END** (kinematics test scene: headless check plus the maintainer's GUI session).
- 2026-10-18 · task 1 · Scene ingestion on `feature/gpu-scene-ingestion` (rebased on dev `3caa72c1`; 8 commits:
  `scenario.py` refactor, 2 CPU loader fixes, `read_scene`, reset v0, `state.load_scene`/`load_problem`,
  `--scene`, docs). **FYI tasks 3 and 4:** `SimulationState.reset()` now restarts from `(X, v0)` with the
  loaded initial velocities (stored as `state.v0`) instead of `(X, 0)`. That affects every reset, in both
  simulators. **No existing Dump changes:** all five store v = 0 (BeamFallingOnBeam, CubeFallingOnCube,
  MediumLongBendingBeam, TwistPullBeam, trampoline). Among scenes, only CubeFallingOnPlaneFast has a nonzero
  v0. A bit-identical GPU check of MediumLongBendingBeam and TwistPullBeam will follow once task 2's run ends.
  **Shared files:** `state.py` (append-only: `load_scene`, `load_problem`, `self.v0`, one line in `reset`) and the
  argument parsing plus problem loading in `ui.py`/`simulate.py`.
- 2026-10-18 · task 3 · **GPU run START** (~6 min, correctness only: bit-identity of the restructured VBD-family loop vs unrolled, beam, contact on/off). Task 2: ignore timings in this window. Will log END.
- 2026-10-18 · task 2 · **GPU run END.** New finding (FYI task 3, who owns the solver kernels): on trampoline, which has no contacts, **contact gathering dominates the vertex solve**. `local_contact_derivatives` plus the half-edge helpers take ~60% of AAAVBD's warp stall samples (VBD ~72%) and about half its instructions (ncu source view with line info, native builtins folded into their callers). Every trampoline vertex is a surface vertex, so each one walks its half-edges and faces and reads reverse-contact prefixes in every sweep. A contact-free experiment that skips the gather is bit-identical: AAAVBD 83.8 → 40.6 ms/step (−52%), VBD 75.2 → 46.3 (−38%). Candidate fix: compute a per-vertex contact count once per constraint-set update and early-out in the vertex solve. **Not implemented** (solver kernels belong to task 3); asking the maintainer. Tooling (`ncu.py --lineinfo` hotspots, report section) is 3 commits on `feature/gpu-profiling` after `7ad6a983`. **Coordinator:** the `capture_while` anatomy needs task 3's branch on dev; I'll run it after that merge.
- 2026-10-18 · coordinator · Merged task 2's ncu hotspot tooling (d05f0987). **Task 2 finding:** contact gathering in the vertex solve takes ~60% of AAAVBD's stalls (VBD ~72%) even with zero contacts; skipping it is bit-identical: AAAVBD 83.8 → 40.6, VBD 75.2 → 46.3 ms/step on trampoline. The owner of the fix is pending the maintainer's decision; it should land after task 3's kernel changes (gpu/vbd/kernels.py). Asked task 3 for a merge-ready slice.
- 2026-10-18 · coordinator · Maintainer wants reusable sim-params files for benchmarks and scenarios. The paper's configs live in `path/to/accelerated-vbd-extras` (parameters/gpu = GPU export format, all 19 files load in the current GPU loader with no unknown keys; parameters/cpu + vbd.sal.h5 = CPU format). Design discussion pending.
- 2026-10-18 · task 1 · **GPU run (correctness only, ~8 min, just finished; logged afterwards because my START
  edit didn't get written).** It overlapped task 3's correctness window only; no timings were affected.
  **START/END.** Results for `feature/gpu-scene-ingestion`:
  - MediumLongBendingBeam and TwistPullBeam, branch vs dev (VBD, 0.3 s): bit-identical, so the reset change
    leaves zero-velocity Dumps unchanged.
  - Scene vs Dump: `--scene CubeFallingOnCube.h5` vs its Dump (0.5 s), and the compressing prism scene vs a Dump
    of it (0.3 s): bit-identical.
  - CubeFallingOnPlaneFast: loads with the "ignoring 1 static mesh collider" warning and starts at its v0 = 10
    (1.054 m in 0.1 s).
  - **Ready for review:** 8 commits on dev `3caa72c1`.
- 2026-10-18 · task 3 · **GPU run END** (restructured-loop bit-identity checks).
- 2026-10-18 · task 2 · **GPU run START** (~30 min, light): validating the contact-gather early-out prototype (bit-identity and timing on trampoline, beam and a contact scene). Will log END.
- 2026-10-18 · task 1 · Maintainer requests (this session):
  - **CPU `simulate.py -o` drive-letter bug:** fixed on `feature/cpu-output-path` (1 commit, `402c0225`). It
    splits on the last `:`. Ready for review.
  - **GPU static colliders, "if it can be done simply".** I think it can, so I'm implementing it on
    `feature/gpu-static-colliders`. Design: collider vertices become element-less FEM nodes appended after the
    bodies (`--scene` only), fixed forever by a "Static colliders" kinematic constraint. They get the
    bodies' mean nodal mass, which only scales the contact penalty's Rayleigh quotients. The contact
    MultiMesh is the tet boundary plus the collider triangles, one component per body and per collider.
    `SimulationState(..., contact_mesh=None)` takes it, append-only. `sd.py`: open meshes (e.g. ground
    planes) sign distances by face normal (`mesh_query_point_sign_normal`) instead of ray parity. No solver
    or contact-constraint kernel changes. **Touches `gpu/contact/mesh/sd.py`** (a per-mesh closed flag and
    one branch in the detection kernel). Closed meshes are unchanged, so Dumps are expected bit-identical
    (to be verified). GPU checks wait for task 2's END.
- 2026-10-18 · task 3 · **GPU run START** (~5 min, correctness only: feasibility test on CubeFallingOnCube, AAAVBD/VBD). Will log END.

- 2026-10-18 · coordinator · Maintainer: (a) dependencies go in the nearest `requirements.txt` (added to Rules). (b) Feasibility check between outer iterations: **normal contact only, ignore friction**; user-prescribed tolerance under `Solver/Convergence` (relayed to task 3). (c) Next, once the task branches are merged: coordinator proposes manual test scenes covering the recent features, then pushes `dev` to the remote, then the benchmark discussion.
- 2026-10-18 · task 3 · **GPU run END** (feasibility-test validation on CubeFallingOnCube/BeamFallingOnBeam; dev-vs-branch contact scans).
- 2026-10-18 · coordinator · **Merged `feature/gpu-static-solves` @43babcfb** into dev (`3aae8ba5`). The 1e-6 to 1e-5 dynamic differences vs dev at gtol=0 come from codegen (the `is_static` branch changes FMA contraction) and Chebyshev's omega recurrence now running in float32 on the device; verified by task 3. Bug ownership: **vertex-SDF sentinel VF slots** (CubeFallingOnCube, also on dev) → task 1, high priority (paper detector). **OGC CUDA 700 on CubeFallingOnCube/AAAVBD** → parked (OGC low priority). Task 2: contact-gather early-out on dev now. Task 1: rebase `feature/gpu-static-colliders` on dev.
- 2026-10-18 · coordinator · **GPU run START** (~3 min, correctness only: Anderson sweep-parity counts on the beam). Will log END.
- 2026-10-18 · coordinator · **GPU run END** (Anderson sweep parity: every p runs exactly n_max_iters × n_subproblem_max_iters sweeps, eager and captured).
- 2026-10-18 · coordinator · Merged `feature/cpu-output-path` (`28d3b2f7`) and `feature/rom-mcmc` (`4029072e`). On dev: **Anderson sweep parity** (`5f24e614`, `c3cca7e5`: `n_subproblem_max_iters` counts sweeps for every VBD-family solver). **TOML sim-params configs** (`ea8641ea`..`1d44eb82`): `config.py`, `configs/README.md`, `configs/defaults.toml`, and `configs/paper/*.toml` (the 19 paper GPU files, bit-identical on load, with Chebyshev/Anderson tables copied from VBD where absent). `--sim-params a.toml b.h5 --set Table.key=value` in simulate/ui/benchmark_static. New dependency `tomli-w==1.2.0` in `python/tools/sim/requirements.txt` (installed in the venv). Waiting on: task 1 (vertex-SDF slots, then colliders), task 2 (contact-gather early-out, tooling rebase).
- 2026-10-18 · task 1 · **GPU run START** (~10 min, correctness only, no timings): vertex-SDF slot check on
  CubeFallingOnCube and BeamFallingOnBeam (dev vs fix). Task 2: ignore timings in this window. Will log END.
- 2026-10-18 · task 2 · **GPU run END.** The contact-gather early-out is ready for review on `feature/gpu-profiling` (`f9f689a3` flag, `2e4498e9` gate; rebased on the latest dev). Changes: `gpu/contact/dynamics.py` (`MeshDynamicsData.in_contact`, flagged after each constraint-set update from the same prefix ranges the gather reads, with no payload reads, so task 1's sentinel fix stays independent) and one guarded condition in each of `gpu/vbd/solver.py` and `gpu/vbd/aaasolver.py`. **Validation:** VBD/AAAVBD/Chebyshev are bit-identical on trampoline and MediumLongBendingBeam; Anderson differs by 2.0e-4 vs its own run-to-run spread of 2.1–2.6e-4. On BeamFallingOnBeam (contacts), flags equal the gathered sets at every step (0 mismatches over 120 steps). **Production (dev, capture_while):** trampoline AAAVBD 58.7 → 28.4 ms, VBD 75.1 → 30.5, Chebyshev 47.8 → 22.2, Anderson 9.1 → 5.0; BeamFallingOnBeam AAAVBD 105.3 → 100.4, VBD 95.1 → 91.5. Report v3: https://claude.ai/artifact/NngLUsxKfkhZz4Fzmx9XuL
- 2026-10-18 · task 1 · **GPU run END** (vertex-SDF checks). **Vertex-SDF slot bug fixed** on `feature/vertex-sdf-slots`
  (3 commits on dev `3aae8ba5`). It's ready for review and independent of the collider work.
  - **Root cause, not the in-kernel tile sort:** `ContactPairs` sized pair types as a fraction of
    vertices/edges, so tiny meshes got **zero capacity** (CubeFallingOnCube: VV = VE = 0, VF = 3).
    `cuda.compute` caches its radix sort by array types, and **a Sort built from empty arrays makes later
    sorts of the same types keys-only.** The VF sort then left its values at the sentinel nF. Reproduced in
    isolation (a 3-item key/value sort is correct, but drops its values if a zero-size Sort was built first).
    Fix `fa980990`: at least 1 slot per type. CubeFallingOnCube now has 0 sentinel steps (was 6) and 79 valid
    VF contacts over 60 steps (was 0). Old and new detection kernels give identical pairs on identical
    positions, so the tile_sort pattern task 3 suspected is fine.
  - **Second bug, `5e74a386`:** the detector wrote past the pair capacity on overflow. BeamFallingOnBeam
    overflows the default VF capacity (923) at step 12. Excess contacts are now dropped instead of
    corrupting memory. **Maintainer decision needed:** the defaults are too small for that scene, and tiny
    scenes get only 1 VV/VE slot. Should the defaults (or a floor) go up?
  - **Third, latent, `cf215b38`:** the mesh loop variable was overwritten in the body, which skips meshes
    in scenes with more than 256 meshes.
  - **FYI everyone:** GPU contact scenes are **not bit-reproducible run to run**. BeamFallingOnBeam differs
    by 3.5e-3 from frame 10 between two runs of the same code (atomic contact order). Compare contact scenes
    by contact sets or tolerances.
  - CubeFallingOnCube still falls through, as the maintainer expects (coincident vertices, sd = 0), on dev
    too. Task 3's sentinel guard in `convergence.py` can stay as a defensive check.
- 2026-10-18 · task 2 · **GPU run START** (~1 h, intermittent): checking Anderson's sweep counts on trampoline, then the AAAVBD kernel changes the maintainer approved (state-load hoisting, neighbour pre-filter) with ncu and timing runs. Will log END.
- 2026-10-18 · coordinator · **Merged `feature/gpu-profiling`** (contact-gather early-out, `1ba967d9`; beam AAAVBD/VBD bit-identical vs dev, re-checked by the coordinator) and **`feature/vertex-sdf-slots`** (`44d473c0`: min 1 slot per pair type, capacity guard, loop-variable fix; CubeFallingOnCube AAAVBD 0.5 s finite with contacts). Open for the maintainer: raise default contact capacities (BeamFallingOnBeam overflows VF 923)? Inclusive scope times (task 2) pending a decision. Task 1: colliders rebase next; possible overflow race in `assemble_contacts` (reverse map vs forward prefix stream) flagged to task 1.
- 2026-10-18 · task 1 · **GPU run START** (short, correctness only; task 2, ignore timings in this window): checks
  for the `assemble_contacts` stream-order fix, then the static-collider correctness runs. Will log END.
- 2026-10-18 · coordinator · Maintainer decisions: **raise default contact capacities** (task 1 measures peaks, sets defaults with headroom; paper configs untouched; regenerate `configs/defaults.toml`). **Inclusive scope times** approved (task 2: derived inclusive busy/launches in report + GUI tree view, new inclusive union field). **Manual test scenes and push wait for task 1's static colliders.**
- 2026-10-18 · coordinator · **GPU run START** (~10 min, correctness only: fused constraint-set update vs dev on beam, trampoline, BeamFallingOnBeam). Timings in this window are unaffected only if you ignore them. Will log END.
- 2026-10-18 · coordinator · **GPU run END** (fused constraint-set update: bit-identical on beam and trampoline, BeamFallingOnBeam within dev's run-to-run noise; plus a 4-solver sweep count on trampoline with `--profile kernels`).
- 2026-10-18 · coordinator · Landed the **fused constraint-set update** (`ae18a043`, bit-identical); task 2 to measure before/after. **Solver-comparison caveat:** trampoline per-solver ms/step are time to pass |g| <= 1e-3 (absolute) at 20x25 sweeps max. Sweeps/step for steps 5-14: VBD 195-495, AAAVBD 146-271, Chebyshev 97-225, Anderson 25-50. They are not equal-accuracy comparisons; reports must show sweeps/step, ms/sweep and the exit reason. `--profile kernels` segfaults at exit (task 2). **New maintainer request: document all tools in-source plus agent skills** (profiling, benchmarks, ...). Doc passes: task 3 (static), task 4 (sampler) now; task 2 (profiling) and task 1 (kinematics) after their current work. The coordinator writes the skills from those docs.
- 2026-10-18 · task 3 · **GPU run START** (~1 min, correctness only: the documented benchmark_static commands on trampoline-test). Ignore timings. Will log END.
- 2026-10-18 · coordinator · **GPU RESERVED for task 2, 11:49–12:34 (exclusive timing window).** Everyone else: no GPU work at all (no correctness runs, no GUI), until task 2 logs END. CPU work (docs, CPU tests, reviews) is fine. Merged the CUPTI exit fix (`f4804414`).
- 2026-10-18 · task 3 · **GPU run END** (documented-command check).
- 2026-10-18 · task 2 · **GPU run START** (11:52, exclusive window until 12:34): fused constraint-set before/after, AAAVBD vs VBD ms/sweep, and the AAAVBD kernel changes.
- 2026-10-18 · task 1 · **Sorry, task 2:** my static-collider correctness runs (under my earlier START entry)
  ran until **11:53:34**, so they overlapped the start of your exclusive window (11:52–11:53:34). Please
  discard timings from those 1.5 minutes. **GPU run END** for task 1. I'm off the GPU until you log END.
- 2026-10-18 · coordinator · Merged `feature/pairs-stream-order` (task 1: reverse-map stream order, VF capacity 0.5) and `feature/sampler-docs` (task 4, docs-only, verified by AST). Skills added: `sim-run`, `sim-sampling`, `sim-profiling` (`.claude/skills/`). Pending: `sim-benchmarks` (needs task 3's static summary and the benchmark methodology), doc passes by task 3 (in progress), task 2 (after inclusive times) and task 1 (after colliders).
- 2026-10-18 · task 2 · **GPU run END** (12:05, exclusive window released early). Results reported to the coordinator.
- 2026-10-18 · coordinator · Task 2's clean-window results: fused constraint-set update −2…−3% on trampoline (AAAVBD 28.1→27.3, VBD 29.9→29.0 ms), BoB within noise; constraint-set boundary gaps 0.70→0.51 ms/step. AAAVBD vs VBD per sweep 72 vs 59 µs (1.22×; the earlier 1.5× came from contaminated windows). **Merged AAAVBD load hoisting** (`d61ee145`, bit-identical, re-checked): trampoline AAAVBD 27.3→26.0 ms, BoB 98.1→93.7 ms. Also merged docfield-strings, static-params-help (task 3) and benchmark-input (task 4); skill `sim-benchmarks` added.
- 2026-10-18 · task 1 · **GPU run START** (~15 min, correctness only, no timings): static/plane collider checks
  (CubeFallingOnPlane family, inclined plane, beam on static beam) and branch-vs-dev bit-identity. Ignore
  timings in this window. Will log END. Maintainer request: **plane colliders** (new scene object, authored in
  the CPU editor; finite or infinite, infinite by default). GPU: pinned quads plus analytic plane detection in
  `sd.py`.
- 2026-10-18 · task 2 · **GPU run START**: GUI (`ui.py`, trampoline) for the maintainer's manual test of the profiling panel. Timings from other sessions are unreliable while it runs. Will log END when the GUI closes.
- 2026-10-18 · coordinator · **Merged `feature/vertex-sdf-bary`** (task 1): VertexSdf reconstructed closest points with the wrong barycentric order (Warp: u·p0 + v·p1 + (1−u−v)·p2). Signed-distance magnitudes and the VF/VE/VV choice were wrong; the sign was right. **All VertexSdf contact trajectories change: regenerate any contact-scene numbers.** Verified against warp/native/mesh.h mesh_eval_position. Also merged task 2's profiling docs and inclusive times.
- 2026-10-18 · task 2 · **GPU run END**: the maintainer's GUI session (profiling panel test) is closed. GPU free.
- 2026-10-18 · task 4 · **CPU run START** (~5–10 min, no GPU): sample benchmark dataset (trampoline, 3 energies × 4 problems, pools of 400) into `data/benchmarks/trampoline-sample` for the maintainer's dataset browser. Will log END.
- 2026-10-18 · task 4 · **CPU run END** (12:36): sample dataset in `data/benchmarks/trampoline-sample` (3 energies × 4 problems, h = 1e-4 and 1e-2). Pools of 400 give split R-hat 1.05 / 1.26 / 1.14 at 50 / 500 / 5000 J, enough to explore but not for the real dataset. The 5000 J level includes an inverted start (min det F −0.43). Now running the maintainer's dataset browser GUI (CPU only).
- 2026-10-18 · task 1 · **GPU run END** (collider/plane correctness runs, started under my START above).
  **Pre-existing bug on dev, found while testing plane colliders: adaptive contact penalties diverge.**
  `--set Contact.ContactDynamics.penalty_adaptivity=1` (TIMESTEP) or `=2` (SUBPROBLEM) gives NaN positions on
  BeamFallingOnBeam from frame 10 (first contacts) with VBD, AAAVBD and Chebyshev, for any `gamman`
  (1, 10, 1e4). AndersonSolver stays finite. The penalty itself is moderate (σₙ ≈ 16 at frame 10), Hi/Qc
  are finite, and x goes NaN at frame 11, so the divergence is in the VBD contact solve, not in the
  Rayleigh-quotient pass. Also, the default `gamman = 1e4` is tuned for CONSTANT (it multiplies σ₀ = dt²).
  The adaptive modes need an O(1) multiplier. **Maintainer (in my session): wants a per-timestep or
  per-subproblem adaptive penalty instead of the constant one.** With TIMESTEP or SUBPROBLEM and gamman = 1,
  a 24 kg-node cube dropped on a plane penetrates at most 1.7 or 0.8 cm, against 19.7 cm with the constant
  penalty. Owner to be decided (solver kernels are task 3's).
- 2026-10-18 · coordinator · **Adaptive contact penalty NaN** (BeamFallingOnBeam, penalty_adaptivity 1/2, VBD/AAAVBD/Chebyshev) assigned to **task 3** (`feature/adaptive-penalty`): root cause plus fix; constant mode must stay bit-identical; propose adaptive-mode gamman defaults and ask the maintainer before changing any default. Merged task 4's benchmark browser and gravity options; h5py added to python/examples/requirements.txt (3.16.0, the venv's version).
- 2026-10-18 · task 3 · **GPU run START** (intermittent, correctness only: adaptive contact penalty NaN investigation on BeamFallingOnBeam/CubeFallingOnCube). Ignore timings. Will log END.
- 2026-10-18 · task 1 · **GPU run START** (~10 min, correctness only): plane-collider scenes (cube on an
  infinite ground plane, fast cube, inclined plane). Ignore timings. Will log END.
- 2026-10-18 · coordinator · Maintainer on solver configs: **params discussion deferred** (the maintainer will say when). Vertex linear solve: **Inverse by default; EVD only when singular Hessians cause problems.** The paper's trampoline configs are **different scenes** from `data/scenarios/trampoline.Dump.h5`. Still open: whether a tuned-parameter layer should also set sweep budgets. h5py pin bumped to 3.16.0 in python/tools/sim/requirements.txt; other pins match the venv.
- 2026-10-18 · coordinator · **Dynamic benchmark runner → task 3** (maintainer request: stop on the real error, not the solvers' checks). One runner: benchmark_static.py → benchmark.py, kind read from the problem file's `problem` attribute; dynamic = one BDF1 step, error vs samples/x_ref (mass-weighted, free DOFs), reference accuracy recorded. Comes after the adaptive-penalty fix (root cause: lagged contact positions xb make the contact coupling Jacobi-like, which diverges once the penalty reaches the vertex stiffness). Merged task 4's sampling-methods reference (python/examples/docs/sampling-methods.md).
- 2026-10-18 · task 1 · **GPU run END** (plane-collider correctness runs). **Ready for review:**
  `feature/plane-colliders`, 12 commits on dev. It includes the static-collider and kinematics-docs work, so
  `feature/gpu-static-colliders` and `feature/kinematics-docs` are its earlier tips.
  - **Plane colliders (maintainer request):** a new scene object (`tools/vbd/ui/plane_collider.py`) with a
    point, a normal, a tangent and half-extents, infinite by default (20 scene diameters). Stored in the
    scene's `plane_colliders` group, authored in the CPU editor (Objects > Add Plane Collider). CPU: tessellated
    into scene-sized cells for OGC (a single huge quad let fast impacts through). GPU: one pinned quad, and
    `sd.py` classifies planar open meshes as `MESH_PLANE` and tests them analytically (unbounded range,
    in-extent test).
  - **Static mesh colliders on the GPU:** pinned element-less nodes plus a "Static colliders" kinematic
    constraint (`colliders.py`, `SimulationState(..., contact_mesh)`).
  - **Results, 2 s, three plane scenes with an infinite plane:** CPU penetration-free. GPU with the adaptive
    per-step penalty (gamman 1): ≤ 7.8 mm / 5.4 cm (10 m/s) / 2.6 mm. With the constant default, heavy nodes
    sink 20 cm, which is task 3's adaptive-penalty work. Scenes without colliders are bit-identical to dev.
  - **Not yet measured:** runtime overhead. It's only a per-mesh kind read for scenes without colliders.
    I'd like a ~5 min quiet GPU window for a dev-vs-branch per-step timing on BeamFallingOnBeam; please
    say when.
- 2026-10-18 · coordinator · **Merged `feature/plane-colliders`** (`b31e53f7`; beam bit-identical, kinematics tests pass). **GPU RESERVED for task 1, 13:16–13:26** (quiet dev-vs-branch timing). Task 3: pause GPU runs in that window.
- 2026-10-18 · task 1 · **GPU run START** (13:16, timing in my reserved window): before (`cda79761`) vs after
  (`b31e53f7`) the collider merge, graph-replay step times. Will log END.
- 2026-10-18 · task 1 · **GPU run END** (13:22). Graph-replay step times (`--profile-out`, 30 warm-up steps,
  median of 2 × 50 steps, AAAVBD, VertexSdf):
  - **No colliders: no overhead.** BeamFallingOnBeam 95.8 ms before vs 95.6 ms after (run medians 97.0/95.4
    vs 95.8/95.5).
  - **With colliders, the cost is the contact solve itself, not the collider representation.** Cube on the
    infinite plane: 9.0 ms vs 1.6 ms for the same cube with no collider (no contact at all). Bending beam
    on the static beam (2177 pinned nodes): 44.3 vs 6.7 ms. The `kernels` breakdown of the plane scene is
    dominated by the AAAVBD sweep with contacts (gathering), with detection at about 4% (eager-inflated
    numbers), which matches task 2's finding that contact gathering dominates sweeps.
- 2026-10-18 · coordinator · **Maintainer decisions on the parameter walkthrough issues:**
  - LLT vertex solve and init_strategy 3/4 now raise errors on the GPU (`31961900`, `985429cb`). Newton line-search backtrack fixed (`6f71bab4`, unchanged at tau = 0.5).
  - CPU vs GPU VBD differing is intended. **Stencil keys belong to AAAVBD only on the GPU** (GUI, serialization, configs) → task 2 (`feature/aaavbd-params`); the coordinator regenerates the configs afterwards.
  - **AAAVBD warm_start_beta honoured on the GPU** (0 reset every subproblem, 1 reset every time step, 2 never; GPU default 2) → task 2.
  - EVD zeroing is OK. gamman meaning change: document it → task 3 (adaptive-penalty branch).
  - **gtol should be relative to the problem** (design pending with the maintainer).
  - **OPEN, tracked, not now: OGC has no capacity guard on pair writes (overflow writes out of bounds).**
- 2026-10-18 · task 1 · **GPU run START** (~5 min, correctness only): finite-plane thickness checks (maintainer
  request: finite planes become plates with a thickness; infinite planes unchanged). Ignore timings. Will log END.
- 2026-10-18 · task 1 · **GPU run END.** **Ready for review:** `feature/finite-plane-thickness`, 2 commits on dev.
  Finite plane colliders get a `thickness` (default: 10% of the smaller half-extent, editable, stored in the
  scene). On the GPU they're plates, while infinite planes stay half-spaces (bit-identical to dev).
  `SimulationState(..., contact_mesh)` is renamed `contact_geometry` (a `colliders.ContactGeometry`).
  New test scene: `data/scenarios/planes/CubePartlyOffFiniteInclinedPlane.h5`. The cube starts partly off a
  finite incline; it used to be launched, and now it slides and falls off.
- 2026-10-18 · task 2 · **GPU run START** (~15 min, correctness only): `feature/aaavbd-params` bit-identity checks (warm_start_beta = 2 vs dev, then 0 and 1) on the beam and the trampoline, AAAVBD. Ignore timings. Will log END.
- 2026-10-18 · task 2 · **GPU run END** (bit-identity checks). **Ready for review:** `feature/aaavbd-params` (4 commits on dev `ef2558a1`, head `f46176c6`). warm_start_beta = 2 (default and explicit) is bit-identical to dev on MediumLongBendingBeam (61 frames) and trampoline (31 frames), AAAVBD. Per step on the beam: 1 = 2 after step 1, differs from step 2; 0 differs within step 1.
- 2026-10-18 · coordinator · Merged `feature/finite-plane-thickness` (task 1, `c06e7c08`) and `feature/aaavbd-params` (task 2, `9598e393`: stencil keys AAAVBD-only; warm_start_beta honoured, GPU default 2). Added the **relative gradient test** `Solver.Convergence.rtol` (`8f9d5329`, off by default; mass-normalized residual vs the free-flight displacement x̃ − xᵗ, the maintainer's design). Regenerated defaults.toml and paper configs (`c4f4249c`, still bit-identical on load). Manual test page v2 published. Default rtol value pending the maintainer.
- 2026-10-18 · task 3 · **GPU run END** (adaptive-penalty investigation and validation).
- 2026-10-18 · task 3 · **GPU run START** (~10 min: GPU Newton SPD correction in dynamics, dev vs branch on beam/trampoline, incl. Newton step times). Will log END.
- 2026-10-18 · task 3 · **GPU run END** (Newton SPD validation).
- 2026-10-18 · coordinator · Merged `feature/adaptive-penalty` (task 3: other-color contact partners read at current positions in adaptive modes; SUBPROBLEM floored; friction adapted; new gamman_adaptive/gammaf_adaptive = 1; constant mode bit-identical; paper configs regenerated `f327a6c2`) and `feature/newton-spd` (GPU Newton SPD-corrects in dynamics too, default Absolute; old dynamic Newton results change by ~1e-6 to 1e-4). Maintainer: the relative gtol may be stricter on fine meshes (mesh-scale and contact residuals); that's accepted. Task 3 → dynamic benchmark runner next.
- 2026-10-18 · task 3 · **GPU run START** (~5 min, smoke tests of the dynamic benchmark runner on data/benchmarks/trampoline-sample). Will log END.
- 2026-10-18 · task 3 · **GPU run END** (dynamic benchmark smoke tests).
- 2026-10-18 · task 3 · **GPU run START** (intermittent, correctness only: adaptive-penalty stability with the new defaults, BeamFallingOnBeam). Will log END.
- 2026-10-18 · coordinator · **Maintainer's GPU defaults** (`state.load_params`, used by simulate/ui/benchmark): VBD family n_max_iters 1 × 100 sweeps; AAAVBD gammadown 0.99, gammadownS 0.95, gammaup(S) 0.5; rtol 1e-2 with displacement_floor 1e-6 m (no division by zero); ftol 0.5; check_interval 20; VertexSdf max_dist 0.05; Newton 1 outer × 10 steps; Chebyshev/Anderson unchanged; single-body scenes skip detection automatically. Tuning guidance in the sim-run skill. **TIMESTEP penalty (chosen default) diverges on BeamFallingOnBeam with these defaults → task 3 investigating (Jacobi xb, gamman_adaptive 0.1); default stays CONSTANT until stable.** Task 3 then: relative gradient check for Newton (same criterion and floor). Benchmarks verified to stop on the error only (VBD loop bypassed, Newton gtol2 forced 0). Merged task 3's dynamic runner (benchmark.py).
- 2026-10-18 · coordinator · **Maintainer: prefer scene files + sim-params files over Dumps** (Dumps only for mid-run resume or legacy). Identical scene files exist for trampoline (`data/scenarios/trampoline.h5`), BeamFallingOnBeam (`data/BeamFallingOnBeam.h5`) and CubeFallingOnCube (`data/CubeFallingOnCube.h5`). Assigned: task 3 `benchmark.py --scene`; task 4 scene support in modalrom/make_benchmark/benchmark_browser; task 1 scene files for MediumLongBendingBeam and TwistPullBeam. Coordinator: switch skills/docs/test page to scenes.
- 2026-10-18 · coordinator · Maintainer: benchmark Newton solves stop at the first rejected step (non-descent or line-search failure) and report a failure status; results record a termination status per (solver, sample) → task 3, with the Newton relative check. Verified: with gtol2 = 0 the benchmark does at most one Newton step between error checks. Dumps are mainly for debugging mid-simulation crashes; scene files build the FemElastoDynamics automatically wherever it makes sense.
- 2026-10-18 · task 1 · **GPU run START** (~5 min, correctness only): `--scene` vs Dump runs for the new
  MediumLongBendingBeam and TwistPullBeam scenes. Ignore timings. Will log END.
- 2026-10-18 · task 1 · **GPU run END.** **Done:** `data/scenarios/MediumLongBendingBeam.h5` and
  `data/scenarios/TwistPullBeam.h5`, written by the new converter on `feature/dump-to-scene` (1 commit, ready for
  review): `python -m python.tools.vbd.dump_to_scene dump.h5 scene.h5`. Recovered inputs: E 1e6, nu 0.45, rho 1000,
  g −9.81. Constraints: Fixed (beam); twist clamp L_ROTATE [0, inf) about the clamp centroid, so it's editor-safe,
  and lift tip TRANSLATE [0.5, 1.5) (TwistPull). Rebuilt FEM equals the Dumps' exactly. 2 s GPU runs, `--scene`
  vs Dump: **bit-identical, all nodes, both scenes**. The CPU simulator loads both.
- 2026-10-18 · coordinator · **h = 1e-4 dynamic benchmark must work** (maintainer: it shows mass regularization's benefit for the VBD family; Newton must run there). Every solver plateaus vs x_ref (VBD 12-19%, Newton 5-8%; reference bound 8.4%). Suspects: inaccurate x_ref (float32 CPU energy), float32 cancellation in Newton's Armijo test at h² = 1e-8, objective mismatch. Task 4 checks the reference (float64 recompute, where rom_reach stops, objective consistency); task 3 checks the GPU side (gradient at x_ref, VBD past its plateau, line-search cancellation).
- 2026-10-18 · coordinator · h = 1e-4: **x_ref is accurate** (task 4: float64 re-solve within 0.00-0.04% of e0). The 8.4% bound was a loose max-entry·sqrt(3Σm) worst case. The plateau is GPU-side → task 3's decisive test: the GPU gradient at x_ref. Task 4: float64 references + `ref_error_mnorm` (accuracy in the error metric), sample set only. Merged task 4's sampler scene support and task 1's dump_to_scene converter (MediumLongBendingBeam.h5, TwistPullBeam.h5).
- 2026-10-18 · task 4 · **CPU run START** (~10 min, no GPU): regenerating `data/benchmarks/trampoline-sample` with exact (float64) references into `trampoline-sample.new`, then swapping it in. Same seed, layout and samples. Will log END.
- 2026-10-18 · task 4 · **CPU run END**: `data/benchmarks/trampoline-sample` regenerated with exact references (branch `feature/exact-references`). Samples, x, v and xtilde are bit-identical to before; x_ref is float64 and accurate to <= 5.4e-9 e0 (new `samples/ref_error_mnorm`), against <= 3.6e-4 e0 before. The old set is in `trampoline-sample.old-float32-refs` (can be deleted).
- 2026-10-18 · coordinator · **h = 1e-4 plateau root cause** (task 3): the absolute vertex-solve singularity check (|det H| <= 1.19e-7) skips light vertices when H ≈ m I. Maintainer's design: relative (Inverse |det H| <= hess_zero·(tr H/3)³, EVD |λ| <= hess_zero·max|λ|) → task 2 (`feature/relative-hess-zero`). Newton's 5-8% plateau is separate (task 3). Merged task 3's adaptive-penalty-stability (gamma_adaptive default 0.1); penalty default stays CONSTANT until TIMESTEP at 0.1 is confirmed stable. trampoline-sample regenerated with exact float64 references (ref_error_mnorm/e0 <= 5.4e-9); old set kept as trampoline-sample.old-float32-refs. Manual test page v4: research scenes (prisms, spaghetti, paper beams/contact scenes).
- 2026-10-18 · task 2 · **GPU run START** (~20 min, correctness/convergence only): `feature/relative-hess-zero` vs dev, beam and trampoline (VBD, AAAVBD) plus dynamic h = 1e-4 and static benchmark smoke runs (2 samples). Ignore timings. Will log END.
- 2026-10-18 · task 1 · **GPU run START** (~15 min, correctness only): plane scenes under the new defaults,
  with the constant vs TIMESTEP/SUBPROBLEM adaptive penalty, to correct the manual-test flags. Ignore timings.
  Will log END.
- 2026-10-18 · task 3 · **GPU run END** (adaptive-penalty stability, and the h = 1e-4 plateau diagnosis). Before reading the assignment to task 2, I had also implemented the relative check on `feature/vertex-solve-scale` (`bc7220d6`, Frobenius scale `|H|_F/sqrt(3)` instead of `tr H/3`). **It duplicates `feature/relative-hess-zero`, and I deleted it. Task 2's branch is the one to take.** Its validation corroborates task 2's: beam and trampoline (VBD, AAAVBD, default params) are bit-identical to dev; trampoline-sample dynamic_h0.0001 (E50, E5000, 2 samples each): VBD, AAAVBD and Anderson reach 95% in ~0.1-2 ms and end at e/e0 ~1e-5 to 1e-4 (before: 0.12-0.19); Chebyshev ends at ~7e-3; h = 1e-2 and static are unchanged. Task 3 → Newton next: the h = 1e-4 line-search plateau, the relative gradient check, and termination statuses.
- 2026-10-18 · task 4 · **CPU run START** (~10 min, CPU only): regenerating `data/benchmarks/trampoline-sample` from the `trampoline.h5` scene with float32 references again (maintainer: no error reduction beyond 99% is reported, so the float64 references are dropped). Will log END.
- 2026-10-18 · task 4 · **CPU run END.** `data/benchmarks/trampoline-sample` regenerated from `trampoline.h5` with float32 references (branch `feature/float32-references`, ready to merge; reverts the float64 references of `cbd3aca2` at the maintainer's request). Inputs are bit-identical to the previous set; references are within 6.5e-4 e0 of exact ones, enough for the 99% reductions the benchmark reports. Previous sets kept as `trampoline-sample.old-float32-refs` and `.old-float64-refs` for the maintainer to delete.
- 2026-10-18 · coordinator · Maintainer answers: housekeeping approved (old sample sets deleted, merged branches removed); **AAAVBD must honour surface_stencil_surface_neighbours_only** → task 2 (after relative hess_zero; task 3's duplicate vertex-solve-scale to be dropped); task 4's judgment calls approved; configs move to the extras repo later; **benchmark methodology still pending, the final step; the maintainer does a manual test pass right before the benchmarks**; future benchmarks will include stiffer materials (e.g. Y = 1e7, `data/scenarios/trampoline.Y1e7.h5`). Merged task 4's float32 references (maintainer reports at most 99% reductions; references within 6.5e-4 e0).
- 2026-10-18 · task 3 · **GPU run START** (intermittent, correctness only: GPU Newton diagnostics on trampoline-sample dynamic_h0.0001, small single solves). Ignore timings. Will log END.
- 2026-10-18 · task 1 · **GPU run END** (plane scenes, new defaults). Plane scenes need `penalty_adaptivity=1` plus
  `gamman_adaptive=1`. At 0.1, the fast cube sinks 0.28 and bounces. At 1, it sinks 0.054, rests and stays finite on all 5
  scenes. The slide-off scene also needs `gammaf_adaptive=0.001`, because the cube creeps instead of sliding.
  **Finding (contact, not task 1's code):** the AL friction term `sigma_f*c_f - lambda_f` in the primal has no cone
  limit, so it acts as viscous drag even at `mu_f=0`. The frictionless cube creeps at ~0.1 m/s and slides only with
  `gammaf=0`. The C++ primal has the same form (`vbd/Core.h:509`), and mu enters only the dual projection. Corrected
  commands were sent to the coordinator.
- 2026-10-18 · task 2 · **GPU run END** (relative hess_zero validation, new defaults, scene files). **Ready for review:** `feature/relative-hess-zero` (2 commits on dev `30fd3934`, head `e46eb244`). Beam and trampoline, VBD and AAAVBD at dt 0.01: bit-identical to dev. Dynamic h = 1e-4 (E50, 2 samples): dev stalls at e/e0 0.125 / 0.191; branch reaches the 95% target in 16 (VBD) / 17 (AAAVBD) sweeps. Static smoke: same iterations to every level as dev.
- 2026-10-18 · coordinator · Merged `feature/relative-hess-zero` (task 2): vertex singularity check relative to H's scale (Inverse |det H| <= hess_zero·|tr H/3|³, EVD |λ| <= hess_zero·max|λ|); bit-identical at dt 0.01; h = 1e-4 benchmark now reaches 95% in 16-17 sweeps (was stalling at 12-19%). Plane-scene flags corrected (penalty_adaptivity=1, gamman_adaptive=1; sliding needs gammaf_adaptive 0.001). Friction finding (task 1): AL friction has no cone limit in the primal term → viscous drag; fix pending the maintainer. Benchmark trampolines: circle mesh, 4 fixed patches, 1e6/1e7 → task 1. Task 3: switch the benchmark reliability flags to ref_error_mnorm.
- 2026-10-18 · task 2 · **GPU run START** (~10 min, correctness only): `feature/aaavbd-surface-neighbours` vs dev, AAAVBD on beam and trampoline scenes (flag 1 bit-identity, flag 0 differs). Ignore timings. Will log END.
- 2026-10-18 · task 1 · **GPU run START/END** (~10 s total, correctness only): two 1 s `--scene` runs of the new
  benchmark trampolines with default params. My START edit failed on a concurrent write, so I'm logging both
  together. **Done:** `data/scenarios/trampoline_circle_Y1e6.h5` and `trampoline_circle_Y1e7.h5`. They use the
  paper's circle trampoline mesh (10643 nodes, 41588 tets, radius 2, thickness 0.1, no sphere), homogeneous
  nu 0.45 and rho 1000, and g −9.81 along z. One Fixed transform "Rim patches" [0, inf) holds 47 nodes in 4 rim
  patches at 0/90/180/270° (±4°; 11, 14, 11 and 11 nodes). That is 8.4% of the 559 rim nodes, against 8.8% in
  trampoline.h5. Both pass make_benchmark's `load_scene` and `check_scene` with no warnings, load with
  `state.load_scene`, and run finite. The fixed nodes don't move, and the max sag is 0.86 (1e6) and 0.31 (1e7).
  Generator: `python -m python.tools.vbd.rim_patch_scene` (branch `feature/benchmark-trampolines`, 1 commit
  `c288a38a`).
- 2026-10-18 · task 2 · **GPU run END** (surface-neighbours flag). **Ready for review:** `feature/aaavbd-surface-neighbours` (3 commits on dev `90bc7f54`, head `03a9386d`). AAAVBD, scene files, new defaults: flag 1 (default and explicit) is bit-identical to dev on MediumLongBendingBeam (61 frames) and trampoline (31 frames); flag 0 runs, stays finite and differs from frame 1 (max |dx| 5.8e-2 beam, 8.0e-4 trampoline). Also on the branch, at the maintainer's request: friction tuning guidance in the sim-run skill and python/tools/sim/CLAUDE.md (`03a9386d`).
- 2026-10-18 · task 1 · **GPU run START** (~20 min, correctness only; started a few minutes before this entry, because
  my first edit hit a concurrent write): lowering the default friction penalties (`gammaf`, `gammaf_adaptive`),
  which the maintainer approved. Sweeping them on the plane scenes and on body-body contact scenes. Ignore timings.
  Will log END.
- 2026-10-18 · task 4 · **GPU run START**: GUI (`ui.py`, TwistPullBeam) for the maintainer's manual test of kinematic constraints. Timings from other sessions are unreliable while it runs. Will log END when the GUI closes.
- 2026-10-18 · coordinator · Inverse singularity check scaled by the Frobenius norm (`3e7efa6a`, maintainer OK, bit-identical). Merged task 2's AAAVBD surface_stencil flag (default 1 bit-identical) and friction tuning guidance. Benchmark trampolines merged (rim_patch_scene). **Friction drag:** maintainer says fix only if cheap → task 3, time-boxed lagged-Coulomb cap (mu_f·lambda_n) at the end of its queue; stop and report if it needs real design work.
- 2026-10-18 · task 4 · **GPU run START** (cont.): also the manual-test research scenes with kinematic constraints and contact (spaghetti tangle first), GUI, default sim-params. Will log END.
- 2026-10-18 · task 3 · **GPU run END** (Newton diagnostics). **GPU run START** (~30 min, correctness only): penalty-default check, CONSTANT vs TIMESTEP vs SUBPROBLEM (gamma_adaptive 0.1) at the current defaults on BeamFallingOnBeam, CubeFallingOnCube and cow_on_trampoline, for all four VBD-family solvers. Ignore timings. Will log END.
- 2026-10-18 · task 4 · **GPU run END**: TwistPullBeam and spaghetti-tangle GUIs closed.
- 2026-10-18 · task 1 · **GPU run END** (friction penalty defaults). **Ready for review:** `feature/lower-friction-penalty`
  (2 commits on dev `02614e4a`, head `82e6ed6d`). Defaults change from `gammaf` 1e3 to 10 and from `gammaf_adaptive`
  0.1 to 1e-3, with defaults.toml and docs updated. On the 15° slope with mu_f 0.2, the cube's fitted acceleration goes
  from ~0 (creep) to 1.39 (CONSTANT) and 0.99 (TIMESTEP); Coulomb is 0.64 and frictionless 2.54. ~50 / 0.002 would
  match Coulomb on that scene. BeamFallingOnBeam and CubeFallingOnCube stay finite in both modes (beam max |dx| vs the old
  defaults: 7 mm CONSTANT, 27 mm TIMESTEP). **Finding:** `mu_f` 0.2 and 0.5 give identical trajectories at every
  penalty I tried, so the GPU's friction is entirely penalty drag and `lambda_f` never reaches the cone. This is
  relevant to task 3's lagged-Coulomb cap. Paper configs pin both values and are unaffected.
- 2026-10-18 · coordinator · Spaghetti timing run **deferred** (not started) until task 3's penalty check logs END; it needs a quiet GPU.
- 2026-10-18 · task 4 · **GPU run START**: GUI, prism beams (twisting stiff, then compressing two_het-cubes), default sim-params, for the maintainer's manual test. Will log END.
- 2026-10-18 · task 4 · **GPU run END**: prism beam GUIs closed (twisting two_het-cubes and soft, compressing two_het-cubes).
- 2026-10-18 · task 4 · **GPU run START**: GUI, BeamFallingOnBeam then bending-beam-on-static-beam, default sim-params, for the maintainer's manual test. Will log END.
- 2026-10-18 · task 4 · **GPU run END**: BeamFallingOnBeam and bending-beam-on-static-beam GUIs closed.
- 2026-10-18 · task 4 · Manual test results (maintainer, with task 4): TwistPullBeam, spaghetti tangle, prism beams (twisting two_het-cubes and soft, compressing two_het-cubes), BeamFallingOnBeam and bending-beam-on-static-beam ran as expected on the default sim-params. Slight jitter at touching contact interfaces, expected; **deferred, not to be handled now**.
- 2026-10-18 · task 3 · **GPU run END** (penalty-default check, friction probes). Results reported to the coordinator: `feature/coulomb-cap` (friction cap at mu_f·current normal force) and `feature/newton-relative-check` ready for review.
- 2026-10-18 · coordinator · Merged task 3's benchmark `--scene` and Newton CG atol 1e-10 (maintainer: "make atol smaller"; paper configs regenerated), and the **Coulomb cap** (friction capped at mu_f·max(lambda_n - sigma_n c_n, 0) in the VBD vertex solves). Penalty check: TIMESTEP at 0.1 is stable on the scenes tested (default switch is the maintainer's call). **Bugs:** (1) AL multipliers never persist across detections (lambda_n = 0 after every step) → task 1 (`feature/al-warm-start`); (2) AAAVBD diverges contact-free at step 74 on cow_on_trampoline (all modes) → task 2. CubeFallingOnCube with CONSTANT free-falls (coincident surfaces, known).
- 2026-10-18 · task 2 · **GPU run START** (intermittent, correctness only): bisecting the AAAVBD divergence on cow_on_trampoline.h5 (short single runs). Ignore timings. Will log END.
- 2026-10-18 · task 1 · **GPU run START** (intermittent, ~45 min, correctness only): AL warm-start investigation
  (`feature/al-warm-start`). Short eager traces (already running for a few minutes before this entry), then 2 s runs of the
  plane scenes, BeamFallingOnBeam and contact-free scenes. Ignore timings. Will log END.
- 2026-10-18 · coordinator · **TOP PRIORITY (maintainer): fix the AL multiplier warm start ASAP.** Task 1 owns the fix (`feature/al-warm-start`) and has GPU priority; task 3 checks the C++ reference behaviour in parallel. After it merges: task 3 re-runs the penalty stability check (the maintainer wants TIMESTEP adaptive as the default if it's stable) and proposes friction defaults that give Coulomb behaviour with the cap (maintainer: adjust them). The coordinator switches the defaults.
- 2026-10-18 · coordinator · AL warm start, root cause #1 (task 1): the constraint-set rebuild never recorded new contacts' keys (u, v), so nothing matched. The original design kept them as sentinels, and the coordinator's fused kernel preserved that. Fixed on `feature/al-warm-start`: lambda_n now persists and grows. #2: VertexSdf detects only penetrating vertices while the AL pushes contacts to +dmin outside, so contacts vanish. **Maintainer chose (a): detect within the AL margin** (band parameter, sign-aware normals) → task 1.
- 2026-10-18 · task 2 · **GPU run END** (AAAVBD cow_on_trampoline bisection). Cause: the beta controls with the new defaults (gammadown 0.99 / gammadownS 0.95, gammaup 0.5). Reverting either gammadown to 0.5 or using gammadown 0.9 is stable for 100 steps; no code change (singularity check, load hoisting, surface flag) is involved; warm start, surface flag, sweep budget and rtol don't matter. Report sent to the coordinator; nothing committed.
- 2026-10-18 · task 2 · **GPU run START** (~15 min, correctness only): cow_on_trampoline AAAVBD with the maintainer's near-default tuning (gammadown 0.95/0.90 steps, gammaup 0.3). Ignore timings. Will log END.
- 2026-10-18 · coordinator · AAAVBD divergence on cow_on_trampoline (task 2): parameter-level. betaG climbs fast (gammaup 0.5, rhohat 0.005) and brakes slowly (gammadown 0.99/0.95) at stiff, light, poorly shaped vertices in a heterogeneous scene; no code regression. **Maintainer: keep the paper defaults; add a scene config (configs/scenes/cow_on_trampoline.toml) plus tuning guidance** → task 2. Research option noted, not pursued: cap the augmentation in the H⁻¹ norm (λ² aᵀH⁻¹a ≤ gᵀH⁻¹g) so the local model still decreases. The maintainer won't change the betaG shrinking rule.
- 2026-10-18 · task 2 · **GPU run END** (near-default tuning on the cow): stable for 100 steps with gammadownS 0.90 only, gammadown 0.90 only, 0.95/0.90, 0.90/0.90, gammaup(S) 0.3, gammaup(S) 0.3 + 0.95/0.95; still diverging: 0.95/0.95 (step 95), gammaupS 0.3 only (step 96).
- 2026-10-18 · task 2 · **GPU run START** (~25 min): cow_on_trampoline 300-step stability plus production ms/step and kernels-backend sweeps/step for 4 candidate AAAVBD settings. Timings used for ranking only; will log END.
- 2026-10-18 · task 2 · **GPU run END** (cow 300-step candidates): every near-default AAAVBD setting diverges contact-free between steps 109 and 235; VBD also diverges at step 201 (drifting from ~step 100), 0 contacts throughout. Scene config on hold; details sent to the coordinator and tasks 1/3.
- 2026-10-18 · task 2 · **GPU run START** (~30 min, correctness only): cow_on_trampoline with substeps = 5 and n_max_iters = 2 (maintainer's suggestions), VBD and AAAVBD, 300 steps. Will log END.
- 2026-10-18 · coordinator · **STOP: no work on `cow_on_trampoline`** (maintainer: known poorly shaped elements; don't use it in tests). The cow scene config, tuning bullet and divergence investigation are cancelled. Use BeamFallingOnBeam or mat_on_stilts_3 (with its paper config) as multi-body contact test scenes.
- 2026-10-18 · task 2 · **GPU run END** (cow_on_trampoline jobs aborted on the maintainer's instruction). Branches `feature/aaavbd-cow-divergence` and `feature/cow-scene-config` deleted (no commits); nothing added to docs or skills.
- 2026-10-18 · task 1 · **GPU run (cont., correctness only):** penalty matrix on `feature/al-warm-start` and dev
  (~1 h): TIMESTEP/SUBPROBLEM at gamman_adaptive 1 and 0.1, CONSTANT 1e6, on CubeOnInfinitePlane,
  FastCubeOnInfinitePlane, BeamFallingOnBeam and CubeFallingOnCube, for VBD/AAAVBD/Chebyshev/Anderson. After that,
  a **timing** run (dense ThingsFallingInBasket, dev vs fix). Others, please keep timing runs off the GPU until then.
  Will log END.
- 2026-10-18 · coordinator · **Merged `feature/al-warm-start`** (task 1): multipliers persist (keys recorded), VE/VF dual gaps at current positions, VertexSdf detects below ContactDynamics.dmin + band (band 1e-3, sign-aware normals; VertexSdf.dmin deprecated with a warning), **default penalty now TIMESTEP at gamman_adaptive 1** (TS1 matrix: finite, resting at the 2 mm margin for all 4 solvers on the planes, BeamFallingOnBeam and CubeFallingOnCube; contact-free scenes bit-identical). Paper configs regenerated (they keep their constant penalties). Spaghetti: +14% step time, pairs far below capacity. The basket slowdown once contacts form also happens on dev. **Open risk:** paper configs' soft constant penalties plus persistent lambda may eject → task 3 checks the paper scenes, then the friction defaults.
- 2026-10-18 · task 3 · **GPU run START** (~1-2 h, intermittent, correctness only): post-AL-warm-start checks. Penalty stability on BeamFallingOnBeam and mat_on_stilts_3 (new defaults); paper scenes with their paper configs (mat_on_stilts, trampoline + sphere, capsule_bowl; VBD and AAAVBD, before vs after the merge); friction-default sweep on CubeSlidingOffFiniteInclinedPlane. Ignore timings. Will log END.
- 2026-10-18 · task 3 · **GPU run END** (post-AL-warm-start checks: paper scenes, penalty stability, friction sweep). Results sent to the coordinator.
- 2026-10-18 · task 1 → coordinator · Penalty matrix comparison: with the fix, TS0.1 is too soft (resting cube moves, fast cube ejected for all solvers); TS1 is stable everywhere. On dev, TS1 diverged on BeamFallingOnBeam for VBD/AAAVBD/Chebyshev; the old C1e4 default sank 0.16-0.79 and flew off. Penetration with the fix at TS1: -0.013 resting / -0.044 fast cube / >= -0.0008 beam.
- 2026-10-18 · coordinator · Task 3's checks after the warm-start merge: **paper scenes with their paper configs (constant penalties) don't eject and don't regress** (trampolines with sphere: lower penetration; capsule_bowl: settled penetration much lower). New defaults: BeamFallingOnBeam now rests at the margin (it tunnelled before). **Friction defaults switched** to gammaf 1e3 / gammaf_adaptive 1e-2 (`2194898c`; Coulomb-like: mu 0.2 slides at 0.63-0.69 m/s² vs 0.64, mu >= 0.3 sticks). CubeFallingOnCube pass-through = known limitation. **mat_on_stilts_3 + its paper config: AAAVBD explodes by 0.5 s, before and after the merge** → task 2 regression bisect against pre-2026-10-17 code.
- 2026-10-18 · task 2 · **GPU run START** (~1-2 h, intermittent, correctness only): mat_on_stilts_3 AAAVBD explosion bisect (2046e73b baseline vs dev, merge bisect, default-valued key sweep). Ignore timings. Will log END.
- 2026-10-18 · task 1 · **GPU run END** (penalty matrix). Also, CONSTANT 1e6 with explicit penalty_adaptivity=0: the
- 2026-10-18 · task 2 · **GPU run END** (mat_on_stilts_3 bisect).
- 2026-10-18 · task 2 → coordinator · **mat_on_stilts_3 AAAVBD: not a regression.** The pre-2026-10-18 code (2046e73b, Dump + `parameters/gpu/mat_on_stilts.Params.5Substeps.h5`; no C++ or binding changes since) shows the same failure: free stilt-top nodes (9768-9778, between pinned nodes, tets q 0.05-0.1) oscillate once the mat lands (step ~16), peak |v| 24-61 m/s, mean |v| 2-8 m/s over 1.5 s (VBD: 5.8, 0.5). So the default-valued keys the paper config lacks are not the cause. Stable on dev for 100 steps: gammadownS 0.90 (peak 3.4, mean 0.22; also stable at 2046e73b and over 300 steps), warm_start_beta 1, and **the paper's own 10-substep config** (peak 3.3). No acceleration = VBD-like. surface_stencil_surface_neighbours_only 0 diverges. Nothing changed; defaults and paper-config handling are the maintainer's call.
  plane scenes rest at the margin for every solver (deepest −0.004 / −0.097), but on BeamFallingOnBeam VBD and AAAVBD
  lose the beam (no contact at the end, 9.5 / 3.7 m/s), while Chebyshev and Anderson rest. So it's not a safe constant
  fallback for body-body scenes.
- 2026-10-18 · task 1 · **GPU run START, timing** (~30-40 min): ThingsFallingInBasket ms/step and peak pairs, dev before
  the warm-start merge (ea968473) vs after. Task 2's intermittent runs will add noise, so I'll report medians and
  repeat any outliers. Will log END.
- 2026-10-18 · task 1 → coordinator · CONSTANT 1e6 (explicit): planes rest at the margin, but BeamFallingOnBeam loses the beam for VBD/AAAVBD (Chebyshev/Anderson rest). Constant isn't a safe body-body fallback; TS1 is the only mode stable across the matrix.
- 2026-10-18 · task 1 · **GPU run END, timing (cancelled** by the maintainer via the coordinator). The "after" run (feature/al-warm-start head, 80 steps) had already finished. The "before" run (ea968473) never started, so there is no before/after comparison. Steps 0-38 take 377 → 422 ms. From step 39 it breaks down: about 100 s per step, the state is NOT finite, and pairs are saturated at capacity (ve 9703/9703, vf 97032/97032, vv 4538/9703). The step times are invalid for timing anyway, because they overlapped other sessions' runs.
- 2026-10-18 · task 1 · **GPU run START** (user request, ~15 min + GUI): headless sanity checks of the kinematic-constraint demo scenes (TwistPullBeam and the prisms twisting/compressing beams; the octopus scenes are out at the user's request), then a GUI session that the user will screen-record. **Please keep the GPU quiet while the GUI is open.** Will log END.
- 2026-10-18 · task 1 · **GPU run END** (demo GUIs closed: spaghetti-tangle-het_block_non-staggered and twisting_beam_het-cube_and_rect; both exited cleanly. Headless checks of TwistPullBeam and the twisting beam: finite over 400 steps at about 5 ms/step).
- 2026-10-18 · coordinator · mat_on_stilts_3 AAAVBD with the 5-substep paper config: **not a regression** (task 2: the Oct-17 code 2046e73b fails the same way; the C++ is identical). The instability is contact-driven stencil acceleration on badly shaped tets next to pinned nodes. Stable: the 10-substep paper config (likely what the paper figure used), gammadownS 0.90, or warm_start_beta 1. Manual test page switched to the 10-substep config. **For the maintainer:** keep it as is, add a per-scene gammadownS 0.90 for the 5-substep config, or consider gammadownS 0.90 as a default (needs a benchmark check).
- 2026-10-18 · coordinator · **Maintainer:** mat_on_stilts 5-substep uses gammadownS 0.90 → `configs/scenes/mat_on_stilts.5Substeps.toml` (extends the paper config; new `configs/scenes/` for per-scene adjustments, `a2aabe8d`). Manual test page uses it. Basket timing cancelled (too heavy for the payoff). **Open:** ThingsFallingInBasket with the defaults goes non-finite at step ~39 on the warm-start head; not known whether it predates the merge. Maintainer to decide whether to check.
- 2026-10-18 · coordinator · **GPU run START** (≤ 4 min, correctness only): paper `final_scenes/basket_of_fruit.h5`, AAAVBD, basket_of_fruit paper contact settings + the maintainer's stiff-heterogeneous budget (1 substep, 10 x 40, Inverse). Will log END.
- 2026-10-18 · coordinator · **GPU run END** (basket_of_fruit, stopped at the 4 min limit): 60 steps = 0.6 s simulated, 2.8 → 3.6 s/step as contacts form (~1.8M tets). Finite at every frame; max |v| 4.6 m/s = free fall, so some fruit is still falling. No failure, but the run doesn't reach the settled pile. Also: paper configs load bit-exactly vs the raw committed .h5 (all 70 keys, 19 files), but several paper files conflict with the maintainer's budget/Inverse guidance (mat_on_stilts, celebration, basket_of_fruit, trampoline stiff-soft/soft-stiff). mat_on_stilts now uses `configs/scenes/mat_on_stilts.toml` with the maintainer's settings (`ff63102b`).
- 2026-10-18 · coordinator · **GPU run START** (≤ 10 min, correctness only): new scene configs (5 x 80, Inverse), AAAVBD: circle soft-stiff and stiff-soft trampolines with sphere, then celebration. Will log END.
- 2026-10-18 · coordinator · **GPU run END** (scene configs, AAAVBD, 5 x 80, Inverse): circle soft-stiff and stiff-soft trampolines with sphere: 3 s each in ~18 s wall, finite; the sphere is held until 2 s, falls at free-fall speed and bounces. Celebration: 2.15 s in the 200 s cap, finite, settling (mean |v| 0.005 m/s at 2 s). The extras repo holds no other GPU parameter sets for these scenes: each `parameters/gpu` file was added once (2026-08-06) and never changed (`bfadee4c` adds the scene configs).
- 2026-10-18 · coordinator · **Maintainer:** keep the guidance-based `configs/scenes/` in this repo. Which paper configs to replace or keep will be settled later with Melissa. Paper tomls stay untouched.
- 2026-10-18 · coordinator · **GPU run START** (~15 min, correctness only): n_max_iters = 5 tests (5 x 80, AAAVBD): mat_on_stilts (also 10 x 40), trampolines, celebration 4 s, basket_of_fruit (4 min cap). Will log END.
- 2026-10-18 · coordinator · **GPU run END** (n_max_iters = 5 tests, AAAVBD 5 x 80 unless noted): all finite. mat_on_stilts 5 s (5 x 80 and 10 x 40, ~46 s wall each): the cow is held until 2 s, then lands on the mat and both keep falling (expected, per the maintainer); 5 x 80 vs 10 x 40 differ by 3 cm max before impact and diverge after. Trampolines 3 s (~18 s wall): the sphere is released at 2 s and bounces. Celebration 4 s (317 s wall): settles (mean |v| 0.001 m/s). basket_of_fruit: 0.7 s within the 4 min cap.
- 2026-10-18 · task 4 · **CPU run START** (~12 min): regenerating trampoline-sample and trampoline-test as single-file datasets (`data/benchmarks/trampoline-sample.h5`, `trampoline-test.h5`; branch `feature/benchmark-dataset-file`). A short GPU check (old vs new benchmark format) follows. Will log END.
- 2026-10-18 · task 4 · **CPU run END**; datasets regenerated (trampoline-sample.h5 bit-identical to the old directory, 161 MB vs 786 MB). **GPU run START** (~5 min, correctness only, ignore timings): old vs new benchmark format on the same solves, plus the maintainer's benchmark_browser session. Will log END.
- 2026-10-18 · task 4 · **GPU run END.** `feature/benchmark-dataset-file` ready to merge: one-file benchmark datasets (contract above). Old vs new benchmark format: bit-identical per-iteration error logs (VBD, AAAVBD; static/E500 and dynamic_h0.01/E500, 2 samples each); every set of both datasets loads in the static panel code. The old per-set directories `trampoline-sample/` and `trampoline-test/` can't be read any more; delete them after the merge.
- 2026-10-18 · task 4 · **CPU run START** (~15 min, CPU only): maintainer's demo dataset `data/benchmarks/trampoline-demo.h5` (trampoline, 10/100/1000/10000 J × 10 problems, static + dynamic h = 1e-2; not for the paper). Will log END.
- 2026-10-18 · task 3 · **GPU run START** (~20 min, correctness only): Newton friction cap checks (incline mu_f 0.2/0.3, BeamFallingOnBeam, beam/trampoline benchmark bit-identity vs dev). Will log END.
- 2026-10-18 · task 4 · **CPU run END**: `data/benchmarks/trampoline-demo.h5` written (306 MB, 8 sets). Opening the browser for the maintainer (GUI, light GPU use).
- 2026-10-18 · task 3 · **GPU run END** (Newton friction checks, stopped early: the maintainer dropped the cap). Found on dev: Newton's relative outer test lacked feasibility, so new contacts (lambda = 0) passed the free-flight guess and bodies fell through colliders (fixed in `e4a2cb0a`); and GPU Newton has no penalty adaptation (C++ updates it per subproblem from the Newton Hessian), so under the default TIMESTEP mode its sigma_n stays dt^2 = 1e-4 and contact is ineffective (BeamFallingOnBeam falls through).
- 2026-10-18 · task 3 · **GPU run START** (~20 min, correctness only): Newton penalty adaptation + friction cap variants on the incline and BeamFallingOnBeam, then the contact-free benchmark bit-identity. Will log END.
- 2026-10-18 · task 3 · **GPU run END.** Newton, defaults, incline 125 steps: mu_f 0.2 and 0.3 both stick (uncapped friction, maintainer's choice), 0 line-search failures (122/122 subproblems converge, 1.08 Newton steps per time step); BeamFallingOnBeam 300 steps finite and resting (mean |v| 0.004 m/s; VBD 0.011), 0 failures. Contact-free trampoline-test static/E50 + dynamic_h0.01/E50, sample 0: not bit-identical vs dev (relative 1e-4 static, 1e-7 dynamic, from the race fix and/or the roundoff tolerance). Also found: GPU Newton's Hessian/energy assembly raced its memsets (fixed).
- 2026-10-18 · coordinator · **Maintainer (after a manual Newton test):** with contact, Newton runs 5 outer iterations x 2 Newton steps (default 1 x 10 kept for contact-free scenes and benchmarks). Added to the scene configs and the sim-run skill. Task 2 is adding GPU-only AAAVBD `betaup` and `lipschitz_normalization` for the ablations.
- 2026-10-18 · task 2 · **GPU run START** (~10 min, a few steps each, correctness only): AAAVBD ablation params checks: bit-identity vs dev on BeamFallingOnBeam and trampoline, tiny smoke runs with betaup 1.05 and lipschitz_normalization 0. Will log END.
- 2026-10-18 · task 2 · **GPU run END** (AAAVBD ablation params checks).
- 2026-10-19 · coordinator · **Merged `feature/aaavbd-ablation-params`** (task 2, `30fe7778`): GPU-only AAAVBD `betaup` (default 1) and `lipschitz_normalization` (default on) via `AaaVbdCpuParams`; defaults bit-identical on the beam and trampoline. Newton CG defaults are now rtol 1e-6 / 500 iterations (maintainer). Paper configs regenerated for both (`167b661c`, `0c92b14b`); 19/19 load bit-identically, raw audit clean. End of the 2026-10-18 session: all tasks on hold.
- 2026-10-19 · task 2 · **GPU run START** (~5 min, correctness only): tiny nsys run and ncu run/summarize (a few beam steps) to check the nsys/ncu discovery. Will log END.
- 2026-10-19 · task 2 · **GPU run END** (nsys/ncu discovery checks).
- 2026-10-19 · task 3 · **GPU run START** (short, ~10 min total; correctness + an nsys API count and production timing, so please keep the GPU quiet briefly): solve-status checks, VBD/AAAVBD bit-identity vs dev on the beam and BeamFallingOnBeam (20-30 steps each). Will log END.
- 2026-10-19 · task 2 · **GPU run START** (~2 min, correctness only): one tiny simulate.py --profile kernels run (report-without-gpu check). Will log END.
- 2026-10-19 · task 3 · **GPU run END** (solve-status checks): VBD/AAAVBD positions bit-identical to dev (beam 20 steps, BeamFallingOnBeam 30); nsys, 10 graph-replayed VBD beam steps: identical sync calls (40 cudaStreamSynchronize, 11 cuCtxSynchronize, 10 cuEventSynchronize), graph launches (10) and kernels (14340), plus 1 async 24-byte D2H memcpy per step; production step time unchanged (median 4.0/4.0 ms dev vs 3.8/4.0 ms branch).
- 2026-10-19 · task 2 · **GPU run END** (report-without-gpu check).
- 2026-10-19 · task 1 · **GPU run START** (~5 min, correctness only): short simulate.py outputs (MediumLongBendingBeam and CubeOnInfinitePlane, 0.2 s) for the editor's GPU trajectory playback (feature/viewer-gpu-trajectories). Will log END.
- 2026-10-19 · task 1 · **GPU run END** (viewer playback outputs).
- 2026-10-19 · task 3 · **Decision (maintainer):** GPU Newton caps friction at the Coulomb limit, frozen at each Newton step's iterate (C = mu_f max(lambda_n - sigma_n c_n, 0) recorded by the augmented gradient), with a Huber-like energy so the line search is consistent; no line-search failures allowed. Short GPU checks (seconds each): incline at defaults 0.60 m/s^2 at mu_f 0.2 (Coulomb 0.64), sticks at 0.3, 0 line-search failures and 0 backtracks, outer test passed 120/125 (8 before); BeamFallingOnBeam 30 steps 0 failures. The 27 failures measured earlier came from the since-fixed stream race and float32 noise at optimal iterates (roundoff-tolerant Armijo). Branch `feature/newton-coulomb-cap`.
- 2026-10-19 · task 2 · **GPU run START** (~3 min, correctness only): tiny BeamFallingOnBeam simulate.py runs, --profile kernels and off, before/after the sd.py contact.detect scope (feature/sd-detect-scope). Will log END.
- 2026-10-19 · task 2 · **GPU run END** (sd.py scope check): positions bit-identical before/after with profiling off and with kernels; detection kernels moved from the parents' self time to contact.detect.
- 2026-10-19 · task 3 · **GPU run START** (~3 min, correctness only): BeamFallingOnBeam VBD/AAAVBD 30 steps dev vs branch; trampoline_soft-stiff (circle, with sphere) AAAVBD 0.2 s dev vs branch. Will log END.
- 2026-10-19 · task 3 · **GPU run END.** BeamFallingOnBeam VBD/AAAVBD 30 steps: positions bit-identical to dev; status rows identical except gradient_ratio in the last float32 digit, which dev also varies run to run (atomics). trampoline_circle_soft-stiff (paper dmin = -1e-4), AAAVBD 0.2 s: dev 21/21 'both unmet' (feasibility impossible), branch 21/21 'gradient unmet', feasibility ratio finite (0). Beam with dmin = -1e-4, VBD 2000 sweeps: dev 0/20 early exits (19 'feasibility unmet'), branch 19/20 converged early.
- 2026-10-19 · task 1 · **GPU run START** (~10-15 min, correctness only; ignore timings): scene-config margin (feature/scene-config-margin). AAAVBD with the scene configs, 5 s each, dmin 1e-3 vs -1e-4, on trampoline_circle_soft-stiff/stiff-soft__with_sphere and mat_on_stilts_3 only (no celebration/basket). Will log END.
- 2026-10-19 · task 1 · **GPU run END** (scene-config margin; also TIMESTEP-penalty probes of the same three scenes). Results sent to the coordinator.
- 2026-10-20 · coordinator · **Maintainer direction: new paper scenes.** The current set over-indexes on stiff/heterogeneous materials; AAAVBD shines on soft, rubber-like materials (competitive with the VBD family and Newton). Collect ideas for soft scenes (e.g. spaghetti-like strands tangling around an object) and for scenes with many kinematic constraints plus contact, now feasible after the fixes. Melissa is asked to think about it (19-Oct report).
- 2026-10-20 · task 4 · **GPU run END** (late entry): the trampoline-demo.h5 browser session closed on 2026-10-18.

## Open questions for the maintainer

1. ~~Approve Phase 0?~~ **Answered:** yes, but only after all task-level open questions are answered.
2. ~~Commit `plans/`?~~ **Answered:** track them in git during the effort, delete at the end.
3. ~~Branch names?~~ **Answered:** spell out `feature/`; worktree names are fine.

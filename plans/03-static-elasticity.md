# Task 3: Static (equilibrium) solves and a convergence benchmark

Status: **not started** · Branch `feature/gpu-static-solves` · Worktree `C:\git\pbat-static`
Read first: `MASTER.md`, `python/tools/sim/CLAUDE.md`, `source/pbat/sim/CLAUDE.md`.

## Goal

A clean way to set up and run **static equilibrium** solves with every GPU solver (Newton, VBD, AAAVBD,
Chebyshev, Anderson), as the basis of a convergence benchmark for the AAAVBD paper.

## The benchmark (maintainer's definition, 2026-10-17)

1. **Initial shapes come from task 4's MCMC sampler.** Shapes are sampled on the quadratic manifold at a
   prescribed deformation energy and lifted to full space (x₀ = X + u(q)). This task **depends on task 4**
   for its inputs; see the sample-file contract in `MASTER.md`.
2. **Each sample is released into a static solve** whose equilibrium is the **rest shape**.
3. **Error is the distance to the rest shape**, e_k = ‖x_k − X‖. The exact norm is still to settle (plain
   L2 over free nodes, or mass-weighted).
4. **The metric is wall-clock time per percentile of error reduction:** the time at which
   e_k ≤ (1 − p)·e₀ for a set of percentiles p (e.g. 10%, 50%, 90%, 99%, 99.9%). Aggregate over many
   samples and energy levels, and compare solvers.

Consequences for the design:
- **The equilibrium must be exactly X**, so benchmark cases run with no external loads (no gravity) and
  Dirichlet nodes held at their rest positions. Confirm with the maintainer.
- **Measuring wall-clock time per percentile needs the error trajectory without distorting the timing.**
  Reading the error back every sweep forces a host sync and rules out graph capture. Options to evaluate:
  - compute e_k on the device every sweep into a preallocated log array, with CUDA-event timestamps, and
    read both back only at the end. This keeps graph capture possible, so it measures production speed;
  - run eagerly with task 2's profiler, simpler but includes launch overhead.

  Agree with the maintainer which one is "the" wall-clock number.
- `check_convergence` can stop on a target percentile of error reduction, in addition to the gradient-norm
  tolerance.

## What exists today (the hack)

- **CPU editor** (`tools/vbd/ui/simulation.py:322` and `:194`): the "Static Elasticity" checkbox sets the
  lumped masses to zero (`fem.m = 0`), switches initialization to `Position`, and (commented out) sets
  `xtilde = X`. The time step and BDF order still apply even though they're meaningless for a static problem.
- **`tools/static_elasticity/elasticity.py`** computes hyperelastic vibration modes and serializes
  mode-deformed configurations, used as initial states. It has its own `parse_archive_path`, still with the
  old Windows-path behavior.

Problems with the hack:
1. With m = 0, VBD's per-vertex 3×3 systems lose the mass term. The maintainer considers this benign: the
   vertex solves already have a pseudo-singularity check (`hess_zero`, `EVertexIntegrationLinearSolver`).
   Zeroing the mass array has a second side effect, though. The GPU time-integration setup computes
   `fext[i] / m[i]`, which divides by zero, and gravity loads expressed as `m·g` vanish. Static mode
   therefore needs a flag, not zeroed masses.
2. The `xtilde = X` variant adds a spurious pull toward the rest shape, which changes the solution.
3. There is no convergence measure: GPU `check_convergence` is still a stub that returns False.
4. Initial configurations, Dirichlet sets and loads are scattered between tools.

## Proposed design (to confirm with the maintainer)

**Explicit problem kind.** Add `ProblemKind {Dynamic, Static}` to the GPU FEM problem, or a thin `StaticProblem`
that shares the FEM data. In static mode the objective is E(x) = Ψ(x) − f_extᵀx + contact, with no kinetic
term and no BDF.

**Pure minimization, no added regularization (maintainer decision).** Static mode drops the inertia term
entirely instead of zeroing masses (gravity loads keep using the masses). VBD relies on its existing
pseudo-singularity check for nearly singular vertex systems. If that turns out not to be enough in practice,
the maintainer prefers an **adaptive** safeguard to a constant regularization term. One option is
Levenberg–Marquardt-style per-vertex damping that only switches on when a local solve fails or doesn't
reduce the local energy, and decays back to zero as the solve converges. Treat that as optional: implement
it only if experiments show the need, and agree on the design with the maintainer first.

**A `StaticSolveSpec`** describing one benchmark case:
- initial configuration: **an MCMC sample from task 4's sample file** (the benchmark's source). Other
  sources (the Dump's `x`, modal perturbations from `elasticity.py`) are useful for debugging,
- Dirichlet set: from the Dump, or from task 1's kinematic constraints evaluated at a frozen time,
- loads: gravity on/off, `f_ext` scaling,
- stopping: relative ‖∇E‖ tolerance, maximum outer iterations and sweeps, maximum GPU time.

Serialize it to HDF5 so cases are reproducible.

**Convergence metric.** Implement GPU `check_convergence` and a gradient-norm kernel (reuse
`gpu/gradient.py`, which already has an elastic gradient), covering the elastic, external and contact terms
over free DOFs. The same metric then serves dynamic solves too, replacing the fixed iteration counts.

**Benchmark runner** (`python/tools/sim/benchmark_static.py`): for each solver and sample, run from the
identical initial state with profiling **off**.
- **VBD family:** capture a fixed sweep budget as a CUDA graph. Each sweep, a small kernel writes the
  mass-weighted error e_k into a preallocated device log, and timestamps come either from per-sweep CUDA
  event nodes inside the graph (`Stream.record_event(event, external=True)`, supported in Warp 1.17 per task
  2) or from the replay's total time interpolated by sweep index. Pick between the two after the
  measurement prototype. Logs are read back only after the replay.
- **Newton:** runs eagerly, logging per Newton iteration.
- **Output:** the time at every 5% of error reduction, plus the full e_k (and optionally ‖∇E‖) curve. Write
  JSON, plot e_k/e₀ vs time and vs sweeps, and publish an HTML artifact.
- Task 2's `profiling.scope` is used only for an optional per-phase breakdown, in a separate eager run.

## Validation

- Static mode on the bending beam under gravity with fixed BCs: all solvers reach the same equilibrium,
  matching Newton to tolerance.
- Gravity loads are unchanged by static mode (no division by zero, no vanishing `m·g`).
- Without contact, compare against a CPU Newton static solve if feasible (`pbat.sim.algorithm.newton`).

## Files

Own: the objective/inertia/time-integration parts of `gpu/elasticity/fem.py`, the static-mode branches in
the solver files, `check_convergence`, `benchmark_static.py`, `panels/static.py`. Coordinate with task 1 on
the Dirichlet lines of the setup kernel, and with task 2 on scope names.

## Open questions for the maintainer

1. ~~Proximal term or pure minimization?~~ **Answered:** pure minimization. VBD's pseudo-singularity check is enough; if a safeguard is ever needed, make it adaptive rather than a constant term.
2. ~~Which convergence metric?~~ **Answered:** wall-clock time per percentile of error reduction, with error = distance to the rest shape. **Norm (2026-10-17): mass-weighted L2 over free nodes**, e = sqrt((x−X)ᵀM(x−X)). **Timing (2026-10-17): graph mode with a device-side error log** for the VBD family: capture a fixed sweep budget, a small kernel writes e_k each sweep into a preallocated device log, CUDA events bracket the replay, and per-percentile times are interpolated from the sweep index. Newton (not capturable) runs eagerly. **Percentiles (2026-10-17): every 5%** (5..95%; if that's too much data, from 60% up). No 99/99.9% tail, by the maintainer's choice. The full e_k log is stored either way.
3. ~~Contact?~~ **Answered (2026-10-17):** contact-free only. The paper's static benchmark never uses contact.
4. ~~CPU parity?~~ **Answered (2026-10-17):** GPU-only. No C++ changes. GPU Newton is the reference solver, and validation checks convergence to X (known exactly).
7. ~~Static Newton with an indefinite Hessian?~~ **Answered (2026-10-17):** per-element PSD projection of the elastic Hessian (clamp negative eigenvalues, as C++ `HyperElasticSpdCorrection` does), behind a param that defaults to on in static mode.
8. `check_convergence` in captured dynamic solves: device-side with `wp.capture_while`, or eager only? **Maintainer undecided (2026-10-17):** wants early termination, but not if splitting the captured region adds execution-time overhead. → Measure before deciding (see question 9).
9. ~~Prototype first?~~ **Answered (2026-10-17):** yes, measure first, comparing a fixed-count captured graph against a `capture_while` graph on MediumLongBendingBeam, with the GPU run announced in the coordination log. **Reuse the GPU Newton's existing convergence check** (`newton/solver.py::check_convergence`: `Gradient.compute` + `_gnorm2_reduce`) instead of writing a new one.
10. ~~Solver hyperparameters?~~ **Answered (2026-10-17):** the maintainer supplies a params file per solver (the `--sim-params` .h5 layout), fixed across samples. No tuning by this task.
11. ~~Meshes?~~ **Deferred (2026-10-17):** decide later. The intended scene is the **trampoline with its outer edge fixed**. See the Progress note about the existing `trampoline.Dump.h5`. **Dirichlet set (2026-10-17): keep the Dump's 4 rim patches as they are.** No full-rim override.
12. ~~Static-mode API?~~ **Answered (2026-10-17):** a flag on the GPU `FemElastoDynamics` (`fem.static`). Setup leaves `xtilde`, `x` and `dmask` untouched. Solvers drop the inertia term, use h² = 1 and add −fext explicitly.
13. Importance weights (`samples/weight_log`) in the aggregation? **Pending:** the maintainer asked what they mean. Plan 04 says the chain targets π(q) ∝ √det(JᵀMJ)·δ(Ψ̂ − E*) directly. If so, the samples are already π-distributed, `weight_log` is a per-sample diagnostic (presumably log √det(JᵀMJ)), and unweighted statistics are correct. Asked task 4 to confirm the semantics.
5. ~~Which initial configurations?~~ **Answered:** MCMC samples from task 4.
6. ~~No gravity, Dirichlet at rest?~~ **Answered (2026-10-17):** yes, the equilibrium is exactly X. Benchmark cases have no external loads, and the Dirichlet nodes are fixed at rest and must pin the rigid modes (validate this). Static mode itself still supports gravity and `f_ext` scaling for other uses.

## Progress

- 2026-10-17 · Investigation findings (the code, not the plan text, is the source of truth):
  - `fext` is integrated from the body-force density (`fext = bg·wg·N` in C++ `SetExternalLoad`) and doesn't depend on `m`. Static mode can use `fext` directly. Zeroing `m` breaks only the `h²·fext/m` term of `xtilde` (division by zero in `_setup_time_integration_optimization_kernel` and in C++ `aext()`).
  - The CPU "Static Elasticity" hack drops gravity entirely. With `m = 0`, `fext` enters only through `xtilde`, so the hack solves min Ψ subject to the Dirichlet BCs. (Its `xtilde = X` line is live, but has no effect with `m = 0`.)
  - `elasticity.py`'s Dump stores the mode-deformed shape in `X` and keeps the original `GNegU`/`wgU`, which encode the true rest shape. `ui.SimulationState.reset()` sets `x ← X`, so in those dumps `X` holds the initial configuration. The benchmark needs the true rest positions separately, which the MCMC contract's `X` provides.
  - On the GPU, `fext` enters the VBD vertex solves, `gradient.py` and Newton's energy only through `xtilde`. Static mode needs an explicit `−fext` term in each of them, and `h2 = 1`.
  - GPU Newton has no SPD projection of the elastic Hessian; C++ Newton has `spd_correction`. In dynamics the mass term hides indefiniteness, but in static mode the SNH Hessian can become indefinite.
  - Warp 1.17 (installed) has `wp.capture_while`/`wp.capture_if`, so a device-side stopping test can stay inside a captured graph.
  - The dmask countdown in the setup kernel subtracts `bt·1000` ms each step. A static solve must not advance it.
- 2026-10-17 · `data/scenarios/trampoline.Dump.h5` exists: one component, a disc of radius 1 m and thickness about 2.6 cm, 6000 nodes, 17673 tets, gravity in `fext`. Its Dirichlet set is **only 4 small rim patches** (43 nodes, about 5° wide, at 0°/90°/180°/270°, active for 36000 s), **not the full outer edge** (about 580 nodes have r > 0.99). It pins the rigid modes, but a benchmark with the "outer edge fixed" needs a new Dump, or a Dirichlet set rebuilt from r ≥ r_rim. Maintainer to decide. (`trampoline.h5` and `trampoline.Y1e7.h5` are editor scenes, not Dumps.)
- 2026-10-17 · The MCMC sample-file contract (MASTER.md, agreed) fits the benchmark. Mass-weighted error uses the Dump's `m`, and the contract's `X` is checked against the Dump's rest positions. Still waiting for task 4 to state the `X`/`samples/x` layouts.

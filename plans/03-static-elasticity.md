# Task 3: Static (equilibrium) solves and a convergence benchmark

Status: **merged (static machinery); benchmark runs on hold** · Branch `feature/gpu-static-solves` · Worktree `C:\git\pbat-static`
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
8. ~~`check_convergence` in captured dynamic solves?~~ **Answered (2026-10-17), after the prototype (see Progress):** **capture_while + fallback.** VBD-family solvers run their outer loop in `wp.capture_while`, using Newton's device-side ‖∇E‖ check (`Gradient` + `_gnorm2_reduce`) against `gtol`. If the active contact detector can't run inside a conditional body (vertex SDF today, because `wp.Mesh.refit` allocates), fall back to the unrolled graph. File the refit allocation upstream with Warp.
9. ~~Prototype first?~~ **Answered (2026-10-17):** yes, measure first, comparing a fixed-count captured graph against a `capture_while` graph on MediumLongBendingBeam, with the GPU run announced in the coordination log. **Reuse the GPU Newton's existing convergence check** (`newton/solver.py::check_convergence`: `Gradient.compute` + `_gnorm2_reduce`) instead of writing a new one.
10. ~~Solver hyperparameters?~~ **Answered (2026-10-17):** the maintainer supplies a params file per solver (the `--sim-params` .h5 layout), fixed across samples. No tuning by this task.
11. ~~Meshes?~~ **Deferred (2026-10-17):** decide later. The intended scene is the **trampoline with its outer edge fixed**. See the Progress note about the existing `trampoline.Dump.h5`. **Dirichlet set (2026-10-17): keep the Dump's 4 rim patches as they are.** No full-rim override.
12. ~~Static-mode API?~~ **Answered (2026-10-17):** a flag on the GPU `FemElastoDynamics` (`fem.static`). Setup leaves `xtilde`, `x` and `dmask` untouched. Solvers drop the inertia term, use h² = 1 and add −fext explicitly.
13. ~~Importance weights?~~ **Resolved by task 4 (2026-10-17):** the dataset is renamed `samples/log_target` (log unnormalized target density, a diagnostic). For MCMC batches (`sampler` = polar/crw, root attr `importance_weights_required = False`), average uniformly. Only `sampler = ray` batches need self-normalized weights exp(log_target − max)/Σ. The loader checks the attribute and refuses such files. Layouts: `X` is (N,3) float32 = `fem.X.T` in Dump node order; `samples/x` is (S,N,3) float32 in the same order (pass `samples/x[s].T` to pbat); basis rows are 3i+d.
5. ~~Which initial configurations?~~ **Answered:** MCMC samples from task 4.
6. ~~No gravity, Dirichlet at rest?~~ **Answered (2026-10-17):** yes, the equilibrium is exactly X. Benchmark cases have no external loads, and the Dirichlet nodes are fixed at rest and must pin the rigid modes (validate this). Static mode itself still supports gravity and `f_ext` scaling for other uses.

14. ~~Benchmark budget?~~ **Answered (2026-10-18):** a per-solve **time cap** (`StaticSolveSpec.max_seconds`, e.g. 60 s). Report the time for each 5% level reached plus the fraction of samples reaching it; unreached levels are NaN. Context: cond(M⁻¹K) = 1.3e7 on MediumLongBendingBeam, so VBD-family solvers need about 80–500 s per sample for 95%, while Newton takes about 3 s.
15. ~~Static-tuned solver params?~~ **Answered (2026-10-18):** the maintainer supplies a `--sim-params` file. Until then, runs use the Dump's (dynamics-tuned) params, and reports say so.

16. ~~When to run the full benchmark?~~ **Answered (2026-10-18):** hold it. Task 4 (MCMC sampler) generates the benchmark dataset of problems first; coordinate with task 4 on that dataset before running.
17. ~~Convergence check interval and the refit blocker?~~ **Answered (2026-10-18):**
    - Every N VBD sweeps (default 20), test C++'s **full-step** criterion ‖g_L‖ ≤ gtol: the Lagrangian gradient with current multipliers, like `ToGradient(bForAugmentedLagrangian=false)`. Passing it ends the subproblem, then finalize runs, then the outer test.
    - The GPU check switches to the Lagrangian form; Newton keeps the AL form for its steps.
    - Restructure the loops: outer iterations unrolled on the host; each subproblem's sweeps replay as one device block inside `capture_if`/`capture_while`; refit/detection stay in the regular graph between conditional nodes.
    - The maintainer wants graph capture to stay close to the efficiency of capturing the whole step. Explanation: https://claude.ai/artifact/A632yEjdeA2JV1cEMPqmPp
    - Evidence for the blocker: whole-step capture with VertexSdf works (the allocation becomes a regular graph memory node); only conditional bodies reject it. `scan_device` calls `wp_alloc_device` per call with no cache or option. The scanned average edge length is used only by `mesh_query_point_sign_normal`, while we call `mesh_query_point` (ray-parity sign). Python doesn't expose the mesh's BVH.
18. ~~Feasibility test?~~ **Answered (2026-10-18):** penetration only, over normal contacts (friction ignored): max_j max(0, dmin − c_j(x)) ≤ ftol · dmin, with ftol relative to dmin (`ConvergenceParams.ftol`, default 0.1, under Solver/Convergence). It's tested between outer iterations together with the gradient test; if it fails, the next subproblem runs. The maintainer asked whether AL complementarity is already enforced by the dual update. Answer given: the update keeps s ≥ 0 and λ ≥ 0, but λĉ = 0 is only its fixed point; the multiplier part is bounded by ‖g_L‖ at converged subproblems, while penetration needs the separate length test.

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
- 2026-10-17 · Task 1's upcoming Dirichlet changes: `dbc`/`ndbc` are removed from the GPU struct (don't use them). The dmask countdown is removed: a nonzero dmask without a kinematics group means permanently fixed. dmask becomes device-written each step by the kinematics kernels. For static solves, freeze kinematic time with `KinematicConstraintSet.advance_time = False` (or set its step counter); nothing special is needed on our side.
- 2026-10-17 · **capture_while prototype** (scratch `proto_capture_while.py`, MediumLongBendingBeam, contact detection deactivated, 100 replays, CUDA-event timing, median ms per step):
  | solver | fixed (current) | fixed + ‖∇E‖ every outer | capture_while, gtol=0 (same work) | capture_while, Dump gtol=1e-3 |
  |---|---|---|---|---|
  | VBD | 110.6 | 108.2 | 48.9 (k=20) | 41.4 (k̄=15.5, min 9) |
  | AAAVBD | 115.5 | 113.1 | 55.8 (k=20) | 22.6 (k̄=7.3, min 5) |
  - The gradient check (reusing Newton's `Gradient` + `_gnorm2_reduce`) costs nothing measurable.
  - With identical work and bit-identical results, the loop-structured graph is about 2.2× faster than the unrolled one: 6669 vs 696 captured launches, and host `capture_launch` takes 24.6 vs 0.27 ms.
  - AAAVBD bakes the host `k` into launches (`k > 0` gates the acceleration), so the first outer iteration is peeled into `capture_if`, and the `while` body uses k = 1.
  - Blocker: `wp.Mesh.refit` (vertex-SDF `detect_contacts`) allocates CUB scan temp storage, which conditional bodies reject. OGC's `wp.Bvh` refit/rebuild is untested.
- 2026-10-18 · **Implementation started** on `feature/gpu-static-solves` (rebased on dev c15c6a11). Commits so far:
  1. `45df6670`: `FemElastoDynamics.static` flag + `h2` property. Setup forces the Position strategy; back-substitution and the BDF step are no-ops; leaving static mode restarts at rest, in place. **The setup kernel is untouched** (task 1 hotspot).
  2. `2878b3d8`: VBD/AAAVBD vertex solves use −fext instead of inertia; penalty Hessians drop the mass; sweeps use `fem.h2`. Chebyshev/Anderson inherit via `vbd.iterate`. Dynamics change only by ≤5 ulp of codegen round-off.
  3. `cba23f65`: `gpu/elasticity/spd.py`, a 9×9 Jacobi eigenvalue filter mirroring C++ `FilterEigenvalues` (NoCorrection/Projection/Absolute).
  4. `64cbe01d`: static branches in `gradient.py` and Newton (energy −fextᵀx, no mass block). Static Newton SPD-corrects d²Ψ/dF² with the CPU params' `spd_correction` (C++ default: Absolute), now serialized. **Applied in static mode only**, as agreed; C++ Newton also filters in dynamics.
- 2026-10-18 · Static validation on the bent MediumLongBendingBeam (tip at 0.3 L, no loads, contact off):
  - VBD-family solvers decrease e monotonically but slowly on the smooth bending mode (3–5% after 1000 sweeps). High-frequency noise drops 94% in 125 sweeps. That's the expected Gauss–Seidel behaviour.
  - Newton is limited by its **hard-coded CG cap of 300 Jacobi-PCG iterations** (`n_lin_max_iters`, `rel_eps_lin`, `abs_eps_lin` in `newton.Params.__init__`, neither from the CPU params nor serialized). With 300 iterations, e only halves every 10 steps. With 3000 it converges quadratically to e/e0 = 4.7e-6 in about 4 steps (940–1470 CG iterations each).
  5. `b24ee2b0`: Newton CG limits as a serialized GPU-only `LinearSolverParams` (maintainer decision, 2026-10-18).
  6. `512d49fb`: `GradientNorm` factored out of Newton (no behaviour change).
  7. `7ad0b610`: device-side convergence + `capture_if`/`capture_while` outer loop for the VBD family, with an unrolled fallback (`ContactDetection.supports_conditional_capture`). Bit-identical with gtol = 0; about 2× faster.
  8. `7c1eba28`: `subproblem_iteration(kp)` per VBD-family solver. Chebyshev ω now lives on the device, so iterations are replayable.
  9. `c6268779`: `static.py`, with the sample loader, `StaticSolveSpec`, `ErrorLog` (globaltimer-bracketed, logging excluded) and `StaticSolver` (capture once, replay per shape).
- 2026-10-18 · First runs on task 4's 30-sample batch (E* = 10 J, beam, Dump params):
  - About 93 µs of solver time per sweep.
  - After 100k sweeps, e/e0 = 0.84 (VBD), 0.67 (AAAVBD), 0.70 (Chebyshev), 0.94 (Anderson).
  - Newton (CG ≤ 3000) reaches 95% in 3.2–3.9 s.
  10. `4cabe1e1`: `benchmark_static.py`. It writes results.json (spec, metadata, per-sample time per 5% level, log-spaced curves, aggregates) and results.h5 (full logs).
  11. `dfad4d6f`: `panels/static.py`, with a static-mode toggle, the load scale and sample loading, and e/e0 shown in the GUI. Logic is tested headlessly; the GUI itself is untested (Claude can't drive it).
- 2026-10-18 · **Validation:**
  - Each VBD vertex gradient (elastic − fext) matches the global static gradient (`GradientNorm`, used by Newton and `check_convergence`) to 9e-7 relative, on an MCMC sample with gravity on.
  - Newton's static energy and gradient agree to <1% by central differences at ε = 3e-5. Below that, float32 energy noise dominates, and it's about 3× larger with the load term −fextᵀx (~10³ J here). Possible improvement, not done: write the load term as −fextᵀ(x − X).
  - Gravity loads stay finite with static mode on (fext used directly, no fext/m).
  - Dynamics stay unchanged (bit-identical or round-off, see commits).
- **Still open:**
  - (a) ~~Warp issue~~: not needed (maintainer, 2026-10-18).
  - (b) The maintainer's static params file.
  - (c) The full benchmark run, 30 samples × 5 solvers × 60 s cap (about 2.5 h exclusive GPU; to be announced).
  - (d) Optional `cpu-gpu-parity` review.
- 2026-10-18 · Rebased on dev c947de87 (task 2's profiling, the step-skip fix, task 4's samplers). Conflicts were only task 2's `@profiling.scoped` decorators next to the new `subproblem_iteration`s; both kept. Tests pass; `capture_while` is still bit-identical to the unrolled loop. Commit `ff4baab7`: batch warnings for R-hat ≥ 1.05 and nonzero sampling gravity.
- 2026-10-18 · **Preliminary benchmark** (5 samples, 20 s cap, Dump params, Newton CG ≤ 3000; re-run on an idle GPU after an overlap):
  - Newton: 95% in a median 3.3 s.
  - Within 20 s, AAAVBD and Chebyshev remove about 52% (median), VBD about 30%, Anderson about 13%.
  - Dashboard: https://claude.ai/artifact/9qaUKXA9X4NQwRfipot8Xd
- **Task 1 note:** set `KinematicConstraintSet.advance_time = False` before a static solve's first capture (it's read on the host at capture time).
- 2026-10-18 · **Full-run sequencing (coordinator):** start the 2.5 h benchmark only after (1) task 2's CUPTI/trampoline run logs END and (2) the coordinator's ~10 min kinematics review logs END. Announce in MASTER.md and check that neither is running.
- 2026-10-18 · **Full benchmark on hold** (maintainer): waits for task 4's benchmark dataset. Coordinating with task 4 directly.
- 2026-10-18 · Task 4's dataset plan:
  - Scene: `data/scenarios/trampoline.Dump.h5:fem` (6000 nodes, 17,673 tets, 43 rim Dirichlet nodes).
  - Microcanonical ensemble, pooled draws with an evenly spaced subset (R-hat < 1.05), no gravity.
  - Proposed location: `data/benchmarks/trampoline/`. E* levels and sample counts are open with the maintainer.
- 2026-10-18 · **Trampoline conditioning (CPU):** cond(M⁻¹K) at rest = 5.9e8 (λ_min = 9.4, λ_max = 5.6e9), 44× the beam's 1.3e7. Expect much slower VBD-family static convergence than on the beam within the same cap, and Newton's Jacobi-PCG needing far more than 3000 iterations. Raised with task 4 and the maintainer.
- 2026-10-18 · **Benchmark plan (maintainer, MASTER.md "Benchmark plan"):** task 4 generates the static and dynamic datasets, task 3 runs the static problems, and the coordinator runs the dynamic ones. The full run waits until the maintainer and coordinator fix the data to report and the plot. Until then, only small tests of the machinery.
- 2026-10-18 · Rebased on dev 91ad5a90 (task 1 kinematics, task 2 milestone 2). Conflicts were in fem.py setup (kinematics update + static Position strategy, both kept), state.py (`_attach_kinematics` + `_construct_convergence_check`, both kept) and ui.py panel hooks (all kept). Commit `Prescribe Dirichlet nodes at a frozen time in static solves`: `StaticSolver` runs setup first, and static `step()` never advances kinematics. Unit tests (including kinematics) pass, and small static solves reproduce the pre-rebase numbers.
- 2026-10-18 · **Task 4 test dataset** `data/benchmarks/trampoline-test/static/{E50,E5000}.h5` (3 problems each) passes the loader.
  - X matches the Dump exactly; Dirichlet drift is 0; only the expected R-hat warning (2.7, 3.9) fires.
  - Deformation energies are 11–26 J (E = 50) and 1.1–1.5 kJ (E = 5000); no inverted tets in these six shapes (min J 0.21).
  - Maintainer decisions relayed by task 4: E levels 50/500/5000 J, 5–10 problems per level to start, **time cap 5–10 s**.
- 2026-10-18 · Small trampoline check (E5000 sample 0, 5 s cap, Dump params):
  - VBD e/e0 = 0.50 (37k sweeps, ~129 µs/sweep); AAAVBD 0.39 (50% at 2.5 s); Newton (CG ≤ 300) 0.35.
  - Much faster than the conditioning extrapolation suggested: cond(M⁻¹K) overstates the difficulty for these modal shapes.
- 2026-10-18 · Commits after the rebase on dev 65fc5fe1:
  - `0e948904`: Lagrangian-form convergence gradient (C++ parity).
  - `3de23271`: check interval and the loop restructure (unrolled outer loop inside capture_if, sweep blocks in capture_while, refit in `prepare_detection`). Bit-identical to unrolled; early exit now works with contact (VBD 59 → 16 ms, AAAVBD 67 → 11 ms per beam step at the Dump's gtol).
  - `43babcfb`: feasibility test, validated against numpy on BeamFallingOnBeam.
  - The branch is merge-ready per the coordinator's review. The branch-vs-dev difference at gtol = 0 is codegen from the static branch (≤ ~1e-5), confirmed earlier.
- 2026-10-18 · **Pre-existing contact bugs found (also on dev), reported to the coordinator:**
  - (a) Vertex-SDF detection counts VF slots holding the sentinel v = nF on CubeFallingOnCube, so consumers read `F[nF]` out of bounds; suspected cause is the block write indexing after `tile_sort` in `sd.py`.
  - (b) OGC crashes with CUDA 700 on CubeFallingOnCube between steps 10 and 20.
- 2026-10-18 · **Merged:** `feature/gpu-static-solves` @43babcfb is on dev as 3aae8ba5. The vertex-SDF sentinel bug goes to task 1; the OGC crash is parked (not used for the paper). The coordinator implements Anderson sweep parity on dev. Next: wait for the maintainer and coordinator to settle the benchmark data, plot and measurement; static benchmark runs stay on hold.
- 2026-10-18 · **Maintainer: no Warp issue about `wp.Mesh.refit`.** The restructured loop records the refit in the regular graph between conditional nodes, so it stays captured. The draft stays unposted in the session scratchpad. The static params file is being prepared via the coordinator; if it's delegated here, this session builds it.
- 2026-10-18 · dev now has Anderson sweep parity (c3cca7e5; `IterationSchedule.n_iterations` + tail) and TOML sim-params configs (`python/tools/sim/config.py`, `configs/README.md`). `benchmark_static.py --sim-params` merges several .toml/.h5 sources left to right, plus repeatable `--set Table.key=value` (eb2662cb). A static params file can be a TOML layer.

# Task 3: Static (equilibrium) solves and a convergence benchmark

Status: **not started** · Branch `feature/gpu-static-solves` · Worktree `C:\git\pbat-static`
Read first: `MASTER.md`, `python/tools/sim/CLAUDE.md`, `source/pbat/sim/CLAUDE.md`.

## Goal

A clean way to set up and run **static equilibrium** solves with every GPU solver (Newton, VBD, AAAVBD,
Chebyshev, Anderson), as the basis of a convergence benchmark for the AAAVBD paper.

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
- initial configuration: the Dump's `x`, rest `X` plus a modal perturbation (reusing `elasticity.py`'s
  modes), or a random perturbation with a fixed seed,
- Dirichlet set: from the Dump, or from task 1's kinematic constraints evaluated at a frozen time,
- loads: gravity on/off, `f_ext` scaling,
- stopping: relative ‖∇E‖ tolerance, maximum outer iterations and sweeps, maximum GPU time.

Serialize it to HDF5 so cases are reproducible.

**Convergence metric.** Implement GPU `check_convergence` and a gradient-norm kernel (reuse
`gpu/gradient.py`, which already has an elastic gradient), covering the elastic, external and contact terms
over free DOFs. The same metric then serves dynamic solves too, replacing the fixed iteration counts.

**Benchmark runner** (`python/tools/sim/benchmark_static.py`): for each solver and case, run from the
identical initial state and record per sweep (or per Newton iteration) ‖∇E‖, E, the cumulative sweep count
and GPU time, using task 2's `profiling.scope` API. Write JSON; plot residual vs sweeps and residual vs GPU
time; publish as an HTML artifact.

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
2. Which convergence metric does the paper use: relative gradient norm, energy decrease, or displacement to a Newton reference?
3. Should static benchmarks include contact, or start contact-free?
4. Do you want CPU parity (a C++ static mode in `pbat.sim.algorithm`), or is GPU-only enough for the paper?
5. Which initial configurations: the modal ones from `elasticity.py`, or others you've used before?

## Progress

(update as you go)

# Solver accuracy vs residual-based termination: notes

Measurements on why a solver's stopping test (its residual) doesn't tell how accurate its time steps are, kept
for the paper's argument for error-to-reference comparisons (the benchmark's design, `benchmark.py`). Each entry
says what was measured, how, the numbers and what they show.

## 1. Early exit biases a frictionless slide by up to 30% (VBD family)

**Setup.** `data/scenarios/planes/CubeOnInfiniteInclinedPlane.h5`: a 10 cm, 1 kg cube (Young's modulus 1e6) on a
15° plane, `mu_f = 0`. Parameters: `configs/scenes/CapsulesIntoBowl.toml` (its contact settings: AL margin
`dmin` 1 cm, VertexSdf band 1 cm, TIMESTEP penalty; budgets: VBD family 10 outer x 20 sweeps, Newton 10 x 2;
`dt` 1e-2). The exact sliding acceleration is g sin 15° = 2.539 m/s². The measured acceleration is
`(v(t2) - v(t1)) / (t2 - t1)` of the mass-weighted center-of-mass (COM) velocity along the slope. With BDF1, a
step's velocity is (x_k - x_k-1) / dt, so frames written at every step (`--fps 100`) give the COM velocity
exactly:

```bash
python -m python.tools.sim.simulate --scene path/to/PhysicsBasedAnimationToolkit/data/scenarios/planes/CubeOnInfiniteInclinedPlane.h5 \
    --sim-params python/tools/sim/configs/scenes/CapsulesIntoBowl.toml --set Contact.ContactDynamics.mu_f=0 \
    --solver AAAVBD --duration 1.5 --fps 100 -o incline.AAAVBD.h5
# early exit off (the full budget every step): add
#   --set Solver.Convergence.rstol=0 --set Solver.Convergence.astol=0 --set Solver.AAAVBD.gtol=0   (Solver.VBD.gtol=0 for VBD)
```

**Results** (2026-10-22). The cube lands by step 10 and slides from then on:

| Solver, stopping rule | a, steps 50-100 (m/s²) | a, steps 100-150 (m/s²) | Sweeps per step, mean | Worst gradient ratio |
|---|---|---|---|---|
| Newton (10 x 2) | 2.539 | 2.539 | | 7.3 |
| AAAVBD, early exit (defaults: rstol 1e-2, astol 0.5) | 1.769 (-30%) | 2.753 (+8%) | 42 of 200 | 1.0 (all pass) |
| VBD, early exit (defaults) | 2.627 (+3%) | 2.822 (+11%) | 21 of 200 | 0.96 (all pass) |
| AAAVBD, full budget | 2.541 | 2.543 | 200 | |
| VBD, full budget | 2.539 | 2.540 | 200 | |

With the full budget both VBD-family solvers are exact. With the default early exit every step passes the
stationarity test, yet the slide is 30% too slow or 11% too fast for a whole second. The error comes from the
stopping rule, not the budget. Conversely, Newton fails its outer test on some steps (gradient ratio up to 7.3),
yet its slide is exact.

**Why the test allows it.** The test bounds the mass-normalized residual RMS_M(M⁻¹g) by max(rstol × RMS_M(x̃ -
xᵗ), astol × |a| × h²) (`gpu/vbd/convergence.py`). Divided by h², that's a residual acceleration (unbalanced force
per mass) of max(rstol |v| / h, astol |a|): about max(1.3, 4.9) m/s² for this slide at 1.3 m/s. Both terms exceed
the 2.5 m/s² being simulated. A rigidly sliding body's residual is nearly uniform, so it goes straight into the
acceleration. The relative term also grows with speed, because the free-flight displacement x̃ - xᵗ is mostly
rigid translation. The test is a stationarity tolerance for the step's deformation, not an accuracy bound on slow
rigid motion such as sliding, rolling or settling.

**For the paper.** Residual ratios below 1 don't certify accurate trajectories, and two solvers with equally
"converged" steps can differ by tens of percent in the motion that matters. That's why the benchmark measures
error to a reference rather than residuals.

## 2. GPU Newton stalls above the VBD family's residuals on contact-rich scenes

**Residuals at the tuned budgets.** CapsulesIntoBowl (`configs/scenes/CapsulesIntoBowl.toml`: VBD family 10 x 20
sweeps, Newton 10 outer x 2 Newton steps, `mu_f` 0.2, default tolerances rstol 1e-2 and astol 0.5), 6 s, 601 steps
(task 2's production runs, 2026-10-22). Final gradient ratio to the tolerance, from the frames' `solve_status`:

| Solver | Converged steps | Gradient ratio p50 | p90 | max |
|---|---|---|---|---|
| VBD | 248 | 1.4 | 4.2 | 9.0 |
| AAAVBD | 218 | 1.6 | 5.5 | 15 |
| Chebyshev | 256 | 1.4 | 4.6 | 7.7 |
| Anderson | 234 | 1.5 | 4.8 | 7.6 |
| Newton | 40 | 5.9 | 13 | 29 |

**A larger Newton budget doesn't help.** From AAAVBD's step-100 state, `mu_f = 0`, Newton with 50 outer x 10 Newton
steps, CG rtol 1e-7 (cap 4000), rstol 1e-3, astol 1e-2 and ftol 0.1:
- No subproblem converged within its 10 Newton steps. There were no line-search backtracks, and CG converged (~400
  iterations, not capped).
- The outer gradient ratio fell from ~1000 to ~160 in 10 outer iterations, then stayed at 156-167 for the next 40.
  With these tolerances 10-50x tighter than the defaults, that's about 3-16 at the defaults, a floor at the level of
  the tuned budget's.
- The feasibility ratio alternated between ~1.05 and ~1.17 from one outer iteration to the next: the contact set
  flip-flops.
- The steps took ~60 s each and never converged.

The floor isn't float32's: the VBD family gets lower residuals with 200 sweeps. A likely cause is the augmented
Lagrangian's non-smoothness for Newton: the slack is held fixed for each Newton step, so contacts in the band act as
two-sided springs whose activity changes between iterations (the one-sided AL idea, TODO in `gpu/newton/solver.py`).

**For the paper.** A "converged" second-order reference isn't available by just raising Newton's budget on
contact-rich scenes. Its residual floor has to be checked, and a reference's own uncertainty should be reported
(e.g. the distance between two independent references).

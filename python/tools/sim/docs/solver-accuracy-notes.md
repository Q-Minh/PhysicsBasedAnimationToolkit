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

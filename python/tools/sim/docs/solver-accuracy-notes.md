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

## 3. High-budget references: a shared residual floor, and GPU Newton dissipates

**Setup.** CapsulesIntoBowl, `mu_f = 0`, 10 steps each from AAAVBD's checkpointed states at 1 s and 3 s (task 2's
production run). Tolerances are 100x tighter than the defaults (rstol 1e-4, astol 5e-3), so nothing exits early and
the reported ratio / 100 is the ratio at the defaults exactly (2026-10-22):

| Configuration | Gradient ratio at the defaults, 1 s (p50 / max) | 3 s (p50 / max) | s/step |
|---|---|---|---|
| Newton, 100 outer x 1 Newton step | 8.6 / 9.1 | 5.2 / 5.6 | ~7-11 |
| AAAVBD, 10 x 200 sweeps | 7.3 / 9.0 | 6.5 / 7.3 | ~1.2-1.5 |
| VBD, 10 x 200 sweeps | 7.2 / 8.5 | 6.8 / 7.4 | ~1.0-1.2 |

With 10-100x the tuned budgets, all three end at ~5-9x the default tolerance. Newton's ratio still creeps down
over its outer iterations at 1 s (~33, 18, 11, 8 at outer iterations 1, 10, 50, 100) and is flat at ~5 from outer
iteration 10 at 3 s.

**Distances between the references** (mass-weighted RMS; the reference steps move the nodes ~23 mm at 1 s and ~8 mm
at 3 s per step):

| Pair | 1 step, 1 s | 1 step, 3 s | 10 steps, 1 s | 10 steps, 3 s |
|---|---|---|---|---|
| VBD vs AAAVBD | 0.043 mm (0.19%) | 0.004 mm (0.05%) | 0.44 mm | 0.18 mm |
| Newton vs either | 0.20 mm (0.85%) | 0.058 mm (0.77%) | 5.4 mm | 0.85 mm |

The capsules carry the Newton difference (14.4 mm after 10 steps at 1 s; tubes 0.8 mm, bowl 0). It's systematic:
after 10 steps, Newton's capsules have 1.7% (1 s) and 1.2% (3 s) less kinetic energy than both VBD-family
references, without friction. Newton dissipates contact motion that the VBD family keeps, consistent with Newton's
slower capsules in side-by-side runs.

**For the paper.** No solver reaches the residual tolerance at these states, so a "converged" reference here means
agreement between two independent high-budget solvers (VBD and AAAVBD agree within 0.05-0.2% of a step), not a
residual test. Residuals at the same level (Newton ~5-9, VBD family ~6-9) come with systematically different motion.

## 4. The residual floor is at a few contacts that end the step without a multiplier

**Setup.** As in 3: one VBD 10 x 200 solve (frictionless) from the 1 s and 3 s states. Then a per-node breakdown of
the outer test's Lagrangian gradient, in residual acceleration |g_i| / (m_i h²). The default tolerance is
4.9 m/s² (2026-10-22, scratch diagnostic).

- **Not float32.** Repeated evaluations differ by 3e-5 m/s². eps × the sum of the absolute values of all terms
  per node bounds rounding at 0.005 m/s², a ratio of 0.001, against the floor's 6-8.
- **Only at contacts.** Capsule contact nodes hold 96.5% (1 s) and 99.3% (3 s) of Σ |g_i|² / m_i. The rest is at
  tube and bowl contact nodes, and 0.0% at nodes without contacts. Those are converged: capsules p50 0.03 m/s²,
  tubes p50 0.16 m/s², bowl 0.002 m/s².
- **A heavy tail.** At capsule contact nodes the residual's p50 is 0.4-0.6 m/s², but its p99 is 1300-2000 and its
  max 3500-3900 m/s². Elastic (186 m/s² RMS) and contact (178) forces nearly cancel, leaving 40 (1 s) and 30 (3 s).
- **Pressed contacts without a multiplier.** Of 3330 (1 s) and 3771 (3 s) vertex-face contacts, 75 and 71 (~2%)
  end the step pressed into the AL margin (c_n < 0, 0.2-2 mm into the 1 cm margin) with lambda_n = 0. Their nodes
  hold 97.6% and 93.9% of the residual. The worst nodes are all such vertices. Most of them had the same
  (vertex, face) key at the step's start, so they aren't new contacts.

The dual update gives every pressed contact lambda_n > 0 (`contact.dynamics._apply_dual_update`). A pressed contact
with lambda_n = 0 after the step's last outer iteration therefore (re)entered the constraint set at the final
detection, after the last dual update: in the sweeps the AL penalty held it, but the test's Lagrangian force
-lambda grad c is 0. A contact that is dropped and found again within a step restarts at lambda = 0. That suggests
fast-sliding capsule vertices flicker between vertex-face, vertex-edge and neighbouring-face pairs across the 10
detections (not verified per detection). So the floor is mostly an artefact of the measure: the test evaluates
the multipliers of a constraint set the step's solve never used.

**For the paper.** Every solver's "unconverged" ratio of 5-9 on this scene comes from ~2% of contacts, not from
unresolved elasticity. A residual-based comparison would rank solvers by how their contact sets flicker at the last
detection. Possible fixes, all of which change the shared stopping test (not done): test before the final
re-detection; carry lambda across a vertex's pair changes; or give contacts new at the final detection
lambda = sigma max(-c_n, 0).

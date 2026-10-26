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
detection. Possible fixes, all of which change the shared stopping test: test before the final re-detection; carry
lambda across a vertex's pair changes; or give contacts new at the final detection lambda = sigma max(-c_n, 0).

**Update (2026-10-22).** The second fix is in: a vertex contact whose partner feature changes to one sharing a mesh
vertex keeps its multipliers (`gpu/contact/constraints.py`, `carry_across_features`; d34e6341), always on. Jumps to a
non-adjacent feature and edge-edge contacts still start at lambda = 0. CapsulesIntoBowl's floor hasn't been
re-measured with it, and DragonImpact's leftover residual (entry 7, mostly pillar-pillar contacts) was measured with it
on.

## 5. AAAVBD loses its edge at impacts: stale stencil gradients after each outer iteration

**Question** (maintainer). AAAVBD's stencil gradient augmentation assumes a smooth energy and gradient. Does it lose
its advantage over VBD, Chebyshev and Anderson at abrupt high-to-low velocity changes, such as the capsules'
impacts?

**Production runs (CPU, existing data).** These are task 2's runs at `mu_f = 0.2`, 591 steps per solver (2026-10-22).
Each step's abruptness is the largest non-gravity acceleration of a capsule's center of mass, |Δv_com/Δt − g|,
taken from the 60 fps frames. Bins use the pooled quantiles:

| Abruptness (m/s²) | Sweeps (AAAVBD / VBD / Chebyshev / Anderson) | Converged % | Gradient ratio p50 |
|---|---|---|---|
| < 17 (lower half) | 114 / 145 / 99 / 105 | 68 / 63 / 73 / 65 | 0.88 / 0.97 / 0.93 / 0.91 |
| 43-76 (p75-p90) | 198 / 194 / 197 / 194 | 1 / 10 / 5 / 6 | 4.7 / 3.7 / 4.1 / 3.9 |
| 76-98 (p90-p97) | 200 / 199 / 194 / 192 | 0 / 3 / 5 / 7 | 6.3 / 4.7 / 5.1 / 4.7 |

At impacts AAAVBD ends with the worst residual. These runs are only weak evidence:
- each solver follows its own trajectory;
- the 200-sweep budget saturates;
- the residual carries entry 4's contact artefact.

**Controlled one-step test.** The setup:
- **States.** 18 states of AAAVBD's production run, regenerated by resuming its step-50 checkpoint and stepping
  continuously; resuming is bit-exact for the VBD family. They span the fast slide, the impacts and the rocking in
  the bowl.
- **Run.** From each state, every solver ran one step on the production schedule (10 outer iterations × 20 sweeps)
  with early exit off. The error was recorded after every sweep.
- **Error.** The M-norm distance to the step's own converged solution x*, VBD with 20 × 200 sweeps. VBD 10 × 200
  lies within 1e-4 to 1e-2 of the initial error e0 from x*, and AAAVBD 10 × 200 within 1.5e-3 to 1.2e-2. So AAAVBD
  converges to the same solution given enough sweeps, and only thresholds of 0.1 e0 and above are used.
- **Solver buffers.** AAAVBD started from its checkpointed buffers, with betaG warm-started per its TIMESTEP policy
  (maintainer: never reset per step). The other solvers' cross-step buffers started fresh: their histories are
  rebuilt every step, and only AAAVBD's betaG carries across steps.
- **Abruptness.** The same measure, computed exactly from vᵗ and the converged step.

Graph-mode step time at this budget: AAAVBD 186 ms, VBD 173, Chebyshev 172, Anderson 182.

Sweeps to reach 0.1 e0, relative to VBD:

| | Smooth steps (< 20 m/s², 7 states) | Impacts (> 40 m/s², 11 states) |
|---|---|---|
| AAAVBD | 4.2x (3.1-4.6) | 1.8x (from not reaching it to 2.8x) |
| Chebyshev | 2.2x | 2.0x (1.7-2.1) |
| Anderson | 2.6x | 1.3x |

The error after the full 200 sweeps, e/e0 (median):

| | Smooth steps | Impacts |
|---|---|---|
| AAAVBD | 0.003 | 0.054 (max 0.22) |
| Chebyshev | 0.006 | 0.012 |
| Anderson | 0.010 | 0.019 |
| VBD | 0.027 | 0.033 |

AAAVBD's error relative to VBD's grows with abruptness: Spearman correlation +0.79 at 20 sweeps and +0.83 at
60 sweeps. Chebyshev's ratio is flat at about 2x. At 6 of the 11 impact states AAAVBD ends further from x* than
plain VBD; there its error bottoms out at 0.06-0.2 e0 and then grows back.
Figure: `data/solver-accuracy/CapsulesIntoBowl.OneStepConvergence.pdf`, with data in the `.json` next to it:
- (a) the error vs sweeps at a smooth state;
- (b) the same at a stagnating impact state;
- (c) the error ratio to VBD at 60 sweeps vs abruptness.

**Mechanism.** AAAVBD's error jumps up by 1.5-5x at the first sweep of an outer iteration, at all 11 impact states
and at none of the smooth ones. The other solvers show no such jump, apart from one each for Chebyshev and Anderson.
At the stagnating states the jumps fall on every second outer iteration (sweeps 21, 61, 101, ...), consistent with
the contact-set flip-flop of entry 2.

At that sweep the augmentation sums the neighbours' stored gradients g_k. Those come from the previous subproblem,
before its dual update and re-detection changed the objective. betaG adapts only from the second sweep of a
subproblem (kp > 0). C++ `vbd::detail::ComputeStencilGradientAugmentation` does the same, and the GPU mirrors it.
At impacts the multipliers and the contact set change the most between outer iterations, so the stale stencil
gradient pushes the iterate away.

betaG doesn't collapse: over the 495 steps, the capsules' surface mean stays at 0.87-0.90 (p10 about 0.80) and the
interior mean at 0.98. Its Spearman correlation with abruptness is −0.31. The shrink rule doesn't switch the
augmentation off where the objective jumps.

**Verdict: supported, with a refined cause.** AAAVBD keeps a 4x lead over VBD on smooth steps (2x over Chebyshev).
At impacts the lead drops to 1.8x, below Chebyshev's 2x, and at half of them AAAVBD falls behind plain VBD. The
non-smoothness that hurts it is the change of objective between outer iterations (multipliers, contact set). The
velocity jump within a step's smooth subproblem doesn't.

Possible fixes, not tried (a TODO in `gpu/vbd/aaasolver.py`):
- skip the augmentation at a subproblem's first sweep (kp = 0), as for k = 0;
- or refresh g_k at the subproblem's start.

**Follow-up: the two fixes** (Q8, 2026-10-22). The same 18 states and protocol, with AAAVBD changed in two ways:
- A: no augmentation at any subproblem's first sweep (kp = 0 for every k, as for k = 0);
- B: a pass that stores every vertex's current g_k, x_k and |H| before the first sweep of each later subproblem.

The unchanged AAAVBD reproduced the numbers above exactly.

| | Sweeps to 0.1 e0 vs VBD, smooth / impacts (median, impact min) | e/e0 at 200 sweeps, impacts (median / max) | Impact states behind VBD | Error jumps at impacts | ms per step |
|---|---|---|---|---|---|
| Current | 4.2x / 1.84x (never) | 0.054 / 0.22 | 6 of 11 | 39 | 182.7 |
| A (skip) | 4.2x / 2.39x (1.10x) | 0.014 / 0.027 | 0 | 1 | 181.9 |
| B (refresh) | 4.4x / 2.10x (never) | 0.069 / 0.27 | 6 | 38 | 185.7 |

A removes the sawtooth and the stagnation at no cost, and keeps the smooth-step lead. At impacts its 2.4x is above
Chebyshev's 2.0x, and its error after 200 sweeps is close to Chebyshev's 0.012.

B doesn't help, and its extra pass costs 2%. The harm isn't the staleness of g_k: it is applying the augmentation at
the first sweep after the objective changed, before betaG can adapt.

## 6. GPU Newton's Coulomb limit: why the per-step freeze stays

**Question.** GPU Newton caps the augmented friction force at a Coulomb limit
C = mu_f max(lambda_n − sigma_n c_n, 0). C is frozen per contact for the time step, at its first positive estimate
(`contact.constraints.frozen_friction_limit`). At an impact that first estimate comes from a deeply penetrating early
iterate, 5-27x the step's final normal force, so impacts get too much friction. Three alternatives were tried
(2026-10-22). The records are on the branches `feature/newton-friction-limit` (F1, F1+A) and
`feature/newton-friction-subproblem-freeze` (F3), both unmerged:
- **F1:** C = mu_f max(lambda_n, 0), the multiplier of the current outer iteration, fixed within the subproblem.
- **F1+A:** F1, plus the live estimate mu_f max(−sigma_n c_n, 0) for contacts with lambda_n = 0.
- **F3:** the per-step freeze's estimate, re-frozen at each subproblem's start.

**Setup.**
- **Incline.** `data/scenarios/planes/CubeOnInfiniteInclinedPlane.h5` (15°), with CapsulesIntoBowl's contact settings
  and Newton 10 outer iterations x 2 Newton steps. Accelerations are fitted over the second and third thirds of each
  run. The impact case lifts the cube 10 cm and launches it at 3 m/s along the slope and 2 m/s into the plane.
- **Capsules.** Newton resumed from AAAVBD's CapsulesIntoBowl checkpoint at step 200, for 15 steps of the 2.1-2.4 s
  impacts.

Sliding acceleration (m/s²) on the incline, with Coulomb in brackets, and the impact window's friction ratio (1 is
Coulomb; AAAVBD 1.01):

| Case | Per-step freeze (kept) | F1 | F1+A | F3 |
|---|---|---|---|---|
| From rest, mu_f 0.2 (0.644) | 0.52 | 0.90 | −0.09 (sticks) | 1.16 |
| From rest, mu_f 0.3 (sticks) | sticks | sticks | −0.39 | sticks |
| Sliding at 3 m/s, mu_f 0.2 (0.644) | 0.67 | 1.02 | −1.47 | 0.85 |
| Sliding at 3 m/s, mu_f 0.3 (−0.30) | −0.56 | +0.43 (never stops) | −2.65 | +0.02 (never stops) |
| Sliding, 1 outer iteration, mu_f 0.2 / 0.3 | 0.68 / −0.32 | 0.33 / −0.94 | 0.11 / −0.36 | as the freeze |
| Impact window friction ratio | 1.25 | 0.90 | 0.88 | 1.02 |
| Steps passing the outer test, from rest at 0.2 | 147 / 150 | 143 | 6 | 143 |

No variant had line-search failures. With one outer iteration per step, F3 is the per-step freeze.

**Mechanisms.**
- **F1 under-frictions sliding.** Part of the normal load is carried by the penalty term sigma_n (−c_n), and
  lambda_n lags it, so mu_f lambda_n is too low. It also runs loaded contacts without friction until their first dual
  update: on the capsules, 18% of loaded contact-subproblems at 10 outer iterations and 29% at 1 (a whole step).
- **A makes Newton stall.** The live limit differs between contacts with and without a multiplier, and it makes the
  line-search energy inexact. The outer test then fails almost every step, and the stalled iterates over-friction.
- **F3 settles into a slower regime.** The per-step freeze and F3 are identical while steps take one outer
  iteration. Once F3's steps need two (from step ~58, every step from 75), each step runs the same cycle (sums over
  the cube's contacts, in units of its normal load):
  - the first subproblem freezes C/mu_f = 1.07;
  - its dual update dips lambda_n from 0.94 to 0.80;
  - the second subproblem re-freezes at 0.79;
  - the last dual update brings lambda_n back to 0.94.

  The step's motion comes from the last subproblem, so the friction ratio settles at 0.65-0.75. It is self-reinforcing:
  weaker friction means faster sliding, the outer test fails after the first subproblem, and a second one runs.
- **The per-step freeze** keeps the step's first, larger estimate. It over-estimates the normal force at impacts
  (1.25), and in steady sliding it stays within 0.8-1.04x Coulomb's acceleration.

**Decision** (maintainer, 2026-10-22, provisional). Keep the per-step freeze: "for Newton, we best over-estimate the
normal contact force". Every alternative that tracks the normal force within a step, from the multiplier or from
re-estimates, under-estimates it while sliding, through the multiplier's lag or a dip between outer iterations. Too
little friction lets bodies slide that should stop. Too much, at an impact, mostly shortens a slide.

## 7. More outer iterations raise the residual on pillar-pillar contact: the penetration flips every outer iteration

**Setup.** The DragonImpact scene draft (`data/scenarios/ArmadilloSpaghetti.DragonImpact.h5`, not yet a `SCENES`
entry): four stacked spaghetti pillars, the armadillo and two dragons, with `configs/scenes/ArmadilloSpaghetti.Fall.toml`
at 10 substeps. One traced step of AAAVBD (Inverse vertex solve) from the production run's states at 6.0 s and 6.8 s,
at outer iterations x sweeps per substep of the same total, early exit off (rstol and astol at 1e-6 x the defaults,
so the ratios below are the logged ones x 1e-6, i.e. at the default tolerances). The ratio is the outer test's
gradient ratio at each substep's end; substep 0 at 6.0 s is left out (its resumed checkpoint carries a converted AL
state). 2026-10-24, scratch diagnostic.

| Outer x sweeps per substep | Finalizes per step | Ratio at 6.0 s, median / max | Ratio at 6.8 s, median / max | End penetration ratio, max (6.0 / 6.8 s) |
|---|---|---|---|---|
| 1 x 100 (production) | 10 | 689 / 1614 | 1251 / 1261 | 0.55 / 0.53 |
| 1 x 1000 | 10 | 679 / 1573 | 1199 / 1208 | not recorded |
| 2 x 50 | 20 | 704 / 888 | 1390 / 1462 | 0.66 / 0.66 |
| 4 x 25 | 40 | 715 / 759 | 1435 / 1499 | 0.64 / 0.73 |
| 5 x 20 | 50 | 800 / 1034 | 1487 / 1499 | 0.61 / 0.49 |
| 2 x 500 | 20 | 720 / 869 | 1370 / 1411 | 0.66 / 0.63 |
| 10 x 100 | 100 | 704 / 786 | 1451 / 1515 | 0.69 / 0.54 |

Against one outer iteration with the same sweeps per substep (1 x 100 for the 100-sweep splits, 1 x 1000 for 2 x
500 and 10 x 100), every split into more outer iterations raises the median ratio: by 2-16% at 6.0 s and 11-21% at
6.8 s. The one-outer budgets' larger maxima at 6.0 s are a single substep (1614 and 1573). More sweeps in one outer
iteration help a little (1 x 1000 vs 1 x 100: -1.5% at 6.0 s, -4% at 6.8 s).

**Per outer iteration** (6.8 s, 10 x 100, substep 0; start, then after each outer iteration's finalization and
multiplier update):
- ratio 2091, 1264, 1313, 1322, 1344, 1365, 1377, 1359, 1384, 1394, 1402;
- penetration ratio 0.74, 0.47, 0.37, 0.46, 0.37, 0.48, 0.36, 0.45, 0.36, 0.56, 0.36.

The first outer iteration reaches the plateau. The next nine raise the ratio from 1264 to 1402 (one dip), while
the penetration alternates between two levels (~0.36 and ~0.45-0.56) from one outer iteration to the next. The same pattern
appears at 6.0 s (substep 1: penetration 0.49, 0.66, 0.44, 0.66, 0.35, 0.62, ...). The multipliers were not
recorded, so the cycle is seen in the penetration, not in lambda itself. It is consistent with the contact-set
flip-flop of entries 2 and 5, here on pillar-on-pillar contact.

**Where.** At each traced step's end, 87-96% of Σ |gᵢ|²/mᵢ is on pillar nodes within 1.5 cm of another pillar, in
every configuration. The median |M⁻¹g| there is 3.7-5.1 mm with one outer iteration per substep and 8-22 mm with
2-10.

**For the paper.** On this scene the residual plateau is not the AL penalty force that more multiplier updates
would remove: extra outer iterations cost a finalization each (dual update and re-detection) and leave the step
further from stationarity. The production budget stays at 1 outer iteration x 100 sweeps per substep.

**The plateau isn't a stall** (2026-10-24, for a stall exit considered and not pursued; TODO in
`gpu/vbd/convergence.py`). Traced per 20-sweep check, 1 x 100 per substep, early exit off:
- From sweep ~41-61 the residual changes by at most ~0.1% per window (VBD's rises slightly), but the iterate keeps
  moving: the largest window movement, on armadillo and dragon nodes, halves every window, and consecutive
  windows point the same way (mass-weighted cosine +0.85...+1 over the top 1% movers). That is converging creep that
  the pillar-contact residual doesn't see, so stopping on the plateau would cut real convergence.
- A movement test against time-step scales (|a| h², the free-flight displacement) stalls stiff bodies first: a sweep
  moves a vertex by about H_i⁻¹ g_i, (1 + (h ω)²) less than the residual's M⁻¹ g. On the 1e8 bending beam
  (`small_beamY1e8.h5`, h = 1e-2, VBD 1 x 4000) the window movement is 1e-2 |a| h² by sweep 200, while the residual
  ratio falls from 494 to 8 over 4000 sweeps.

## 8. Unexplained: a gap of exactly −dmin, then a reversed normal, at a vertex-edge to vertex-face hand-off

An observation without an explanation yet; nothing was changed over it.

**Setup.** The DragonImpact production run (`configs/scenes/ArmadilloSpaghetti.DragonImpact.toml`: AAAVBD, 10 substeps
of 1 x 100, dmin = 1 cm, ftol 0.5, so the penetration tolerance is 5 mm). Steps 481-490 replayed eagerly from the
4.8 s checkpoint, with the convergence check's per-pair penetration (dmin minus the pair's gap) recorded per
substep; the replay matched the run bit for bit (max |dx| = 0 at step 481). 2026-10-25, scratch diagnostic.

**What was seen.** Two penetration spikes, at the same armadillo edge node and one pillar2 vertex each, shortly after
the dragons hit:
- Step 482, substep 8: one vertex-edge pair, pillar2 vertex 101870 against armadillo edge 201706-201708, has a gap of
  −0.009998 m (penetration ratio 4.0, i.e. −dmin to 0.02%). The next substep, the same vertex is in a vertex-face
  pair against an adjacent armadillo face, 1.6 mm outside it (ratio 1.68), with a normal of (−0.80, −0.47, −0.36)
  against the edge pair's (0.83, 0.55, 0.04). The vertex moved about 1.1 cm in that substep (11.3 m/s).
- Step 484, substep 1: pillar2 vertex 118799 against armadillo edge 201708-201734, gap −0.009974 m (ratio 3.99).
  The next substep it is a vertex-face pair at −7.8 mm (ratio 3.55) with a normal of (−0.82, −0.56, 0.09) against the
  edge pair's (0.53, 0.84, 0.13), then every pair is below 0.31 from substep 3 on.

The run went on to 18 s with no other step over 2 (the watcher stopped on the first spike; the stop was later relaxed
to 10 consecutive steps over 2).

**Not explained.** The gap is measured along the pair's normal stored at detection, with its barycentric coordinates
held (`_ve_penetration` in `gpu/vbd/convergence.py`). A gap within 0.3% of −dmin, twice, suggests a vertex near the
activation distance measured against a normal of the opposite orientation rather than a 2 cm overlap, but that was
not checked (the stored normals and the pairs' unsigned distances were not recorded). Whether the vertex really
crossed the armadillo's surface is also open.

## 9. Low budgets at 10 substeps fail through the twist or the impact, without warning from the residual

**Setup.** The DragonImpact production config (`configs/scenes/ArmadilloSpaghetti.DragonImpact.toml`: 10 substeps of
1 outer iteration x 100 sweeps, launch at 4.49 s), with only the sweep budget lowered,
`--set Solver.<S>.n_subproblem_max_iters=20` or `40`, on the code of the 100-sweep runs (c649f50a). Runs started at
t = 0 or resumed from each solver's own 100-sweep checkpoint at 4.4 s (step 440), so AAAVBD's betaG stayed warm
started. A run stopped on a non-finite state, on dropped contact pairs, or on a penetration ratio over 2 for 10
consecutive steps; inverted elements were only logged (2026-10-26).

| Budget, start | Solver | End | Where | Before it, in the frames |
|---|---|---|---|---|
| 10 x 20, t = 0 | AAAVBD | non-finite at step 70 (twist) | armadillo underside | 10-18 inverted armadillo elements from step 60 (0 at 100 sweeps) |
| 10 x 20, t = 0 | VBD | dropped pairs at step 67 (twist) | armadillo underside against pillars 1-3 | 2-5 inverted from step 62, 59 and a penetration ratio of 4.98 at step 66 |
| 10 x 20, 4.4 s | AAAVBD | non-finite at step 500 (impact) | dragonL's tail the fastest feature | nothing: inversions as at 100 sweeps, feasibility ratio 0.42-0.59 (0.15) |
| 10 x 20, 4.4 s | VBD | dropped pairs at step 480 (impact) | dragonL landing on the armadillo underside | 54 m/s at step 479 (11.8 m/s at 100 sweeps) |
| 10 x 40, 4.4 s | AAAVBD | non-finite at step 494 (impact) | dragonR hitting pillar 1 | step 493: 4 inverted in pillar 1 and 3 in dragonR at the contact |
| 10 x 40, 4.4 s | VBD | 18 s, every gate passed (entry 12) | | |

**How a run fails.** An eager replay of AAAVBD's twist failure (10 x 20, from its step-60 checkpoint, a convergence
check after every sweep, early exit off) reproduced the run's frames bit for bit:
- Step 68's last two substeps already have 15 and 18 inverted armadillo elements and 16 and 34 m/s.
- Step 69's first substep starts with normal pair counts (2617 vertex-face pairs against a capacity of 166,435), then
  diverges within its sweeps: the largest displacement since the substep's start is 5.6 cm after sweep 2, 27 cm after
  sweep 3 and 21 m after sweep 4, at armadillo surface nodes next to the inverted elements (1.0 and 1.9 cm from one
  for two of them); 57 elements are inverted after sweep 2 and 4439 after sweep 10.
- Only the substep's final detection, on the exploded positions, overflows (294,679 vertex-face pairs requested). The
  first non-finite value comes in the next substep, after sweep 19.

So the inversions come first, then a divergence within one substep's sweeps, and the contact overflow last. The other
failures show the same end: a step requesting 2.6-5.2x the vertex-face capacity, which took 186-1933 s against 3-15 s
for the step before.

**The residual didn't warn.** Before the failing step the gradient ratios were in the 100-sweep runs' range: AAAVBD
from 4.4 s at 20 sweeps 195-300 (about the same at 100), VBD 150-450 (140-290), AAAVBD at 40 sweeps 155-340 (143-400).
Only the failing step itself jumped (AAAVBD at 40 sweeps: 6480 at step 493, against 309).

**Relation to entry 5.** Entry 5's error jumps come from the stencil augmentation at the first sweep of a subproblem
after an outer iteration's dual update and re-detection. These runs have one outer iteration per substep, and since
8cce9b3d (in the code these runs used) AAAVBD skips the augmentation at every subproblem's first sweep. The failures
don't single out AAAVBD either: VBD failed first in two of the three pairs (steps 67 vs 70, 480 vs 500), AAAVBD in the
third (494, while VBD reached 18 s). betaG (read only) ran higher at 20 sweeps: in the step-60 checkpoints, the
armadillo's surface median was 0.963 against 0.876 at 100 sweeps (p95 1.0 against 0.946), and the interior median
0.995 against 0.984. Nothing unusual came before the blow-up. After it, gammadown lowered the surface p5 to 0.74 and
then 0.48. betaG at the runaway nodes themselves wasn't recorded.

**For the paper.** On this scene a budget below 10 x 100 isn't safe through the twist and the impact, for VBD and
AAAVBD alike, and the stopping test's residual doesn't flag it. A failure is local (a few inverted elements, or one
contact) and becomes a divergence within one substep. The five failures happened at three places: the armadillo
underside (entry 10), dragonL's tail and dragonR against pillar 1.

Data (`data/scenarios/runs/`): `ArmadilloSpaghetti.DragonImpact.<AAAVBD|VBD>.ShortBudget.*` (20 sweeps from t = 0),
`.ShortBudget.From4.4s.*` (20 sweeps from 4.4 s), `.MidBudget.From4.4s.*` (40 sweeps from 4.4 s), and the replay's
per-sweep records, `ArmadilloSpaghetti.DragonImpact.AAAVBD.ShortBudget.Diagnostic.h5`.

## 10. A weak spot on the armadillo's underside: a finer mesh that low budgets leave least converged

**Question.** Three of entry 9's five failures ran away at the same few armadillo nodes, with either solver and from
t = 0 or from 4.4 s. Is that part of the mesh defective?

**Region.** The armadillo nodes within 10 cm of the rest centroid (−0.83, 1.17, −0.73) of the 12 nodes that ran away
first in those failures (186045, 186086, 186314, 186410, 186426, 186488, 186519, 187265, 187351, 187391, 215950,
221111): 614 nodes (408 on the surface) in 2446 of the armadillo's 162,385 elements, on its underside. It isn't held
(the nearest held node is 45 cm away), and its material is the armadillo's uniform one. CPU analysis of the rest mesh
and the saved frames (2026-10-26, scratch script).

**Mesh.** Rest-mesh percentiles, region against the rest of the armadillo:

| | p1 | p5 | p50 | p95 |
|---|---|---|---|---|
| Element volume (cm³) | 0.016 / 0.034 | 0.032 / 0.068 | 0.15 / 0.30 | 0.50 / 2.2 |
| Min dihedral angle (°) | 19.7 / 26.0 | 29.9 / 33.4 | 46.4 / 47.6 | 59.6 / 60.0 |
| Radius ratio (3 r_in / r_circ) | 0.33 / 0.46 | 0.55 / 0.60 | 0.79 / 0.81 | 0.94 / 0.94 |

- The region is meshed about 2x finer: half the median element volume, a quarter of the p95.
- Its low-quality tail is somewhat worse, but it has no outlier: its worst element (12.9°, radius ratio 0.19) is better
  than the armadillo's worst (11.1°, 0.11).
- Smaller elements of the same material make it the stiffest part of the armadillo relative to its mass. As a proxy,
  a node's summed element stiffness μ h_e (h_e an element's size) over its mass has p50 5.3e7 against 3.4e7 and p95
  1.5e8 against 8.2e7, about 1.6-1.8x. Per sweep it is therefore the slowest part of the armadillo to converge.

**Load in the 100-sweep runs.** The region is barely strained and rarely in contact:
- Contact with the pillars (every 40 steps through the twist, then every 10 steps to 5.0 s): AAAVBD's region never
  comes within 15 mm of a pillar (39-379 mm). VBD's touches pillars 0-2 around step 80 only (10 mm), then stays
  86-368 mm away.
- Deformation: in both runs the largest principal stretch is at most 1.41 at steps 60-69 of the twist and 1.02 at
  4.4 s, against 1.5-4.7 in the rest of the armadillo, which also holds its known inverted elements. At 4.8 s, VBD's
  dragonL lands on the region (up to 1.51; J ≥ 0.81); AAAVBD's dragonL lands 33 cm away.

**At low budgets** the region deforms much more before each failure:
- AAAVBD, 20 sweeps from t = 0: largest stretch 1.91 at step 60 (1.11 at 100 sweeps), 22 at step 69. By then the
  region has drifted ~35 cm from its 100-sweep position and presses on pillars 0-1 (105 surface nodes within 10 mm).
- VBD, 20 sweeps from t = 0: 1.07 at step 60, 25 at step 66, pressed on pillars 1-3 (170 surface nodes).
- VBD, 20 sweeps from 4.4 s: 1.22 at step 470, 63 at step 479, under dragonL's landing.
- AAAVBD from 4.4 s, at 20 and 40 sweeps, failed elsewhere (dragonL's tail, dragonR against pillar 1); the region
  stayed at ≤ 1.07 up to the failure at 20 sweeps.

**Reading.** A weak spot that low budgets expose, not a defective element: the finer mesh makes the region the least
converged part of the armadillo at a fixed budget, and it breaks when a load reaches it (pillar contact during the
twist, a dragon's landing), while at 100 sweeps it is neither strained nor loaded. Remeshing it was not tried.

## 11. DragonImpact's residual floor sits on persistent contacts, at the same level for every solver

**Setup.** One traced 10 x 100 step each, eager, with the convergence check after every sweep and early exit off,
from AAAVBD's production states at 4.85 s, 7.0 s and 10.0 s, for AAAVBD and for VBD from the same state. The traced
AAAVBD steps reproduced the production frames bit for bit. Ratios are the production gradient ratio to the default
tolerance, medians over the step's 10 substeps. At every detection the trace also classified each vertex contact:
the same (u, v) key as before, carried to an adjacent feature by `carry_across_features`, or new (2026-10-26).

| Frame | Solver | R0 | R20 / R100 | R40 / R100 | R100 | After the final detection | Sweeps to 90 / 99 % of the drop |
|---|---|---|---|---|---|---|---|
| 4.85 s | AAAVBD | 319 | 1.107 | 1.034 | 95 | 155 | 7 / 38 |
| 4.85 s | VBD | 289 | 1.029 | 1.015 | 93 | 146 | 6 / 30 |
| 7.0 s | AAAVBD | 1464 | 1.011 | 0.999 | 521 | 888 | 7 / 16 |
| 7.0 s | VBD | 1470 | 0.984 | 0.996 | 537 | 895 | 6 / 10 |
| 10.0 s | AAAVBD | 1359 | 1.008 | 1.000 | 477 | 825 | 7 / 14 |
| 10.0 s | VBD | 1354 | 0.979 | 0.994 | 491 | 830 | 5 / 9 |

At 4.85 s the two solvers' substeps stop sharing a state within the step (AAAVBD's dragonL spike starts in substep 7),
so that frame's medians compare different states; at 7.0 and 10.0 s, R0 agrees within 1%.

- **The plateau comes early.** Both solvers reach it within about 10-40 sweeps, VBD in fewer. At 7.0 and 10.0 s VBD's
  ratio rises slightly after sweep 20.
- **Finalization raises it 1.6-1.7x.** The dual update and the final detection take the ratio from R100 to the value
  the outer test sees.
- **Nodes without contacts aren't converged either, but hold almost none of it.** Their own ratio at sweep 100 is
  4.0 / 6.1 (AAAVBD / VBD) at 4.85 s, 15.5 / 15.6 at 7.0 s and 13.1 / 13.2 at 10.0 s. AAAVBD is no faster there: at
  sweep 20 its contact-free ratio is 2.5-6x its sweep-100 value, VBD's about 1.5x.

**Where the plateau sits** (shares of Σ |gᵢ|²/mᵢ, medians; kept means the same key or carried, changed means new at
the detection that made the set):

| Frame | Solver | At sweep 100: no contact / kept / changed | After the final detection: no contact / kept / changed |
|---|---|---|---|
| 4.85 s | AAAVBD | 0.2 / 92.2 / 7.6 % | 0.2 / 97.3 / 2.0 % |
| 4.85 s | VBD | 0.3 / 91.8 / 7.8 % | 0.5 / 96.6 / 2.7 % |
| 7.0 s | both | 0.1 / 99.9 / 0.0 % | 0.0 / 99.5 / 0.5 % |
| 10.0 s | both | 0.1 / 99.9 / 0.0 % | 0.0 / 99.1-99.2 / 0.8-0.9 % |

**Contact churn per detection** (20 per traced step; fractions of the new set, dropped as a fraction of the previous):

| Frame | Same / carried / new / dropped, median | At the step's last detection |
|---|---|---|
| 4.85 s (both solvers) | 97.0-97.2 / 2.1-2.2 / 0.5 / 0.5 % | 95.6-95.9 / 4.1-4.3 / 0.04-0.07 / 0.9-1.0 % |
| 7.0 s | 98.6 / 1.4 / 0.03 / 0.03 % | 97.1 / 2.8 / 0.01 / 0.1 % |
| 10.0 s | 98.7 / 1.3 / 0.04 / 0.03 % | 97.5 / 2.5 / 0.01 / 0.07 % |

The few contacts new at the last detection (3-20) are armadillo-dragon and pillar-armadillo pairs.

**Same floor for three solvers.** End-of-solve ratios over steps 480-490 (110 substeps each), from the production
frames' `solve_status` and a GPU Newton probe (2 outer iterations x 2 Newton steps per substep, CG never at its cap,
from AAAVBD's 4.8 s checkpoint): medians AAAVBD 160, VBD 194, Newton 171. Every substep lies between 98 and 368,
except three of AAAVBD's at steps 481-483 (600-5306), around its penetration spikes (entry 8). The trajectories differ
after step 480, which starts AAAVBD and Newton from AAAVBD's checkpoint and VBD from its own, nearly identical state:
there Newton ends at 117 against 164 for both VBD-family solvers. A ±1-ulp perturbation of every free coordinate
(2026-10-24) put float32's floor at ratios of 10.6 (6.0 s) and 25 (6.8 s), against entry 7's plateaus of 689 and 1251.

**Relation to entries 4 and 7.** On CapsulesIntoBowl (entry 4) the floor sat at the ~2% of contacts that end the step
pressed without a multiplier, mostly re-entered at the final detection. On DragonImpact it doesn't: 99% or more sits
on contacts whose key persisted or was carried, and new contacts hold 0.5-2.7% after the final detection. This
matches entry 7, where 87-96% of the residual is on pillar nodes within 1.5 cm of another pillar and more outer
iterations raise it. A possible reading, not measured (the trace recorded keys, not multipliers): during the sweeps
the AL penalty holds these persistent contacts, while the outer test's Lagrangian gradient counts only -lambda grad c,
so whatever part of a contact force the multiplier hasn't absorbed shows as residual.

**For the paper.** On this scene the stopping test reports a floor that AAAVBD, VBD and GPU Newton share, well above
float32's and located on persistent contacts. Its level depends on the frame: as the outer test sees it (after the
final detection), about 150-200x the tolerance around the impact (the 4.85 s trace, steps 480-490) and 800-900x at
7.0 and 10.0 s (689-1251 at 6.0-6.8 s in entry 7). It ranks no solver, and a solver ending at the floor has reached it
within the first tens of sweeps.

Data (`data/scenarios/runs/`): `ArmadilloSpaghetti.DragonImpact.<AAAVBD|VBD>.At<4.85|7.0|10.0>s.AAAVBDState.SweepTrace.h5`
(per-sweep checks and per-detection churn rows); the Newton probe, `ArmadilloSpaghetti.DragonImpact.Newton.From4.8s.Probe.h5`.

## 12. VBD at 10 x 40 on DragonImpact: stable to 18 s, 0.86x the step time, a different landing

**Setup.** VBD at 10 substeps of 1 x 40 sweeps (`--set Solver.VBD.n_subproblem_max_iters=40`), from its own
100-sweep checkpoint at 4.4 s (entry 9) to 18 s, against the 100-sweep production runs of VBD and AAAVBD
(`configs/scenes/ArmadilloSpaghetti.DragonImpact.toml`), all on c649f50a. The 40-sweep run first stopped at step 526
on the watcher's speed rule (a node over 15 m/s for 10 steps) while finite; it was resumed from its step-520
checkpoint with that rule logged only, and the other stops kept (2026-10-26). Speeds below are mass-weighted RMS speeds
from consecutive frames.

**Stability.** Every gate passed (4.4-7.0 s, 7.0-12.0 s and 12.0-18.0 s), with no non-finite frame and no dropped
pairs. The 100-sweep columns come from those runs' gates, which cover 4.4-12.0 and 12.8-18 s for VBD and 4.8-18 s for
AAAVBD.

| 4.4-18 s | VBD 10 x 40 | VBD 10 x 100 | AAAVBD 10 x 100 |
|---|---|---|---|
| Max penetration ratio | 0.75 (step 470) | 0.70 | 4.0 (two single steps, entry 8) |
| Inverted armadillo elements, every 10th frame, max | 3 | 7 | 4 |
| Longest run of frames over 15 m/s (peak) | 19 (31.9 m/s) | 7 (25.4 m/s) | 4 (29 m/s) |
| Median step time, steps 441-1799 | 13.24 s | 15.38 s | 15.62 s |
| GPU time, steps 441-1799 | 5.0 h | 5.8 h | 5.9 h |

The speed rule's own measure (the end-of-substep node velocity) ran 26 steps over 15 m/s at the 40-sweep run's
impact (steps 517-542, peak 35.7 m/s in dragonR); no streak reached 10 steps after 5.42 s. The 40-sweep frames also
have up to 4 inverted dragonR elements at steps 500 and 517-520, gone by the next frames; the 100-sweep runs' other
bodies weren't counted.

**Step time.** Cutting the sweeps by 60% saves 14% (0.86x VBD 100, every leg 0.86-0.87x). A kernels profile of one
AAAVBD 10 x 100 step at 7.0 s (`ArmadilloSpaghetti.DragonImpact.AAAVBD.At7.0s.Kernels.Profile.json`) shows why:
73% of the GPU time is one kernel, the vertex SDF narrow phase `_detect_vertex_mesh_contacts`
(`gpu/contact/mesh/sd.py`), at 574 ms per launch, twice per substep (the initial and the final detection). The sweeps
take 27%, 4.24 ms each.

**Landing.** Each dragon's tilt at 18 s, from its best-fit rotation from the rest shape, against a frame where the
renders show it on its feet (dragonL: VBD 100 at 18 s; dragonR: VBD 40 at 18 s); 0° is on its feet, ~180° on its back:

| | VBD 10 x 40 | VBD 10 x 100 | AAAVBD 10 x 100 |
|---|---|---|---|
| dragonL tilt | 7° | 0° | 116° |
| dragonR tilt | 0° | 112° | 127° |
| dragonR CoM path, 7-12 s / 12-18 s | 0.22 / 0.03 m | 1.89 / 0.12 m | 0.40 / 0.04 m |
| dragonL CoM path, 7-12 s / 12-18 s | 0.15 / 0.05 m | 0.08 / 0.02 m | 0.15 / 0.05 m |

At 40 sweeps both dragons land on their feet; at VBD 100 dragonR ends on its side and slides about 2 m on it between
7 and 13 s; at AAAVBD 100 both end on their side or back. The poses are set by 7.0 s and hold to 18 s. The three runs
land three different ways, so the landing also differs between the two 100-sweep solvers.

**Settling** (RMS speed, mean over 17-18 s, m/s; time under 1 cm/s within 5-18 s):

| | VBD 10 x 40 | VBD 10 x 100 | AAAVBD 10 x 100 |
|---|---|---|---|
| Pillars (mid-spans) | 0.193 | 0.294 | 0.311 |
| Armadillo | 0.401 | 0.886 | 0.896 |
| dragonL | 0.0134 (0.24 s under) | 0.0042 (5.0 s, from 13.7 s on) | 0.0098 (2.1 s) |
| dragonR | 0.0090 (1.5 s) | 0.0081 (1.8 s, from 17.7 s on) | 0.0105 (0.8 s) |

- The pillars and the armadillo move at about the same speed in VBD 100 and AAAVBD 100. In VBD 40 the pillars move
  at about two thirds of that from about 10 s on, and the armadillo at about half from about 6 s on. Neither falls
  under 1 cm/s in any run.
- The dragons' landing poses and slides differ across all three runs (Landing, above).

**For the paper.** On this scene VBD at 10 x 40 stays stable where every 10 x 20 run and AAAVBD at 10 x 40 failed
(entry 9), at 0.86x the step time, because contact detection, not the sweeps, dominates the step. It isn't at parity
with the 100-sweep run: its landing, its dragons' settling and its pillar and armadillo motion differ. The two
100-sweep solvers also land differently from each other, so a single landing outcome can't serve as a reference here.

Data (`data/scenarios/runs/`): `ArmadilloSpaghetti.DragonImpact.VBD.MidBudget.From4.4s.h5` (Leg1 up to step 520, then
Leg2-4) and its merged `.Profile.json`; the 100-sweep runs `ArmadilloSpaghetti.DragonImpact.<VBD|AAAVBD>.h5` and their
`.Profile.json`.

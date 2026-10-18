# Task 4: MCMC sampling of quadratic-manifold shapes at fixed energy

Status: **not started** · Branch `feature/rom-mcmc` · Worktree `C:\git\pbat-mcmc`
Read first: `MASTER.md` (rules). This task is independent of the GPU simulator work.

## Goal (first milestone)

Extend the quadratic-manifold example with a mode that samples reduced coordinates q whose deformation
energy equals a user-set value E*, and renders each sample in Polyscope. The GUI gets an energy input and a
"Sample" button.

This is groundwork for a later convergence benchmark: MCMC over reduced coordinates **and velocities** at
fixed total mechanical energy (kinetic + potential). The maintainer will provide more details later,
including notes from an earlier discussion.

**Task 3 depends on this task.** Its static convergence benchmark uses these samples as initial shapes,
lifted to full space (x₀ = X + u(q)) and released into static solves whose equilibrium is the rest shape.
So beyond the GUI, this task must **export samples to a file** following the sample-file contract in
`MASTER.md`, and the sampler must be usable headless (from a script, without the GUI) to generate
benchmark batches.

## What exists today

- The quadratic-manifold example is now **`python/examples/modalrom.py`**. It was renamed from
  `quadmanifold.py` in commit `bc888bac`; there is no `quadmanifold.py` any more.
- It computes vibration modes (`vibration_modes`), modal derivatives (`modal_derivatives`) and the maps
  `linear_map(U, q)` and `quadratic_map(U, Q, q)`, with deflation/orthogonalization of the modal-derivative
  basis and Dirichlet support. The GUI steps through modes and animates them with a wave signal
  (amplitude and frequency inputs).

## Proposed design (to confirm with the maintainer)

**Target set:** the level set {q : Ψ(u(q)) = E*}, where u(q) is the quadratic-manifold displacement and Ψ the
full-space hyperelastic energy.

**Samplers**, behind one small interface so they can be swapped:
1. **Ray sampler (baseline, exact level-set hits).** Draw a random direction d in q-space (isotropic, or
   whitened by the reduced stiffness), then solve the scalar equation Ψ(u(r·d)) = E* for r > 0 by
   bracketing and bisection or Newton. Simple and robust, but it is not uniform on the level set, so it
   isn't MCMC. It's useful for seeing what shapes the manifold can produce.
2. **Constrained random-walk MCMC.** A Metropolis step on the level set: perturb q, then project back onto
   the level set (radially, or along ∇Ψ), with the Jacobian and reversibility corrections a constrained
   sampler needs (e.g. a RATTLE-style projection). It targets a well-defined distribution, such as the
   uniform measure on the level set.
3. **Later, for the benchmark:** microcanonical sampling over (q, q̇) with
   H = ½ q̇ᵀ M_r(q) q̇ + Ψ(u(q)) = E*. M_r(q) = J(q)ᵀ M J(q) is the configuration-dependent reduced mass
   matrix of the quadratic map. Out of scope until the maintainer provides details.

**Implementation notes:**
- Ψ(u(q)) needs the full-space energy at each evaluation. Use pbatoolkit's CPU hyperelastic energy
  (the example already has a `potential` function), or a small Warp kernel if sampling gets slow.
- GUI: an E* input, a sampler choice, the number of samples per click, a "Sample" button, and a list or
  slider to page through stored samples. Show the achieved energy and the acceptance rate (MCMC).

## Validation

- Every displayed sample satisfies |Ψ − E*| ≤ tol·E*.
- MCMC: acceptance rate in a sensible range, and the energy stays on the level set over long chains.
- Visual sanity: small E* gives near-linear modal shapes, large E* shows quadratic (modal-derivative) effects.

## Files

Own: `python/examples/modalrom.py` and any new modules (e.g. `python/examples/rom_mcmc.py`). Touches no
simulator files.

## Open questions for the maintainer

1. ~~Which distribution?~~ **Answered (2026-10-17):** the mass-metric microcanonical measure,
   π(q) ∝ √det(J(q)ᵀ M J(q)) · δ(Ψ̂(q) − E*), with J(q) = U + Q:q and Ψ̂(q) = Ψ(X + u(q)) − Ψ(X). The weight is
   pluggable; this one is the default and the one written to task 3's files.
5. ~~Sampler?~~ **Answered:** implement **both**, side by side: polar Metropolis (random walk on directions θ
   with an exact radial root r(θ), density p(θ) ∝ w · r^(m−1) / ∂_rΨ̂) and a constrained random walk with a
   projection and a reversibility check (Zappa, Holmes-Cerfon & Goodman 2018). The ray sampler stays as a
   browsing baseline.
6. ~~Fix the modal derivatives?~~ **Answered:** yes. Divide by 2h, and project the derivatives M-orthogonally
   off span(U), keeping the vibration-mode derivatives (K − ω²M). Also pass `w` to `modal_derivatives`
   explicitly and drop the dense `np.eye(nfree) @ Thetared`.
7. ~~Reduced space?~~ **Answered:** the quadratic map u = Uq + ½Q:qq over the first `-k` modes, excluding
   ω≈0 (rigid) modes. The linear map Uq is selectable for comparison. The augmented [U Z] basis is not sampled.
8. ~~Sample-file contents beyond the MASTER.md draft?~~ **Answered:** the draft, plus chain metadata (attrs:
   weight kind, burn-in, thinning, acceptance rate, ESS; dataset `samples/weight_log`), plus the ROM basis
   (`basis/U`, `basis/Q`, `basis/omega`), so samples can be re-lifted or checked without recomputing the MDs.
   `samples/x` is stored as float32. One E* per file.
9. ~~Default for headless batches?~~ **Answered:** polar Metropolis (`--sampler` selects the constrained random
   walk), with automatic thinning by the estimated integrated autocorrelation time so the S samples are about
   independent.
10. ~~Headless entry point?~~ **Answered:** flags on `modalrom.py` (`--headless`, `--energy`, `--samples`,
    `--sampler`, `--seed`, `-o`), sharing the ROM setup with the GUI. The samplers and export live in
    `rom_mcmc.py`.
2. ~~Deformation energy only?~~ **Answered:** yes. E* is the deformation energy Ψ(X+u(q)) − Ψ(X), in
   **absolute joules**. Velocities come later.
3. ~~Save samples for task 3?~~ **Answered:** yes, samples seed task 3's static benchmark. Agree on the sample-file contract in `MASTER.md` with the maintainer and task 3.
4. Please share the notes from your earlier discussion when it's time for the velocity part.

## Progress

- 2026-10-17 · Investigated `modalrom.py` and prototyped on MediumLongBendingBeam (scratch only, no tracked edits).
  Findings: (a) `modal_derivatives` divides the central difference by h instead of 2h, so Q is 2x too large.
  (b) The vibration-mode-derivative system K - w_i^2 M blows up for near-degenerate mode pairs (beam: 1.294/1.303,
  7.59/7.65, 19.61/19.75). M-orthogonal projection of the MDs off span(U) fixes it: for mode 0 at 0.3 m tip
  displacement, the energy drops from 93.5 J (linear) to 11.6 J, with optimal scale c=1. (c) The Stable
  Neo-Hookean rest energy is nonzero (4.96e-4 J), so deformation energy = Psi(X+u) - Psi(X). (d) The energy is
  monotone along random rays (the level set is star-shaped), so I propose a polar Metropolis sampler with an
  exact radial root. (e) Energy plus gradient costs under 1 ms on the CPU, and the MDs for 20 modes take 14 s.
- 2026-10-17 · Findings, the proposed design and question batch 1 published at
  https://claude.ai/artifact/FzhhvttWedq3qMSLwu2GbE
- 2026-10-17 · Batches 1 and 2 answered (questions 1, 2 and 5–10 above). Only question 4 (the velocity notes) is
  still open, and it doesn't block the first milestone. The proposed contract additions are posted in the
  MASTER.md coordination log for task 3.
- 2026-10-17 · Decision: keep pbat's float32 energy (no float64 port). Its absolute noise floor is about
  2–4e-4 J on the beam, so tolerances must account for it. Correction: pbat's 3D Stable NH is exactly 0 at rest,
  so the 4.96e-4 J "rest energy" is float32 round-off.
- 2026-10-17 · Committed `2046e73b` (rom_mcmc.py + tests) and `7ab8b152` (MD fix, headless mode, GUI panel),
  rebased on dev after Phase 0; 6 unit tests pass. On the beam at E*=10 J, polar and crw agree within about
  2σ, with energies within 5e-5 of E*. A stubbed GUI smoke test passes.
- 2026-10-18 · Committed `eb6c4785`: noise-aware tolerances. Each chain measures the energy noise (residual
  of a cubic fit over ±1e-3 relative along a ray; ±1e-6 was too small, since float32 positions didn't change)
  and keeps tolerances at 10× it. The radial root caches evaluations, fixing a brentq sign-flip failure under
  noise. Also committed `f291e2da`: GUI/headless report the noise and the chain's settings. 8 unit tests pass.
  On the beam, the noise is about 8e-5 J at every E*, with 0 star violations at 0.1, 2 and 10 J. The warning
  fires below about 0.8 J. Contract wording in MASTER.md corrected.
- **Merge-ready** (first milestone). Still open: question 4 (velocity notes) for the (q, q̇) milestone.

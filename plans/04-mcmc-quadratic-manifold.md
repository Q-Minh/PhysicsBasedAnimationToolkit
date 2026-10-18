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

1. Which distribution should the samples follow: uniform on the energy level set, or something else (e.g. weighted by the reduced mass matrix)?
2. Is E* the deformation energy only for this milestone, with velocities added later?
3. ~~Save samples for task 3?~~ **Answered:** yes, samples seed task 3's static benchmark. Agree on the sample-file contract in `MASTER.md` with the maintainer and task 3.
4. Please share the notes from your earlier discussion when it's time for the velocity part.

## Progress

(update as you go)

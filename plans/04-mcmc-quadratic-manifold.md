# Task 4: MCMC sampling of quadratic-manifold shapes at fixed energy

## Handoff (2026-10-20)

Read with: `python/examples/docs/sampling-methods.md` (methods, §8 benchmark problems, §11 code map), the skills
`.claude/skills/sim-sampling` and `sim-benchmarks`, and the MASTER.md contract sections ("MCMC sample files",
"Benchmark datasets"). This handoff only records what those don't.

**1. Scope and ownership** (all under `python/examples/`):
- `rom_mcmc.py`: samplers (polar, crw, ray, microcanonical), `run_chains` + split-R̂, energy-noise
  estimation, `write_samples`/`write_basis` (the sample-file contract).
- `modalrom.py`: ROM setup (mass-normalized modes, modal derivatives), GUI, headless sampling, `load_fem`
  (scene files and `file.h5:fem`).
- `rom_reach.py`: shape reachability (full-order static Newton, fits, ROM equilibria) and the
  `static_equilibrium` solver reused for the dynamic-benchmark references.
- `make_benchmark.py`: the benchmark dataset generator; `benchmark_browser.py`: the Polyscope dataset browser.
- Tests: `test_rom_mcmc.py`, `test_rom_reach.py`, `test_make_benchmark.py`, `test_benchmark_browser.py`.
- Reader side of the dataset format I wrote, now shared with task 3: `python/tools/sim/static.py`
  (`split_batch_path`, `is_dataset`, `dataset_sets`, `load_sample_batch`, `TestSampleBatches`),
  `python/tools/sim/benchmark.py` (`--dataset`/`--sets`), `python/tools/sim/panels/static.py` (set combo).

**2. Current state**
- Everything is merged into dev. Merges: `4029072e` (rom-mcmc: milestones 1–3, BDF1 references, generator),
  `3a975263` (sampler-docs), `8bf175b0` (benchmark-input), `534bdcd3` (browser), `45d38f60` (per-set gravity),
  `863b1b2b` (sampler-param-docs), `22220cec` (scene files), `cbd3aca2` (float64 references) reverted by
  `f17533d1` (float32 references), `0947c3fd` (single-file datasets).
- Worktree `pbat-mcmc` (location in CLAUDE.local.md), branch `feature/benchmark-dataset-file` at `eea8601d`
  (= dev at the time, no unmerged commits). Resume from a fresh branch off dev.
- Untracked `plans/mcmc-sampling-conservative-system.txt` in the worktree is the maintainer's notes: leave it.
- Datasets in the main checkout's `data/benchmarks/` (all trampoline, `data/scenarios/trampoline.h5`):
  - `trampoline-sample.h5`: 50/500/5000 J × 4 problems, static + `dynamic_h0.0001` + `dynamic_h0.01`, pools of 400.
  - `trampoline-test.h5`: 50/5000 J × 3 problems, pools of 40 (R̂ meaningless; for quick checks).
  - `trampoline-demo.h5`: 10/100/1000/10000 J × 10 problems, static + `dynamic_h0.01`; the maintainer's demo
    for collaborators, **not for the paper**.

**3. Key decisions (maintainer unless noted)**
- Target: the Liouville-measure microcanonical ensemble over (q, q̇) at total energy E (K + U); velocities
  exported as `samples/qdot` only. Microcanonical is the preferred sampler; polar for fixed deformation
  energy; crw only to cross-check polar; ray only for browsing.
- Quadratic map over mass-normalized non-rigid modes; pbat float32 energies (no float64 port), so all
  tolerances are noise-aware. No full-order Newton inside sampling.
- 4 chains from q = 0, pooled; problems are evenly spaced pool subsets, **paired** across static and dynamic
  sets. Static sets sample without gravity (`pools/pool`), dynamic sets with the scene gravity
  (`pools/pool_gravity`), so they are different pools when gravity is on.
- Dynamic problems: one BDF1 step, x̃ = x + h v + h² g; h = 1e-4 and 1e-2. Energies up to ~1e4 J, since
  static solves from large deformations pass through all scales.
- **Float32 references:** reported error reductions stop at 99%, so float32 Newton references (within
  6.5e-4 e0 of exact ones, recorded per sample in `ref_error_mnorm`) suffice. The float64 port was reverted.
- **One HDF5 file per benchmark**, pools inside, the FemElastoDynamics stored as `fem/`, old directory format
  dropped with no compatibility code. `benchmark.py` takes solver parameters only from `--sim-params`/`--set`.
- Scene inputs: scene files preferred over Dumps. Constraints are checked by the time-span rule (active over
  [0, max dt] ⇒ Fixed and covering the span); multi-body scenes are rejected.

**4. Verified facts and gotchas**
- R̂ is not yet good enough for a real dataset: pools of 400 give split-R̂ 1.05 / 1.26 / 1.14 at 50 / 500 /
  5000 J (trampoline-sample), 1.04–1.21 on the demo. The microcanonical chain is cheap (~1 ms/step) but
  correlated (τ ≈ 135 steps on the beam). Target R̂ < 1.05 means larger pools or more/longer chains.
- Inversions: min det F < 0 from ~5000 J (5000 J: some starts; 10000 J: 3/10 static, 4/10 dynamic starts on the
  demo). Allowed by design (Stable NH), recorded in `samples/min_J`.
- With gravity in U, dynamic low-energy sets have U < 0 (demo 10 J: U/E from −4.4 to −1.55); this is expected.
- Reference accuracy (float32, demo): ≤ 4.8e-4 e0 at 10 J, decreasing to ≤ 8.4e-5 e0 at 10000 J.
- The h = 1e-4 plateau seen earlier was GPU-side, not the reference (float64 re-solve agreed to 0.04% e0).
- Energy noise floor: ~8e-5 J on the beam; sampling below ~0.8 J there triggers the noise warning.
- Dataset format: `X` and `basis/` are HDF5 hard links in every batch group; `make_benchmark.py` writes `fem/`
  through `pbat.io.Archive` first and must release it (`gc.collect()`) before h5py reopens the file.
  `split_batch_path` ignores a drive-letter colon; `load_sample_batch` refuses a dataset root and names a set.
- **Stale contract text:** MASTER.md's older "Benchmark dataset (task 4 …)" bullet (directory layout,
  `pool_file`, `ref_newton_correction`, an absolute path) predates the single-file format; the "Benchmark
  datasets" bullet is current. The coordinator should drop or mark the old one.

**5. Open items** (each waits on the maintainer)
- **Full benchmark dataset:** on hold until the benchmark design is fixed (MASTER.md "Benchmark plan"). Next:
  get the energies, problems per level, sets (h values) and scenes (`trampoline.h5`, `trampoline.Y1e7.h5`,
  circle variants `TrampolineCircle.{1MPa,10MPa}.h5`), then size pools/chains for R̂ < 1.05 and validate on
  1 static + 1 dynamic problem before the full run (announce CPU runs in MASTER.md).
- `modalrom.py --sampler` CLI default is still polar while the docs prefer microcanonical: ask whether to switch.
- Question 4 (velocity notes) is moot unless the maintainer still has notes; milestone 2 used the decisions above.
- Optional: a better microcanonical proposal to cut τ (only if requested).

**6. How to resume**
- Tests (venv Python, see CLAUDE.local.md): from `python/examples`,
  `python -m unittest test_rom_mcmc test_rom_reach test_make_benchmark test_benchmark_browser` (38 tests,
  ~11 s); from the repo root, `python -m unittest python.tools.sim.static.TestSampleBatches`.
- Smallest generator check (CPU, ~1 min): `python make_benchmark.py -i path/to/PhysicsBasedAnimationToolkit/data/scenarios/trampoline.h5
  -o out.h5 --energies 50 --problems 1 --pool 40 --dt 1e-2`. `--pool`, `--chains`, `--burn-in` and `--pilot`
  size the chains (defaults: pools of 400, 4 chains).
- Browse a dataset: `python benchmark_browser.py -d path/to/dataset.h5`. Benchmark one: see the
  `sim-benchmarks` skill (`benchmark.py --dataset … --sets 'static/*'`).

---

Status: ~~not started · Branch `feature/rom-mcmc`~~ (superseded by the Handoff above) · Worktree `pbat-mcmc`
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

*Superseded: see the answered questions below and `python/examples/docs/sampling-methods.md`.*

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
   *Superseded: milestone 2 was built on the 2026-10-18 decisions; see the Handoff.*

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
- *(Superseded: everything is merged; see the Handoff.)* **Merge-ready** (first milestone). Still open: question 4 (velocity notes) for the (q, q̇) milestone.
- 2026-10-18 · Milestone 1 merged into dev (`f291e2da`). The maintainer found an ImGui ID clash: the "Sample"
  button and slider shared a label. Fixed with PushID/PopID in `837ab6cf`, also merged into dev.
- 2026-10-18 · Profile, quadratic-manifold status and velocity plan published:
  https://claude.ai/artifact/ABsVM6TwtGpAnuEacBVx9w. The quadratic map was already the headless default; only
  the GUI started on Linear. Fixed in `1e67c1d9` (GUI defaults to Quadratic).
- 2026-10-18 · **Decisions:**
  - **Speed-ups:** implement items 1–4. (1) Assemble each dK/dq_j once in `modal_derivatives`. (2) Evaluate
    u(r d) = r a + r² b in closed form along rays. (3) Shorter warm-up: 300 burn-in + 500 pilot. (4) Radial root
    tolerance at the noise floor.
  - **No full Newton solves for sampling.** The maintainer only wants them where sample efficiency justifies
    it, e.g. sampling loads and solving for equilibria.
  - **Milestone 2 (fixed total energy):** Liouville measure dq dp, so π(q) ∝ √det M_r (E−U)^((d−2)/2) on
    U < E, and q̇ = √(2K) L z/|z| with LLᵀ = M_r⁻¹. Gravity is off by default; an opt-in flag adds −f_gᵀu(q)
    with f_g = M g in full space. Inverted elements are allowed, with each sample's min J recorded.
  - **Milestone 2 chains and export:** 4 independent chains from q = 0, pooled, with split-R̂ reported. Export
    `samples/qdot` (S, m) only, with no full-space velocities; consumers lift them with the stored basis.
- 2026-10-18 · Speed-ups 1–4 merged into dev (`800e87ef`): a 100-sample batch takes ~9–11 s instead of ~31 s.
- 2026-10-18 · **Milestone 2 implemented** (commits `4ddea087`..`240dcbe1`):
  - `MicrocanonicalMetropolis` (sampler "microcanonical"), `run_chains` with split-R̂, opt-in gravity via
    `fext`, and per-sample min det(F). The export adds `samples/qdot`, `samples/min_J` and `basis/M`.
  - The GUI shows U and K and draws velocity vectors.
  - 12 unit tests pass, including a configuration-dependent-mass test whose negative control (dropping
    √det M_r) fails as it should.
  - On the beam at E = 10 J (200 samples, 4 chains, 16 s): K + U = E to 5e-16 from the file alone;
    U/E has mean 0.498 and std 0.111, against Beta(10, 10)'s 0.5 and 0.109 for a 20-mode harmonic system;
    split-R̂ 1.044; no inverted samples.
  - The chain is cheap per step (~1 ms) but correlated (τ ≈ 135 steps). A better proposal could be a later
    improvement.
- 2026-10-18 · **Milestone 3 (shape reachability) added by the maintainer.**
  - Prototype and design: https://claude.ai/artifact/3JwUPvjYCLm2Q2kUmrH4MQ
  - Prototype finding on the beam under gravity: the quadratic manifold contains the bent shapes
    geometrically (best fit within 0.4% M-error at 0.71 m tip sag). The ROM equilibrium is too stiff, though:
    it sags 33% less, and the geometric best fit carries much more energy (locking-like).
  - Fitting to linear-mode targets can't enhance them, because Q ⊥_M U; the enhanced shape is just u(η).
  - **Decisions:** support all three target types: gravity, region force (slab/box region via sliders) and
    handle displacement. Each is turned into a target by a full-order static solve, reusing
    `pbat.math.optimization.Newton`. Show three fits: geometric best fits (U, [U Z], quadratic), the
    energy-regularized fit ½‖u(q)−u_t‖²_M + αΨ̂(q) with an α slider, and the ROM equilibrium under the same
    load. No headless sweep.
- 2026-10-18 · **Mass-normalized modes (maintainer):** switch to UᵢᵀMUᵢ = 1 as the only behavior, with a
  scale-free finite-difference step in `modal_derivatives` and a relative Z cutoff. Files record
  `mode_normalization`. The GUI wave amplitude becomes the peak displacement in metres.
- 2026-10-18 · Mass normalization committed in `c03f8cd4`. U equals the old U rescaled, Q equals the old Q
  rescaled entrywise (s_i·s_j) to 0.3% (float32 factorization accuracy), sampled shapes are unchanged, and
  there are 74 Z modes instead of 75.
- 2026-10-18 · **Milestone 3 implemented** (`32806b99`, `8817fd36`, `b40545ed`):
  - `rom_reach.py`: box regions, gravity and region loads, a load-stepped full-order static Newton solve
    (pbat's Newton bindings are commented out, so it's a short Python loop with backtracking), and the
    fits `linear_fit`, `manifold_fit` (α = 0: Levenberg–Marquardt; α > 0: energy-regularized L-BFGS) and
    `rom_equilibrium` (handles by mass-weighted penalty, κ = 100 ω_max²).
  - GUI panel in `modalrom.py` with the target as a transparent ghost and per-vertex error.
  - 8 new unit tests: closed-form static solves, recovery of reachable targets, the regularization
    trade-off, and analytic ROM equilibria.
  - Beam, gravity 0.4 (tip sag 0.366 m, target 7.87 J):
    - U best fit: 0.8% error but 152 J of energy.
    - Quadratic best fit: 0.1% and 8.4 J.
    - Quadratic ROM equilibrium: 11% error.
    - Linear ROM equilibrium: 68% error.
    - Regularized fit: α ≈ 0.001–0.01 is the useful range; α = 1 collapses the shape.
  - Beam, handle tip −0.5 m: quadratic ROM equilibrium 4.9% error, linear 7.3% but with 97 J against 26 J.
  - The GUI was only smoke-tested with stubs; the maintainer's interactive check is pending.
- 2026-10-18 · Milestones 2 and 3 merged into dev (`c947de87`); the maintainer is very satisfied. Gravity is now
  available in the GUI sampler (`5a4b2d3d` single-crossing star check, `e86d5630` GUI controls).
- 2026-10-18 · **Next objective: generate the benchmark dataset** (see MASTER.md "Benchmark plan"). Task 3 runs
  the static problems and the coordinator runs the dynamic ones. No full runs until the reported data is fixed.
  - Design (prototype): https://claude.ai/artifact/Vo7K55wZefigfxatyBEmCu
  - **Maintainer decisions:**
    - Scene `trampoline.Dump.h5:fem`, microcanonical samples, paired across all sets.
    - **Large energies:** up to ~1e4 J was used on the beam. Static solves from large deformations pass
      through mid and low deformation, so one benchmark covers all scales.
    - **Start small:** 5–10 problems per level, then iterate. The solver time cap should be 5–10 s, not
      60 s (for task 3).
    - **Time steps h = 1e-4 and 1e-2 s.** 1e-2 is the usual graphics step without substepping; 1e-4
      corresponds to 100 substeps.
    - **Dynamic problems:** x̃ = x + h v + h² g with the scene gravity, plus a reference solution x⋆ from a
      converged full-order CPU Newton solve.
  - Trampoline at high energy (short chains): at 50 / 500 / 5000 J the median peak displacement is 14 / 24 /
    36 cm, min det F is 0.82 / 0.69 / −1.6 (5000 J inverts some elements) and U/E ≈ 0.3. These are the default
    levels.
  - **Generator implemented** (`d1f03bbe`, `e45a286c`, `391b994f`; `python/examples/make_benchmark.py`).
    - It writes pooled samples, static problems, and BDF1 dynamic problems per h, all paired.
    - Small test set: `data\benchmarks\trampoline-test\` (2 energies × 3 problems, 51 s CPU).
      *Superseded: now the single file `data/benchmarks/trampoline-test.h5`.*
    - Checked: X matches the Dump, Dirichlet DOFs are at rest, x̃ = x + hv + h²g to 6e-8, and the reference
      gradient drops 3–5 orders to the float32 noise floor (~3 N). Reference accuracy is ~3.5–4 decades of
      the step size.
    - Full generation waits until the coordinator relays the data requirements.
- 2026-10-18 · **Documentation pass** (coordinator request) on `feature/sampler-docs`, 5 commits
  (`03c746e5`..`80f78b58`), docs only, verified by comparing ASTs without docstrings:
  - Module docstrings with usage, help with units and defaults, public API docstrings, and the full
    sample-file contract in `rom_mcmc.write_samples`.
  - **Sampler guidance (maintainer: microcanonical preferred):** microcanonical by default (velocities,
    deformation scales spread; 4 chains, ≥400 samples, rhat < 1.05); polar when every shape needs the same
    deformation energy; crw to cross-check polar; ray only for browsing.
  - The CLI default of `modalrom.py --sampler` is unchanged (polar), since this was a docs-only pass.
  - Skill summary sent to the coordinator.
- 2026-10-18 · **Benchmark references back to float32** (maintainer: no error reductions tighter than 99%), merged
  as `feature/float32-references`. References are within 6.5e-4 e0 of exact ones; `trampoline-sample` regenerated
  from `trampoline.h5`. **Maintainer confirmed** two judgment calls: scene constraints are checked by the time-span
  rule (active during [0, max dt] ⇒ Fixed and covering the span), and multi-body scenes are rejected. On hold.

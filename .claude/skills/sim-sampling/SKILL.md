---
name: sim-sampling
description: Sample deformed shapes (and velocities) of an elastic body with the quadratic-manifold modal ROM MCMC samplers in python/examples (modalrom.py GUI or headless, rom_mcmc.py), check sample quality, and generate static/dynamic benchmark problem sets with make_benchmark.py, and browse them with benchmark_browser.py. Use when asked to draw shape samples at an energy, pick a sampler, inspect or validate a sample batch, build benchmark datasets, or explore/visualize a benchmark dataset.
---

# Shape sampling (modal ROM MCMC)

The sampling tools live in `python/examples`, everything runs on the **CPU**, and the in-source docs are complete:
- `modalrom.py`: GUI and headless CLI.
- `rom_mcmc.py`: samplers and the sample-file contract in `write_samples`.
- `rom_reach.py`: full-order equilibrium solves.
- `make_benchmark.py`: problem-file layout.
- `benchmark_browser.py`: Polyscope browser for a generated dataset.

Read those docstrings for anything not covered here. The derivations, algorithms, parameters and validation of every sampler and of shape matching are in `python/examples/docs/sampling-methods.md`, the maintainer's technical reference and the source of truth. Update it in the same change as any sampler, fit or benchmark-definition change, keeping its equation tags and citations consistent. https://claude.ai/artifact/8EuUcmLsX882p2y5Qsh67n is a rendered snapshot of an earlier version.

Run from `python/examples` with the dev venv's Python (Python <= 3.12; its location is machine-specific, see `CLAUDE.local.md`). Prefer **scene files** (e.g. `data/scenarios/trampoline.h5`), which are built into a FemElastoDynamics the way the CPU editor builds them. A stored FemElastoDynamics (`filepath:h5path`, e.g. `trampoline.Dump.h5:fem`) works too and gives identical results. `modalrom.py` also accepts a tetrahedral mesh, with material set by `-m`/`-Y`/`-n`; `make_benchmark.py` rejects meshes. Scenes are rejected, with a message naming the culprit, when they have static colliders, more than one body, or constraints that move nodes or start or end during the problems' time span [0, max(`--dt`)]. Constraints entirely outside that span are ignored. Of the current scenarios, `trampoline.h5` and `trampoline.Y1e7.h5` qualify, and so do the benchmark's circle trampolines `trampoline_circle_{Y1e5,Y1e6,Y1e7,Y1e8,het}.h5`. These share one mesh (10643 nodes, 41588 tets) and 4 fixed rim patches (47 nodes); `het` has a stiff 1e8 center (10049 tets) inside a soft 1e6 outer ring (31539 tets). They are git-ignored; regenerate them with `python -m python.tools.vbd.rim_patch_scene` (the commands are in its docstring). `data/` exists only in the main checkout, so from `python/examples` or a worktree point to it as `path/to/PhysicsBasedAnimationToolkit/data/...`.

## Which sampler

| Sampler | Target | Cost and mixing | Use for |
|---|---|---|---|
| `microcanonical` (**preferred**) | States (q, q̇) at total energy E = K + U. Shapes span U < E, with velocities | ~1 ms/step but correlated (τ ≈ 100–250). Use 4 chains and ≥ 400 samples | Benchmarks (static and dynamic), most uses |
| `polar` | Shapes exactly at potential energy E* | Mixes fast (τ ≈ 10–20) | When every shape must have the same deformation energy |
| `crw` | Same as polar | About 2× polar's cost | Cross-checking polar |
| `ray` | Biased (needs importance weights) | Cheap | Browsing only |

`modalrom.py`'s CLI default is still `polar`, so pass `--sampler microcanonical` explicitly.

`--energy` means **E\*** (potential energy) for polar/crw/ray, but the **total** energy for microcanonical. At large E, U/E ≈ 0.3.

## Commands

```bash
# GUI: mode animation, fixed-energy sampling (energy, sampler, gravity, Sample, slider, velocity arrows),
# shape reachability (load or handle target -> "Solve target" -> linear/quadratic fits and ROM equilibrium)
python modalrom.py -i path/to/PhysicsBasedAnimationToolkit/data/scenarios/trampoline.h5

# Headless microcanonical batch (ensemble="total", writes samples/qdot), prints acceptance, thinning, tau, ESS, split R-hat
python modalrom.py -i <scene>.h5 --headless -o batch.h5 --sampler microcanonical --energy 10 --samples 400

# Headless fixed-potential batch, every shape at U = E* (ensemble="potential")
python modalrom.py -i <scene>.h5 --headless -o batch.h5 --sampler polar --energy 10 --samples 100

# Benchmark dataset: ONE file with a group per named scene (-i name=scene.h5, repeated), each holding its FEM
# (<scene>/fem), <scene>/pools/pool/E{E} (sampled without gravity) and <scene>/pools/pool_gravity/E{E} (with gravity),
# and the problem sets <scene>/static/E{E} and <scene>/dynamic_h{h}/E{E}. By default static problems are sampled
# without gravity and dynamic ones with it, from separate pools; the dynamic h sets always share samples, and all sets
# share them when both gravity settings match. Defaults: 10000 J, 10 problems, h = 1e-4 and 1e-2
python make_benchmark.py -i Y1e6=<scene_Y1e6>.h5 -i Y1e7=<scene_Y1e7>.h5 --label "Y1e6=Y = 1e6" --label "Y1e7=Y = 1e7" -o <dataset>.h5
python make_benchmark.py -i <name>=<scene>.h5 -o <dataset>.h5 --dt              # static only
python make_benchmark.py -i <name>=<scene>.h5 -o <dataset>.h5 --problems 2 --pool 40 --chains 2 --burn-in 100 --pilot 100   # smoke test

# Browse a generated dataset: scene, problem set, energy and problem selectors; start state (colored by displacement
# or det F), rest-shape ghost, velocity arrows, and for dynamic problems the inertial target x~ and reference x*;
# energies, min det F, reference-solve stats, and the pool's R-hat, ESS and U/E histogram
python benchmark_browser.py -d path/to/PhysicsBasedAnimationToolkit/data/benchmarks/trampolines-test.h5
```

Translating requests:
- "N samples": `--samples N` (headless) or `--pool N` (`make_benchmark`).
- "Energies a, b": `--energies a b`.
- "M problems per energy": `--problems M`. It must not exceed `--pool` and should stay well below the pool's ESS. For example, 100 problems at 10000 J: `--energies 10000 --problems 100 --pool 2000`.
- "Time steps h1, h2": `--dt h1 h2`.
- "Static only": `--dt` with no value.
- "With gravity": `--gravity gx gy gz` (headless) or the GUI checkbox. In `make_benchmark.py`, use `--static-gravity`/`--no-static-gravity` (default off) and `--dynamic-gravity`/`--no-dynamic-gravity` (default on); they add the scene's gravity potential to the sampled total energy.
- "Reproducible": `--seed`.
- "More modes": `-k`.
- "Scenes A and B": `-i A=<a>.h5 -i B=<b>.h5`; a plot label per scene: `--label "A=..."`.
- "Show / explore / visualize the dataset": `benchmark_browser.py -d <dataset>.h5`. The mesh comes from the selected scene's stored FEM.

`make_benchmark.py -h` lists the rest (`--chains`, `--burn-in`, `--pilot`, `--eps`, `--newton-iters`).

**Full benchmark dataset generation is on hold** until the maintainer fixes the data requirements. Small smoke-test sets are fine. A sample dataset (trampoline, 50/500/5000 J × 4 problems, pools of 400, h = 1e-4 and 1e-2, no gravity) is `data/benchmarks/trampoline-sample.h5`; `trampoline-test.h5` is a smaller one (50/5000 J × 3 problems).

## Output files

A batch is a `modalrom.py` sample file's root, or a pool or problem set group of a benchmark dataset. Every batch follows the sample-file contract (`rom_mcmc.write_samples`):
- `X` (N,3) rest positions, `samples/x` (S,N,3) float32 positions, plus `samples/q`, `energy`, `log_target`, `samples/min_J`.
- `basis/{U,Q,omega,M}`.
- Attributes: `target_energy`, `rhat`, `ensemble`, `gravity`, `mode_normalization="mass"`, `importance_weights_required`.

A benchmark dataset (`make_benchmark.py`, format version 2) is one file with a group per named scene:
- `<scene>/fem/` (the scene's FemElastoDynamics, which the benchmark solves its problems on), and `<scene>/X`, `<scene>/basis/` stored once per scene; each of the scene's groups' `X` and `basis` are hard links to them. Scene attributes: `input`, optional `label`, `sets`, `pools`.
- Root attributes `dataset_format = "pbat-benchmark-dataset"`, `dataset_version = 2`, `scenes`, `sets` (full paths, e.g. `Y1e6/static/E10000`, `Y1e6/dynamic_h0.01/E10000`), `pools`, and the generator's settings.
- Problem sets add `problem`, `scene`, `fem` (its scene's FEM group, e.g. `Y1e6/fem`), `pool` (the pool group, e.g. `Y1e6/pools/pool/E10000`), `pool_indices`, `gravity` (the acceleration in the sampled potential; zero without), and `scene_dt`/`scene_bdf_steps` (the input's integrator, recorded only; dynamic problems are always BDF1 steps with the `--dt` values).
- Version 1 files (one scene at the root, e.g. the old `trampoline-*.h5`) aren't read any more.
- Dynamic sets also add `h`, `aext`, and `samples/{v, xtilde, x_ref, ref_error_mnorm, ref_newton_*, ref_min_J}`. `x_ref` solves the step for the stored `xtilde`, to pbat's float32 energy noise floor: errors of up to about 7e-4 of the initial error e0 on the trampoline, which is enough for error reductions up to 99%. `ref_error_mnorm` estimates each problem's reference error in the benchmark's metric.

Use `samples/x`, not `samples/q`: modes have been mass-normalized since `c03f8cd4`, so `q` from older files isn't comparable.

## Quality checks (report them with every batch)

- **`rhat < 1.05`.** Otherwise raise `--samples`/`--pool` or `--burn-in`. Microcanonical with 100 samples can exceed 1.05 (e.g. 1.08).
- **`energy_noise_warning = False`.** pbat's energy is float32, with ~1e-4 J absolute noise, so energies ≲ 1 J on the beam can't be resolved.
- **`star_violations = 0`** for polar.
- **`importance_weights_required = False`**, i.e. not a ray batch.
- **`samples/min_J`.** Values ≤ 0 mean inverted elements. They're allowed, but report them.
- **Dynamic sets.** `samples/ref_error_mnorm` is each reference's accuracy in the benchmark's error metric (the M-norm of its last Newton correction over free DOFs). Compare it with e0 = |x − x_ref|_M; it's up to ~6e-4 e0 on the trampoline (it can underestimate the true error by up to ~3×). Don't report error reductions beyond 99%. Don't compare `ref_newton_correction`, a max entry in m, with M-norms.

## Pitfalls

- **Gravity** is off in `modalrom.py` unless `--gravity` (or the GUI checkbox). In `make_benchmark.py`, the default is sampling without gravity for static problems and with gravity for dynamic ones. Static problems are always solved without loads (equilibrium exactly X); dynamic `xtilde` always includes the scene's gravity. With gravity in the sampled potential, U and U/E can be negative (sagged shapes), and K = E − U can exceed E.
- **Mesh inputs** work in `modalrom.py` only (`-m`/`-Y`/`-n` set the material). They have no Dirichlet constraints: the 6 rigid modes are excluded, and reachability load targets need constraints. `make_benchmark.py` needs a scene file or a stored FemElastoDynamics.
- **Scene loading** imports `python/tools/sim/state.py` (and Warp, without initializing the GPU) and the editor's `python/tools/vbd` modules, so the tools must run from a full checkout.
- **`make_benchmark.py` overwrites** the `--output` file.
- **Loading a set elsewhere:** `static.load_sample_batch("<dataset>.h5:Y1e6/static/E10000")` (GPU tools); the GUI's Static Equilibrium panel lists a dataset's sets.
- **GUI costs:** the first Sample click pays for burn-in and the pilot run, and "Solve target" blocks the UI for 2–6 s.
- **Shared venv:** never reinstall pbatoolkit from a worktree.

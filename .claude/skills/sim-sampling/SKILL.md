---
name: sim-sampling
description: Sample deformed shapes (and velocities) of an elastic body with the quadratic-manifold modal ROM MCMC samplers in python/examples (modalrom.py GUI or headless, rom_mcmc.py), check sample quality, and generate static/dynamic benchmark problem sets with make_benchmark.py. Use when asked to draw shape samples at an energy, pick a sampler, inspect or validate a sample batch, or build benchmark datasets.
---

# Shape sampling (modal ROM MCMC)

The sampling tools live in `python/examples`, everything runs on the **CPU**, and the in-source docs are complete:
- `modalrom.py`: GUI and headless CLI.
- `rom_mcmc.py`: samplers and the sample-file contract in `write_samples`.
- `rom_reach.py`: full-order equilibrium solves.
- `make_benchmark.py`: problem-file layout.

Read those docstrings for anything not covered here.

Run from `python/examples` with `C:\git\python-3.12\Scripts\python.exe` (Bash: `/c/git/python-3.12/Scripts/python.exe`). Inputs are `file.h5:h5path` (an editor Dump's `fem` group) or a tetrahedral mesh. Use absolute paths: `data/` exists only in the main checkout.

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
python modalrom.py -i C:/git/PhysicsBasedAnimationToolkit/data/scenarios/trampoline.Dump.h5:fem

# Headless microcanonical batch (ensemble="total", writes samples/qdot), prints acceptance, thinning, tau, ESS, split R-hat
python modalrom.py -i <Dump>.h5:fem --headless -o batch.h5 --sampler microcanonical --energy 10 --samples 400

# Headless fixed-potential batch, every shape at U = E* (ensemble="potential")
python modalrom.py -i <Dump>.h5:fem --headless -o batch.h5 --sampler polar --energy 10 --samples 100

# Benchmark problem sets: pool/E{E}.h5, static/E{E}.h5, dynamic_h{h}/E{E}.h5 (the same samples in every set)
python make_benchmark.py -i <Dump>.h5:fem -o <dir> --energies 50 500 5000 --problems 10 --dt 1e-4 1e-2
python make_benchmark.py -i <Dump>.h5:fem -o <dir> --dt              # static only
python make_benchmark.py -i <Dump>.h5:fem -o <dir> --problems 3 --pool 40 --chains 2 --burn-in 100 --pilot 100   # ~1 min smoke test
```

Translating requests:
- "N samples": `--samples N` (headless) or `--pool N` (`make_benchmark`).
- "Energies a, b": `--energies a b`.
- "M problems per energy": `--problems M`.
- "Time steps h1, h2": `--dt h1 h2`.
- "Static only": `--dt` with no value.
- "With gravity": `--gravity gx gy gz` (headless) or the GUI checkbox.
- "Reproducible": `--seed`.
- "More modes": `-k`.

`make_benchmark.py -h` lists the rest (`--chains`, `--burn-in`, `--pilot`, `--eps`, `--newton-iters`).

**Full benchmark dataset generation is on hold** until the maintainer fixes the data requirements. Small smoke-test sets are fine.

## Output files

All batches follow the sample-file contract (`rom_mcmc.write_samples`):
- `X` (N,3) rest positions, `samples/x` (S,N,3) float32 positions, plus `samples/q`, `energy`, `log_target`, `samples/min_J`.
- `basis/{U,Q,omega,M}`.
- Root attributes: `target_energy`, `rhat`, `ensemble`, `gravity`, `mode_normalization="mass"`, `importance_weights_required`.
- Problem files add `problem`, `pool_file`, `pool_indices`.
- Dynamic files also add `h`, `aext`, and `samples/{v, xtilde, x_ref, ref_newton_*, ref_min_J}`.

Use `samples/x`, not `samples/q`: modes have been mass-normalized since `c03f8cd4`, so `q` from older files isn't comparable.

## Quality checks (report them with every batch)

- **`rhat < 1.05`.** Otherwise raise `--samples`/`--pool` or `--burn-in`. Microcanonical with 100 samples is typically about 1.08, which fails.
- **`energy_noise_warning = False`.** pbat's energy is float32, with ~1e-4 J absolute noise, so energies ≲ 1 J on the beam can't be resolved.
- **`star_violations = 0`** for polar.
- **`importance_weights_required = False`**, i.e. not a ray batch.
- **`samples/min_J`.** Values ≤ 0 mean inverted elements. They're allowed, but report them.
- **Dynamic files.** `samples/ref_newton_correction` should be far below |x_ref − x|: about 3.5–4 decades on the trampoline.

## Pitfalls

- **Gravity** is off unless `--gravity` (or the GUI checkbox). Static benchmark problems must be gravity-free, so their equilibrium is exactly X. Dynamic sets put the scene's gravity only into `xtilde`.
- **Mesh inputs** (not a Dump) have no Dirichlet constraints: the 6 rigid modes are excluded, and reachability load targets need constraints.
- **`make_benchmark.py` overwrites** existing files in `--output`.
- **GUI costs:** the first Sample click pays for burn-in and the pilot run, and "Solve target" blocks the UI for 2–6 s.
- **Shared venv:** never reinstall pbatoolkit from a worktree.

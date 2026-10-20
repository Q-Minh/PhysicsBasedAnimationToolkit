---
name: sim-benchmarks
description: Run the solver convergence benchmarks of the GPU simulator — generate static/dynamic problem sets from MCMC shape samples, run the static and dynamic benchmark (benchmark.py) over solvers, scenes and samples with chosen accuracy levels, budgets and parameters, read its results and plot them (benchmark_plot.py). Use when asked to benchmark solvers, compare convergence or time-to-accuracy, rerun a benchmark case, or vary benchmark parameters.
---

# Solver benchmarks

**The methodology is the maintainer's "Benchmark design" (plans/MASTER.md)**: three benchmarks (static, h = 1e-4, h = 1e-2), scenes on the x axis, accuracy levels, incremental time per level as a fraction of a per-scene budget of k x Newton's time to 99%. **Pilot and full runs are scheduled with the maintainer: don't launch them, nor full dataset generation, without the go-ahead.** Smoke tests on a few samples are fine.

Run from the repo root with the dev venv's Python (Python <= 3.12; its location is machine-specific, see `CLAUDE.local.md`). Scene data lives in the main checkout's git-ignored `data/`; from a worktree, point to it as `path/to/PhysicsBasedAnimationToolkit/data/...`.

## Pipeline

1. **Problems.** `python/examples/make_benchmark.py` draws microcanonical samples on the quadratic-manifold ROM and writes **one dataset file** holding:
   - `fem/`: the FemElastoDynamics the problems are posed on. The benchmark solves on it, so it needs no scene.
   - `pools/pool/E{E}`, `pools/pool_gravity/E{E}`: all the samples.
   - `static/E{E}`: release the shape with no loads; the solution is the rest shape X.
   - `dynamic_h{h}/E{E}`: one BDF1 step from (x, v), with Newton reference solutions `samples/x_ref`.

   The root attribute `sets` lists the problem sets. See the `sim-sampling` skill for options and quality checks (R-hat < 1.05). Small existing datasets: `data/benchmarks/trampoline-test.h5` (3 problems per set) and `trampoline-sample.h5` (4).
2. **Static benchmark.** `python -m python.tools.sim.benchmark` releases each sample with every solver and logs the error to the rest shape after every iteration. Per solve it records when each accuracy level (error reduction 1 − e/e0, default 50, 60, 70, 80, 85, 90, 95, 99%) is first crossed and the incremental time per level.
3. **Dynamic benchmark.** The same runner on a `dynamic_h{h}` set: one BDF1 step of the set's h toward its `xtilde`, with the error measured against `samples/x_ref`. The kind and h come from the set.
4. **Figures.** `python -m python.tools.sim.benchmark_plot results.h5 -o out/fig/benchmark` draws one 3D grouped bar chart per benchmark (static, each dynamic h).

One run covers every set of the dataset, or those matching `--sets`. The sets of one kind and h form a **benchmark** (`static`, `dynamic_h0.0001`, `dynamic_h0.01`), one figure each. Its **cases** (x axis) are the scenes: a set's `scene` attribute, or, for datasets without one (single-scene version 1), the stem of the dataset's `input` scene file. A scene with several selected energy levels gives one case per level (`<scene>_E<E>`).

## Running

Error: e = sqrt((x − x*)ᵀ M (x − x*)) over free nodes (lumped mass), with x* = X (static) or x_ref (dynamic). Each solve stops at the highest level (`--target-reduction`) or at its time budget or iteration cap. VBD-family solves run as one captured CUDA graph. Every iteration is timed from device timestamps, with the logging excluded, also inside replayed graph blocks.

**Level attribution:** an iteration's time goes to the highest level it crosses. The levels it jumps over are reached with zero time (`skipped`), e.g. a Newton step from 0 to 80% puts its whole time on the 80% bar. A level crossed after the budget doesn't count as reached.

**Budgets:** with `--budget-from pilot.h5`, each case's budget is `--budget-factor` (default 2) times the `--budget-statistic` (default median) of Newton's time to the pilot's highest level in that case. Without a pilot, or for cases where no pilot Newton solve reached that level (with a warning), the budget is `--max-seconds`. Each case records its `budget` and `budget_source`. The pilot is an ordinary results file, from a Newton-only run with a generous `--max-seconds`.

```bash
# One quick solve (smoke test)
python -m python.tools.sim.benchmark --dataset data/benchmarks/trampoline-test.h5 \
    --sets static/E50 --solvers AAAVBD --sample-end 1 --max-seconds 5 -o out/one.h5

# Pilot (Newton only), then every solver with budgets of 2x Newton's median time to 99%
python -m python.tools.sim.benchmark --dataset DATASET.h5 --solvers Newton --max-seconds 60 \
    --sim-params python/tools/sim/configs/benchmark/shared.toml -o out/pilot.h5
python -m python.tools.sim.benchmark --dataset DATASET.h5 --budget-from out/pilot.h5 \
    --sim-params python/tools/sim/configs/benchmark/shared.toml --scene-params Y1e8=y1e8.toml -o out/results.h5
python -m python.tools.sim.benchmark_plot out/results.h5 -o out/fig/benchmark

# Only some sets (fnmatch patterns over the names in the dataset's `sets` attribute)
python -m python.tools.sim.benchmark --dataset DATASET.h5 --sets "*/static/*" "Y1e6/dynamic_h0.01/*" -o out/some.h5

# Parameter variations: configs layered left to right, then single-key overrides
python -m python.tools.sim.benchmark ... --sim-params base.toml static.toml \
    --set Solver.Newton.LinearSolver.n_max_iters=3000 --set Solver.AAAVBD.rhohat=0.9

# Reproducible case: save the spec (dataset, sets, sample range, caps), rerun it
python -m python.tools.sim.benchmark ... --save-spec out/case.h5 -o out/a.h5
python -m python.tools.sim.benchmark --spec out/case.h5 -o out/b.h5
```

The problems are posed on the dataset's stored FEM (per scene), whose Dirichlet nodes stay fixed at rest. No scene or Dump is passed. **Parameters:** `--sim-params` applies to every scene, then `--scene-params SCENE=A.toml[,B.toml]` (once per scene), then `--set`. `configs/benchmark/shared.toml` holds the settings every scene shares: AAAVBD's shared parameter set and the Inverse vertex solve for every VBD-family solver.

Translating requests:

| Request | Flag |
|---|---|
| "Only the static problems" / "only h = 0.01 at 500 J" | `--sets "static/*"` / `--sets dynamic_h0.01/E500` (multi-scene datasets prefix the scene: `"*/static/*"`, `Y1e6/dynamic_h0.01/*`; default: every set) |
| "Only solvers X, Y" | `--solvers X,Y` (from VBD, AAAVBD, Chebyshev, AndersonSolver, Newton; default all, in that order) |
| "Samples i to j" | `--sample-begin i --sample-end j+1` (`-1` = all), in every selected set |
| "Accuracy levels L1, L2, ..." | `--levels 0.5,0.9,0.99` (error reductions; the highest is the default target) |
| "Budget k x Newton" | `--budget-from pilot.h5 --budget-factor k` (`--budget-statistic mean` for the mean) |
| "Time budget T s per solve" | `--max-seconds T` (default 60), without a pilot |
| "Iteration budget N" | `--max-iterations N`: counts VBD sweeps, **Anderson steps** (p sweeps each when `vbd_iters_per_anderson_iter = p`) or Newton steps |
| "Stop at 95%" | `--target-reduction 0.95` (default: the highest level) |
| "With a load" | the StaticSolveSpec's `fext_scale` (via `--spec`). Static benchmark problems are load-free by design |
| "Use parameters P" / "change key K" / "for scene S" | `--sim-params P.toml ...` / `--set K=V` / `--scene-params S=P.toml` (see the `sim-run` skill; check with `python -m python.tools.sim.config show`) |
| "Coarser/finer graph blocks" | `--block N` (default 16, rounded up to a multiple of Anderson's m). Iterations a block runs past the stop are dropped, so it doesn't change the results |
| "Other camera" | `benchmark_plot --elev E --azim A` (defaults 28, -62) |

`benchmark.py`'s module docstring has the full details.

## Outputs

- **`results.h5`** (layout in `benchmark_results.py`'s docstring):
  - root attributes: levels, budget factor, statistic and source, dataset, device, commit, date, parameter sources, solvers, Newton CG limits; the `StaticSolveSpec` and `dataset_attrs` groups;
  - `/<benchmark>/<case>`: the case's scene, label, energy, set, `budget` and `budget_source`;
  - `/<benchmark>/<case>/<solver>`: per sample, e0, iterations, sweeps, solver time, final e/e0, `termination` (`target`, `iteration_cap`, `time_cap`, `non_finite`, or Newton's `newton_non_descent`, `newton_line_search_failed` or `newton_zero_gradient`; Newton stops at its first rejected step, at the float32 floor usually `newton_line_search_failed`), and per sample and level `crossing_time`, `crossing_iteration`, `increment`, `reached` and `skipped` (NaN or -1 when not reached);
  - `aggregate/{increment,crossing_time}/{count,mean,std,min,max,median,q25,q75}` per level, over the samples that reached it;
  - `logs/<sample>/{error,time,sweeps}`: the full per-iteration logs.
- **Dynamic cases** add each sample's reference accuracy in the error's M-norm (`reference_error_bound`, `reference_relative_error_bound`), from the set's `samples/ref_error_mnorm`, and `resolvable`: the levels that accuracy resolves.
- **Figures** (`benchmark_plot.py`): per benchmark, `<prefix>_<benchmark>.pdf` and `.png`. x = case, depth = level, height = mean increment / budget, one bar per solver per cell, translucent min-max boxes, `n/N` labels where not every sample reached the level, no bar where none did.

When comparing solvers, use the time to reach each error level, not the time per iteration. Also report iterations and sweeps. Anderson iterations aren't sweeps when p > 1.

## Pitfalls

- **Contact** is always off, for both problem kinds.
- **Dynamic h:** dynamic problems take h from the set and override `Integration.dt`, the substeps and the BDF order. At small h (1e-4), steps start close to their solution (e0 ~ 1e-3), so float32 rounding floors the error around e/e0 ~ 1e-5 to 1e-4.
- **Solver parameters:** they come only from `--sim-params`, `--scene-params` and `--set`; without them, the code defaults apply (`configs/defaults.toml`). Pass the scene's config (e.g. `configs/paper/<scene>.toml`) for meaningful solver parameters.
- **Stopping:** solver `gtol`, `n_max_iters` and `n_subproblem_max_iters` don't stop benchmark solves; the target and the caps do. Newton's CG absolute tolerance is set to 0 (its relative one stays), since in h²-scaled units at small h the gradient drops below it after one step. Vertex-solve and acceleration parameters still matter.
- **Newton** runs eagerly with host syncs, so its times are noisier. Static problems need `Solver.Newton.LinearSolver.n_max_iters` of about 1500–3000; the default 500 leaves Newton steps inexact.
- **Quiet GPU:** timings are only valid while nothing else runs. Reserve the GPU in `plans/MASTER.md` while parallel sessions are active.
- **Problem sets:** sets that need importance weights (ray sampler) are refused. R-hat ≥ 1.05 and nonzero gravity only warn. Static problems should be gravity-free.
- **First solve:** the first solve per solver and set includes kernel compilation and graph capture in its host time, but not in the reported times.
- **Not sim-params:** `fem.static` and the StaticSolveSpec aren't part of the config system.

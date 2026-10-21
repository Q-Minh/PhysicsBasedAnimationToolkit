---
name: sim-benchmarks
description: Run the solver convergence benchmarks of the GPU simulator — generate static/dynamic problem sets from MCMC shape samples, run the static and dynamic benchmark (benchmark.py) over solvers, scenes and samples with chosen accuracy levels, budgets and parameters, read its results, plot them (benchmark_plot.py) and play its solver trajectories side by side (benchmark_viewer.py). Use when asked to benchmark solvers, compare convergence or time-to-accuracy, rerun a benchmark case, vary benchmark parameters, or watch what the solvers do during a benchmark solve.
---

# Solver benchmarks

**The methodology is the maintainer's "Benchmark design" (plans/MASTER.md)**: three benchmarks (static, h = 1e-4, h = 1e-2), scenes on the x axis, accuracy levels per benchmark, incremental time per level as a fraction of Newton's own time T_N on the same problem (per sample, no pilot). **Full runs are scheduled with the maintainer: don't launch them, nor full dataset generation, without the go-ahead.** Smoke tests on a few samples are fine.

Run from the repo root with the dev venv's Python (Python <= 3.12; its location is machine-specific, see `CLAUDE.local.md`). Scene data lives in the main checkout's git-ignored `data/`; from a worktree, point to it as `path/to/PhysicsBasedAnimationToolkit/data/...`.

## Pipeline

1. **Problems.** `python/examples/make_benchmark.py` draws microcanonical samples on the quadratic-manifold ROM and writes **one dataset file** with one group per named scene (version 2; contract in plans/MASTER.md), each holding:
   - `<scene>/fem/`: the FemElastoDynamics the scene's problems are posed on. The benchmark solves on it, so it needs no scene file.
   - `<scene>/pools/pool/E{E}`, `<scene>/pools/pool_gravity/E{E}`: all the samples.
   - `<scene>/static/E{E}`: release the shape with no loads; the solution is the rest shape X.
   - `<scene>/dynamic_h{h}/E{E}`: one BDF1 step from (x, v), with Newton reference solutions `samples/x_ref`.

   The root attribute `sets` lists the problem sets (full paths), and a scene group's optional `label` names it in the figures. Version-1 files (one scene at the root) still run, but none are kept. See the `sim-sampling` skill for options and quality checks (R-hat < 1.05). Small existing dataset: `data/benchmarks/trampolines-test.h5` (scenes Y1e6 and Y1e7, 2 problems per set).
2. **Static benchmark.** `python -m python.tools.sim.benchmark` releases each sample with every solver and logs the error to the rest shape after every iteration. Per solve it records when each accuracy level (error reduction 1 − e/e0, default decades: 50, 90, 99, 99.9, 99.99, 99.999% static, up to 99.99% dynamic; per benchmark with `--levels static=...`) is first crossed and the incremental time per level.
3. **Dynamic benchmark.** The same runner on a `dynamic_h{h}` set: one BDF1 step of the set's h toward its `xtilde`, with the error measured against `samples/x_ref`. The kind and h come from the set.
4. **Figures.** `python -m python.tools.sim.benchmark_plot results.h5 -o out/fig/benchmark` draws one 3D grouped bar chart per benchmark (static, each dynamic h). `python -m python.tools.sim.benchmark_curves results.h5 -o out/fig/curves --problems static/Y1e6 dynamic_h0.01/Y1e7` draws convergence curves (e/e0, log, against time / T_N, one curve per solver, one panel per problem) from the logs, for the tails the bars summarize; run the benchmark with `--budget-factor 5` or so for them to extend past T_N.
5. **Trajectories.** `python -m python.tools.sim.benchmark_viewer results.h5 [--benchmark static --case Y1e6 --sample 0]` plays a `--snapshots` run's trajectories in Polyscope (without a file, open one with Load). It shows every solver side by side, in sync in solver time. Translucent overlays at every copy, each with its own checkbox and a color legend, show the problem: for dynamic problems the reference x* (`x_ref`, on by default), the inertial target x~ (`xtilde`) and the start state x (`samples/x`, the solvers' frame 0); for static problems the rest shape X = x* (off by default). Copies are colored by per-vertex error |x − x*| or displacement on one shared scale. Each solver row gives e/e0, the level reached and buttons to jump to its level crossings, and a log10(e/e0) plot follows the rows. The controls are the CPU editor's trajectory viewer's: Frame slider, Autoplay at F / video_seconds fps (the solve slowed down by `playback_factor`, shown with the solver time and its fraction of T_N), and Screenshot (one `<frame:08d>.png` per frame, into the working directory, for videos). The mesh and x* come from the results file's `dataset` attribute; pass `--dataset path/to/...` if that file has moved. It needs no Warp or GPU.

One run covers every set of the dataset, or those matching `--sets`. The sets of one kind and h form a **benchmark** (`static`, `dynamic_h0.0001`, `dynamic_h0.01`), one figure each. Its **cases** (x axis) are the scenes: a set's `scene` attribute, or, for datasets without one (single-scene version 1), the stem of the dataset's `input` scene file. A scene with several selected energy levels gives one case per level (`<scene>_E<E>`).

## Running

Error: e = sqrt((x − x*)ᵀ M (x − x*)) over free nodes (lumped mass), with x* = X (static) or x_ref (dynamic). Each solve stops at the benchmark's highest level (`--target-reduction`), at its time budget or iteration cap, or, for Newton, at its stall test. VBD-family solves run as one captured CUDA graph. Every iteration is timed from device timestamps, with the logging excluded, also inside replayed graph blocks.

**Level attribution:** an iteration's time goes to the highest level it crosses. The levels it jumps over are reached with zero time (`skipped`), e.g. a Newton step from 0 to 80% puts its whole time on the 80% bar. A level crossed after the budget doesn't count as reached.

**Budgets, per problem (Newton normalization; the maintainer's design):** Newton solves each problem first, to the target, until its **stall test** stops it, or at most `--max-seconds` (default 60 s, a practical cap). Its time T_N to the highest level it reached is every other solver's budget on that problem, times `--budget-factor` (default 1), and every solver's increments there are divided by T_N (`normalized_increment`): Newton's reached increments sum to 1. Each solve records its `budget` and `normalization` (T_N).

- **Stall test** (Newton only): stop once the last K iterations improved the best error by less than delta per iteration on average, min_{j≤c} e_j > (1 − delta)^K min_{j≤c−K} e_j (best-so-far, so float32 noise can't restart it); `--newton-stall-iterations K` (0 = off, default 2), `--newton-stall-reduction delta` (default 0.02). On the 15 floor-check problems no level was lost for K in 2–5 and delta up to ~0.04; larger deltas stop Newton early on static het, whose damped steps reduce e by only 4–9% for ~60 steps, and on static Y1e5's last level. The stall's extra K steps cost wall time only: T_N is a crossing time.
- **Fixed budgets:** `--budget [BENCHMARK[/CASE]=]SECONDS` (repeatable; the case's entry, then its benchmark's, then a bare number for every case) gives a case a fixed budget instead: every solver, Newton included, gets it, the increments are divided by it, and Newton isn't needed (e.g. smoke tests).

```bash
# One quick solve (smoke test)
python -m python.tools.sim.benchmark --dataset data/benchmarks/trampolines-test.h5 \
    --sets Y1e6/static/E10000 --solvers AAAVBD --sample-end 1 --budget 5 -o out/one.h5

# The benchmark: per problem, Newton first, then every other solver within Newton's time T_N; the default
# levels (static to 99.999%, dynamic to 99.99%); Newton's CG cap is 1000 for the static Y1e8 and het problems
python -m python.tools.sim.benchmark --dataset DATASET.h5 --sim-params python/tools/sim/configs/benchmark/shared.toml \
    --scene-params static/Y1e8=python/tools/sim/configs/benchmark/newton_cg1000.toml \
    --scene-params static/het=python/tools/sim/configs/benchmark/newton_cg1000.toml -o out/results.h5
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

The problems are posed on the dataset's stored FEM (per scene), whose Dirichlet nodes stay fixed at rest. No scene or Dump is passed. **Parameters:** `--sim-params` applies to every scene, then `--scene-params SCENE=A.toml[,B.toml]` (every benchmark of the scene), then `--scene-params BENCHMARK/SCENE=A.toml[,B.toml]` (one benchmark's problems of it, e.g. `static/Y1e8=`), then `--set`; each key at most once. Each case records its files in `scene_params`. `configs/benchmark/shared.toml` holds the settings every scene shares: AAAVBD's shared parameter set, the Inverse vertex solve for every VBD-family solver, and Anderson's 2 sweeps per step. `configs/benchmark/newton_cg1000.toml` raises Newton's CG cap to 1000, for `static/Y1e8` and `static/het`.

Translating requests:

| Request | Flag |
|---|---|
| "Only the static problems" / "only h = 0.01 at 500 J" | `--sets "static/*"` / `--sets dynamic_h0.01/E500` (multi-scene datasets prefix the scene: `"*/static/*"`, `Y1e6/dynamic_h0.01/*`; default: every set) |
| "Only solvers X, Y" | `--solvers X,Y` (from VBD, AAAVBD, Chebyshev, AndersonSolver, Newton; default all, in that order) |
| "Samples i to j" | `--sample-begin i --sample-end j+1` (`-1` = all), in every selected set |
| "Accuracy levels L1, L2, ..." / "for the static figure only" | `--levels 0.5,0.9,0.99` / `--levels static=0.5,0.9,0.99,0.999` (error reductions; a benchmark's own entry wins; its highest level is its default target) |
| "Budget k x Newton" | `--budget-factor k` (default 1; per problem, k x Newton's T_N there) |
| "Newton's cap" / "Newton's stall test" | `--max-seconds S` (default 60) / `--newton-stall-iterations K --newton-stall-reduction D` (K = 0: off) |
| "Time budget T s per solve" | `--budget T`, or per benchmark/case `--budget static=T`, `--budget static/Y1e8=T` (instead of Newton's times; Newton not needed) |
| "Iteration budget N" | `--max-iterations N`: counts VBD sweeps, **Anderson steps** (p sweeps each when `vbd_iters_per_anderson_iter = p`) or Newton steps |
| "Stop at 95%" | `--target-reduction 0.95` (default: the benchmark's highest level) |
| "With a load" | the StaticSolveSpec's `fext_scale` (via `--spec`). Static benchmark problems are load-free by design |
| "Use parameters P" / "change key K" / "for scene S" / "for scene S, static only" | `--sim-params P.toml ...` / `--set K=V` / `--scene-params S=P.toml` / `--scene-params static/S=P.toml` (see the `sim-run` skill; check with `python -m python.tools.sim.config show`) |
| "Coarser/finer graph blocks" | `--block N` (default 16, rounded up to a multiple of Anderson's m). Iterations a block runs past the stop are dropped, so it doesn't change the results |
| "Other camera" / "solver X in front" | `benchmark_plot --elev E --azim A` (defaults 28, -62) / `--solver-order A,B,...` (back to front; default: taller bars further back, by the 75th percentile of each solver's bar heights per figure) |
| "Show / watch what the solvers do" | `benchmark_viewer results.h5` on a `--snapshots` run |
| "Trajectories for a video" | `--snapshots`: F = `--snapshot-video-seconds` x `--snapshot-fps` frames (default 5 s x 24 = 120; `--snapshot-frames F` overrides) at equal solver-time intervals over the problem's budget (factor x T_N, Newton's frames included; or the fixed budget), plus level crossings, for `--snapshot-fraction` of each set's samples (default 1: all; ~F x N x 12 bytes each, ~15 MB on the trampolines; the run prints its disk estimate first). `--snapshot-memory-mb` (default 32) caps the device ring of `--snapshot-ring-ratio` x F slots (default 2) |

`benchmark.py`'s module docstring has the full details.

## Outputs

- **`results.h5`** (layout in `benchmark_results.py`'s docstring):
  - root attributes: the `--levels` options, budget factor and options, Newton's stall test, dataset, device, commit (and `git_dirty`), date, parameter sources, solvers, software versions (`version_*`: Warp, CUDA driver and toolkit, pbatoolkit); the `StaticSolveSpec` and `dataset_attrs` groups;
  - `/<benchmark>`: its `levels`;
  - `/<benchmark>/<case>`: the case's scene, label, energy, set, `budget` (fixed budgets only), `budget_source`, `max_seconds` (Newton's cap), `scene_params`, every solver parameter of the case's simulation as TOML (`sim_params_toml`), Newton's CG limits (`newton_cg_*`), and the GPU's state at the case's start and end (`gpu_start_*`, `gpu_end_*`: clocks, temperature, power, pstate and active clock-event (throttle) reasons, from nvidia-smi, found via `--nvidia-smi`, `NVIDIA_SMI`, `PATH` or the standard install locations);
  - `/<benchmark>/<case>/<solver>`: per sample, e0, iterations, sweeps, solver time, host time (the solve's wall-clock time: minus the solver time, the per-solve overhead), `budget`, `normalization` (T_N), final e/e0, `termination` (`target`, `iteration_cap`, `time_cap`, `non_finite`, `stalled` (Newton's stall test), or Newton's `newton_non_descent`, `newton_line_search_failed` or `newton_zero_gradient`), and per sample and level `crossing_time`, `crossing_iteration`, `increment`, `normalized_increment`, `reached` and `skipped` (NaN or -1 when not reached);
  - `aggregate/{increment,normalized_increment,crossing_time}/{count,mean,std,min,max,median,q25,q75}` per level, over the samples that reached it;
  - `logs/<sample>/{error,time,sweeps}`: the full per-iteration logs; for Newton also `logs/<sample>/newton/{status,cg_iterations,cg_capped,cg_residual,cg_tolerance,backtracks}`, one entry per Newton step.
- **Dynamic cases** add each sample's reference accuracy in the error's M-norm (`reference_error_bound`, `reference_relative_error_bound`), from the set's `samples/ref_error_mnorm`, and `resolvable`: the levels that accuracy resolves.
- **Trajectories** (`--snapshots`): `/<benchmark>/<case>/<solver>/trajectories/<sample>/{iteration, t, t_target, x, level_frames, level_exact}`, with the attribute `playback_factor` = video seconds / the problem's budget (per trajectory; how much the video slows solver time down, e.g. 200 at a 25 ms budget), frame k of every solver at solver time k x budget / F (true-speed side-by-side playback), plus the end (at most the budget) and each level crossing within it. About F x N x 12 bytes per solve on disk. Snapshots are taken in the same timed run, outside the timed span, from a fixed device ring (at most `--snapshot-memory-mb`); results are unchanged.
- **Figures** (`benchmark_plot.py`): per benchmark, `<prefix>_<benchmark>.pdf` and `.png`. x = case, depth = the benchmark's levels, height = mean increment / T_N (or / the fixed budget), one bar per solver per cell, translucent min-max boxes, `n/N` labels where not every sample reached the level and always at the highest level (also `0/N`, so a stalled solver shows), empty cells where no sample reached the level or every one that did jumped over it, and a red x instead of bars where some sample's reference can't resolve the level (dynamic `resolvable`).

When comparing solvers, use the time to reach each error level, not the time per iteration. Also report iterations and sweeps. Anderson iterations aren't sweeps when p > 1.

## Pitfalls

- **Contact** is always off, for both problem kinds.
- **Dynamic h:** dynamic problems take h from the set and override `Integration.dt`, the substeps and the BDF order. At small h (1e-4), steps start close to their solution (e0 ~ 1e-3), so float32 rounding floors the error around e/e0 ~ 1e-5 to 1e-4.
- **Solver parameters:** they come only from `--sim-params`, `--scene-params` and `--set`; without them, the code defaults apply (`configs/defaults.toml`). Pass the scene's config (e.g. `configs/paper/<scene>.toml`) for meaningful solver parameters.
- **Stopping:** solver `gtol`, `n_max_iters` and `n_subproblem_max_iters` don't stop benchmark solves; the target and the caps do. Newton's CG absolute tolerance is set to 0 (its relative one stays), since in h²-scaled units at small h the gradient drops below it after one step. Vertex-solve and acceleration parameters still matter.
- **Newton** runs eagerly with host syncs, so its times, and the budgets T_N they set, are noisier. At a float32 floor its line search accepts steps within the energy's roundoff, so it keeps stepping: the stall test ends it. Its CG cap (`Solver.Newton.LinearSolver.n_max_iters`) trades exact steps for time: the pilot (2026-10-20) found the default 500 fastest to 99%, except on the static Y1e8 and het problems (1000, `newton_cg1000.toml`). Re-check it when the levels change.
- **Quiet GPU:** timings are only valid while nothing else runs. Reserve the GPU in `plans/MASTER.md` while parallel sessions are active.
- **Problem sets:** sets that need importance weights (ray sampler) are refused. R-hat ≥ 1.05 and nonzero gravity only warn. Static problems should be gravity-free.
- **First solve:** the first solve per solver and set includes kernel compilation and graph capture in its host time, but not in the reported times.
- **Not sim-params:** `fem.static` and the StaticSolveSpec aren't part of the config system.

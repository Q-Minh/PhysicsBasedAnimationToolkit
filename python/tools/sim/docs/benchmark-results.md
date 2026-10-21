# Solver convergence benchmark: data and scripts

The solver convergence benchmark compares five GPU solvers (VBD, AAAVBD, Chebyshev, Anderson and Newton) on many
sampled problems. It measures how much solver time each needs to remove a given fraction of its initial error. This
note describes the two files a run involves and the scripts that read them. It applies to the full run of
21 October 2026, on the five circle trampolines.

| File | What it holds | Size |
|---|---|---|
| `data/benchmarks/trampolines.h5` | **Dataset:** the problems (start states, inertial targets, reference solutions), per scene and problem set | 1.1 GB |
| `data/benchmarks/results/trampolines.h5` | **Results:** per problem and solver, the full per-iteration error log, level crossings, budgets, terminations, Newton's per-step records, and video frames for some problems | 8.8 GB |

Both are HDF5 and need only `h5py` and `numpy` to read. Neither is in git (`data/` is git-ignored), so they're
shared as files. The scripts are in this repository and run from its root as modules (`python -m python.tools.sim.<name>`).

## 1. Concepts

- **Scenes.** Five circle trampolines on the same mesh, 10643 nodes (47 of them fixed) and 41588 tetrahedra.
  They differ only in material:
  - `Y1e5`, `Y1e6`, `Y1e7` and `Y1e8`: uniform Young's modulus 1e5 to 1e8 Pa;
  - `het`: a stiff center (1e8 Pa, 10049 tets) inside a soft ring (1e6 Pa, 31539 tets).

  All use Poisson's ratio 0.45 and Stable Neo-Hookean elasticity. The benchmark has no contact.
- **Problem sets and benchmarks.** Each scene has three problem sets of 32 problems each, sampled at a total energy of
  10000 J: `static`, `dynamic_h0.01` and `dynamic_h0.0001`. The sets of one kind form a benchmark (one figure); each
  scene is a case within it.
  - **static:** release a deformed shape x with no load. The solution x\* is the rest shape X.
  - **dynamic (h):** one implicit-Euler step of size h from a state (x, v), with gravity. The solution x\* is a
    full-order Newton reference, `x_ref`.
- **Error.** e = ‖x − x\*‖_M is the lumped-mass norm over free nodes, logged after every solver iteration. The
  benchmark scores the relative error e/e0.
- **Accuracy levels.** Error reductions 1 − e/e0, in decades:
  - static: 50, 90, 99, 99.9, 99.99 and 99.999%;
  - dynamic: up to 99.99%.

  A level is **crossed** at the first logged iterate that reaches it. Its **increment** is the time since the
  previous level's crossing; an iterate that jumps several levels gives its time to the highest one, and the others
  are **skipped** with zero time.
- **Budgets and normalization.**
  - On every problem, Newton runs first: until it reaches the top level, its stall test stops it, or at most 60 s.
  - Its time to the highest level it reached is that problem's **T_N**.
  - Every other solver gets a budget of 2 × T_N on that problem (`budget_factor = 2`).
  - Times are reported over T_N (`normalized_increment`), so Newton's increments on a problem sum to 1. The bar
    charts score within 1 × T_N by re-reading the logs; the curves and iteration tables use the full 2 × T_N.
- **Reference accuracy.** The dynamic references are float32 Newton solves, so they resolve errors only down to
  about 1e-4 × e0. `resolvable` says, per problem and level, whether the reference is accurate enough to score that
  level. The figures leave a level out of a case when most problems' references can't resolve it.
- **What counts as solver time.** Only the solve itself, timed on the device. Logging, snapshots and warm-up are
  excluded; `host_time` is the wall-clock time around the whole solve.

## 2. Results file layout

```
/                                   root attributes (2.1)
/StaticSolveSpec                    the run's specification, as attributes
/dataset_attrs                      the dataset's root attributes (sampler settings, commit)
/<benchmark>                        static | dynamic_h0.0001 | dynamic_h0.01; attrs: cases, levels
/<benchmark>/<case>                 Y1e5 | Y1e6 | Y1e7 | Y1e8 | het; attrs (2.2)
/<benchmark>/<case>/<solver>        VBD | AAAVBD | Chebyshev | AndersonSolver | Newton; attrs: solver, n_samples
    per-problem arrays (2.3)
    aggregate/<quantity>/<statistic>
    logs/<sample:06d>/...           the full per-iteration log of each solve (2.4)
    trajectories/<sample:06d>/...   video frames, for 8 of the 32 problems (2.5)
```

Iterate cases in file order through the `benchmarks` and `cases` attributes (`benchmark_results.iter_cases`).

### 2.1 Root attributes

| Attribute | Meaning |
|---|---|
| `format`, `version` | `pbat-benchmark-results`, 1 |
| `benchmarks`, `solvers` | the benchmarks and solvers, in figure order |
| `budget_factor` | 2.0: other solvers' budget over T_N |
| `levels_options` | the levels per benchmark, as command-line options |
| `newton_stall_iterations`, `newton_stall_reduction` | Newton's stall test: stop when the best error improved by less than 2% per iteration on average over the last 2 iterations |
| `dataset` | the dataset path, relative to the repository root |
| `run_file`, `run_toml`, `run_toml_resolved` | the run file (`python/tools/sim/configs/benchmark/trampolines.toml`): its path, its text, and its resolved parameters |
| `sim_params`, `budget_options` | the parameter sources and fixed budgets given on the command line (none here) |
| `date`, `device`, `git_commit`, `git_dirty`, `version_warp`, `version_cuda_driver`, `version_cuda_toolkit`, `version_pbatoolkit` | provenance |
| `note_trajectories_relabelled` | trajectories were renamed to their own samples after the run (see 4) |

### 2.2 Case attributes

`benchmark`, `scene`, `label` (the scene's display name), `set` (the dataset set, e.g. `het/dynamic_h0.01/E10000`),
`problem` (`static` or `dynamic`), `h`, `energy` (J), `budget_source` (how budgets were set), `max_seconds` (Newton's
cap), the case's Newton CG settings (`newton_cg_rtol`, `newton_cg_n_max_iters`, `newton_cg_atol`), and
`sim_params_toml`, every solver parameter the case ran with.

### 2.3 Per-solver arrays

S = 32 problems; L = 6 levels (static) or 5 (dynamic). Row i of every array is problem `sample[i]`.

| Dataset | Shape | Meaning |
|---|---|---|
| `sample` | (S,) | the problem's index in the dataset set |
| `e0` | (S,) | initial error ‖x0 − x\*‖_M |
| `final_relative_error` | (S,) | e/e0 at the end of the solve |
| `iterations`, `sweeps` | (S,) | solver iterations (log entries − 1) and VBD sweeps (0 for Newton) |
| `solver_time`, `host_time` | (S,) | timed solve (s); wall-clock time around it (s) |
| `budget` | (S,) | the solve's time budget (s): 60 for Newton, 2 × T_N for the others |
| `normalization` | (S,) | T_N (s), the same for every solver on a problem |
| `termination` | (S,) | `target` (reached the top level), `time_cap`, `stalled` (Newton's stall test), `iteration_cap`, `non_finite`, or for Newton `newton_non_descent`, `newton_line_search_failed`, `newton_zero_gradient` |
| `deformation_energy`, `min_J` | (S,) | the start state's potential energy (J) and smallest det F |
| `reference_error_bound`, `reference_relative_error_bound` | (S,) | dynamic only: the reference's accuracy, absolute and over e0 |
| `crossing_time` | (S, L) | solver time (s) of each level's first crossing; NaN if not reached |
| `crossing_iteration` | (S, L) | log entry of the crossing; −1 if not reached |
| `increment`, `normalized_increment` | (S, L) | time attributed to the level (s), and the same over T_N |
| `reached`, `skipped` | (S, L) | the level was reached; reached with zero time by an iterate that also crossed a higher level |
| `resolvable` | (S, L) | dynamic only: the reference resolves the level |
| `aggregate/<q>/<stat>` | (L,) | for q in `increment`, `normalized_increment`, `crossing_time`: count, mean, std, min, max, median, q25, q75 over the problems that reached each level, within the run's 2 × T_N |

### 2.4 Logs

`logs/<sample>/` holds one entry per iterate, entry 0 being the start state:

| Dataset | Meaning |
|---|---|
| `error` | e after each iteration (absolute; divide by `error[0]` for e/e0) |
| `time` | cumulative solver time (s) |
| `sweeps` | cumulative VBD sweeps (an Anderson step counts its 2 sweeps; 0 for Newton) |

For the VBD family, an iteration is one sweep (one Anderson step for Anderson). For Newton it's one Newton step, and
`logs/<sample>/newton/` adds one entry per step attempted:
- `status`: `stepped`, `converged`, or why the step was rejected (`non_descent`, `line_search_failed`);
- `cg_iterations`: the step's CG iterations, which come in multiples of 10 because CG checks convergence every 10;
- `cg_capped`: whether CG hit its cap;
- `cg_residual` and `cg_tolerance`: the final CG residual and the tolerance it was tested against;
- `backtracks`: the line search's step halvings.

Log entry j + 1 follows the j-th `stepped` entry.

### 2.5 Trajectories

Problems 0, 4, 9, 13, 18, 22, 27 and 31 of every set have video frames, for every solver: about 120 frames evenly
spaced in solver time over the 2 × T_N budget, plus each level's crossing.

| Dataset or attribute | Meaning |
|---|---|
| `iteration` (F,) | the log entries the frames show |
| `t`, `t_target` (F,) | each frame's solver time, and the time it stands for (NaN for crossing-only frames) |
| `x` (F, N, 3) float32 | node positions, in the dataset FEM's node order |
| `level_frames`, `level_exact` (L,) | the frame of each level's crossing (−1: not reached), and whether it's the crossing's own state |
| attrs `video_seconds`, `playback_factor` | the frames play in 5 s, i.e. slowed down `playback_factor` times relative to solver time |
| attrs `requested_frames`, `ring_slots`, `stride`, `device_bytes` | snapshot-ring bookkeeping |

## 3. Dataset layout (briefly)

`data/benchmarks/trampolines.h5` has one group per scene. Each holds:
- `fem`: the FEM, a serialized `pbatoolkit` `FemElastoDynamics`;
- `X` (N, 3): the rest positions;
- `basis/`: the 20-mode quadratic manifold the problems were sampled on;
- `pools/`: the sampler's chains;
- one group per problem set, `static/E10000`, `dynamic_h0.01/E10000` and `dynamic_h0.0001/E10000`, each with
  `samples/`.

`samples/` holds, per problem (S = 32, N = 10643):
- states: `x`, plus for dynamic sets `v`, `xtilde` (the inertial target) and `x_ref` (the reference solution), all (S, N, 3);
- the reference solve's diagnostics: `ref_error_mnorm`, `ref_newton_iterations` and the other `ref_*` arrays;
- `energy` and `min_J`.

The set's attributes document its definitions (`problem_definition`, `energy_definition`, `layout`) and the sampler's
quality (`rhat`, `ess`). The sampling method is described in `python/examples/docs/sampling-methods.md` (section 8:
benchmark problems).

## 4. Caveats

- **float32 floors.** At h = 1e-4, and for the stiffer dynamic cases, every solver levels off near e/e0 ≈ 1e-4,
  where the float32 reference stops resolving errors. Use `resolvable` before reading 99.99% results.
- **Problems a solver didn't reach count as unreached, not as slow.** Statistics are over the problems that reached
  a level; always read them with the count n/N.
- **Trajectory labels.** The writer first stored the 8 trajectories as samples 0–7. They were renamed in place to
  their true samples (fixed in commit 84e164f1). The logs and timings were never affected.
- **`git_dirty = True`** only reflects an untracked file in the checkout: the tracked code was at `git_commit`.
- **Machine paths.** Some dataset attributes (`input`) hold paths on the machine that generated it; ignore them.

## 5. Reading the results in Python

```python
import h5py
import numpy as np

with h5py.File("data/benchmarks/results/trampolines.h5", "r") as f:
    g = f["dynamic_h0.01/Y1e7/AAAVBD"]
    levels = f["dynamic_h0.01"].attrs["levels"]           # (L,) error reductions
    t_n = g["normalization"][()]                          # (S,) Newton's time per problem
    within = g["crossing_time"][()] <= t_n[:, None]       # (S, L) reached within 1 x T_N (NaN compares False)
    print("problems reaching each level within T_N:", within.sum(axis=0))

    log = g["logs/000000"]                                # one solve's full log
    e = log["error"][()] / log["error"][0]                # e/e0 per iterate
    t = log["time"][()] / t_n[0]                          # solver time over T_N
    sweeps = log["sweeps"][()]
    j = g["crossing_iteration"][0, 3]                     # log entry of the 99.9% crossing (-1: not reached)
    print("sweeps to 99.9%:", sweeps[j] if j >= 0 else None)
```

`python/tools/sim/benchmark_results.py` has the shared helpers: `iter_cases`, `level_crossings`, `rescore` (re-score
at another budget factor) and `aggregate`. It has no GPU dependencies.

## 6. Scripts

Every script below reads existing files: none of them re-runs solves, and only the benchmark itself needs a GPU.
Commands run from the repository root.

| Script | Output | Command |
|---|---|---|
| `benchmark.py` | runs the benchmark and writes a results file (GPU, about 5 h for the full run) | `python -m python.tools.sim.benchmark --run python/tools/sim/configs/benchmark/trampolines.toml -o data/benchmarks/results/trampolines.h5` (add `--dry-run` to check it, `--list` to list the dataset's sets) |
| `benchmark_plot.py` | 3D bar charts, one per benchmark: per scene × level × solver, the mean incremental time over T_N, with min–max boxes and n/N labels | `python -m python.tools.sim.benchmark_plot RESULTS.h5 -o FIGURES/trampolines_bars [--budget-factor 1] [--elev 28 --azim -62]` |
| `benchmark_curves.py` | convergence curves, e/e0 against time over T_N on log-log axes, one panel per problem: one sample, or with `--all-samples` the min–max band and median over all problems | `python -m python.tools.sim.benchmark_curves RESULTS.h5 -o FIGURES/trampolines_curves_Y1e6 --problems static/Y1e6 dynamic_h0.01/Y1e6 dynamic_h0.0001/Y1e6 [--sample 0 \| --all-samples] [--xscale linear]` |
| `benchmark_iterations.py` | tables of iterations to each level, median [25th–75th percentile] VBD sweeps, or Newton steps and CG iterations, with n/N, as Markdown and LaTeX | `python -m python.tools.sim.benchmark_iterations RESULTS.h5 -o TABLES/iterations [--budget-factor 2]` |
| `benchmark_viewer.py` | interactive Polyscope viewer of the trajectories: the solvers side by side, in sync, with the reference, inertial target and start-state overlays, error colors and per-level jumps | `python -m python.tools.sim.benchmark_viewer RESULTS.h5 [--dataset path/to/trampolines.h5]` |
| `python/examples/benchmark_browser.py` | interactive Polyscope browser of the dataset: each problem's start state, velocities, inertial target and reference, with energies and sampler diagnostics | `cd python/examples && python benchmark_browser.py -d path/to/trampolines.h5` |

- **Plotting dependencies:** the plotting and table scripts need `h5py`, `numpy` and `matplotlib`. The viewers also
  need `polyscope`, and the dataset browser needs `pbatoolkit` and `scipy`.
- **Moved files:** the viewer finds the dataset through the results file's `dataset` attribute, a path relative to
  the repository root. When the files are elsewhere, pass `--dataset`.
- **Module docstrings:** each script's docstring has the full options and the exact definitions.

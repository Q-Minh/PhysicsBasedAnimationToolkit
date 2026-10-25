# Solver convergence benchmark: data and scripts

The solver convergence benchmark compares five GPU solvers (VBD, AAAVBD, Chebyshev, Anderson and Newton) on many
sampled problems. It measures how much solver time each needs to remove a given fraction of its initial error. This
note describes the files a run involves and the scripts that read them. It applies to the full run of
21 October 2026, on the five circle trampolines.

| File | What it holds | Size |
|---|---|---|
| `data/benchmarks/results/trampolines-thin.h5` | **Thin results, for sharing:** every number of the full results, with each solve's log thinned and no video frames (see 2.6). The plotting and table scripts read it like the full file and give the same bars and tables. | 42 MB |
| `data/benchmarks/results/trampolines.h5` | **Full results:** per problem and solver, the full per-iteration error log, level crossings, budgets, terminations, Newton's per-step records, and video frames for some problems | 8.8 GB |
| `data/benchmarks/trampolines.h5` | **Dataset:** the problems (start states, inertial targets, reference solutions), per scene and problem set. Only the viewers need it. | 1.1 GB |

All are HDF5 and need only `h5py` and `numpy` to read. None is in git (`data/` is git-ignored), so they're
shared as files. The scripts are in this repository and run from its root as modules (`python -m python.tools.sim.<name>`).
**If you received the thin file,** sections 2.1 to 2.4 and 2.6 describe it, and section 6 lists the plotting scripts.

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
    logs/<sample:06d>/...           the per-iteration log of each solve (2.4; thinned in the thin file, 2.6)
    trajectories/<sample:06d>/...   video frames, for 8 of the 32 problems (2.5; not in the thin file)
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
| `thin` | thin file only: how it was thinned (2.6) |

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

### 2.6 The thin file

`trampolines-thin.h5` (`python -m python.tools.sim.benchmark_thin`) has the same layout and every attribute and
per-problem array of the full file, with two differences:

- **No `trajectories/`** (video frames).
- **Thinned logs.** A log of more than 1000 entries keeps:
  - its start and end;
  - each level's first crossing;
  - the entries nearest to 1000 times spaced evenly on a log-time axis.

  Shorter logs, Newton's included, are whole, and Newton's per-step records are copied as they are.
  `logs/<sample>/entry` gives each kept entry's index in the full log. In total, 846k of 143M entries are kept.

What stays exact and what doesn't:
- **Exact:** level crossings, so re-scoring at any budget factor gives the same bars, and the iteration tables
  are identical. `benchmark_thin` checks both when it writes the file.
- **Approximate:** curves are exact at the kept entries and only skip points between them. At about 1000 points
  per solve, the plots look identical.
- **`crossing_iteration` keeps its meaning,** the full log's entry (iteration count) at each crossing. Its row in a
  thinned log is `np.searchsorted(entry, crossing_iteration)`.

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
    j = g["crossing_iteration"][0, 3]                     # full-log entry of the 99.9% crossing (-1: not reached)
    if j >= 0 and "entry" in log:                         # thin file: the crossing's row in the thinned log
        j = np.searchsorted(log["entry"][()], j)
    print("sweeps to 99.9%:", sweeps[j] if j >= 0 else None)
```

`python/tools/sim/benchmark_results.py` has the shared helpers: `iter_cases`, `level_crossings`, `rescore` (re-score
at another budget factor) and `aggregate`. It has no GPU dependencies.

## 6. Scripts

Every script below reads existing files: none of them re-runs solves, and only the benchmark itself needs a GPU.
Commands run from the repository root. The plot, curve and table scripts work on the thin file as on the full one;
the viewers need the full results and the dataset.

| Script | Output | Command |
|---|---|---|
| `benchmark.py` | runs the benchmark and writes a results file (GPU, about 5 h for the full run) | `python -m python.tools.sim.benchmark --run python/tools/sim/configs/benchmark/trampolines.toml -o data/benchmarks/results/trampolines.h5` (add `--dry-run` to check it, `--list` to list the dataset's sets) |
| `benchmark_plot.py` | 3D bar charts, one per benchmark: per scene × level × solver, the mean incremental time over T_N, with min–max boxes (`--counts` adds n/N labels) | `python -m python.tools.sim.benchmark_plot RESULTS.h5 -o FIGURES/trampolines_bars [--budget-factor 1] [--counts] [--elev 28 --azim -62]` |
| `benchmark_frames.py` | the bar charts' example frames (Polyscope, in the profiling snapshots' style, camera and crop fitted to the shapes): per benchmark, one sample's initial iterate and reference solution, a homogeneous case colored by the error and the heterogeneous one by Lamé λ, with their colorbars' rendered shades and, on dynamic initial iterates, the start velocity as arrows (length = \|v\| × the JSON's `arrow_seconds`); cases and samples per benchmark (`[BENCHMARK=]VALUE`; defaults: Y1e5 static, Y1e7 dynamic, het; sample 0 static, `auto` dynamic: the largest kinetic energy × velocity coherence); `benchmark_plot.py --frames` places them as a grid left of each chart, as tall as it (the chart unchanged, the figure wider): `--columns [BENCHMARK=]PAIRS` picks the pairs per chart, by default both on the static chart and one per stacked dynamic chart (error on h = 1e-2, material on h = 1e-4), the two single-column grids equally wide. Camera: `--elev`, `--azim`. Regenerate both after a benchmark re-run | `python -m python.tools.sim.benchmark_frames RESULTS.h5 -o FIGURES/frames/trampolines_frames [--dataset path/to/trampolines.h5] [--error-case Y1e7] [--material-case het] [--sample 0]`, then `benchmark_plot RESULTS.h5 -o FIGURES/trampolines_bars --width 5.5 [--no-legend] --frames FIGURES/frames/trampolines_frames [--dataset path/to/trampolines.h5]` (the paper: the legend on the static chart only) |
| `benchmark_curves.py` | convergence curves, e/e0 against time over T_N on log-log axes, one panel per problem: one sample, or with `--all-samples` the min–max band and median over all problems | `python -m python.tools.sim.benchmark_curves RESULTS.h5 -o FIGURES/trampolines_curves_Y1e6 --problems static/Y1e6 dynamic_h0.01/Y1e6 dynamic_h0.0001/Y1e6 [--sample 0 \| --all-samples] [--xscale linear]` |
| `benchmark_iterations.py` | tables of iterations to each level, median [25th–75th percentile] VBD sweeps, or Newton steps and CG iterations, with n/N, as Markdown and LaTeX | `python -m python.tools.sim.benchmark_iterations RESULTS.h5 -o TABLES/iterations [--budget-factor 2]` |
| `benchmark_thin.py` | the thin copy of a results file (2.6), checked against the full one | `python -m python.tools.sim.benchmark_thin RESULTS.h5 -o RESULTS-thin.h5 [--max-entries 1000]` |
| `benchmark_viewer.py` | interactive Polyscope viewer of the trajectories: the solvers side by side, in sync, with the reference, inertial target and start-state overlays, error colors and per-level jumps | `python -m python.tools.sim.benchmark_viewer RESULTS.h5 [--dataset path/to/trampolines.h5]` |
| `benchmark_video.py` | MP4s of the viewer's side-by-side comparisons, one per benchmark: every trajectory sample (or per case the N with the largest initial error, `--samples-per-case N`), each opening on its start state held for a second, with a color bar under the solver labels (one height for the whole run), titled and labeled in Linux Biolinum (6.1) | `python -m python.tools.sim.benchmark_video RESULTS.h5 -o VIDEOS/trampolines [--benchmarks static] [--cases Y1e5] [--samples 0] [--samples-per-case 3] [--label-y Y] [--error-fraction ...]` |
| `python/examples/benchmark_browser.py` | interactive Polyscope browser of the dataset: each problem's start state, velocities, inertial target and reference, with energies and sampler diagnostics | `cd python/examples && python benchmark_browser.py -d path/to/trampolines.h5` |

- **Plotting dependencies:** the plotting and table scripts need `h5py`, `numpy` and `matplotlib`. The viewers also
  need `polyscope`, and the dataset browser needs `pbatoolkit` and `scipy`.
- **Moved files:** the viewer finds the dataset through the results file's `dataset` attribute, a path relative to
  the repository root. When the files are elsewhere, pass `--dataset`.
- **Module docstrings:** each script's docstring has the full options and the exact definitions.

### 6.1 Videos

The viewer and the videos color each vertex by its error |x − x\*| from 0 to the problem's initial error. A
solver that stalls at e/e0 ~ 0.1 then looks almost as converged as one at 1e-5, so for the problems where a solver
stays unconverged the color scale ends at a fraction of the initial error instead (`--error-fraction
[BENCHMARK/]CASE[:SAMPLE]=F`, repeatable, the most specific entry winning; values above it saturate). Each video
frame has a viridis color bar from 0 to that max (Polyscope's table), centered under the solver labels and
captioned "Error |x - x\*|", its max marked "≥" with a triangle past it when the scale is clamped.

The solver labels sit at one height in every clip of every benchmark a run records: just below the lowest point
the recorded meshes reach on screen over all their frames, which the script prints with the clip and frame setting
it. For the 30 kJ videos it is y = 657 px, set by dynamic_h0.01/Y1e5 sample 27 at a later frame, not a start
state: its solution x\* sits 11 px lower on screen than its start, and AAAVBD's iterate at 1.4 ms dips 2.4 px more.
A run recording a subset (`--benchmarks`, `--cases`, `--samples`) computes it over that subset, so a partial
re-record should pass the full recording's height with `--label-y`.

The 30 kJ videos use:

```bash
python -m python.tools.sim.benchmark_video data/benchmarks/results/trampolines.30kJ.h5 \
    -o data/benchmarks/results/videos/trampolines.30kJ --samples-per-case 3 \
    --error-fraction static/Y1e5=0.2 --error-fraction static/Y1e6=0.2 \
    --error-fraction static/Y1e7=0.5 --error-fraction static/Y1e7:0=0.2 \
    --error-fraction static/het=0.5 --error-fraction static/het:9=0.3 \
    --error-fraction dynamic_h0.01/Y1e8=0.05 --error-fraction dynamic_h0.01/Y1e8:13=0.03 \
    --error-fraction dynamic_h0.01/het=0.05 --error-fraction dynamic_h0.01/het:22=0.03 \
    --error-fraction dynamic_h0.01/het:27=0.03
```

How they were chosen, from each trajectory sample's last frame (2 x T_N): a problem is clamped only if some solver
ends unconverged (e/e0 > 1e-2); the fraction is the largest of 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03 that puts that
solver's largest vertex error between 0.3 and 1 of the scale (visible, not saturated), while the converged solvers
stay dark. A case takes the value most of its samples agree on, and a sample outside that band gets its own.

The videos show 3 of each case's 8 trajectory samples (`--samples-per-case 3`): those with the largest initial
error, the largest vertex error |x(0) − x\*| (the color scale's 100%), in sample order; the script prints its pick
with the errors. The fractions were chosen over all 8 samples, and every picked sample is in the band at its
fraction (0.34–0.85 of the scale); the entries of dynamic_h0.01/Y1e8 sample 13 and het sample 22 match unpicked
samples and only apply in the viewer.

| Problems | Fraction | Why |
|---|---|---|
| static/Y1e5 | 0.2 | Anderson ends at e/e0 0.09–0.17 (its largest vertex error 7–12% of the initial error), VBD at 0.03–0.14: 0.36–0.61 of the scale |
| static/Y1e6 | 0.2 | Anderson 0.09–0.29, VBD 0.06–0.24 (largest vertex errors 7–18%): 0.34–0.89 of the scale |
| static/Y1e7 | 0.5 | Anderson 0.11–0.31 (largest vertex errors 15–29%): 0.31–0.58 of the scale |
| static/Y1e7, sample 0 | 0.2 | its largest vertex error is only 7.5%: 0.15 of the scale at 0.5 |
| static/Y1e8 | 1 | Anderson 0.43–0.60 (largest vertex errors 33–72%) already shows at the full scale |
| static/het | 0.5 | Anderson 0.26–0.61 (largest vertex errors 13–41%): 0.46–0.82 of the scale |
| static/het, sample 9 | 0.3 | its largest vertex error is 13%: 0.26 of the scale at 0.5 |
| dynamic_h0.0001/* | 1 | every solver converges (e/e0 at most 8e-4) |
| dynamic_h0.01/Y1e5, Y1e6, Y1e7 | 1 | every solver converges (e/e0 at most 6e-4) |
| dynamic_h0.01/Y1e8 | 0.05 | Anderson ends at e/e0 0.011–0.042 and VBD up to 0.036 (largest vertex errors 1–5%), AAAVBD and Newton at ~1e-4 |
| dynamic_h0.01/Y1e8, sample 13 | 0.03 | its largest vertex error is 1.3%: 0.25 of the scale at 0.05 |
| dynamic_h0.01/het | 0.05 | VBD ends at 0.021–0.064 and Anderson at 0.011–0.028 (largest vertex errors 1.2–4.8%), AAAVBD and Newton at ~1e-4 |
| dynamic_h0.01/het, samples 22, 27 | 0.03 | their largest vertex errors are 1.2–1.3%: 0.25–0.27 of the scale at 0.05 |

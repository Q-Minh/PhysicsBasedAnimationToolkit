---
name: sim-benchmarks
description: Run the solver convergence benchmarks of the GPU simulator — generate static/dynamic problem sets from MCMC shape samples, run the static and dynamic benchmark (benchmark.py) over solvers and samples with chosen budgets, targets and parameters, and read its results. Use when asked to benchmark solvers, compare convergence or time-to-accuracy, rerun a benchmark case, or vary benchmark parameters.
---

# Solver benchmarks

**The methodology is still being decided with the maintainer**: what data to report, the exact plot, and how execution time is measured. **Don't launch full benchmark runs or full dataset generation without the maintainer's go-ahead.** Smoke tests on a few samples are fine.

Run from the repo root with `/c/git/python-3.12/Scripts/python.exe` (Bash). Scene data lives in the main checkout's `data/`, so use absolute paths from worktrees.

## Pipeline

1. **Problems.** `python/examples/make_benchmark.py` draws microcanonical samples on the quadratic-manifold ROM and writes:
   - `pool/E{E}.h5`: all the samples.
   - `static/E{E}.h5`: release the shape with no loads; the solution is the rest shape X.
   - `dynamic_h{h}/E{E}.h5`: one BDF1 step from (x, v), with Newton reference solutions `samples/x_ref`.

   See the `sim-sampling` skill for options and quality checks (R-hat < 1.05). A small existing test set is `data/benchmarks/trampoline-test`.
2. **Static benchmark.** `python -m python.tools.sim.benchmark` releases each sample with every solver and logs the error to the rest shape after every iteration. It reports the time and iterations needed to reach each 5% level of error reduction.
3. **Dynamic benchmark.** The same runner on a `dynamic_h{h}` file: one BDF1 step of the file's h toward its `xtilde`, with the error measured against `samples/x_ref`. The kind and h come from the file.

## Static benchmark

Error: e = sqrt((x − X)ᵀ M (x − X)) over free nodes (lumped mass). Each solve stops at e ≤ (1 − target) e0 or at a cap. VBD-family solves run as one captured CUDA graph. Times come from device timestamps taken around each iteration, so the logging itself is excluded.

```bash
# One quick solve (smoke test)
python -m python.tools.sim.benchmark --fem-elasto-dynamics C:/git/PhysicsBasedAnimationToolkit/data/scenarios/trampoline.Dump.h5:fem \
    --samples C:/git/PhysicsBasedAnimationToolkit/data/benchmarks/trampoline-test/static/E50.h5 \
    --solvers AAAVBD --sample-end 1 --max-seconds 5 -o out/one.json

# Every solver x every sample, with a scene's parameters
python -m python.tools.sim.benchmark --fem-elasto-dynamics DUMP.h5:fem --samples BATCH.h5 \
    --sim-params python/tools/sim/configs/paper/<scene>.toml --max-seconds 10 -o out/all.json

# Parameter variations: configs layered left to right, then single-key overrides
python -m python.tools.sim.benchmark ... --sim-params base.toml static.toml \
    --set Solver.Newton.LinearSolver.n_max_iters=3000 --set Solver.AAAVBD.rhohat=0.9

# Reproducible case: save the spec, rerun it
python -m python.tools.sim.benchmark ... --save-spec out/case.h5 -o out/a.json
python -m python.tools.sim.benchmark --fem-elasto-dynamics DUMP.h5:fem --spec out/case.h5 -o out/b.json
```

Translating requests:

| Request | Flag |
|---|---|
| "Only solvers X, Y" | `--solvers X,Y` (from VBD, AAAVBD, Chebyshev, AndersonSolver, Newton; default all, in that order) |
| "Samples i to j" | `--sample-begin i --sample-end j+1` (`-1` = all) |
| "Time budget T s per solve" | `--max-seconds T` (default 60) |
| "Iteration budget N" | `--max-iterations N`: counts VBD sweeps, **Anderson steps** (p sweeps each when `vbd_iters_per_anderson_iter = p`) or Newton steps |
| "Reduce the error by 99%" | `--target-reduction 0.99` (default 0.95) |
| "With a load" | the StaticSolveSpec's `fext_scale` (via `--spec`). Static benchmark problems are load-free by design |
| "Use parameters P" / "change key K" | `--sim-params P.toml ...` / `--set K=V` (see the `sim-run` skill; check with `python -m python.tools.sim.config show`) |
| "Coarser/finer graph blocks" | `--block N` (default 16, rounded up to a multiple of Anderson's m; can overshoot the stop by up to one block) |

`benchmark.py`'s module docstring has the full details.

## Outputs

- **`results.json`:**
  - the spec, plus metadata: scene, sim-params sources, sample attributes, Newton CG limits, device, commit, date;
  - per solver and sample: e0, iterations, sweeps, solver time, final e/e0, time and iterations to each 5% level (null if never reached), deformation energy, min det F, and a log-spaced curve;
  - per level: aggregates (median, q25, q75, fraction of solves that reached it);
  - per sample, `termination`, and per solver, the count per reason: `target`, `iteration_cap`, `time_cap`, `non_finite`, or Newton's `newton_non_descent`, `newton_line_search_failed` or `newton_zero_gradient`. Newton stops at its first rejected step; at the float32 floor that is usually `newton_line_search_failed`.
- **Dynamic batches** add the problem kind and h to the metadata, and to each sample a `reference` with its accuracy in the error's M-norm (`error_bound`, `relative_error_bound`). It comes from the file's `samples/ref_error_mnorm`, or for older files without it from a loose worst case on `newton_correction` (`error_bound_source` says which). Per level, `reliable` counts the samples whose reference resolves that level.
- **`results.h5`** next to it: the spec and the full logs at `<solver>/<sample>/{error,time,sweeps}`.

When comparing solvers, use the time to reach each error level, not the time per iteration. Also report iterations and sweeps. Anderson iterations aren't sweeps when p > 1.

## Pitfalls

- **Contact** is always off, for both problem kinds.
- **Dynamic h:** dynamic problems take h from the file and override `Integration.dt`, the substeps and the BDF order. At small h (1e-4), steps start close to their solution (e0 ~ 1e-3), so float32 rounding floors the error around e/e0 ~ 1e-5 to 1e-4.
- **Solver parameters:** without `--sim-params`, only the code defaults apply (`configs/defaults.toml`), not the Dump's `vbd/params`. Pass the scene's config (e.g. `configs/paper/<scene>.toml`) for meaningful solver parameters.
- **Stopping:** solver `gtol`, `n_max_iters` and `n_subproblem_max_iters` don't stop benchmark solves; the target and the caps do. Newton's CG absolute tolerance is set to 0 (its relative one stays), since in h²-scaled units at small h the gradient drops below it after one step. Vertex-solve and acceleration parameters still matter.
- **Newton** runs eagerly with host syncs, so its times are noisier. Static problems need `Solver.Newton.LinearSolver.n_max_iters` of about 1500–3000; the default 300 leaves Newton steps inexact.
- **Quiet GPU:** timings are only valid while nothing else runs. Reserve the GPU in `plans/MASTER.md` while parallel sessions are active.
- **Sample batches:** batches that need importance weights (ray sampler) are refused. R-hat ≥ 1.05 and nonzero gravity only warn. Static problems should be gravity-free.
- **First solve:** the first solve per solver includes kernel compilation and graph capture in its host time, but not in the reported times.
- **Not sim-params:** `fem.static` and the StaticSolveSpec aren't part of the config system.

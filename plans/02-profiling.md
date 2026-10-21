# Task 2: Solver profiling and cost breakdowns

Status: **on hold** (see Handoff) · Worktree `pbat-profiling` (location in `CLAUDE.local.md`)
Read first: `MASTER.md` (rules, ownership, GPU-sharing rule), `python/tools/sim/CLAUDE.md`.

## Handoff (2026-10-21)

Task 2 is **on hold**: no unmerged work, no uncommitted worktree changes. The how-to lives in
`python/tools/sim/CLAUDE.md` ("Profiling", AAAVBD bullets, convergence paragraph) and
`.claude/skills/sim-profiling/SKILL.md`. This section records state, decisions and gotchas only. Since 2026-10-20 task 2
also did Newton CG tuning for the benchmark, the stationarity test (rstol/astol) and the scene configs (§2, §5).

### 1. Scope and ownership

- **Profiling stack:**
  - `gpu/common/profiling.py` (scopes; backends `off`/`kernels`/`nvtx`) and `gpu/common/cupti.py` (ctypes CUPTI binding);
  - `panels/profiling.py`;
  - `profiling/` (`results`, `scope_paths`, `report`, `nsys`, `venvrun`, `anatomy`, `ncu`, `nsight`, `overheads`);
  - `simulate.py`'s `--profile*` flags;
  - the scope lines in the solver and contact files.
- **AAAVBD GPU code I changed:**
  - `gpu/vbd/aaasolver.py`: loads hoisted, `betaup`, `lipschitz_normalization`;
  - the contact-gather early-out (`MeshDynamicsData.in_contact`);
  - the relative `hess_zero` check;
  - `surface_stencil_surface_neighbours_only` on the GPU;
  - `gpu/vbd/params.py`: `AaaVbdCpuParams`, `STENCIL_GRADIENT_ACCELERATION_KEYS`;
  - the AAAVBD groups in `state.py` and `configs/defaults.toml`.

### 2. Current state

- **Merged into dev:**
  - Profiling milestones: M1 (`1d52c8ff`), M2 `02d75771`, M3 (`7ad6a983`).
  - feature/gpu-profiling follow-ups:
    - ncu hotspots `d05f0987`;
    - gather early-out `1ba967d9`;
    - CUPTI exit `5035b905`;
    - AAAVBD load hoisting `d61ee145`;
    - inclusive scopes `ff3c486a`;
    - docs `abe37d42`;
    - tooltips `b5a5450c`.
  - aaavbd-params `9598e393`, relative-hess-zero `7ee9ee3a`, aaavbd-surface-neighbours `9a62fcf6`, aaavbd-ablation-params `30fe7778`.
  - profiling-tool-discovery `006e7450`, report-without-gpu `feb00581`, sd-detect-scope `15340ab4`.
  - 2026-10-20/21: anderson-deterministic-gram `c0d38305` (MASTER decision #11 A), anderson-notes `836eb1e4`,
    scene-configs `b6ba92da`, newton-init-strategy `2a2b61dc`, astol-noise-floor `90efbda5` (also reverts Newton's
    forced first subproblem), astol-acceleration `38385d36`.
- **Unmerged:** nothing.
- **Leftovers:** my local branches `feature/anderson-deterministic-gram`, `feature/anderson-notes`,
  `feature/scene-configs`, `feature/newton-init-strategy`, `feature/astol-noise-floor` and `feature/astol-acceleration`
  are merged into dev and can be deleted with `git branch -d`.
- **Worktree vs dev:** `pbat-profiling` is on `feature/astol-acceleration` at `bf346098`, clean, 0 commits ahead and 32
  behind dev (`b4e73feb`). **Resume from:** `git checkout --detach dev` there (location in `CLAUDE.local.md`), then branch
  `feature/<topic>` from the latest dev.
- **Reports:**
  - trampoline breakdown https://claude.ai/artifact/LHMcGJs6Z6N89iruhYLxYz
  - AAAVBD overheads v3 https://claude.ai/artifact/NngLUsxKfkhZz4Fzmx9XuL
  - solver-costs sample https://claude.ai/artifact/9RuazLzk7xVFNzB2DFZRLH
  - collaborator index https://claude.ai/artifact/57XN1mBRfqnYHfMRdhuays

### 3. Key decisions (maintainer)

- **What to report:** GPU busy time per phase only (as if written in CUDA), with no host, Python or launch overhead. Profiling runs eagerly.
- **Measurement tools:**
  - The `kernels` backend is CUPTI: it replaced Warp activity timing, which was 4x off from launch latency.
  - Paper figures come from nsys.
  - Clocks are unlocked; validate against nsys instead.
- **Contact:** VertexSdf only. The fused vertex solve is one slice, "elastic + contact".
- **Goal:** find AAAVBD's overheads and make it faster. The cheap kernel changes were approved and are done.
- **AAAVBD parameters:**
  - Stencil keys are AAAVBD-only on the GPU.
  - `warm_start_beta` GPU default: 2.
  - `betaup` (1) and `lipschitz_normalization` (on) are GPU-only ablation switches, bit-identical at their defaults.
- **Tuning:**
  - Never change the betaG shrink rule.
  - Tune near the defaults: gammadown 0.99 → 0.95 → 0.90, gammaup no lower than ~0.3.
  - For contact instabilities, change mainly gammadownS and gammaupS.
  - No per-scene tuning.
  - Defaults are the maintainer's call: report, don't change.
- **cow_on_trampoline:** stopped. Don't resume it or mention it in docs or skills.

### 4. Verified facts and gotchas

- **Eager vs graph.** Profiles are eager, so they include no graph gaps. Production cost is graph replay (`--profile off --profile-out`). `profiling.anatomy` explains the difference: steady `capture_while` sweeps are nearly gap-free, and the gaps sit at the boundary phases.
- **CUPTI vs nsys** (beam, AAAVBD): identical activity counts. Sweep within 0.6%, step within 2.9%.
  - Sub-µs activities read ~0.2 µs long, so the contact phases come out 27–40% high: use nsys for those.
  - `kernels` and nsys can't share a process: use `--profile nvtx` under nsys.
- **Busy vs union time.** `busy_ms` double-counts concurrent streams (contact uses 4x6, Newton assembly 6); `union_ms` doesn't.
- **Scope attribution:**
  - A kernel goes to the innermost scope open at its launch.
  - Off-mode scopes leave graphs and positions identical.
  - Never nest a scope inside a same-named scope: use an unscoped helper (`Sd._refit`).
  - VertexSdf's detection is all under `<phase>/contact.detect`. Under `solve.finalize_subproblem` that's 2 calls per outer iteration (prepare + detect).
- **Nsight on Windows:**
  - The venv's `python.exe` stub hangs under nsys, hence `profiling/venvrun.py`.
  - `PYTHONPATH` alone breaks `cuda.compute`.
  - `ncu.bat` goes through cmd.exe, which strips `^` from kernel regexes, so `find_tool` uses `ncu.exe`.
  - ncu runs at base clocks: compare kernels with each other only.
- **Reading results needs no GPU.** `profiling/results.py` and `report.py` must not import warp, cupy, pbatoolkit, h5py or nvtx (enforced by `TestReportImports`).
- **Per-step times mean "time to pass gtol", not equal accuracy.** Show sweeps/step, outer iterations, ms/sweep and the exit reason.
- **AAAVBD now:**
  - per sweep: ~1.17x VBD;
  - vertex solve (ncu): 9.2 µs vs VBD 7.6 µs;
  - latency-bound: ~74% long-scoreboard stalls;
  - grid-limited occupancy: 667 one-warp blocks per color.
- **Speedups since 2026-10-17** (trampoline AAAVBD graph replay): ~108 → ~26 ms/step. They came from:
  - the constraint-set copy fix (`a9abadc1`, −23%);
  - the gather early-out (−52%);
  - `capture_while`;
  - the load hoisting (−5%);
  - the fused constraint-set update (`ae18a043`, −2…3%).
- **Determinism.** The beam and the trampoline are bit-reproducible for VBD, AAAVBD and Chebyshev. BeamFallingOnBeam VBD/AAAVBD was also bit-identical run to run in 10–30-step checks.
- **mat_on_stilts_3, AAAVBD, 5-substep paper config: not a regression.** The Oct-17 code fails the same way.
  - Cause: free stilt-top nodes between pinned nodes (tet quality 0.05–0.1), plus contact, destabilize the stencil acceleration.
  - Stable: the 10-substep config, gammadownS 0.90, or warm_start_beta 1.
  - `surface_stencil_surface_neighbours_only` 0 diverges.
- **Parameter plumbing:**
  - nanobind `vbd::Params` rejects new attributes, hence the `AaaVbdCpuParams` wrapper, which forwards to `.vbd_params`. A Python subclass leaks warnings at exit.
  - Use `state.load_params(fem)`.
- **Windows shell:** pass h5 paths in Windows drive form with forward slashes, not Git Bash's `/<drive>/...` form. Heredocs mangle backslashes, so write scripts to files.

### 5. Newton CG and the convergence test (2026-10-20/21)

- **Final Newton CG table** (benchmark trampolines; pinned per case in `configs/benchmark/trampolines.toml`; quiet-machine
  re-time at 21:08 on 2026-10-20; noise 2–9%, so differences under ~10% are ties):

  | Benchmark | Y1e5 | Y1e6 | Y1e7 | Y1e8 | het |
  |---|---|---|---|---|---|
  | static (rtol 1e-6) | cap 500 | cap 1000 | cap 1000 | cap 1000 | cap 4000 |
  | dynamic h = 1e-2 | rtol 1e-2, cap 4000 | rtol 1e-2, cap 4000 | rtol 1e-2, cap 4000 | default (rtol 1e-6, cap 500; 1e-2/4000 is +9%) | rtol 1e-2, cap 4000 |
  | dynamic h = 1e-4 | default | default | default | default | default |

  - Static rtol 1e-3 gains nothing and stalls on Y1e8 and het. At h = 1e-4 Newton takes 1–3 steps, so CG settings barely
    matter.
  - Newton timings need a quiet CPU as well as a quiet GPU: CG syncs with the host every 10 iterations, and a CPU-loaded
    sweep inflated the large caps.
  - Static Newton's slowness is diagnosed, not a bug: inexact Jacobi-CG steps plus a long projected-Newton phase (static
    het's damped steps reduce the error by only 4–9% for ~60 steps).
  - The default CG cap is now 2000 (task 3, `b48ab7f0`); the benchmark cases stay pinned.
- **Stationarity test** (`gpu/vbd/convergence.py`, VBD family and Newton, dynamic problems): passes when
  RMS_M(M⁻¹g) ≤ max(rstol · RMS_M(x̃ − xᵗ), astol · |a| · h²).
  - |a| is the mass-weighted RMS of fext/m over free nodes (`external_acceleration`; 9.81 without a load), and h is
    `fem.bdf.h` (dt / substeps).
  - Defaults rstol 1e-2, astol 0.5: the absolute term scales like one step's gravity sag, so it's time-step and
    scene-size invariant, and with astol < 1 a step that hasn't moved can't pass.
  - Only the host-side tolerance changed; the device kernels didn't. Static problems use `gtol`. The benchmark runs
    with the test off.
  - History: astol·L (scene bounding box) let Newton pass from xᵗ before any step (2.8 mm > h²g ≈ 1 mm); 1e-6 was much
    too slow; 1e-4·L was not time-step invariant. A forced first Newton subproblem was added and reverted.
  - **Changed meaning:** astol used to be a fraction of L. Generated configs were regenerated, but a hand-written file
    with astol = 1e-4 is now extremely strict.
  - Stiff scenes near rest may still run their full budget when their float32 floor, about (h ω)² ulps, exceeds both
    terms.
- **Newton init:** `Solver.Newton.init_strategy` (default Position) through `SimulationState.step_init_strategy()`.
- **Scene configs:** `scene_configs.py` (temporary, CPU-only, `make`/`check`) writes `configs/scenes/*.toml` from the
  paper configs plus the durable rules. Regenerate with CUDA hidden (`CUDA_VISIBLE_DEVICES=`): first `paper_configs
  make`, then `scene_configs make`, then both `check`s.

### 6. Instabilities: heterogeneous trampoline + sphere

These weren't diagnosed by task 2; this is what I know, with the sources.

- At the 2 mm VertexSdf band, the sphere impact exploded on the heterogeneous trampoline; at 5 mm (`c2b6223b`) VBD and
  AAAVBD still exploded on the second bounce without substeps. The sphere-trampoline configs now use a 1 cm band with
  max_dist 50 cm (maintainer, 2026-10-21; `scene_configs.py` rules `SPHERE_BAND`, `TRAMPOLINE_DETECTION`). In my
  cheap-scene check (2026-10-20), the stiff sphere didn't pass through at 2 mm.
- Newton on the heterogeneous trampoline + sphere is unstable from the second contact frame, also at mu_f = 0 (MASTER.md
  open item 1b). The heterogeneous soft-stiff contact scenes use Newton 30 × 1 with strict CG (rtol 1e-9, 4000
  iterations; `9a43b8e4`).
- For the friction instability at the stiff center, task 3's frozen Coulomb limit is merged (`3fbd1caf`), frozen at
  each contact's first positive estimate (`4841ae50`), with the unset-limit fix (`39a629a9`; plan 03 has the details).
  Since the instability also happens at mu_f = 0, friction isn't the whole story.
- Next, with the maintainer: a short headless Newton run of that scene with per-step diagnostics (line-search
  failures, CG exit, contact count and penetration at the second contact frame).

### 7. Open items (all wait for the maintainer)

0. **Next work once the maintainer is back** (MASTER.md "Open items"): refine the benchmark figures; AAAVBD ablations
   (protocol not agreed yet, item 1 below); the instabilities on the paper scenes (§6).
1. **AAAVBD ablations** (MASTER.md, "Planned: AAAVBD ablation studies"):
   - The switches exist: `lipschitz_normalization`, `betaup` (beta > 1, e.g. 1.05), `warm_start_beta`, and the gammas.
   - Next: propose a protocol, e.g. `python.tools.sim.benchmark` time-to-error-level with `--set Solver.AAAVBD.<key>=...` on 1 static + 1 dynamic set.
2. **H⁻¹-norm augmentation cap** (idea, not pursued): scale the stencil augmentation `a` so that `λ² aᵀH⁻¹a ≤ gᵀH⁻¹g`, so the local model still decreases. The shrink rule stays untouched.
   - Next, only if asked: a GPU-only AAAVBD switch using the vertex solve's 3x3 H, tried on mat_on_stilts_3 with the 5-substep config.
3. **mat_on_stilts_3, 5-substep config:** the maintainer picks one:
   - leave it as is;
   - add a per-scene gammadownS 0.90;
   - make gammadownS 0.90 the default (benchmark check first).
4. **Performance ideas:**
   - an early-out for the per-sweep dual updates (4 kernels even with no contacts);
   - pre-filtering the AAAVBD surface neighbours;
   - fusing the remaining boundary copies and fills;
   - the acceleration functions (~25% of vertex-solve stalls).

   Next: size them with `profiling.anatomy` and `ncu --lineinfo` on dev.
5. **Recheck on dev:** 21 MEM_ALLOC/MEM_FREE node pairs and ~2,500 memcpy nodes per captured step. These were measured before the later changes.
6. **Manual test for the maintainer:** the GUI profiling panel's Tree mode.
7. **Residual-based float32 noise floor** for the stationarity test (future work, not implemented): a backward-error
   estimate from the absolute values of the force terms (inertial, elastic, contact) that sum to g. Only if the
   maintainer asks.
8. **Not checked on the GPU:** the astol·|a|·h² change was validated by CPU tests and config checks only (maintainer
   rule for config/default changes). The first interactive scene run should look at the Solve status ratios.
9. **Unanswered:** whether to delete the old collaborator-index artifact (57XN1mBRfqnYHfMRdhuays). Do nothing unless the
   maintainer answers.

### 8. How to resume

- Use the venv from `CLAUDE.local.md`. Scene data is in the main checkout's `data/`.
- **Cheap scenes:** `data/scenarios/MediumLongBendingBeam.h5` (contact-free), `data/BeamFallingOnBeam.h5` (contact), `data/scenarios/trampoline.h5`.
- No heavy scenes or long runs without the maintainer. Log GPU runs (START/END) in MASTER.md.
- **Tests:**
  ```bash
  python -m unittest python.tools.sim.gpu.common.profiling python.tools.sim.gpu.common.cupti \
      python.tools.sim.profiling.results python.tools.sim.profiling.nsight python.tools.sim.gpu.vbd.params
  # convergence test and configs (CPU, with CUDA hidden: CUDA_VISIBLE_DEVICES=)
  python -m unittest python.tools.sim.gpu.vbd.convergence python.tools.sim.config
  ```
  pytest isn't installed; use `unittest`.
- **Smallest profile check** (seconds; read the JSON with `results.scope_tree`):
  ```bash
  python -m python.tools.sim.simulate --scene path/to/PhysicsBasedAnimationToolkit/data/BeamFallingOnBeam.h5 \
      --solver AAAVBD --duration 0.1 --profile kernels --profile-warmup 2 --profile-out out/k.json -o out/k.h5
  ```
- **Bit-identity:** the same run with profiling off, before (`git checkout HEAD~1 -- <file>`) and after. Then compare every float dataset of the two `.h5` files with `np.array_equal`.
- **More recipes:** nsys, anatomy, ncu, report and production timing are in the sim-profiling skill; benchmarks in sim-benchmarks.

## Goal

Measure where a time step's GPU time goes, by algorithm phase, for every GPU solver (Newton, VBD, AAAVBD,
Chebyshev, Anderson). Produce per-solver pie charts and cross-solver comparisons that hold up in a paper.

## Facts that shape the design

> Superseded (2026-10-20): the in-process backend uses CUPTI, not Warp activity timing (Handoff §4).

- **Warp activity timing** (`wp.timing_begin`/`wp.timing_end`, or `wp.ScopedTimer(cuda_filter=...)`)
  records the GPU duration of each *eager* kernel launch, memset and memcpy, in launch order. A captured CUDA
  graph is recorded as a single `TIMING_GRAPH` entry, with no per-kernel detail.
- **`wp.ScopedCapture` itself provides no timings.** It only records the graph.
- **Production runs use graph capture** (`ui.py` and `simulate.py` capture `substeps × (setup → solve → step)`),
  which removes host launch overhead. Eager timings include gaps where the GPU waits for the host, so eager
  *wall* time overstates the real cost.
- Graph-safe instrumentation: CUDA events recorded inside a capture become graph nodes only when recorded
  as external events. Check whether `wp.record_event` does this during capture before relying on it.
- `nvtx` is already pinned in `python/tools/sim/requirements.txt`, and `ui.py` contains a commented-out
  `wp.ScopedTimer(..., use_nvtx=True)`. Nsight Systems with `--cuda-graph-trace=node` shows per-kernel
  times even inside graphs, which makes it the reference to validate against.
- Newton isn't graph-capturable (`supports_graph_capture == False`), so it always runs eagerly. Comparisons
  must account for that, for example by reporting Newton's eager GPU busy time next to the others' graph times.

## Proposed design

> Superseded (2026-10-20): backends are off / kernels (CUPTI) / nvtx, see `python/tools/sim/CLAUDE.md` and Handoff.

**Scope API** (the stub lands in Phase 0, in `python/tools/sim/gpu/common/profiling.py`):
`with profiling.scope("anderson.gram"):` with hierarchical dotted names, a no-op when profiling is off, so
production graphs are unaffected. Suggested phase taxonomy (adjust as needed):
`step.setup`, `contact.detect`, `contact.filter`, `contact.constraint_set`, `contact.dual`,
`contact.penalty`, `vbd.sweep`, `aaavbd.sweep`, `anderson.gram`, `anderson.solve`, `anderson.mix`,
`chebyshev.mix`, `newton.assemble`, `newton.linear_solve`, `newton.line_search`, `step.finalize`.

**Backends**, selected at run time:
- `off` (default).
- `phases`: CUDA events at scope boundaries, eager mode. Gives per-phase GPU timeline durations, including idle gaps.
- `kernels`: Warp activity timing, eager mode. Each kernel's busy time is attributed to the innermost open
  scope by tracking launch order. This is the most accurate breakdown of GPU work.
- `nvtx`: NVTX ranges per scope, for Nsight Systems; works in graph mode too.

**Profiling disables graph capture (maintainer decision).** When any backend other than `off` is active,
`SimulationState.step` runs eagerly even for capturable solvers. All solvers, Newton included, are then
measured the same way.

**Measurement protocol per solver and scene:**
1. Warm up (compile), excluding the first steps.
2. Eager mode with the `kernels` backend: per-phase GPU busy time over N steps. This gives the pie charts
   and cross-solver comparisons.
3. Optionally, as context only: an unprofiled graph-mode run timed with events around `capture_launch`,
   to show how much host launch overhead graph capture removes in production.

Record GPU model, driver, Warp version and solver parameters with every result.

**Outputs:**
- `simulate.py --profile {off,phases,kernels,nvtx} --profile-out results.json`: per-step, per-phase times.
- A report script: pie chart per solver, stacked bars across solvers, phase time per sweep. The maintainer
  prefers rendered pages for results, so publish comparisons as an HTML artifact.
- A live table in a `panels/profiling.py` GUI panel.

## Validation

> Superseded (2026-10-20): done; CUPTI is within 0.6% (sweep) / 2.9% (step) of nsys (Handoff §4).

- Per step, the sum of phase busy times plus unattributed time equals the activity-timing total.
- On one scene, compare against Nsight Systems: per-phase totals within a few percent.
- With profiling off, captured graphs and step times are unchanged (no overhead).

## Files

> Superseded (2026-10-20): see Handoff §1.

Own: `gpu/common/profiling.py`, `panels/profiling.py`, report scripts, `--profile` flags. Add `scope(...)`
lines in solver and contact files, but don't restructure those files. Tasks 1 and 3 instrument their own new code.

## Open questions for the maintainer

1. ~~Which comparisons matter most?~~ **Answered (2026-10-17): cost breakdown.** Per-step GPU cost by phase at
   the production parameters, plus normalized cost per VBD sweep / Newton iteration. Time-to-tolerance stays
   in task 3's graph-mode benchmark.
2. ~~Nsight Systems as reference?~~ **Answered (2026-10-17):** it is already installed (2026.1.3). **Warp + nsys
   import:** the in-process `kernels` backend (Warp activity timing plus per-scope event brackets, with the
   remainder reported as "untimed") serves the GUI and quick runs. **Paper figures come from nsys:** NVTX
   scopes, then a script attributes every kernel in the nsys sqlite export to its innermost scope. That covers
   CUB/cupy kernels and concurrent streams exactly. No new dependencies.
3. ~~Host-side view?~~ **Answered (2026-10-17): GPU only.** Time things as if the code were written directly in
   C++/CUDA: report GPU busy time per phase, excluding host launch overhead, Python time and host-sync gaps
   (relevant to Newton). Drop protocol step 3 (eager vs graph replay wall time) as a reported result.
4. ~~Lock GPU clocks?~~ **Answered (2026-10-17): no lock, validate.** Keep default clocks. Compare per-kernel
   averages from an eager run against an nsys graph-mode trace (`--cuda-graph-trace=node`) once, and report the
   difference.
5. ~~Scenes?~~ **Answered (2026-10-17): mostly trampoline** (`data/scenarios/trampoline.Dump.h5`), but scene
   choice is deferred: "for now, I just want to see profiling work." Develop on trampoline (MediumLongBendingBeam
   for quick checks).
6. ~~Contact detection?~~ **Answered (2026-10-17): VertexSdf only.**
7. ~~Contact cost inside the fused vertex-solve kernel?~~ **Answered (2026-10-17): one fused slice,**
   "vertex solve (elastic + contact)". Dual updates, detection, filtering and constraint-set updates get their own
   slices. No ablation, no split kernels.
8. ~~Solver parameters?~~ **Answered (2026-10-17): equal sweep budget.** VBD, AAAVBD, Chebyshev and Anderson
   run the same total number of VBD sweeps per step. Newton keeps its defaults. Default budget: the Dump's
   `n_max_iters × n_subproblem_max_iters`, with Anderson's `n_subproblem_max_iters` set to budget / `p`.
9. ~~First milestone?~~ **Answered (2026-10-17): CLI + report first.** Scope backend, scopes in all solvers,
   `simulate.py --profile`, the nsys importer, and a published HTML report (pies plus stacked bars) on
   trampoline. The GUI panel (`panels/profiling.py`) comes in milestone 2.
10. ~~Backend set?~~ **Answered (2026-10-17): `off` / `kernels` / `nvtx`.** `phases` is dropped. Per-scope event
    brackets are folded into `kernels` as a diagnostic for spotting untimed work.

## Progress

> Superseded (2026-10-20): history up to the 2026-10-18 docs pass; later work and the current state are in Handoff §2.

- 2026-10-17 · Investigation findings:
  - **Warp activity timing** (`warp/native/cuda_util.h:359`) brackets each *Warp* launch, memset and memcpy
    with a CUDA event pair and skips capturing streams. It returns durations without timestamps.
    **Not covered:** `cuda.compute` (CCCL) sort, lower_bound, scan and reduce in `contact/mesh/pairs.py`,
    `contact/dynamics.py` and `contact/mesh/ogc.py`, plus cupy copies.
  - **First nsys probe** (MediumLongBendingBeam, AAAVBD, VertexSdf, 30 steps, graph mode with node trace):
    1.70 s of kernel time. `_accelerated_vertex_solve_kernel` is 85.8% (130,500 launches, avg 11.2 µs,
    ~4,350 per step). Non-Warp kernels (CCCL radix sort and binary search, cupy copies) are ~4%. Contact dual
    updates are ~5%. Detection is ~1.5%.
  - **Concurrent streams:** contact work runs on 4×6 `wp.Stream`s (`contact/constraints.py:105`) and Newton
    triplet assembly on 6 streams, so per-kernel durations summed per scope overcount the timeline.
  - **Newton** syncs to the host every iteration (`.numpy()` on the gradient norm, slope and energy;
    `warp.optim.linear.cg` with `use_cuda_graph=False`). Its kernels are all Warp (`warp.sparse`, `warp.optim`).
  - **In-graph events:** `Stream.record_event(..., external=True)` (`warp/_src/context.py:5335`) records
    external event nodes during capture. Relayed to task 3 via the coordinator.
  - **nsys on Windows:** the venv's `python.exe` launcher stub hangs under nsys. Launch the base interpreter
    with a wrapper that sets `sys.prefix` and calls `site.addsitedir` on the venv. `PYTHONPATH` alone breaks
    `cuda.compute` (cuda.pathfinder can't find nvrtc). The wrapper will be committed with the profiling scripts.
- 2026-10-17 · All open questions answered (batches 1 to 3, see above). Waiting for Phase 0, then rebase onto
  `dev` and implement milestone 1.
- 2026-10-17 · Coordinator confirmed the Phase 0 stub contract: `profiling.scope(name: str)` is the only public
  API, a cheap no-op when off (no allocations, no device work, capture-safe). The coordinator will message when
  Phase 0 is done.
- 2026-10-18 · **Milestone 1 implemented** on `feature/gpu-profiling` (14 commits, rebased on dev `800e87ef`):
  - **Backend (`gpu/common/profiling.py`):** `scope`, the `scoped` decorator, `push`/`pop`, `set_backend`,
    `begin_step`/`end_step`, with backends OFF/KERNELS/NVTX. Unit tests are in the module.
  - **Scopes:** in the contact, VBD-family and Newton files.
  - **Eager profiling:** `state.step` runs eagerly while profiling. The substep loop moved into
    `SimulationState._advance`.
  - **`simulate.py`:** `--profile`, `--profile-out` and `--profile-warmup`. Warm-up runs unprofiled in graph
    mode. With profiling off, `--profile-out` records production (graph-replay) step times.
  - **Tools:** `profiling/nsys.py` (runner plus importer), `profiling/venvrun.py`, `profiling/report.py`, and docs
    in `python/tools/sim/CLAUDE.md`.
- 2026-10-18 · Validation:
  - **Kernels backend vs nsys** (beam, AAAVBD, same eager steps): Warp activity timing gives 208 ms/step, nsys
    54 ms/step. The sweep alone is 110 vs 48 ms, the contact phases 15–30× too high. The cause is launch
    latency: event pairs start on an idle GPU before the kernel is submitted. Eager nsys sweep time (47.8 ms)
    matches graph-mode nsys (49 ms), so eager kernel durations are representative. **Quote only nsys numbers.**
  - **Profiling off leaves production unchanged:** captured graphs have identical node counts and types on the
    merge base and the branch, for VBD, AAAVBD, Chebyshev and Anderson (11,978–14,898 nodes on the beam), and
    trajectories are bit-identical. Replay timings agree within run-to-run noise (±5 ms).
  - **Side finding:** every captured step contains 21 MEM_ALLOC/MEM_FREE node pairs (allocations during capture)
    and ~2,500 memcpy nodes.
  - Not done yet: the "per-phase totals within a few percent of nsys" check for the kernels backend. It fails
    by construction (see above).
- 2026-10-18 · **Trampoline report** (5 solvers, nsys, steps 10–14, Dump parameters = 500 sweeps/step):
  https://claude.ai/artifact/LHMcGJs6Z6N89iruhYLxYz
  - **VBD family:** 65–72 ms/step of kernel work vs ~100 ms/step graph replay, i.e. ~30 ms/step of graph
    scheduling overhead. The sweep is 78–88% of the work.
  - **Contact:** VertexSdf detection costs ~6 ms/step even with zero contacts (trampoline has none).
  - **Anderson:** the Gram matrix plus solve costs 3.4 ms/step.
  - **Newton:** converged after one outer iteration (13.6 ms/step), so it isn't at equal accuracy.
- ~~Open: should the inaccurate `kernels` backend stay?~~ **Answered (2026-10-18): switch the in-process
  backend to CUPTI.** `cupti-python` is Linux-only, so I'll write a small ctypes binding to the CUPTI activity
  API: kernel, memcpy and memset records with GPU start/end times, attributed to scopes through external
  correlation IDs. **DLL source (maintainer): pip `nvidia-cuda-cupti-cu12==12.9.79`**, which matches the venv's
  CUDA 12.9. Add it to `requirements.txt`; the coordinator installs it in the shared venv. This is milestone 2
  work, together with the GUI panel.
- 2026-10-18 · **Maintainer direction:** iterate on the breakdowns and visuals so the maintainer can suggest data
  points. The goal is to find AAAVBD's performance overheads and make it faster, since AAAVBD is the technique
  to publish. **Order: milestone 2 first** (CUPTI backend plus GUI panel), then **milestone 3: AAAVBD analysis**
  with two new data points, chosen by the maintainer:
  - **Sweep anatomy:** import an nsys graph-mode trace with node-level timing and show one production sweep as
    kernel time vs gaps, by node type.
  - **Nsight Compute** on the VBD and AAAVBD vertex-solve kernels: occupancy, stall reasons, memory traffic.
  First observations from the trampoline traces (to re-measure, since task 1's runs overlapped):
  - **Launches:** 9 colors of ~667 vertices. Each color launch is one partial wave (667 blocks × 32 threads,
    128 registers/thread) and takes ~12.7 µs (VBD) vs ~14.0 µs (AAAVBD), +12%.
  - **Graph replay:** 206 µs/sweep vs 144 µs/sweep of kernel work, so ~60 µs/sweep goes to gaps.
  - **Fixed nodes per sweep:** 4 dual-update kernels (even with no contacts), an `xb` memcpy, and memsets.
- 2026-10-18 · Milestone 1 merged into dev (`1d52c8ff`). **Milestone 2 implemented** on `feature/gpu-profiling`
  (4 commits after `7c485e35`):
  - **CUPTI binding:** `gpu/common/cupti.py`, a ctypes binding to the activity API with external correlation.
    The DLL is resolved through `importlib.metadata` from `nvidia-cuda-cupti-cu12==12.9.79`, which is pinned.
  - **Backend switch:** the `kernels` backend now runs on CUPTI and reports union/span like nsys.
  - **GUI panel:** `panels/profiling.py` (live rolling-window table, JSON export), plus 3 small edits in
    `ui.py`.
  - **Validation vs nsys** (beam, AAAVBD, idle GPU):
    - Activity counts are identical (11,946 per step).
    - The sweep is within 0.6% (48.06 vs 47.76 ms/step) and the total within 2.9% (55.7 vs 54.1).
    - Sub-µs activities read ~0.2 µs longer in CUPTI, which makes the tiny-kernel contact phases 27–40% higher.
      Use nsys for those.
  - **Tested:** the GUI panel's step hook and export were tested headlessly. Drawing it needs the
    interactive GUI (maintainer smoke test).
- 2026-10-18 · Milestone 2 merged into dev. **Milestone 3 (AAAVBD analysis)** on `feature/gpu-profiling`,
  7 commits after `92791e16`:
  - **Tools:** `profiling/anatomy.py` (nsys graph-mode sweep anatomy: per-step budget, boundary phases, steady
    sweep node by node), `profiling/ncu.py` (Nsight Compute summary), `profiling/overheads.py` (report), and a
    capture range for production runs in `simulate.py`.
  - **Report:** https://claude.ai/artifact/NngLUsxKfkhZz4Fzmx9XuL
  - **Findings (trampoline, AAAVBD, 107 ms production step):**
    - **Step budget:** sweep kernels 65.8 ms, sweep gaps 1.1, boundary kernels 8.1, boundary gaps 29.6,
      other 2.2.
    - **Boundary gaps:** almost all of the boundary gap time sits before the constraint-set update's memcpy and fill
      nodes (~10 µs each).
    - **Root cause:** `ConstraintSet.update_constraint_set` repeats its copies and fills 4× (loop over all 4
      stream lists).
    - **Measured fix:** a 2-line change (one stream list) is bit-identical and gives AAAVBD 108.1 → 83.5 ms
      (−23%), VBD 98.7 → 74.9 (−24%). **Not committed** (not my file). Patch in scratchpad.
    - **Remaining:** ~7.6 ms/step of boundary gaps would mostly go with a fused copy/fill kernel.
    - **Steady sweeps:** nearly gap-free (0.17 µs between color launches).
    - **AAAVBD vs VBD kernel (ncu, base clocks):** +7% duration, +13% instructions, ~+28% DRAM traffic.
      Both are latency-bound: 73–74% long-scoreboard stalls, 83% of cycles with no eligible warp.
    - **Occupancy:** limited by the grid (667 one-warp blocks per color, 0.87 waves, 24% achieved occupancy),
      not by registers.
  - **Possible contact bug on dev** (logged for the coordinator): CubeFallingOnCube + VertexSdf detects 0
    contacts, and the top cube falls through.
- 2026-10-18 · **Maintainer decisions:** commit the constraint-set fix. The coordinator landed it on dev as
  `a9abadc1`, using one stream per field. The cube fall-through is expected (coincident surfaces, zero
  penetration), not a bug. Milestone 3 tools were merged (`7ad6a983`). Next directions: kernel source hotspots
  and the `capture_while` anatomy.
- 2026-10-18 · **Kernel source hotspots** (3 commits after `7ad6a983`):
  - **Tooling:** `ncu.py --lineinfo` (venvrun's `PBAT_WARP_LINEINFO` hook) reports stall samples, stall reasons and
    instructions per Python line and function, with native builtins folded into their callers by address.
  - **Finding:** on contact-free trampoline, contact gathering (`local_contact_derivatives` plus half-edge
    helpers) is ~60% of AAAVBD's kernel stall samples (VBD ~72%) and ~half of its instructions.
  - **Skip-gather experiment** (bit-identical, contact-free only): AAAVBD 83.8 → 40.6 ms/step (−52%), VBD
    75.2 → 46.3 (−38%).
  - **Proposed fix (not implemented, task 3 owns the solver kernels):** a per-vertex contact count per
    constraint-set update, with an early-out in the vertex solve.
  - **To recheck:** AAAVBD is faster than VBD with the gather skipped.
  - **Report:** v2 of https://claude.ai/artifact/NngLUsxKfkhZz4Fzmx9XuL
  - **Pending:** the `capture_while` anatomy waits for task 3's merge.
- 2026-10-18 · **Contact-gather early-out implemented** (maintainer: task 2 implements it), on dev after task 3's merge:
  - **Commits:** `f9f689a3`, `2e4498e9`. `MeshDynamicsData.in_contact` is flagged per constraint-set update from the
    gather's own prefix ranges, plus one guarded condition per vertex solve.
  - **Validation:**
    - VBD/AAAVBD/Chebyshev are bit-identical on trampoline and the beam. Anderson is within run-to-run noise.
    - On BeamFallingOnBeam, flags exactly match the gathered sets.
  - **Production timings (dev, capture_while):**
    - Trampoline: AAAVBD 58.7 → 28.4 ms, VBD 75.1 → 30.5, Chebyshev 47.8 → 22.2, Anderson 9.1 → 5.0.
    - BeamFallingOnBeam: −4…−5%.
  - **`capture_while` anatomy** (dev graph, AAAVBD trampoline): step 18.3 ms traced, gaps 1.2 ms (7%),
    ~237 sweeps per step (convergence test).
  - **ncu after the early-out:** AAAVBD's vertex solve takes 10.0 µs vs VBD's 7.6 µs (+32%, +36% instructions). The
    acceleration functions hold ~25% of the stalls.
  - **Ideas, not implemented:** hoist AAAVBD's per-vertex state loads, pre-filter surface neighbours, early-out for
    the dual updates.
  - **Report:** v3 of the same link.
- 2026-10-18 · **Maintainer: implement the cheap AAAVBD changes.** Commits `8439a205` (unconditional `gk[j]` load in the
  stencil augmentation) and `e2bc4fe4` (load `betaG/gk/xk/Hnk[i]` before the gathers).
  - **Validation:** bit-identical on trampoline and the beam.
  - **ncu:** 10.5 → 9.2 µs per launch.
  - **Exclusive-window replay:** trampoline AAAVBD 27.3 → 26.0 ms (−5%), BeamFallingOnBeam 98.1 → 93.7 (−4.5%).
- 2026-10-18 · **Anderson question (maintainer):** on trampoline, Anderson exits after 2 outer iterations (50
  sweeps/step) at gtol 1e-3, vs AAAVBD at 221–475 and VBD at 345–500. Against Newton, Anderson's position error is
  close to AAAVBD's (e.g. 4.0e-2 vs 3.0e-2 at frame 25) and below VBD's. Newton is a rough reference (1 outer
  iteration). **Per-step times are "time to pass |g| ≤ gtol", not equal-accuracy comparisons.** Reports must
  show sweeps/step, outer iterations/step, ms/sweep and the exit reason (coordinator).
- 2026-10-18 · **Exclusive window** (11:52–12:05):
  - **Fused constraint-set update (`ae18a043`):** −2…−3% on trampoline. Boundary constraint-set nodes go from 13.8
    to 3.6.
  - **Per-sweep cost:** AAAVBD is 1.22× VBD per sweep on trampoline (72 vs 59 µs; 1.17× with the changes).
  - **CUPTI exit segfault:** fixed (`f4804414`, merged).
  - **Next:** inclusive scope times (maintainer-approved via the coordinator).
- 2026-10-18 · **Inclusive scope times** (6 commits, merged):
  - **Stored:** `union_incl_ms` from raw intervals in the CUPTI recorder and the nsys importer.
  - **Derived:** `results.inclusive` and `results.scope_tree`.
  - **Views:** a scope tree in `report.py` and a Tree mode in the GUI panel.
  - **Tests:** unit-tested. The GUI Tree mode is on the maintainer's manual test list.
- 2026-10-18 · **Docs pass** (2 commits): docstrings for all public profiling API and the non-trivial helpers, and module
  docstrings matching their sections. Corrections for `.claude/skills/sim-profiling/SKILL.md` were sent to the
  coordinator, who owns the skill.

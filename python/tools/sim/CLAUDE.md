# python/tools/sim — Warp GPU simulation stack

This is the active development area. It implements a GPU FEM elastodynamics solver with augmented-Lagrangian (SAL) contact in **NVIDIA Warp** (plus cupy and `cuda.compute`/cuda-cccl for scans, reductions and sorts). Each solver mirrors a C++ algorithm in `source/pbat/sim/algorithm/` (see `source/pbat/sim/CLAUDE.md`). The two sides should agree closely but don't have to match exactly. When they diverge, check whether it's intentional before "fixing" either side.

Pinned dependencies are in `requirements.txt` (warp-lang, cupy, cuda-cccl, polyscope, h5py, ...). `pbatoolkit` must also be installed, built from this repo; see the `pbat-build` skill.

## Running

All commands run from the repo root, as modules:

```bash
python -m python.tools.sim.ui --fem-elasto-dynamics scene.h5:group/path [--sim-params params.h5]
python -m python.tools.sim.gpu.elasticity.fem     # modules with a unittest.TestCase run their tests via __main__
python -m unittest python.tools.sim.gpu.vbd.params
```

- Code layout: `state.py` holds `SimulationState`, the loaders and the HDF5 (de)serialization, shared by `ui.py` (GUI) and `simulate.py` (headless). GUI features live in `panels/<feature>.py` (see `panels/__init__.py`). Profiling scopes come from `gpu/common/profiling.py`.
- The UI is interactive (Polyscope + imgui), so Claude can't drive it. Use the headless counterpart instead:

  ```bash
  python -m python.tools.sim.simulate --fem-elasto-dynamics data/CubeFallingOnCube.Dump.h5:fem -o out.h5 [--sim-params p.h5] [--solver AndersonSolver] [--cd VertexSdf] [--duration 2] [--fps 60]
  ```

  It reuses `state.SimulationState` and the GUI's parameter (de)serialization. Its output uses the same layout as the GUI's Serialization feature:
  - `FemElastoDynamics/{X,E,m,dmask,...}`
  - `sim/<frame:08d>/x`
  - `params/` holding `fps`, `solver` and `cd` attributes, plus the full parameter groups.
- Instead of `--fem-elasto-dynamics dump.h5:fem`, both `ui.py` and `simulate.py` accept `--scene scene.h5`, a CPU editor scene file (`state.load_scene`, built like the CPU editor through `tools/vbd/scenario.py`). Static mesh colliders are ignored with a warning, since GPU contact has no static geometry. Settings come from `--sim-params` or the defaults either way. `reset()` restarts from the rest positions with the loaded initial velocities.
- Example Dump files live in `data/*.Dump.h5` and `data/scenarios/*.Dump.h5`. `data/` is git-ignored but present locally.
- The FEM must already be sorted by connected component. `SimulationState` raises an error otherwise.

## Relationship to `python/tools/vbd` (the CPU predecessor)

`ui.py` is the GPU successor of `python/tools/vbd/`, which pairs an online simulator (`editor.py`) with an offline one (`simulate.py`). Both run the **C++ solvers** through the `pbatoolkit` CPU API, so they're the natural source of reference trajectories when checking GPU results.

- **Input files.** The GPU UI takes as input the HDF5 file written by the editor's **Dump** button (`tools/vbd/ui/simulation.py::_serialize_problem` → `solvers/*.py::serialize_problem`). That file contains the groups `fem`, `contact`, `vbd/params` (plus solver-specific groups such as `vbd/anderson_params`) and `kinematics`, and an `initialization_strategy` metadata entry. Launch with `--fem-elasto-dynamics dump.h5:fem`. Changing how C++ `FemElastoDynamics` (or the dumped `MeshDynamics`/`Params`) serializes therefore affects both tools.
- **Kinematic constraints** (procedurally animated Dirichlet BCs, the editor's `TransformLibrary`) exist on both sides with the same closed-form semantics; see "Kinematic constraints" below. Other editor features may still be missing on the GPU, so check before assuming one exists in `tools/sim`.

## Architecture

**GPU objects are built from CPU objects.** Every GPU object is constructed from its `pbatoolkit` (C++) counterpart. The CPU object owns configuration, adjacency, coloring and (de)serialization. The GPU class copies them into a `@wp.struct` `*Data` and exposes it as `.data` for kernels:

| GPU (Warp) | Built from (pbatoolkit) | C++ reference |
|---|---|---|
| `gpu.elasticity.fem.FemElastoDynamics` | `pbat.sim.dynamics.FemElastoDynamics` | `sim/dynamics/FemElastoDynamics.h` |
| `gpu.integration.bdf.Bdf` | (dt, order) | `sim/integration/Bdf.h` |
| `gpu.vbd.params.Params` / `ChebyshevParams` / `AndersonParams` | `pbat.sim.algorithm.vbd.Params` (+ `ChebyshevParams`/`AndersonParams`) | `sim/algorithm/vbd/{Core,Chebyshev,Anderson}.h` |
| `gpu.newton.solver.Params` | `pbat.sim.algorithm.newton.Params` | `sim/algorithm/newton/Core.h` |
| `gpu.contact.multimesh.MultiMesh` | `pbat.sim.contact.MultiMesh` | `sim/contact/MultiMesh.h` |
| `gpu.contact.dynamics.MeshDynamics` | GPU-only `Params` (DocField) | `sim/contact/MeshDynamics.h` |

Enum values cross the boundary as `wp.constant(int(pbat....Enum.X.value))` (see the `VLS_SOLVER_*` constants in `gpu/vbd/params.py`), so the C++ enums stay the source of truth.

**Solver protocol.** Every solver class (`VbdSolver`, `AaaVbdSolver`, `ChebyshevSolver`, `AndersonSolver`, `NewtonSolver`) implements `solve(fem, contact, cd, params) -> bool` and the property `supports_graph_capture`. Their outer loops have the same shape as C++ `vbd::Solve`:

```
initialize_solve → for k < n_max_iters: [check_convergence] → prepare_subproblem
  → solve_subproblem (kp sweeps: contact.update_dual(slack only) → iterate)
  → finalize_subproblem (full dual update → cd.filter_step → cd.detect_contacts → contact.update_constraint_set)
→ fem.back_substitute_velocities() → cd.on_time_step_ended()
```

Shared steps are defined in `gpu/vbd/solver.py` and imported by the accelerated variants. `linearize_constraints` is still a TODO stub on the GPU. The UI calls `fem.setup_time_integration_optimization(...)` before `solve` and `fem.step()` after it.

**Convergence and early exit.** The VBD-family solvers share `vbd.solver.solve_outer_loop`:
- It tests C++'s criterion `|g_L|^2 <= gtol^2` on the device: g_L is the Lagrangian gradient at the current multipliers, without the AL penalty term (`gpu/vbd/convergence.py::ConvergenceCheck`, built on `gpu/gradient.py::GradientNorm`, which Newton uses in AL form). The test runs between outer iterations and every `ConvergenceParams.check_interval` sweeps (default 20) inside a subproblem. `SimulationState` builds one check and shares it among the VBD-family params (`params.convergence`).
- Outer iterations are unrolled on the host, each inside `wp.capture_if`. Inside a subproblem, the distinct first iterations are recorded once, then a block of identical iterations replays in `wp.capture_while`. Each solver describes this with `schedule() -> IterationSchedule` (prefix, period, sweeps per iteration); AAAVBD's first iteration, Chebyshev's first two and Anderson's first m are distinct.
- Conditional graph bodies must not allocate. `ContactDetection.prepare_detection()` runs between the conditional nodes; vertex SDF's `wp.Mesh.refit` lives there, since it allocates scan temporaries. A detector without `supports_conditional_capture` (OGC, untested) keeps its whole detection and filter outside conditionals.
- At equal work, results are bit-identical to the unrolled loop (Anderson within its run-to-run noise), and the graph is smaller and faster.

**Static problems.** With `fem.static` set, the objective is Ψ(x) − fextᵀx: solvers drop the inertia term, use `fem.h2 = 1` instead of `BetaTilde²`, and static Newton SPD-corrects d²Ψ/dF² (`gpu/elasticity/spd.py`). Its CG limits are `newton.solver.LinearSolverParams`.

**How solver names map between GPU and CPU:**
- `VbdSolver` is plain VBD.
- `AaaVbdSolver` adds per-vertex *stencil gradient acceleration* (`betaG`, `rhohat`, `gammaup`/`gammadown`, separate interior/surface values in `vec2f`). In C++ this acceleration is built into `vbd::Iterate` itself.
- Chebyshev and Anderson wrap the VBD sweep, matching C++ `Chebyshev.h` and `Anderson.h`. The GPU Anderson solves its least-squares problem by normal equations or CG (`AndersonLinearSolver`), whereas the CPU version uses a COD.
- `NewtonSolver` assembles the sparse Hessian from triplets and has `supports_graph_capture == False`.

**Kernel conventions.**
- **Thread layout.** The vertex solves launch `dim = n_verts_in_partition * 32` with `block_dim = 32`. Each block handles one vertex: threads stride over its adjacent elements and contacts, the partial sums are reduced with `wp.tile` + `wp.tile_reduce`/`wp.tile_sum`, and thread 0 does the 3x3 solve and writes `fem.x[i]`.
- **Gauss-Seidel ordering.** Color partitions run sequentially from the host (`params.Pptr` stays on the CPU; `Padj` is on the device).
- **Contact lagging.** `params.xb` is a snapshot of `x` taken before each sweep and is used to evaluate contacts.
- **Precision.** Everything is `float32` (`wp.vec3f`, `wp.mat33f`). The custom matrix and vector types are in `gpu/types.py`.
- **Ported math.** `gpu/elasticity/snh.py` (Stable Neo-Hookean) and `gpu/elasticity/chain.py` (∂Ψ/∂F → ∂/∂x) are direct ports of `physics/StableNeoHookeanEnergy.h` and `fem/DeformationGradient.h`.

**Small dense linear algebra inside the solver** (e.g. the Anderson least-squares step in `gpu/vbd/aasolver.py`):
- **Use Warp tile ops.** `wp.tile_matmul`, `wp.tile_cholesky`, `wp.tile_cholesky_solve` and `tile_*_solve` are backed by MathDx and capture cleanly.
- **Avoid CuPy's cuBLAS and cuSOLVER.** CuPy raises `NotImplementedError` for any cuBLAS or cuSOLVER call during stream capture, even through its low-level `cupy_backends` bindings.
- **Kernels with shape-specialized tiles.** Kernels whose tile shapes depend on a parameter (e.g. the Anderson window `m`) are generated per value with `module="unique"`. They must be created before capture starts (`AndersonParams` builds them in its constructor), so that capture's module force-load includes them.
- **Anderson solve modes.**
  - The Gram matrix of `[F f]` comes from one tiled pass. `FfT` stores `F^T` and `f` as one `(m+1, 3N)` buffer.
  - It is then solved by Cholesky, which regularizes only when a pivot falls below the relative threshold `cod_numerical_zero`, or by unregularized Jacobi-PCG.
  - `vbd_iters_per_anderson_iter` (`p`, GPU only) makes each Anderson iteration apply `p` VBD sweeps. `n_subproblem_max_iters` counts Anderson iterations, so a subproblem runs `n·p` sweeps.
  - Tests: `python -m unittest python.tools.sim.gpu.vbd.aasolver`.

**CUDA graph capture.** `state.SimulationState.step` records one `wp.ScopedCapture` of `substeps × (setup → solve → step)` and replays it. Capturing doesn't execute anything, so the first step launches the graph right after recording it. That means `solve()` on a capturable solver must:
- not allocate,
- not copy device data back to the host,
- not use Python control flow that depends on device values.

Loop counts come from host-side params, so changing them requires a recapture. The UI sets `state.capture = None` on reset. Handles must stay stable across steps, which is why `cd.register_handles(...)` is called once and buffers are preallocated with capacity multipliers (`gpu.contact.mesh.pairs.Params`). Work that is independent across contact types runs on separate `wp.Stream`s, one per VV/VE/VF/EE.

## Kinematic constraints

`gpu/kinematics/constraints.py` (module docstring has the exact semantics) mirrors the CPU editor's `TransformLibrary.evaluate` (`python/tools/vbd/ui/utils/transform_library.py`):

- **Data.** A `KinematicConstraint` prescribes the motion of a node group (`Fixed`, `Translate`, `Rotate`, `RadialTranslate`) over a half-open time window `[begin, begin + duration)`, rounded to the step grid. A `KinematicConstraintSet` holds all of them on the device.
- **Evaluation.** Positions are evaluated in closed form from per-node anchors (the position when a node's constrained run starts), composing overlapping constraints in order, so nothing drifts. Constrained nodes get the analytic prescribed velocity, so they keep moving when released.
- **Ownership of `dmask`.** `FemElastoDynamics.setup_time_integration_optimization` calls `kinematics.update(...)` first, which rewrites `dmask` and sets `x`/`v` of constrained nodes. Their `xtilde` is then their end-of-step position. `fem.step()` advances the device step counter. Time therefore lives in a device array and replays correctly in the captured graph. `advance_time = False` freezes it (needs a recapture).
- **Loading.** `state.load_kinematic_constraints` reads the Dump's `kinematics` group (written by `TransformLibrary.export_kinematic_constraints`, including `t0` and the anchors of runs open at Dump time). Dumps without one treat nonzero `dmask` as permanently fixed. `SimulationState` attaches a fresh set to every GPU FEM (construction and `reset`).
- **Contact.** `ContactDetection.register_handles(..., dmask)` makes `filter_step` skip constrained nodes, like C++ `RestoreFeasibility(x, dmask)`.
- **GUI.** `panels/kinematics.py` lists, toggles (on the device, no recapture) and highlights constraints.
- **Tests.** `python -m unittest python.tools.sim.gpu.kinematics.constraints` checks the GPU against the CPU editor, graph replay, releases and anchors.

## Contact on the GPU

The GPU uses the same contact model family as C++ `MeshDynamics`: an augmented Lagrangian with slack `s`, multipliers `lambda_n`/`lambda_f`, a decay weight `gamma`, and Coulomb friction in a 2D tangent basis. The code is organized differently from the C++ side, though:

- **`contact/mesh/cd.py`.** The abstract `ContactDetection` is pluggable, with implementations `ogc.Ogc` (Offset Geometric Contact, with BVHs over edges and faces) and `sd.Sd` (vertex vs. `wp.Mesh` SDF). The UI defaults to `VertexSdf`. Its `detect_contacts` postcondition requires pairs to be unique and sorted by `v` per `u`, followed by a call to `contacts.assemble_contacts()`. `filter_step` modifies `x` to keep steps feasible, which is the GPU analogue of `RestoreFeasibility`.
- **`contact/mesh/pairs.py`.** `ContactPairs` stores forward and reverse VV/VE/VF/EE pairs plus per-contact bases. Reverse pairs let a vertex solve gather the contacts in which it's the edge or face side.
- **`contact/constraints.py`.** `ConstraintSet` holds per-type dual state and warm-starts it across constraint-set updates.
- **`contact/dynamics.py`.** `MeshDynamics` handles dual updates and adaptive penalties. GPU-only differences from C++:
  - `PenaltyAdaptivity` (CONSTANT/TIMESTEP/SUBPROBLEM) uses a max-Rayleigh-quotient reduction into `sigma_n`/`sigma_f`.
  - `min_sigma_n` sets a floor on the normal penalty.
  - The penalty starts at `dt^2`.
- **`contact/debug/contact.py`.** Polyscope browsers for inspecting individual contact stencils.

## Params, UI and serialization

- **GPU-only parameter classes** use `common.fields.DocField` descriptors (a default value plus a docstring that becomes the imgui tooltip) together with `SerializableMixin`, which serializes them to h5 attributes.
- **UI widgets.** `panels.common.draw_params` reflects over objects: every public, settable `float`/`int`/`bool`/`Enum` attribute gets a widget. So a new C++ `Params` field bound with `def_rw`, or a new `DocField`, appears in the UI without any UI code.
- **CPU param serialization.** CPU param objects are serialized by the same reflection (`serialize_vbd_cpu_params`, etc. in `gpu/vbd/params.py`). The group layout (`Solver/<SolverType.name>`, `Contact/...`, `Integration`) is defined in `state.serialize_params`/`deserialize_params`.
- **GPU mirror updates are manual.** A field that kernels need must be copied into the `*Data` struct in the GPU `Params.__init__`. A new solver also has to be registered in `state.py` (`SolverType`, both `SimulationState.__init__` and `reset()`, `serialize_params`/`deserialize_params`) and in `ui.py` (`SOLVER_SUB_PARAMS`).

## Profiling

- **Scopes.** `gpu/common/profiling.py` provides `with profiling.scope("vbd.sweep"):`, the `@profiling.scoped("name")` decorator, and `push`/`pop` for inline regions (Newton's loop). Names are dotted and nest at run time into paths such as `solve/solve.finalize_subproblem/contact.detect`. Every GPU activity is attributed to the innermost scope open at its launch. With the backend `OFF` (the default), scopes are no-ops and safe inside graph capture.
- **Execution.** Any other backend makes `SimulationState.step` run eagerly, since per-kernel timing needs eager launches. `simulate.py --profile-warmup N` runs the first N steps unprofiled, with graph capture, then switches profiling on.
- **Nsight Systems is the reference.** `python -m python.tools.sim.profiling.nsys run -o out/prefix -- <simulate.py args>` runs `simulate.py --profile nvtx` under nsys, limits the capture to the measured steps, and imports the trace into `out/prefix.json`. Every kernel is included (Warp, CUB, cupy) with true GPU durations. nsys hangs on the venv's `python.exe` launcher, so the runner launches the base interpreter through `profiling/venvrun.py`.
- **Kernels backend (in-process, CUPTI).** `--profile kernels` times every kernel, memset and memcpy with CUPTI (`gpu/common/cupti.py`, a ctypes binding to the DLL from the pinned `nvidia-cuda-cupti-cu12` wheel; `cupti-python` has no Windows wheels). It attributes each activity to its scope through CUPTI external correlation ids. Don't use Warp's activity timing (`wp.timing_begin`) for this: in eager mode each measurement includes launch latency, which inflated totals about 4× on the beam. CUPTI and nsys can't run in the same process.
- **Production timing.** `simulate.py --profile-out f.json` with profiling off records each step's GPU time from CUDA events around the step (graph replay for capturable solvers).
- **GUI panel.** `panels/profiling.py` ("Profiling" tree node) toggles the CUPTI backend and shows a rolling window of per-phase GPU busy time. Export writes JSON for the report. `ui.py` wraps `state.step()` in `profile_step(state)`.
- **Report.** `python -m python.tools.sim.profiling.report a.json b.json ... -o report.html` takes one profile per solver and any production runs. It writes stacked bars, donuts and tables.
- Tests: `python -m unittest python.tools.sim.gpu.common.profiling`.

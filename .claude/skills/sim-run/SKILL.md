---
name: sim-run
description: Run regular elastodynamics simulations with the GPU simulator (python/tools/sim; headless simulate.py or the interactive ui.py) or, when a scene must be authored or a C++ reference trajectory is needed, the CPU editor/simulator (python/tools/vbd). Covers choosing inputs (editor Dumps vs scene files), solvers and contact detection, layered TOML parameter configs and --set overrides, kinematic constraints, static mode in the GUI, and reading outputs. Use when asked to simulate a scene, change simulation parameters, compare GPU and CPU runs, or author a new scene.
---

# Running simulations

Run everything from the repo root with the dev venv's Python (Python <= 3.12; its location is machine-specific, see `CLAUDE.local.md`). Module names below are run with `python -m`.

## GPU or CPU?

| Need | Use |
|---|---|
| Fast simulation, the paper's results, profiling, benchmarks | **GPU**: `python.tools.sim.simulate` (headless) or `python.tools.sim.ui` (GUI) |
| Author or edit a scene (bodies, materials, colliders, procedural motion) | **CPU editor**: `python.tools.vbd.editor`, then save the scene or press **Dump** |
| Reference trajectory from the C++ solvers (ground truth when checking GPU changes) | **CPU offline**: `python.tools.vbd.simulate` |
| Broyden-accelerated VBD | CPU only |

Default to the GPU. Use the CPU side only for authoring and reference runs.

## Inputs

The GPU simulators take exactly one of:

- **`--fem-elasto-dynamics file.h5:fem`**: an editor Dump, written by the CPU editor's **Dump** button. It holds `fem`, `contact`, solver params and the `kinematics` group (procedural constraints).
- **`--scene scene.h5`**: a CPU editor scene file, built the same way as the CPU editor builds it. Static mesh colliders and plane colliders (infinite by default, authored in the editor's Objects tab) are simulated too. The default adaptive contact penalty handles hard impacts on them.

**Prefer scene files (`--scene`) plus `--sim-params`** (maintainer's preference). `python -m python.tools.vbd.dump_to_scene dump.h5 scene.h5` turns a Dump taken at t = 0 back into a scene file. Use a Dump mainly to debug a crash that happens mid-simulation: it captures the state at that moment (current x and v, the kinematics' `t0` and open anchors), so the run resumes exactly there. The GPU never reads a Dump's solver parameters. Scenes live in `data/*.h5` and `data/scenarios/`; Dumps are `*.Dump.h5`. `data/` is git-ignored. From a worktree, point to the main checkout's data, e.g. `path/to/PhysicsBasedAnimationToolkit/data/scenarios/MediumLongBendingBeam.Dump.h5:fem`.

Commonly used scenes:

| Scene | Path | Notes |
|---|---|---|
| Bending beam | `--scene data/scenarios/MediumLongBendingBeam.h5` | Clamped beam, no contact, deterministic |
| Trampoline | `--scene data/scenarios/trampoline.h5` | Contact-free, deterministic, the profiling workhorse |
| Beam on beam | `--scene data/BeamFallingOnBeam.h5` | Real contacts; not bit-reproducible run to run |
| Cube on cube | `--scene data/CubeFallingOnCube.h5` | Tiny contact scene. Coincident surfaces pass through with VertexSdf; this is expected |
| Twist and pull | `--scene data/scenarios/TwistPullBeam.h5` | Kinematic constraints: rotating clamp, lifted then released tip |

## Headless GPU runs

```bash
python -m python.tools.sim.simulate --scene data/scenarios/MediumLongBendingBeam.h5 \
    -o out/beam.h5 --solver AAAVBD --duration 2 --fps 60 \
    --sim-params python/tools/sim/configs/paper/beam1e6.GroundTruth.toml --set Solver.AAAVBD.n_max_iters=2
```

| Option | Default | Meaning |
|---|---|---|
| `-s/--solver` | `AAAVBD` | `VBD`, `AAAVBD`, `Chebyshev`, `AndersonSolver`, `Newton` |
| `--cd` | `VertexSdf` | `VertexSdf` (default, the one the paper uses) or `OGC` (low priority, known crashes) |
| `-d/--duration` | `10` | Simulated seconds |
| `--fps` | `60` | Frame rate of the saved positions (independent of `dt`) |
| `-o/--output` | required | Output `.h5`, overwritten. Windows drive letters are fine |
| `--sim-params` | none | `.toml` configs and/or `.h5` parameter files, merged left to right |
| `--set Table.key=value` | none | Single-parameter override after `--sim-params`, repeatable. `value` is a TOML literal |
| `--profile*` | off | See the `sim-profiling` skill |

Translating requests:
- "Use solver X": `--solver`.
- "Run for N seconds": `-d N`.
- "Save every frame at 100 fps": `--fps 100`.
- "Use the paper's settings for scene S": `--sim-params python/tools/sim/configs/paper/<S>.toml`.
- "Change parameter P": `--set` (see the table below).
- "N substeps": `--set Integration.substeps=N`.

Without `--sim-params`, parameters are the code defaults (`configs/defaults.toml`), not the Dump's solver params.

## Parameters

Parameter files are layered TOML; the full guide is `python/tools/sim/configs/README.md`.
- Tables mirror the HDF5 groups (`[Solver.VBD]`). Enums are integers and bools are 0/1.
- A file can `extends = ["base.toml"]`.
- Unknown and missing keys raise a `ConfigWarning`.
- `python -m python.tools.sim.config show a.toml --set ...` prints the merged result.
- `python -m python.tools.sim.config convert p.h5 p.toml` converts a GUI export.
- **Never edit `configs/paper/*.toml`** (several were hand-tuned by a collaborator). Make a new file that `extends` one.

Frequently changed keys:

| Key | Meaning |
|---|---|
| `Integration.dt`, `Integration.substeps` | Time step and substeps per frame step |
| `Solver.<S>.n_max_iters` | Outer (augmented-Lagrangian) iterations per step, for `<S>` in `VBD`, `AAAVBD`, `Chebyshev`, `AndersonSolver` |
| `Solver.<S>.n_subproblem_max_iters` | VBD **sweeps** per subproblem, the same unit for every VBD-family solver (including Anderson) |
| `Solver.<S>.gtol` | Early exit when the Lagrangian gradient norm ≤ gtol (absolute); `0` disables early exit |
| `Solver.Convergence.check_interval` | Sweeps between convergence tests inside a subproblem (default 20) |
| `Solver.Convergence.ftol` | Feasibility tolerance between outer iterations: max penetration into the contact margin ≤ ftol·max(\|dmin\|, 1e-6 m), normal contact only |
| `Solver.AndersonSolver.m`, `.vbd_iters_per_anderson_iter` | Anderson window; sweeps per Anderson step (p). The budget stays n sweeps: n // p Anderson steps, then n % p plain sweeps |
| `Solver.Chebyshev.rho` | Chebyshev spectral radius estimate |
| `Solver.<S>.vlinsolve` | Per-vertex 3x3 solve: 0 Inverse (**default; use it**), 1 LLT, 2 QR, 3 EVD. Switch to EVD only when singular vertex Hessians cause problems |
| `Solver.Newton.n_max_iters`, `Solver.Newton.newton.n_max_iters`, `Solver.Newton.LinearSolver.*` | Newton's outer (AL) iterations; Newton steps per outer iteration; CG limits |
| `Contact.ContactDynamics.*`, `Contact.CDType.VertexSdf.*` | Contact penalties, friction `mu_f`, AL margin `dmin`; vertex SDF detects below `dmin + VertexSdf.band` (`VertexSdf.dmin` is deprecated) |
| `Contact.ContactDynamics.penalty_adaptivity`, `.gamman_adaptive` | 0 constant, 1 per time step, 2 per subproblem. Default: 1 (TIMESTEP), with `gamman_adaptive` 1. Adaptive penalties scale the largest vertex-Hessian Rayleigh quotient by `gamman_adaptive` (C++'s default 1). The contact multipliers are warm-started across steps, so softer penalties (`gamman_adaptive` 0.1, or CONSTANT at `gamman` 1e4) eject resting bodies |
| `Contact.ContactStorage.n_*_contact_capacity` | Contact-pair capacity multipliers. Contacts beyond capacity are dropped |

## Tuning (maintainer's guidance)

The GPU defaults (`state.load_params`, listed in `configs/defaults.toml`) follow this guidance. Use it when writing scene configs:

- **Outer iterations:** `Solver.<S>.n_max_iters = 1` for the VBD family (the default). Use more when a step needs several contact detection passes to resolve the constraint set without tunnelling, e.g. fast impacts or contact-rich scenes. The paper's contact scenes use 5 outer iterations (5 x 80 sweeps). Keep the total sweep budget below in mind.
- **Contact-free scenes:** keep `n_max_iters = 1` and turn contact detection off. Single-body scenes without colliders do this automatically, because vertex SDF has no self-contact. Otherwise set `--set Contact.CDType.VertexSdf.deactivate=1`.
- **Sweep budget:** the total over all outer iterations, subproblems and Anderson steps (`n_subproblem_max_iters` when `n_max_iters = 1`; default 100).

  | Mesh | Sweeps |
  |---|---|
  | Soft, low-res | ~50 |
  | Soft, higher-res | ~100 |
  | Stiffer (Young's modulus ~1e7) | ~200 |
  | Stiff, heterogeneous | ~400 |

  With substeps, count the sweeps across all of them.
- **Early exit:** `Solver.Convergence.rstol = 1e-2` and `astol = 1e-4` (defaults, dimensionless) are the relative and absolute stationarity tolerances of a mass-normalized gradient test, for the VBD family and Newton: it passes when RMS(M⁻¹g) ≤ max(rstol × the step's free-flight displacement, astol × L), L being the mean bounding-box diagonal of the scene's bodies. The absolute term keeps stiff scenes near rest from stalling above what float32 resolves; such scenes (Y ≈ 1e8 regions resting in contact) need 5e-4 to 1e-3, set in their scene configs (the Solve status panel shows astol × L and the residual ratio). It's resolution-independent for smooth residuals and stricter on fine meshes for contact or mesh-scale residuals, which is accepted. `ftol = 0.5` (default) allows half the contact margin between outer iterations (half of `|dmin|` for a negative margin, as in the paper configs; at least 1e-6 m).
- **AAAVBD:** the defaults are gammadown 0.99, gammadownS 0.95, gammaup(S) 0.5 and warm_start_beta 2. The scene configs share `gammaupS = 0.3` (0.5 also works well); lower it further for a contact-rich scene only if necessary. `betaup` (default 1) and `lipschitz_normalization` (default 1) are GPU-only ablation switches, not tuning knobs: keep them at their defaults, which match C++.
- **Prefer not to tune AAAVBD's own parameters per scene** (maintainer): betaG0, gammaup/down(S), rhohat(S), warm_start_beta, betaup, lipschitz_normalization should preferably stay one shared set across scenes, since the Lipschitz normalization is meant to make them problem-invariant, a selling point of the method. Tune budgets (outer iterations, sweeps, substeps), convergence thresholds, contact detection and penalty scale first; change an AAAVBD parameter for a scene only if necessary, and say why in the config.
- **Newton:** 1 outer iteration x 10 Newton steps by default (contact-free scenes, benchmarks). With contact, use 5 outer iterations x 2 Newton steps (more contact detection passes, the same step count), as the scene configs do.
- **Vertex solve:** Inverse (default). Use EVD only when singular vertex Hessians cause problems.
- **Paper scenes:** `configs/scenes/*.toml`, one per paper scene, are complete parameter sets: the paper config plus this guidance (TIMESTEP penalty, a positive AL margin `dmin = 1e-3` (the paper files' `-1e-4` dates from penetration-only detection), VertexSdf band 2 mm (5 mm for the sphere on the trampolines), Newton 5 x 2 with contact, Inverse, AAAVBD's shared gammas, `astol = 5e-4` for stiff scenes near rest, and 1 substep of 5 x 80 sweeps for the stiff heterogeneous contact scenes). Each header lists what differs from the paper config and why. They're generated by `python -m python.tools.sim.scene_configs make`: change its `SCENES` table, not the files. Several `configs/paper` files don't follow it (fewer sweeps, EVD), so prefer the scene config when one exists.
- **Contact penalty:** the default is TIMESTEP adaptive at `gamman_adaptive = 1`. Resting contacts settle at the margin `dmin` (2 mm). Use SUBPROBLEM only when it is clearly worth it for a scene (it gives the same results on the test scenes). Don't use CONSTANT (`penalty_adaptivity = 0`) without a stated justification (maintainer): soft constant penalties let resting bodies sink or eject with persistent multipliers, and caused centimetre impact penetration on the paper scenes; as a reference, it needs `gamman` around 1e6. The paper parameter files all use CONSTANT, so set `penalty_adaptivity = 1` in a scene config (as `configs/scenes/` does).
- **Friction:** the dual update projects `lambda_f` onto the Coulomb cone, and the VBD vertex solves cap the primal friction force at `mu_f` times the current normal force, so `mu_f` decides sliding vs sticking. The defaults, `gammaf = 1e3` (constant mode) and `gammaf_adaptive = 1e-2`, reproduce Coulomb on a 15° slope (tan = 0.27): the cube slides at 0.63-0.69 m/s² for `mu_f = 0.2` (Coulomb 0.64) and sticks for `mu_f` >= 0.3, while BeamFallingOnBeam stays stable. Much softer friction penalties make high-`mu_f` bodies creep; much stiffer ones make low-`mu_f` bodies too sticky (e.g. `gammaf_adaptive = 0.1` gives 0.45 m/s² at `mu_f = 0.2`). `gammaf = 0` removes friction. GPU Newton caps friction too (at the normal force of each Newton step's iterate, with a matching energy for its line search): it slides at 0.60 m/s² for `mu_f = 0.2` and sticks at 0.3 on the same slope, with no line-search failures. C++ Newton doesn't cap.

## Interactive GPU runs

```bash
python -m python.tools.sim.ui --scene data/scenarios/TwistPullBeam.h5 [--sim-params p.toml ...] [--set KEY=VALUE]
```

Claude can't drive the GUI. Hand the command to the user and say what to look at. Panels:
- **Simulation**: integration, solver and contact parameters, all editable live. Changes reset the captured CUDA graph.
- **I/O**: import `.toml`/`.h5` parameters; export TOML when the file name ends in `.toml`.
- **Serialization**: record the trajectory to HDF5, in the same layout as `simulate.py`.
- **Kinematic constraints**: list, enable or disable (no recapture needed) and highlight constrained nodes.
- **Static Equilibrium**: with "Static equilibrium" on, each step runs the solver on min Ψ(x) − s·fextᵀx from the current shape. You can load MCMC sample shapes and watch e/e0 against the rest shape.
- **Profiling**: live per-phase GPU time (see the `sim-profiling` skill).

## Kinematic constraints

Procedurally animated Dirichlet nodes (Fixed, Translate, Rotate, RadialTranslate) over half-open time windows. They come from the Dump's `kinematics` group (the editor's transform library) or from a scene file. A Dump without the group treats nonzero `dmask` as permanently fixed. They are evaluated in closed form, so nothing drifts, and released nodes keep their prescribed velocity. GPU and CPU semantics match. Details: `python/tools/sim/CLAUDE.md` ("Kinematic constraints") and the docstring of `gpu/kinematics/constraints.py`. To author constraints, edit the transform library in the CPU editor and Dump again. The `sim-kinematics` skill covers authoring, semantics and pitfalls in detail.

## Outputs

`simulate.py` and the GUI's Serialization write:

```
FemElastoDynamics/{X (N,3), E (E,4), lame_mu, lame_lambda, m, dmask}
sim/<frame:08d>/x                     (N,3) positions per saved frame
sim/<frame:08d>/solve_status/         simulate.py only: the solves (time steps or substeps) since the previous
                                      frame; frame 0 has none, and steps after the last frame aren't stored
  vbd                                 VBD family: step, substep, solver, outer, sweeps, reason, gradient_ratio,
                                      feasibility_ratio, max_outer, max_sweeps
  newton_solves, newton_subproblems   Newton: outer, converged; per subproblem status, newton_steps,
                                      max_cg_iters, cg_capped, backtracks (+ budgets)
  attrs solve_end, subproblem_status  enum names of reason / status; lost (solves the ring overwrote)
params/                               attrs fps, solver, cd + every parameter group (convertible to TOML)
```

**Playing an output in the CPU editor.** Open the editor (`python -m python.tools.vbd.editor`), load the scene the output was simulated from (Scene > Session tab > **Load session**), switch to the Simulation mode and press **Load Trajectory** in its Integration tab (Group `sim`, the default). GPU frames are read with h5py: the editor loads whichever FEM nodal fields a frame has (`x`, `v`, `atfd`, `fext`, `xtilde`; the GPU tools write only `x`) and leaves the rest. `v` is zeroed when absent, so the kinetic energy isn't meaningful for these files, and a warning under the frame slider says the frames are partial. The static colliders' nodes, which come after the bodies' nodes, are dropped. A different scene (node count or rest positions) is rejected with an error. **Dump** then writes the loaded frame for the GPU simulators.

**Solve status, for tuning budgets and tolerances.** `simulate.py` prints a summary at the end: how many VBD-family solves converged early vs hit the outer-iteration budget (and whether the gradient test, feasibility or both were unmet), mean/max outer iterations and sweeps against the budget, and the worst final ratios to the tolerances (gradient measure / tolerance, penetration / (`ftol * max(|dmin|, 1e-6 m)`); at most 1 passes). For Newton: subproblem statuses (stepped, converged, non-descent, line-search failed), Newton steps, CG iterations and cap hits, and line-search backtracks. The GUI's "Solve status" panel shows the last step per substep and the same summary over a rolling window. A solve that keeps hitting the budget with `gradient_ratio` well above 1 needs more sweeps or outer iterations (or looser `Solver.Convergence.rstol`/`astol`); a feasibility ratio above 1 points at contact (penalty, detection). GPU Newton also runs its outer test after the last subproblem, as C++. On the 15° incline cube at the defaults, 120 of 125 Newton solves pass the outer test.

```python
import h5py
import numpy as np
with h5py.File("out/beam.h5") as f:
    frames = sorted(f["sim"])
    x_last = f["sim"][frames[-1]]["x"][()]
    vbd = np.concatenate([f["sim"][k]["solve_status"]["vbd"][()] for k in frames
                          if "vbd" in f["sim"][k]["solve_status"]])   # VBD-family solves, all frames
```

To rerun with a run's exact parameters: `--sim-params out/beam.h5:params`. It warns about the unknown `fps`/`solver`/`cd` keys, which is harmless.

## CPU editor and reference runs

```bash
python -m python.tools.vbd.editor                     # author scenes; Save, or Dump for the GPU simulators
python -m python.tools.vbd.simulate --scene data/BeamFallingOnBeam.h5 -s vbd -d 2 -o out/cpu.h5:sim \
    [-p params.h5] [--overrides vbd.n_max_iters=20]
```

- **Solvers:** `vbd`, `anderson_vbd`, `broyden_vbd`, `chebyshev_vbd`, `newton`. `--spec` prints the parameter specification.
- **Output:** `-o` takes `file.h5:group`.
- **Parameters:** CPU parameter files use the CPU format (root attributes `dt`, `bdf_scheme`, ... plus per-solver groups). They are **not** the GPU TOML configs.
- **Comparing with the GPU:** CPU and GPU solvers mirror each other but aren't bit-identical.

## Pitfalls

- **Shared GPU:** while parallel sessions run, log `GPU run START`/`END` in `plans/MASTER.md`, and don't take timings while another session uses the GPU.
- **Contact scenes** aren't bit-reproducible run to run (atomics). Compare them by tolerance or contact sets. The beam and trampoline are deterministic.
- **FEM ordering:** the FEM must be sorted by connected component; `SimulationState` raises otherwise.
- **`gtol` early exit:** per-step time depends on the stopping rule. Don't compare solver speeds from ms/step at a loose `gtol` (see the `sim-benchmarks` skill).
- **OGC:** has known robustness issues (a CUDA 700 crash on CubeFallingOnCube with AAAVBD). Prefer VertexSdf.

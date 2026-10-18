---
name: sim-run
description: Run regular elastodynamics simulations with the GPU simulator (python/tools/sim; headless simulate.py or the interactive ui.py) or, when a scene must be authored or a C++ reference trajectory is needed, the CPU editor/simulator (python/tools/vbd). Covers choosing inputs (editor Dumps vs scene files), solvers and contact detection, layered TOML parameter configs and --set overrides, kinematic constraints, static mode in the GUI, and reading outputs. Use when asked to simulate a scene, change simulation parameters, compare GPU and CPU runs, or author a new scene.
---

# Running simulations

Run everything from the repo root with the venv's Python: `/c/git/python-3.12/Scripts/python.exe` in Bash, `C:\git\python-3.12\Scripts\python.exe` in PowerShell. Module names below are run with `python -m`.

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
- **`--scene scene.h5`**: a CPU editor scene file, built the same way as the CPU editor builds it. Static mesh colliders are ignored with a warning, because GPU contact has no static geometry yet (task 1 is adding them).

Scenes live in `data/*.h5` / `data/*.Dump.h5` and `data/scenarios/`. `data/` is git-ignored. From a worktree, use absolute paths such as `C:/git/PhysicsBasedAnimationToolkit/data/scenarios/MediumLongBendingBeam.Dump.h5:fem`.

Commonly used scenes:

| Scene | Path | Notes |
|---|---|---|
| Bending beam | `data/scenarios/MediumLongBendingBeam.Dump.h5:fem` | Clamped beam, no contact, deterministic |
| Trampoline | `data/scenarios/trampoline.Dump.h5:fem` | Contact-free, deterministic, the profiling workhorse |
| Beam on beam | `data/BeamFallingOnBeam.Dump.h5:fem` | Real contacts; not bit-reproducible run to run |
| Cube on cube | `data/CubeFallingOnCube.Dump.h5:fem` | Tiny contact scene. Coincident surfaces pass through with VertexSdf; this is expected |
| Twist and pull | `data/scenarios/TwistPullBeam.Dump.h5:fem` | Kinematic constraints: rotating clamp, lifted then released tip |

## Headless GPU runs

```bash
python -m python.tools.sim.simulate --fem-elasto-dynamics data/scenarios/MediumLongBendingBeam.Dump.h5:fem \
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
| `Solver.Convergence.ftol` | Feasibility tolerance between outer iterations: max penetration into the contact margin ≤ ftol·dmin, normal contact only |
| `Solver.AndersonSolver.m`, `.vbd_iters_per_anderson_iter` | Anderson window; sweeps per Anderson step (p). The budget stays n sweeps: n // p Anderson steps, then n % p plain sweeps |
| `Solver.Chebyshev.rho` | Chebyshev spectral radius estimate |
| `Solver.<S>.vlinsolve` | Per-vertex 3x3 solve: 0 Inverse (**default; use it**), 1 LLT, 2 QR, 3 EVD. Switch to EVD only when singular vertex Hessians cause problems |
| `Solver.Newton.n_max_iters`, `Solver.Newton.LinearSolver.*` | Newton steps; CG limits |
| `Contact.ContactDynamics.*`, `Contact.CDType.VertexSdf.*` | Contact penalties, friction `mu_f`, margin `dmin`, detection distance |
| `Contact.ContactStorage.n_*_contact_capacity` | Contact-pair capacity multipliers. Contacts beyond capacity are dropped |

## Interactive GPU runs

```bash
python -m python.tools.sim.ui --fem-elasto-dynamics data/scenarios/TwistPullBeam.Dump.h5:fem [--sim-params p.toml ...] [--set KEY=VALUE]
```

Claude can't drive the GUI. Hand the command to the user and say what to look at. Panels:
- **Simulation**: integration, solver and contact parameters, all editable live. Changes reset the captured CUDA graph.
- **I/O**: import `.toml`/`.h5` parameters; export TOML when the file name ends in `.toml`.
- **Serialization**: record the trajectory to HDF5, in the same layout as `simulate.py`.
- **Kinematic constraints**: list, enable or disable (no recapture needed) and highlight constrained nodes.
- **Static Equilibrium**: with "Static equilibrium" on, each step runs the solver on min Ψ(x) − s·fextᵀx from the current shape. You can load MCMC sample shapes and watch e/e0 against the rest shape.
- **Profiling**: live per-phase GPU time (see the `sim-profiling` skill).

## Kinematic constraints

Procedurally animated Dirichlet nodes (Fixed, Translate, Rotate, RadialTranslate) over half-open time windows. They come from the Dump's `kinematics` group (the editor's transform library) or from a scene file. A Dump without the group treats nonzero `dmask` as permanently fixed. They are evaluated in closed form, so nothing drifts, and released nodes keep their prescribed velocity. GPU and CPU semantics match. Details: `python/tools/sim/CLAUDE.md` ("Kinematic constraints") and the docstring of `gpu/kinematics/constraints.py`. To author constraints, edit the transform library in the CPU editor and Dump again.

## Outputs

`simulate.py` and the GUI's Serialization write:

```
FemElastoDynamics/{X (N,3), E (E,4), lame_mu, lame_lambda, m, dmask}
sim/<frame:08d>/x                     (N,3) positions per saved frame
params/                               attrs fps, solver, cd + every parameter group (convertible to TOML)
```

```python
import h5py
with h5py.File("out/beam.h5") as f:
    frames = sorted(f["sim"])
    x_last = f["sim"][frames[-1]]["x"][()]
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

# Simulation-parameter configs

TOML files that set the GPU simulators' parameters (`ui.py`, `simulate.py`, `benchmark.py`). The code is in `../config.py`.

These files are cached here for now. Their long-term home is `C:\git\accelerated-vbd-extras` (the paper's repository), next to the HDF5 originals in `parameters/gpu`.

## Layout

| Path | Contents |
|---|---|
| `defaults.toml` | Every parameter the code serializes, with its code default. Use it as a reference or as an `extends` base. |
| `paper/*.toml` | The paper's GPU parameter files (`accelerated-vbd-extras/parameters/gpu/*.h5`), converted and completed. See "Paper configs". |

## Format

A config mirrors the HDF5 layout that `state.serialize_params` writes (and the GUI's "Export Parameters"):

- each group is a table: `Solver/VBD` is `[Solver.VBD]`, `Solver/Newton/newton/line_search` is `[Solver.Newton.newton.line_search]`;
- each attribute is a key;
- enums are integer values and bools are `0`/`1`, as in HDF5. Enum values are those of the C++ enums (e.g. `vlinsolve = 3` is `VertexIntegrationLinearSolver.EVD`) or of the GPU `DocField` enums. The GUI shows the names.

Floats are written exactly. A value that came from a C++ `float` therefore reads `0.0010000000474974513` rather than `0.001`. Don't shorten such values in converted files: the host uses some of them in double precision (`Integration.dt`, contact `dmin`), so shortening changes results. In your own configs, write whatever values you want.

## Using configs

```bash
python -m python.tools.sim.simulate --fem-elasto-dynamics dump.h5:fem -o out.h5 \
    --sim-params python/tools/sim/configs/paper/beam1e6.GroundTruth.toml my_tweaks.toml \
    --set Solver.VBD.gtol=1e-4 --set Solver.Convergence.check_interval=10
```

`--sim-params` and `--set` work the same way in `ui.py` and `benchmark.py`. The GUI's I/O panel imports `.toml` and `.h5` files, and exports TOML when the file name ends in `.toml`.

Sources are merged in this order, with later values winning:

1. Each `--sim-params` source, left to right. A source is a `.toml` config or an HDF5 parameter file (`file.h5`, or `file.h5:group`, e.g. `out.h5:params` from a `simulate.py` output).
2. Within a TOML file, the files listed in its optional `extends = ["base.toml", ...]` come first, in order (paths are relative to that file), and then the file's own values on top. `extends` can be nested, but cycles are an error.
3. Each `--set Table.Sub.key=value`, left to right. `value` is a TOML literal (`1e-4`, `20`, `true`).

The merged result is then checked against the keys the current code serializes. You get a `ConfigWarning` for:

- **unknown keys**, e.g. typos or parameters that were removed. These are ignored.
- **missing keys**, which keep their current values (the code defaults). A table missing altogether is reported once.

To make warnings fatal, run with `python -W error::python.tools.sim.config.ConfigWarning`. To print the merged result without running anything:

```bash
python -m python.tools.sim.config show a.toml b.toml --set Solver.VBD.gtol=1e-4
```

## Making a config

- **Partial override:** write only the keys you change and `extends` the config you start from:

  ```toml
  extends = ["paper/beam1e6.GroundTruth.toml"]

  [Solver.AndersonSolver]
  vbd_iters_per_anderson_iter = 4
  ```

- **From an HDF5 file**, e.g. a GUI export: `python -m python.tools.sim.config convert params.h5 params.toml`.
- **From the GUI:** I/O > Export Parameters, with a `.toml` file name.

The checks compare against the code's keys, not against `defaults.toml`. Even so, regenerate `defaults.toml` when you add a parameter, so it stays a complete reference. To regenerate it, write `config.dumps(config.params_dict(state))` from a `SimulationState` built with `state.load_params` (the GPU defaults, which differ from the C++ ones).

## Paper configs

`paper/*.toml` are the hand-tuned GPU parameter files from `accelerated-vbd-extras/parameters/gpu`. Several were tuned by hand by a collaborator for specific tests.

- **Never edit them, and never modify the HDF5 originals.** To change something, `extends` the paper config in a new file.
- **Provenance:** each file's header records its source file, the extras commit, the source's SHA-256 prefix and the date.

The conversion followed these rules:

1. Every value in the source is kept as is. Loading a paper `.toml` gives the same parameters, bit for bit, as loading its `.h5`.
2. If the source has no `Solver.Chebyshev` / `Solver.AndersonSolver` table, the new table copies the scene's `Solver.VBD` table: the same sweep budget and vertex solve, since both solvers accelerate plain VBD sweeps. It then adds the C++ defaults for `rho` (Chebyshev) and `m`, `beta`, `cod_numerical_zero` (Anderson). These tables are **not hand-tuned**, and each header says so.
3. Every other key the code serializes but the source lacks takes its code default. The header lists these keys, e.g. `Solver.Convergence.*` (`check_interval`, `ftol`), `Solver.Newton.LinearSolver.*`, `Solver.Newton.spd_correction`, and the GPU-only Anderson keys `linear_solver` and `vbd_iters_per_anderson_iter`.
4. Keys the code no longer serializes are dropped and listed in the header: since 2026-10-18 the stencil gradient acceleration keys (betaG0, rhohat(S), gammaup(S), gammadown(S), warm_start_beta, surface_stencil_surface_neighbours_only) exist only under `Solver.AAAVBD`, so they're dropped from the VBD, Chebyshev and Anderson tables. The GPU never used them there, so loading is still bit-identical to the `.h5`.

`beam1e6.GroundTruth.toml` comes from the extras working-tree version of its `.h5` (uncommitted when converted). That version already has hand-made Chebyshev and Anderson tables, so rule 2 did not apply to it.

`parameters/gpu/beam.GroundTruth.Notes.txt` in the extras repository describes how the beam budgets were chosen.

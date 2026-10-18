# python/tools/vbd — CPU simulators (predecessor of tools/sim)

These are the online (`editor.py`) and offline (`simulate.py`) simulators built on the **CPU** `pbatoolkit` API (`pbat.sim.algorithm.{vbd,newton}`, `pbat.sim.contact.MeshDynamics`). New work happens in `python/tools/sim/` (GPU, Warp). This folder still matters for two reasons:

1. **It produces the GPU simulator's inputs.** The editor's **Dump** button (`ui/simulation.py::_serialize_problem`) writes an HDF5 file through the selected solver's `ui/solvers/*.py::serialize_problem`: `fem`, `contact`, `vbd/params` (non-minimal), solver-specific params, and an `initialization_strategy` metadata entry. Then `python -m python.tools.sim.ui --fem-elasto-dynamics dump.h5:fem` reads it.
2. **It produces reference results.** `simulate.py` runs headless CPU simulations and writes per-frame `fem`/`contact` archives. Its `--help` lists the options (`--solver`, `--duration`, `--output`, `--convergence`, ...). Use it as ground truth when validating GPU changes.

Features that exist only here (for now): kinematic/procedural Dirichlet constraints (`_apply_procedural_constraints`), the Broyden-accelerated VBD solver UI, and convergence analysis (`ui/convergence.py`). When dumping, procedural constraints are baked into static "release after X ms" Dirichlet constraints.

Run from the repo root: `python -m python.tools.vbd.editor` and `python -m python.tools.vbd.simulate -h`. Dependencies are in `requirements.txt`.

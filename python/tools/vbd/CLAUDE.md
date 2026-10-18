# python/tools/vbd — CPU simulators (predecessor of tools/sim)

These are the online (`editor.py`) and offline (`simulate.py`) simulators built on the **CPU** `pbatoolkit` API (`pbat.sim.algorithm.{vbd,newton}`, `pbat.sim.contact.MeshDynamics`). New work happens in `python/tools/sim/` (GPU, Warp). This folder still matters for two reasons:

1. **It produces the GPU simulator's inputs.** The editor's **Dump** button (`ui/simulation.py::_serialize_problem`) writes an HDF5 file through the selected solver's `ui/solvers/*.py::serialize_problem`: `fem`, `contact`, `vbd/params` (non-minimal), solver-specific params, an `initialization_strategy` metadata entry, and the `kinematics` group (`TransformLibrary.export_kinematic_constraints`: every transform as a kinematic constraint, plus the Dump time and open anchors). Then `python -m python.tools.sim.ui --fem-elasto-dynamics dump.h5:fem` reads it.
2. **It produces reference results.** `simulate.py` runs headless CPU simulations and writes per-frame `fem`/`contact` archives. Its `--help` lists the options (`--solver`, `--duration`, `--output`, `--convergence`, ...). Use it as ground truth when validating GPU changes.

Procedural Dirichlet constraints (`ui/utils/transform_library.py`) are evaluated in closed form by `TransformLibrary.evaluate`, with the same semantics as the GPU's `gpu/kinematics`: half-open windows on the step grid, per-node anchors, overlapping transforms composed in library order, and prescribed velocities for constrained nodes. L_TRANSLATE moves each node radially at `speed`; `migrate_l_translate.py` converts scenes authored with the old group-normalized behavior. Tests: `python -m unittest python.tools.vbd.ui.utils.transform_library`.

Features that exist only here (for now): the Broyden-accelerated VBD solver UI and convergence analysis (`ui/convergence.py`).

Scene files (bodies, static colliders, transform library) are read by `scenario.py` (`read_scene`, `make_fem_dynamics_object`, `make_contact_dynamics_object`, `load_simulation_scenario_from_scene`), shared by the editor, the offline simulator and the GPU simulators' `--scene`. Constraint nodes are mapped by body name, and a missing `static_mesh_colliders` group means no colliders.

Run from the repo root: `python -m python.tools.vbd.editor` and `python -m python.tools.vbd.simulate -h`. Dependencies are in `requirements.txt`.

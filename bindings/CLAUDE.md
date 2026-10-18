# bindings — nanobind Python bindings (`pbatoolkit`)

`bindings/pypbat/` builds the extension module `_pbat` (target `PhysicsBasedAnimationToolkit_Python`), which Python imports as `from pbatoolkit import pbat`. The directory tree mirrors `source/pbat/`. Each submodule has `X.h`/`X.cpp` with a `void Bind(nanobind::module_& m)`, and the module tree is assembled in `PythonBindings.cpp` and the per-directory aggregators (e.g. `sim/algorithm/vbd/Vbd.cpp` calls `BindCore`, `BindChebyshev`, `BindAnderson`, `BindBroyden`). New files must be listed in that directory's `CMakeLists.txt`, on `target_sources(PhysicsBasedAnimationToolkit_Python ...)`. The pure-Python package is `python/pbatoolkit/` (`pbatoolkit.py.*` holds the Python-side helpers).

## Conventions

- **Library.** Use nanobind (`namespace nb = nanobind`), not pybind11. Eigen arrays pass as `nb::DRef<Eigen::... const>` for zero-copy input.
- **Naming.** Python names are snake_case: functions (`vertex_element_adjacency_graph`), builder methods (`WithVertexColors` → `with_vertex_colors`), arguments (`nb::arg("n_nodes")`), and fields (`def_rw("n_max_iters", &Params::nMaxIters)`). Enums drop the `E` prefix (`EVertexIntegrationLinearSolver` → `VertexIntegrationLinearSolver`).
- **Builder setters** return `Params&` with `nb::rv_policy::reference_internal`, so calls chain in Python: `Params().with_...().construct()`.
- **Templates.** The C++ solvers are templates, so the bindings fix concrete types. For example, `sim/algorithm/vbd/Core.cpp` uses `ElasticEnergyType = physics::StableNeoHookeanEnergy<3>` over `FemElastoDynamics<Tetrahedron<1>, 3, ...>`. Bind each step function (`initialize_solve`, `iterate`, `prepare_subproblem`, `finalize_subproblem`, `solve`, ...) individually so Python can drive or inspect the loop.
- **Docstrings.** Give every function and field a docstring. `ui.py` in `tools/sim` shows `__doc__` as imgui tooltips.
- **Serialization.** Params that need it expose `serialize(archive, minimal=True)`/`deserialize(archive)` over `pbat.io.Archive` (HDF5 through HighFive).
- **Warp mirrors.** Field names bound here are read by name in `python/tools/sim/gpu/**/params.py` (e.g. `params.GVGe`, `params.vlinsolve`, `params.rhohatS`) and reflected over by the UI. Renaming a field breaks those mirrors, so grep `python/tools` for the old name.
- **Stubs.** `PhysicsBasedAnimationToolkit_PythonStub` regenerates `.pyi` stubs with nanobind stubgen.
- **Rebuilding.** After editing bindings, rebuild and reinstall the extension with the `pbat-build` skill before running any Python.

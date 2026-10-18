---
name: pbat-python-api
description: End-to-end checklist for adding or changing pbatoolkit's Python API (a C++ solver/contact/dynamics parameter, function, or enum) and propagating it through bindings, the Warp GPU mirrors in python/tools/sim/gpu, the sim UI, and HDF5 serialization. Use whenever a change touches source/pbat/sim/** and must be visible from Python.
---

# Changing the pbatoolkit Python API

A parameter or function exposed to Python passes through up to six places. Work through them in order, and at the end report which ones you touched and which you deliberately skipped.

## 1. C++ (`source/pbat/sim/...`)

- **Entry points.**
  - VBD: `algorithm/vbd/Core.h` (`Params`). Accelerated variants have their own `*Params` in `Chebyshev.h`, `Anderson.h` and `Broyden.h`.
  - Newton: `algorithm/newton/Core.h`.
  - Contact: `contact/MeshDynamics.h` (`MeshDynamics::Params`).
  - FEM problem: `dynamics/FemElastoDynamics.h`.
- **New `Params` field:**
  - Add it with a default value and a `///<` doc comment, plus a `With*` setter if it belongs to a builder group.
  - Validate it in `Construct(bValidate)` if that applies.
  - Add it to `Serialize`.
  - Add it to `Deserialize`, guarded with `HasMetaData`, so `.h5` files dumped by older versions (editor "Dump" files, `--sim-params` files) still load.
- **Placement and docs.** Put per-vertex or per-element math in `Kernels.h` as `PBAT_HOST_DEVICE` functions using `mini::` types. Update the Doxygen comments.
- **Tests.** Add or extend a doctest in the matching `.cpp`.

## 2. Bindings (`bindings/pypbat/sim/...`, same relative path)

- **Fields.** Use `def_rw("snake_name", &Params::member, "doc")`. Use snake_case names. The docstring becomes the UI tooltip.
- **Setters.** Bind with `nb::rv_policy::reference_internal` and snake_case `nb::arg`s, with defaults matching the C++ ones.
- **Enums.** Use `nb::enum_<E...>(m, "NameWithoutE")`.
- **New functions.** Bind them for the instantiated types only (`StableNeoHookeanEnergy<3>`, `Tetrahedron<1>`). Copy the existing lambdas in `sim/algorithm/vbd/Core.cpp`.
- **New files.** Add them to that directory's `CMakeLists.txt` and call their `Bind*` from the aggregator.

## 3. Rebuild

Use the `pbat-build` skill. Check that the new symbol is visible in the venv: `python -c "from pbatoolkit import pbat; print(pbat.sim.algorithm.vbd.Params().<field>)"`.

## 4. Warp mirror (`python/tools/sim/gpu/...`)

This step applies only if GPU kernels need the value.
- Add a typed member to the `@wp.struct` `*Data` class, e.g. `ParamsData` in `gpu/vbd/params.py`, with a comment pointing to the C++ source.
- Copy the value from the CPU object in the GPU class's `__init__`, casting explicitly (`float(...)`, `int(...)`).
- Map enums through `wp.constant(int(pbat....Enum.X.value))` constants.
- Values that only drive host-side loops stay as Python attributes rather than struct members. `Pptr` is an example.

## 5. UI and serialization (`python/tools/sim/state.py`, `ui.py`)

- **Automatic coverage.** `draw_params` and `serialize_*_cpu_params`/`deserialize_*_cpu_params` reflect over public, settable scalar and enum attributes. A bound `def_rw` field or a `DocField` therefore gets a widget and h5 persistence automatically. Confirm that the new field's type is one of float, int, bool or Enum.
- **When a field is needed on the GPU.** The GPU `Params` is rebuilt from the CPU object only in `SimulationState.__init__`/`reset()`. Check that UI edits trigger a reset (and a recapture: `state.capture = None`), or the GPU keeps stale values.
- **New solver or param group.** Register it in `state.py` (`SolverType`, `SimulationState.__init__` and `reset()`, `serialize_params`/`deserialize_params`) and in `ui.py` (`SOLVER_SUB_PARAMS`).

## 6. Other callers

- Grep `python/` for the changed or renamed name. Expect hits in `python/tools/vbd/**` (CPU editor and simulator, which produce the GPU inputs), `python/examples/`, and `python/tools/sim/gpu/**`.
- If the serialized layout of `FemElastoDynamics`, `MeshDynamics` or `vbd::Params` changes, the editor's Dump files and the GPU UI's loaders (`load_fem_dynamics`, `load_vbd_params`) must still agree.

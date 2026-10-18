---
name: warp-port
description: Port or sync a C++ solver/contact feature from source/pbat/sim into the NVIDIA Warp GPU implementation in python/tools/sim/gpu (or add a new GPU solver), following the repo's kernel, graph-capture, and CPU-mirror conventions, and validate it against the CPU reference. Use when implementing GPU counterparts of C++ algorithms or when GPU and CPU results diverge.
---

# Porting C++ simulation code to Warp

Read `python/tools/sim/CLAUDE.md` and `source/pbat/sim/CLAUDE.md` first if they aren't already in context.

## 1. Locate both sides

- **C++ reference.** Find the function in `source/pbat/sim/algorithm/<solver>/` (`Core.h` for control flow, `Kernels.h` for per-vertex math) or in `contact/MeshDynamics.h`.
- **Existing GPU counterpart.** Look in `python/tools/sim/gpu/<solver>/`. Shared outer-loop steps live in `gpu/vbd/solver.py`. Kernel helpers live in `gpu/vbd/kernels.py`, `gpu/elasticity/{snh,chain,fem}.py` and `gpu/contact/*`.
- **Reuse.** Before writing a new kernel, check for an existing helper (`local_elastic_derivatives`, `local_contact_derivatives`, `integrate_positions`, `add_inertia_derivatives`, `gpu/common/{scan,reduce,sort,search}.py`).
- **Intentional differences.** Some divergences are deliberate: pluggable CD, `PenaltyAdaptivity`, Anderson's normal-equations/CG solve, and stencil acceleration living in `AaaVbdSolver` only. Keep them unless asked otherwise.

## 2. Mirror data, don't recompute it

- Configuration, adjacency and coloring come from the CPU `pbatoolkit` object. Copy them into a `@wp.struct` in the GPU class's `__init__` (see `gpu/vbd/params.py::Params`).
- If the CPU object lacks something you need, add it in C++ and expose it with the `pbat-python-api` skill, instead of reimplementing the preprocessing in Python.

## 3. Kernel conventions

- **Precision.** Use `float32` throughout (`wp.vec3f`, `wp.mat33f`, plus the custom types in `gpu/types.py`). Positions are `wp.array[wp.vec3f]` with shape `(N,)`.
- **Per-vertex block reductions.** Launch with `dim = n_items * block_dim` and `block_dim = 32`. Compute `block_id = tid // block_dim` and `local_tid = tid % block_dim`, stride over adjacency with `local_tid`, reduce with `wp.tile(..., preserve_type=True)` + `wp.tile_reduce(wp.add, ...)`/`wp.tile_sum`, and let only `local_tid == 0` write.
- **Gauss-Seidel.** Launch color partitions sequentially from the host using `params.Pptr` (a host numpy array) and `Padj` (on the device).
- **Ported math.** Keep `@wp.func` math as a line-for-line port of the C++ `PBAT_HOST_DEVICE` kernel, and name the C++ source in the docstring. That keeps diffs against C++ reviewable.
- **Concurrency.** Use one `wp.Stream` per contact type (VV/VE/VF/EE) for independent work.

## 4. Graph-capture safety

Capturable solvers (`supports_graph_capture == True`) must not do any of the following inside `solve()` or anything it calls:
- allocate arrays,
- call `.numpy()` or `int(device_array[...])`, or synchronize,
- branch on device values,
- change array handles.

Preallocate in `__init__`/`construct`, using capacity multipliers for variable-size buffers, and use device-side reductions (`gpu/common/reduce.py`) instead of host reads.

Library calls under capture:
- **Allowed:** `cuda.compute` (CCCL) algorithms with preallocated temp storage, and Warp tile ops (`tile_matmul`, `tile_cholesky*`, `tile_*_solve`, `tile_reduce`, ...).
- **Not allowed:** CuPy's cuBLAS and cuSOLVER bindings, which refuse to run during capture.
- **Tile shapes are compile-time.** Specialize kernels per size with `module="unique"` and create them before capture begins. Mask runtime-sized subsets, like the active window `mk` in Anderson, inside the kernel. If a feature truly needs host decisions, set `supports_graph_capture` to False, as `NewtonSolver` does.

## 5. Wire it up

For a new solver class with `solve(fem, contact, cd, params) -> bool` and `supports_graph_capture`, register it in `ui.py`: `SolverType`, `SOLVER_SUB_PARAMS`, `SimulationState.__init__` and `reset()`, and `_serialize_params`/`_deserialize_params`.

## 6. Validate against the CPU

- **Unit tests.** Add a `unittest.TestCase` in the module with an `if __name__ == "__main__": unittest.main()` guard, following `gpu/elasticity/fem.py`. Compare `@wp.func` outputs with the bound C++ function or with finite differences.
- **Solver-level check.** Write a scratch script, outside the repo, that:
  1. loads a dumped problem (`pbat.io.Archive(path)` → `pbat.sim.dynamics.FemElastoDynamics().deserialize(archive["fem"])`),
  2. builds the GPU objects the way `SimulationState.__init__` does, and runs N steps,
  3. runs the same N steps with the CPU solver through `pbat.sim.algorithm.<solver>` (as `python/tools/vbd/simulate.py` does),
  4. compares `x` and the per-step residual or energy.
  Expect float32-level differences and contact-model differences. Report the magnitudes instead of asserting exact equality.
- **Running anything.** Use the `pbat-build` skill's environment notes (venv, and the workaround for the intermittent DLL import failure). For a second opinion on fidelity, ask the `cpu-gpu-parity` agent to review the diff.

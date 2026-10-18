---
name: pbat-build
description: Build the PBAT C++ library/tests, or rebuild and reinstall the pbatoolkit Python extension into the dev venv, and import it reliably on Windows. Use after any change under source/ or bindings/, before running Python that depends on the change, or when `import pbatoolkit` fails with "DLL load failed".
---

# Building PBAT and pbatoolkit

## Python environment

- Use the venv at **`C:\git\python-3.12`**. pbatoolkit supports Python ≤ 3.12, so the system `python` (3.14) won't work.
  - Bash: `/c/git/python-3.12/Scripts/python.exe`
  - PowerShell: `& C:\git\python-3.12\Scripts\Activate.ps1`
- The installed `pbatoolkit` is a CUDA build, but its CUDA runtime is linked statically, so CUDA doesn't need to be on `PATH`. Smoke test: `/c/git/python-3.12/Scripts/python.exe -c "from pbatoolkit import pbat; print(pbat.gpu)"`.
- **Former intermittent import failure, fixed 2026-10-17.** About 5% of imports used to fail with `DLL load failed while importing _pbat`, because Embree's `/DEPENDENTLOADFLAG:0x2000` hid the package directory from embree4's lookup of `tbb12.dll`.
  - `cmake/dependencies.cmake` now resets the flag on the `embree` target.
  - To check a build: `dumpbin /loadconfig <site-packages>\pbatoolkit\embree4.dll` should show `0000 Dependent Load Flag`, and `ctypes.WinDLL(r"<...>\embree4.dll")` should succeed.
  - If the failure ever comes back, suspect a new dependency built with a nonzero DependentLoadFlags value.
- **HDF5/zlib are linked statically.** This comes from `cmake/triplets/x64-windows.cmake`. pbatoolkit therefore ships no `hdf5.dll`/`z.dll`, and it no longer collides with h5py's copies in either import order. If an `hdf5.dll` reappears in the package, it's probably stale and was copied from the build tree's `lib/Release/`.

## C++ library and tests

```powershell
cmake --preset=user                     # configures ./build (VS multi-config, MSVC /WX)
cmake --build build --target PhysicsBasedAnimationToolkit_Tests --config Release
.\build\bin\Release\PhysicsBasedAnimationToolkitTests.exe -tc="[sim]*"
```

- **Build times.** The template-heavy headers (`sim/algorithm/vbd/Core.h`, `contact/MeshDynamics.h`) are slow to compile. Run builds with a long timeout or in the background.
- **Warnings are errors.** Fix warnings rather than suppressing them. Use the `pbat/warning/*.h` push/pop headers only for third-party or generated code.

## Python extension (bindings)

The Python package installs through scikit-build-core (`pyproject.toml`, which builds `PhysicsBasedAnimationToolkit_Python` into `build/{wheel_tag}`). Run from the repo root with the venv's pip:

```bash
/c/git/python-3.12/Scripts/python.exe -m pip install . -v --config-settings cmake.args="--preset=user"
```

- **What the preset provides.** The `user` preset (in `CMakeUserPresets.json`, which is local and not tracked by git) enables SuiteSparse, METIS, Tracy and the MSVC `/WX` warnings. It does **not** enable CUDA. The CUDA C++ path (`pbat.gpu`) doesn't currently build against CUDA 13.3: CCCL requires `/Zc:preprocessor`, and cuda-api-wrappers v0.8.1 predates CUDA 13. The GPU work in `python/tools/sim` uses Warp and doesn't need it. Don't add extra `-D` flags on the command line. If the configuration needs to change, change the preset.
- **Where it builds.** scikit-build-core builds into `build/cp312-cp312-win_amd64/`, a separate tree from the `build/` tree used for C++ tests.
- **Thread count.** If the user gives a thread count `N`, pass it through `build.tool-args`, the same way `.github/workflows/{pyinstall,wheels}.yml` do. With no `N`, omit the setting; MSVC's `/MP` in the `msvc` preset then uses all cores.
  - Windows (MSBuild + MSVC): `--config-settings build.tool-args="/p:CL_MPCount=N"`. This caps the number of `cl.exe` processes that `/MP` spawns per project.
  - Linux/macOS (Ninja/Make): `--config-settings build.tool-args="-jN"`.
  - Example: `... -m pip install . -v --config-settings cmake.args="--preset=user" --config-settings build.tool-args="/p:CL_MPCount=8"`.
- **Thread count for the C++ test tree:**
  - Windows: `cmake --build build --config Release --target PhysicsBasedAnimationToolkit_Tests -- /p:CL_MPCount=N`.
  - Linux/macOS: `cmake --build build --parallel N`.
  - On Windows, `--parallel N` becomes MSBuild `/m:N`, which builds N *projects* concurrently. Combined with `/MP` inside each project, that oversubscribes the CPU. Prefer `CL_MPCount` there.
- **Build time and output.** A full rebuild takes a long time, so run it in the background and keep a log.
- **HDF5 linkage.** The overlay triplet `cmake/triplets/x64-windows.cmake` (enabled through `VCPKG_OVERLAY_TRIPLETS` in the `default` preset) links hdf5, zlib and libaec statically. That avoids DLL name clashes with h5py.
- **Re-check the PATH step after reinstalling.** A full reinstall takes many minutes. Afterwards, run the PATH-prefixed import check above, then the target script (e.g. `python -m python.tools.sim.gpu.elasticity.fem`).
- **Regenerating stubs.** Build target `PhysicsBasedAnimationToolkit_PythonStub` to regenerate the `.pyi` stubs.

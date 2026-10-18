# source/pbat/sim — C++ simulation (reference implementation)

These C++ solvers are the ground truth that the Warp GPU stack in `python/tools/sim/gpu` mirrors. The Python API (`pbat.sim.*`) is a thin nanobind layer over this directory (see `bindings/CLAUDE.md`).

## Layout

- `algorithm/`: the current API. Free functions over `(fem, contact, params[, accel])`.
  - `vbd/`: Vertex Block Descent, in `Core.h` (`Params`, `Integrate`/`Solve`/`Iterate`/...) and `Kernels.h` (`PBAT_HOST_DEVICE` per-vertex math). Accelerated variants are `Chebyshev.h`, `Anderson.h` and `Broyden.h`, each with its own `*Params` and overloads that take it as an extra argument. Enums are in `Enums.h`.
  - `newton/`: Newton's method with a sparse Hessian (`Core.h`), used as the **ground-truth** solver. It has a pluggable linear solver (`ELinearSolver`), SPD correction, and OGC step truncation (`EOgcTruncationStrategy`).
  - `pd/`: projective dynamics (quasi-Newton, Valanis-Landel).
  - `common/Common.h`: `common::FemElastoDynamics<TEnergy>`, an alias that fixes the problem type to **linear tets, 3D, `Scalar`/`Index`**.
- `dynamics/FemElastoDynamics.h` holds the FEM problem: mesh, `x`/`v`, mass `m`, the inertial target `xtilde`, the Dirichlet mask `dmask`, per-quadrature-point Lamé parameters (`lamegU`), and a `bdf` integrator. It provides `ToElasticGradient`, `ToMomentumGradient`, `Step`, `BackSubstituteIntegratedPositionsIntoVelocities`, and `Serialize`/`Deserialize`.
- `integration/Bdf.h` implements BDF time integration (`BetaTilde`, `Inertia`, `CurrentState`).
- `contact/MeshDynamics.h` is **the entry point for contact** (see below). Also in `contact/`: `MultiMesh.h` (the surface of each connected component, half-edges, and `GXV` mapping volume vertices to surface vertices, `-1` for interior ones), `Constraints.h` and `Friction.h` (per-primitive constraint and friction functions), `Potentials.h`, and `ogc/` (Offset Geometric Contact detection: `Params`, `Input`, `State`). `MeshSdfContact.h` and `MultibodyMeshMixedCcdDcd.h` are separate, older contact paths.
- `vbd/` and `xpbd/` (directly under `sim/`) hold the **legacy** `Data`/`Integrator` API, which is being deprecated. Don't extend them. New work goes in `algorithm/`.

## Solve structure (shared by VBD and Newton)

```
Integrate:  InitializeSolve → Solve → fem.Step()
InitializeSolve: contact query radius from |xtilde - xt|; UpdateConstraintSet(xt); RestoreFeasibility(x)
Solve: for k < nMaxIters                      // outer sequential augmented Lagrangian (SAL) loop
         LinearizeConstraints(fem, contact)   // freeze contact normals/gradients at x
         if CheckConvergence: break           // ‖elastic + momentum + contact grad‖ < gtol
         PrepareSubproblem                    // e.g. penalty update via Rayleigh quotients
         while kp < nSubproblemMaxIters: Iterate   // VBD: graph-colored Gauss–Seidel over vertices
         FinalizeSubproblem                   // UpdateDual<Slack|LagrangeMultiplier|Decay>; RestoreFeasibility; UpdateConstraintSet(x)
       fem.BackSubstituteIntegratedPositionsIntoVelocities()
```

**VBD `Iterate`** (`vbd/Core.h`) works as follows:
- It first calls `contact.UpdateDual<Slack>` and snapshots `x` into `params.xb`, so contacts are lagged.
- It then processes the color partitions (`Pptr`/`Padj`) in sequence, with a TBB `parallel_for` within each partition.
- Each vertex's 3x3 system combines three terms built in `BuildVertexEquation`:
  - elastic terms from `AccumulateElasticEnergy` (over the `GVGp`/`GVGe`/`GVGilocal` adjacency), scaled by `h²` (`h = bdf.BetaTilde()`),
  - contact terms from `AccumulateContactEnergy`, which loops over the vertex's forward PP/PE/PT contacts and, through its half-edges (`GVHEp`/`GVHEadj`), the reverse EP/TP/EE contacts,
  - inertia.
- Stencil gradient acceleration (`ComputeStencilGradientAugmentation`, the `betaG` family of parameters) is applied inside every `Iterate`. On the GPU it's a separate solver, `AaaVbdSolver`.
- The solve itself is `kernels::IntegratePositions` with `EVertexIntegrationLinearSolver`.

**Problem instantiation.** The templates are generic over `physics::CHyperElasticEnergy`, but the bindings instantiate only `StableNeoHookeanEnergy<3>` on `Tetrahedron<1>`.

## Contact model (`contact/MeshDynamics.h`)

`MeshDynamics<Scalar, Index>` owns:
- the dynamic and static `MultiMesh`es,
- the OGC detection state,
- four `ConstraintSet`s: point-point, point-edge, point-triangle and edge-edge. Each is a `graph::DenseAdjacencySet` whose per-contact columns are, by index: `lambda`, slack `s`, decay `gamma`, `chat`, `gradc`, `c(x)`, the tangent basis `T`, the tangential weights `W`, `c_f`, `dhat`, `lambda_f`, `sigma`, `sigma_f`.

Contacts are traversed with `ForEach{PointPoint,PointEdge,EdgePoint,PointTriangle,TrianglePoint,EdgeEdge}Contact` and `ForAllContacts`. Stencil positions are loaded with `LoadStencil`.

Its `Params` are set through builder methods:
- `WithOgcParams` (contact radius `r`, query radius `rq`),
- `WithNormalContact(kc)`,
- `WithFrictionalContact(mu, epsv)`,
- `WithQueryRadiusInitialization(rqstart, betarq)`,
- `WithSequentialAugmentedLagrangian(gamma, gammaf, dmin, decay, decaylo)`.

The normal penalty is `gamma * kc` and the friction penalty is `gammaf * kc`. `vbd::ESALPenaltyStiffness` selects global or local max-Rayleigh-quotient penalty updates.

**Differences on the GPU** (`python/tools/sim/gpu/contact`):
- Detection is pluggable: OGC or vertex-SDF.
- Penalty adaptivity uses a GPU-only enum.
- Penalties are floored at `min_sigma_n`.
- Dual state is warm-started across constraint-set rebuilds.

Expect small numerical differences in the contact model between the two sides. Treat large ones as bugs.

## Conventions here

- New solver features belong in `algorithm/<solver>/`. Reusable per-vertex or per-element math goes in `Kernels.h` as `PBAT_HOST_DEVICE` functions using `mini::` types, so it stays portable to CUDA and stays easy to port to Warp.
- Instrument new phases with `PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.<solver>.<Function>")`.
- Any new `Params` field needs three things:
  - a `With*` setter (or a direct `def_rw` binding),
  - (de)serialization in `Serialize`/`Deserialize`, written with `HasMetaData` guards on read so old `.h5` dumps still load,
  - exposure in `bindings/pypbat/sim/...`.
- Doctests for these templates go in the matching `.cpp` file (e.g. `algorithm/vbd/Core.cpp`). Run them with `-tc="[sim]*"`.

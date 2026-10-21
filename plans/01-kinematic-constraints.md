# Task 1: Kinematic Dirichlet constraints on the GPU

## Handoff (2026-10-21)

Read first: `MASTER.md` (rules), `python/tools/sim/CLAUDE.md` ("Kinematic constraints", contact), `python/tools/vbd/CLAUDE.md`,
skills `sim-kinematics` and `sim-run`. Local locations (venv, extras repo, main checkout with `data/`) are in `CLAUDE.local.md`.

### 1. Scope and ownership
- **Kinematic constraints:** `python/tools/sim/gpu/kinematics/constraints.py`, `python/tools/sim/panels/kinematics.py`, the Dirichlet
  parts of `gpu/elasticity/fem.py`; CPU side `python/tools/vbd/ui/utils/transform_library.py` (closed form, Dump export),
  `vbd/migrate_l_translate.py`.
- **Scene ingestion and colliders:** `state.load_scene`/`load_problem`, `python/tools/sim/colliders.py`, `vbd/scenario.py`
  (`read_scene`, `make_fem_dynamics_object`, `static_meshes`), `vbd/ui/plane_collider.py`, `vbd/dump_to_scene.py`,
  `vbd/rim_patch_scene.py` (benchmark trampolines).
- **AL contact warm start and VertexSdf detection** (shared with task 3): `gpu/contact/constraints.py` (multiplier snapshot by
  (u, v) key), `gpu/contact/mesh/sd.py` (detection band, sign-aware normals), the `gpu/contact/dynamics.py` penalty defaults.
- **Editor trajectory viewer:** `vbd/ui/trajectory.py`, `vbd/ui/simulation.py::_draw_trajectory_ui`/`_set_trajectory_fields`.
- **Scene configs' contact margin:** `python/tools/sim/configs/scenes/*.toml` (`[Contact.ContactDynamics] dmin`).

### 2. Current state
- Everything is merged into dev. Kinematics/contact merges: `c41bc1ad` kinematic constraints · `65fc5fe1` GPU reads CPU
  scene files · `b31e53f7` plane/static mesh colliders · `c06e7c08` finite planes are plates · `b1cfe462` dump_to_scene ·
  `0d1ad502` rim_patch_scene · `bb05a622` lower friction penalties · `ae0b35b5` AL warm start + detection band + TIMESTEP
  default · `6d29fbe0` and `148d9113` viewer plays GPU outputs · `42c4e719` scene-config margin 1e-3.
- **20-21 Oct (benchmark viewer and material views, all on dev):**
  - `python/tools/sim/benchmark_viewer.py` (new, skill `sim-benchmarks`): side-by-side solver trajectories of a results
    file (`ea85fb4a`), overlays of x*, xtilde, start state (dynamic) and X (static) (`3045712a`), hide all solver meshes
    (`4c7e6c10`), material color modes (`bc062b16`) and lumped mass mode (`985529c4`).
  - `python/tools/vbd/ui/material.py` (new, numpy only; tests `python -m unittest python.tools.vbd.ui.material`): the CPU
    editor's names, log10(value + 1) scale, Young's range (5, 9) and colormaps, shared by every mesh GUI (`8ba139ac`).
    Masses show as log10(m) over their 2-98th percentiles, with massless nodes at the smallest mass (`c27d92c6`).
  - The same Lame mu, lambda, Young's modulus and Mass quantities in the GPU GUI (`c2000639`), the CPU editor (`d78a32b0`,
    adds Young's), the modalrom GUI (`9719f01e`) and the benchmark browser (`dd995243`, `aa6e5d77`).
  - modalrom: a .msh input's lumped mass matrix was not diagonal; now row sums (`40888576`, test
    `python/examples/test_modalrom.py`).
  - The viewer crash on static sample 4 was the mislabelled trajectories (MASTER); my frame-dropping branch was deleted
    unmerged. The coordinator later changed the viewer's default error colors (`4b4cd56a`) and added `benchmark_video.py`.
- No unmerged branches and no uncommitted changes. The `pbat-kinematics` worktree is detached at `84e164f1`, 13 commits
  behind dev `b4e73feb` and none ahead. To resume: `git switch -c feature/<new> dev` there.

### 3. Key decisions (maintainer's unless noted)
- **Constraints** (Q1-Q11 below): closed form from per-node anchors, composed in library order; half-open windows on the step
  grid; released nodes keep the prescribed velocity; per-node `L_TRANSLATE`; the GPU GUI only displays and toggles (authoring
  stays in the CPU editor); no countdown reader; a nonzero `dmask` without a `kinematics` group means fixed forever.
- **Contact:** the multipliers persist across steps, which needs a stiff penalty, so the default is TIMESTEP adaptive at
  `gamman_adaptive = 1`, C++'s value. Softer penalties (0.1, CONSTANT 1e4) eject resting bodies, because λ/σ sets the reach.
  The friction model is known to be erroneous, so the maintainer chose small friction penalties (gammaf 10, adaptive 1e-3) to
  limit their effect; task 3 then added the Coulomb cap.
- **Scene configs:** `dmin = 1e-3` in `configs/scenes/` (maintainer approved 1-2 mm). Never edit `configs/paper/*`.
- **Viewer:** keep it simple (read `x` with h5py); the partial-frames warning is fixed text and doesn't name the missing members.
- **Scenes:** don't use `cow_on_trampoline` or the octopus scenes. No heavy scenes (ThingsFallingInBasket, celebration,
  basket_of_fruit) without the maintainer; the basket timing was cancelled. For demos, launch the GUI right away, without long
  offline pre-checks.

### 4. Verified facts and gotchas
- **Node order:** GPU FEM = the bodies' nodes in scene file order (the editor loads them the same way), then the static
  colliders' element-less nodes. Planes add nodes too (CubeOnInfinitePlane: 8 + 4). `SimulationState` never reorders; without
  collider geometry it only asserts that the input is sorted by connected component.
- **VertexSdf:** detects below `ContactDynamics.dmin + VertexSdf.band` (defaults 2e-3 + 1e-3). `VertexSdf.dmin` is deprecated:
  it warns and is ignored (`config.DEPRECATED`). `detector._sds` holds end-of-step per-vertex signed distances, and `max_dist`
  (0.05) where nothing is near. There is no self-contact, so single-body scenes without colliders turn detection off.
- **Normals:** VF uses the face normal, VE the edge pseudonormal, VV the angle-weighted vertex pseudonormal. A blanket flip
  made near-surface contacts attractive (that was the reason for the old penetration-only `dmin = -1e-4`).
- **Scene vs Dump:** scenes carry the transform library and colliders, and colliders only work via `--scene`. Dumps carry a
  `kinematics` group (t0 + open anchors) and no colliders. The output's `FemElastoDynamics/dmask` is only the initial set.
- **Viewer formats:** CPU frames are `sim/<f>/pbat.sim.dynamics.FemElastoDynamics` (deserialized). GPU frames are
  `sim/<f>/x` plus a `solve_status` subgroup, with the FEM once at the root (`FemElastoDynamics/{X,E,...}`). Load the
  scene in the editor before the trajectory.
- **Same mesh:** TwistPullBeam and MediumLongBendingBeam share the same 1804-node mesh, so either scene accepts the other's outputs.
- **Contact runs** aren't bit-reproducible (atomics). Contact-free scenes are bit-identical across the contact changes.
- **Warp:** a multiple-return `wp.func` can't be unpacked into indexed targets (`a[k], b[k] = f()`). Use separate statements.
- **CPU reference runs** (`vbd.simulate`) need `-p` with root attrs `dt`, `bdf_scheme`, `fem_dynamics_init_strategy` plus
  `Solver/VBD` and `Contact` groups. None ships in the repo, so serialize `pbat.sim.algorithm.vbd.Params()` and
  `pbat.sim.contact.MeshDynamicsParams()` into a scratch file.
- **Windows:** stopping a background shell can leave its `python.exe` child running. Check for and kill orphaned processes
  before the next GPU run.

### 5. Open items (each with a next step)
1. **mat_on_stilts_3 with `configs/scenes/mat_on_stilts.toml`** (AAAVBD, 1 substep, 5 x 80): after the 2 s release, the mat
   penetrates about 2 cm, loses contact by about 3.5 s and falls (z −26 m at 5 s). This happens at dmin 1e-3 and −1e-4, and
   with the TIMESTEP penalty, while the config header says it was stable. Next: check whether it tunnels or slides off
   (play the output in the editor); compare with the 10-substep paper config; bisect from `ff63102b`/`78232143`, the commits that introduced it and changed it to 5 x 80.
2. **Paper penalty too soft:** CONSTANT `gamman` 2e4 (σ = 2 at dt 0.01) gives 1-5 cm impact penetration on the circle
   trampolines with sphere. TIMESTEP 1 fixes stiff-soft (deepest +0.4 mm, rests at +1.0 mm) but goes non-finite on
   soft-stiff at step 258, just after the impact. Next: maintainer decision; investigate the soft-stiff blow-up (substeps,
   gammaupS).
3. **ThingsFallingInBasket** with the defaults goes non-finite at about step 39 on the warm-start head (ve/vf at capacity,
   about 100 s per step). Whether this predates the merge is unknown. Next: only with the maintainer; 45 steps of `ea968473`
   would settle it.
4. **CubeFallingOnCube** passes through with VertexSdf on both dev and the fix (a known limitation of vertex-only detection).
5. **CONSTANT 1e6** isn't a safe body-body fallback: BeamFallingOnBeam loses the beam with VBD/AAAVBD.
6. **Older gaps** (still open): GPU outputs store only the initial `dmask`; GPU VBD sags about 6x more than CPU VBD on the
   held beam (a solver difference, not kinematics); OGC crashes once bodies touch (low priority).

7. **Viewer and results:** the full results file `data/benchmarks/results/trampolines.h5` has snapshots for samples 0, 4,
   9, 13, 18, 22, 27, 31 only. Its `dataset` attribute is repo-relative, so outside the main checkout pass `--dataset`.
   The material views were checked by screenshot only; the maintainer hasn't reviewed them interactively.
8. **Trampoline pass-through check** (sphere through the plate): cancelled; the maintainer tests the trampolines
   interactively. Nothing to do unless asked.

### 6. How to resume
- Run from the repo root with the venv's python; `data/` lives in the main checkout.
- Tests: `python -m unittest python.tools.sim.gpu.kinematics.constraints python.tools.vbd.ui.utils.transform_library
  python.tools.vbd.ui.plane_collider python.tools.sim.gpu.elasticity.fem python.tools.sim.config python.tools.sim.status`.
  The contact modules have no unit tests; use the scene checks below.
- Smallest checks:
  - `python -m python.tools.sim.simulate --scene data/scenarios/MediumLongBendingBeam.h5 -o out.h5 -d 0.2` (contact-free; the
    output can be played in the editor).
  - `data/scenarios/planes/{CubeOnInfinitePlane,FastCubeOnInfinitePlane}.h5` should rest at +2 mm.
  - `--fem data/BeamFallingOnBeam.Dump.h5:fem` (body-body).
  - Kinematics: `data/scenarios/TwistPullBeam.h5`, `prisms/*/twisting_beam_*.h5`, `spaghetti/spaghetti-tangle-het_block_*.h5`.
- Cheap contact scenes (about 50-100 ms/step): `path/to/accelerated-vbd-extras/final_scenes/trampolines/trampoline_circle_{soft-stiff,stiff-soft}__with_sphere.h5`
  with `configs/scenes/trampoline_{soft-stiff,stiff-soft}.toml`, and `final_scenes/mat_on_stilts_3.h5` with
  `configs/scenes/mat_on_stilts.toml`. The sphere or mat is released at 2 s, so simulate about 5 s.
- Measuring (scratch scripts are gone; rebuild them from these APIs):
  - Deepest and resting gaps: `state.detector._sds`.
  - Pair counts vs capacity: `state.contact.contacts.read_data[0].{vv,ve,vf,ee}.prefix[-1]` and `.capacity`.
  - Solve statuses: `state.status_log.format_summary()`.
  - Overrides: `config.load_sim_params(state, [toml], ["Table.key=value"])`.
- Log GPU START/END in `MASTER.md`. Timing runs need a quiet GPU.

---

Status (superseded by the Handoff above): **merged into dev (`c41bc1ad`); GUI panel check by the maintainer pending** · Branch `feature/gpu-kinematic-constraints` · Worktree `pbat-kinematics`
Read first: `MASTER.md` (rules, ownership), `python/tools/sim/CLAUDE.md`, `python/tools/vbd/CLAUDE.md`.

## Goal

Support every procedural Dirichlet constraint that the CPU editor supports, in the GPU simulator
(`ui.py` and `simulate.py`), with the motion evaluated on the GPU inside the captured CUDA graph (no
per-step CPU/GPU round trips). Replace the current "release after X ms" hack. Give the feature proper names
in the new simulator.

## What exists today

> Superseded (2026-10-20): describes the pre-implementation code (countdown hack, incremental motion), all replaced.

**CPU editor** (`python/tools/vbd/ui/utils/transform_library.py`, `TransformLibrary`):
- Each `PrimitiveTransform` has a name, a time window `[begin, begin + duration]` in seconds, and per-mesh
  Dirichlet node lists. A node is constrained only while some transform affecting it is active; outside the
  window it is free.
- Kinds (`TransformType`):
  - `FIXED`: hold position.
  - `G_ROTATE`: rotate about an axis through the world origin at `degrees_per_second`.
  - `L_ROTATE`: rotate about an axis through `origin`.
  - `TRANSLATE`: move along `direction` at `speed`.
  - `L_TRANSLATE`: move radially away from `origin` in the xy-plane at `speed`.
- Motion is applied **incrementally**: each step applies the per-step delta (rotation by `ω·dt`, translation
  by `v·dt`) to the current positions (`apply_procedural_constraints` in `tools/vbd/simulate.py:409`,
  `_apply_procedural_constraints` in `tools/vbd/ui/simulation.py:545`). This accumulates rounding drift and
  depends on the step history.
- The transform library is serialized with the editor's scene/session, not with the Dump.

**The release hack** (what the GPU sim supports now):
- On Dump, the editor writes each constrained node's remaining active time in **milliseconds** into `dmask`
  (`_apply_release_dirichlet_constraints`, `tools/vbd/ui/simulation.py:557`). Motion is not exported.
- The GPU kernel `_setup_time_integration_optimization_kernel` (`gpu/elasticity/fem.py`) decrements it every
  step by `round(beta_tilde · 1000)`. Note that this uses BDF's `beta_tilde` rather than `dt`, so for BDF
  order > 1 the release time is wrong (for BDF2, `beta_tilde = 2/3 dt`).

**GPU Dirichlet handling:** `dmask` (nonzero means constrained) is read by the VBD kernels (skip node), the
time-integration setup (`xtilde = current position`) and velocity back-substitution (free nodes only).
`dbc`/`ndbc` partition free and constrained nodes; check which solvers use them (Newton's Hessian
assembly is a likely user). If the constrained set changes over time, that partition must be maintained on
the device too.

## Maintainer decisions so far

- **Fix the drift in both simulators, keeping feature parity.** Switch the CPU editor/offline simulator
  (`python/tools/vbd`) and the new GPU simulator to the same closed-form evaluation. The CPU side keeps its
  existing names (`TransformLibrary` etc.); only the new simulator gets new names.
- **Fix the release-time bug** (countdown decremented by `beta_tilde` instead of `dt`). The countdown goes
  away with the new design, but if a backward-compatible reader for old Dumps keeps any countdown logic, it
  must use `dt`.

## Proposed design (to confirm with the maintainer)

> Superseded (2026-10-20): implemented as answered in the questions below; current semantics in `python/tools/sim/CLAUDE.md` and the `sim-kinematics` skill.

**Naming.** In the new simulator, call these *kinematic constraints*: a `KinematicConstraint` is a prescribed
motion of a node group over a time window, and a `KinematicConstraintSet` holds all of them. Kinds: `Fixed`,
`Translate`, `Rotate` (an axis plus a pivot; the world-origin version is just pivot 0), `RadialTranslate`.
The old editor keeps its names; only the data it exports maps onto the new ones.

**Closed-form, absolute motion.** Evaluate the prescribed position from the node's anchor position at the
start of the window, x_D(t) = T(t − t₀) · x_D(t₀), instead of applying per-step deltas. That gives no drift,
exact positions at any time, and a prescribed velocity that is available analytically.

**Device-side time.** A host float passed as a kernel argument gets frozen into the captured graph. Keep the
simulation time (or step counter) in a device array, advanced by a kernel at the end of each step, and have
all kinematic kernels read it.

**GPU data (structure of arrays).** Per-constraint arrays (kind, begin, duration, axis, pivot, speed/ω), plus
a node → constraint mapping. The old editor lets a node belong to several transforms (applied in sequence).
Either support composition in a fixed order, or require at most one active constraint per node at any time
and reject overlaps at load time. Ask the maintainer which.

**Per-step kernels**, in the time-integration setup:
1. Recompute `dmask` from the active windows (this replaces the countdown).
2. Set prescribed positions for active nodes, as both `x` and `xtilde`.
3. Rebuild `dbc`/`ndbc` on the device if any solver needs it, using a scan from `gpu/common/scan.py`.

At release, give the node its prescribed velocity so it doesn't jerk. Today, velocities are only
back-substituted for free nodes.

**Data flow.** Extend the editor's Dump (`tools/vbd/ui/simulation.py::_serialize_problem` →
`solvers/*.py::serialize_problem`) to write a `kinematics` group: constraint parameters plus node indices
in global (concatenated) numbering. The GPU loader reads it into a `KinematicConstraintSet`. Old Dumps that
only carry the millisecond countdown still load: convert each node's countdown into a `Fixed` constraint
with that duration.

**GUI.** At minimum, show and toggle each constraint and highlight its nodes (`panels/kinematics.py`).
Editing constraints inside the GPU UI is optional; ask the maintainer.

## Validation

> Superseded (2026-10-20): done, see Progress; current checks are in Handoff §6.

- Unit tests: each kind's kernel against a NumPy closed form; window boundaries; the release velocity.
- Graph-capture replay over many steps: positions follow the closed form (this catches frozen-time bugs).
- Parity: a scene with rotating and translating constraints, run with the CPU `tools/vbd/simulate.py` and
  with the GPU `simulate.py`, comparing constrained-node trajectories exactly and free-node behavior
  qualitatively. Expect the CPU's incremental application to differ from the closed form at rounding level.
- `MediumLongBendingBeam` (fixed BCs) behaves as before.

## Files

> Superseded (2026-10-20): see Handoff §1.

Own: new `python/tools/sim/gpu/kinematics/`, `panels/kinematics.py`, the Dirichlet parts of
`gpu/elasticity/fem.py`, `python/tools/vbd/` (the closed-form fix in `ui/utils/transform_library.py`, the
application sites in `ui/simulation.py` and `simulate.py`, and the Dump writer). Append-only: `state.py`,
`simulate.py` in `python/tools/sim`.

Note on CPU parity: closed-form evaluation needs each constrained node's anchor position at the start of its
window. The CPU code currently transforms the *current* positions every step, so it must record anchors when
a window opens. Check that this doesn't change the editor's interactive behavior (scrubbing the trajectory,
reset), and validate the old and new CPU paths against each other before relying on them for GPU parity.

## Open questions for the maintainer

1. ~~Are the names `KinematicConstraint` and `KinematicConstraintSet` acceptable, or do you prefer another (e.g. prescribed motion, scripted BCs)?~~
   **Answered (2026-10-17):** yes. Kinds: `Fixed`, `Translate`, `Rotate` (axis + pivot; covers G_ROTATE and
   L_ROTATE), `RadialTranslate`. The CPU editor keeps `TransformLibrary` and its names.
2. ~~Compose overlapping constraints on the same node, or forbid overlaps?~~ **Answered (2026-10-17):** compose,
   in library order, from a per-node anchor, each constraint with its own clamped active time
   (τ_k = clamp(t, b_k, e_k) − clamp(t_a, b_k, e_k)). Existing scenes rely on motion + Fixed overlaps.
3. ~~Should the GPU GUI edit constraints, or only display and toggle the ones from the Dump?~~
   **Answered (2026-10-17):** display + toggle. `panels/kinematics.py` lists each constraint (kind, window,
   parameters), highlights its nodes, enables/disables it (recapture on toggle). Authoring stays in the CPU editor.
4. ~~Keep reading the millisecond-countdown format from existing Dumps, or regenerate them?~~
   **Answered (2026-10-17):** drop the countdown hack entirely, with no countdown reader. Fully support what the
   CPU simulator supports, in a GPU-friendly way. (Follow-up on Dumps without a kinematics group: see 10.)
5. ~~At release, should a node keep its prescribed velocity or start at rest?~~ **Answered (2026-10-17):**
   prescribed velocity, in both simulators. Constrained nodes store the analytic prescribed velocity every
   step (the CPU sets `fem.v` after `constrain()`, which zeroes it). In "move, then hold" scenes this is zero
   at release anyway; "stop, then release" is authored with a trailing `Fixed` hold.
6. ~~L_TRANSLATE semantics.~~ **Answered (2026-10-17):** the maintainer let Claude decide the intent.
   Finding: the CPU code divides by the Frobenius norm of the whole `3 x n` group (`np.linalg.norm` without
   `axis`), so it is a uniform xy-scaling about `origin`, not a per-node translation. Decision: L_TRANSLATE /
   `RadialTranslate` moves each node away from the local `origin` in the xy-plane at `speed` units/s along its
   own direction (nodes on the axis don't move), in both simulators. Closed form: x + speed·τ·r̂(x_a).
7. ~~What to do with scenes tuned under the old L_TRANSLATE behavior?~~ **Answered (2026-10-17):** one-off
   migration script that rewrites `speed` to speed·mean|r_j|/ρ₀ (exact when the group's nodes share one radius,
   as on rims), backing up the originals. Affected: `cow_on_trampoline`, `octo-sling-1m-tilted`,
   `octopus-trampoline` (in `data/scenarios`, git-ignored).
8. ~~Window boundaries?~~ **Answered (2026-10-17):** half-open in both simulators. A node is constrained during
   step n when begin <= t_n < begin + duration, and the motion is clamped so it lasts exactly `duration`. The
   CPU's current closed interval gives one extra step of motion.
9. ~~Mid-simulation Dump?~~ **Answered (2026-10-17):** the Dump exports the current simulation time and the
   anchors of open constraints. The GPU clock starts at that time, so constraints resume exactly.
10. ~~Dumps without a kinematics group?~~ **Answered (2026-10-17):** a nonzero `dmask` means permanently fixed
    (C++'s own meaning, no countdown). `MediumLongBendingBeam` and `trampoline` behave as before, and the
    `dmask = 1` Dumps (`BeamFallingOnBeam`, `CubeFallingOnCube`) are fixed again, as their scenes intend. Re-dump
    the files that have scenes so they carry the new group.
11. ~~GPU `dbc`/`ndbc`?~~ **Answered (2026-10-17):** remove them from `FemElastoDynamicsData` (and from the
    unit-test assertion). No GPU kernel reads them; every solver uses `dmask`.

## Progress

(update as you go: decisions taken, commits, test results)

- 2026-10-17 · Investigation done; findings and closed forms at https://claude.ai/artifact/35eiBVHW5nDs1qxRH24tNG
  (the L_TRANSLATE section there describes the *old* behavior; the decision in question 6 replaces it). Key facts:
  - No GPU kernel reads `dbc`/`ndbc`; Newton and VBD use `dmask` only, so no device-side partition is needed.
  - C++ `Constrain()` zeroes Dirichlet velocities, and the CPU calls it every step, so both simulators
    currently release nodes at rest.
  - The Dump countdown ignores `begin` and takes the max `duration` per node.
  - Existing scenes overlap motion + `Fixed` on the same nodes ("move, then hold").
  - Implementation sketch: integer step counter in a device array (time in float64 in-kernel); per-node CSR
    list of constraint ids; per-node anchor position and anchor step, written on the first step of a
    constrained run; per-step kernel sets `dmask`, x, xtilde and v of constrained nodes. The Dump
    exports one constraint per transform, with node indices in global (concatenated) numbering.
- 2026-10-17 · All open questions answered. Waiting for Phase 0.
- 2026-10-17 · For task 3's static solves: `KinematicConstraintSet` gets a flag that stops the device step
  counter from advancing (time frozen), plus a way to set the counter. Agreed via the coordinator.
- 2026-10-18 · Implemented on `feature/gpu-kinematic-constraints`, rebased on dev `7c485e35` (task 2 merged).
  14 commits; my first-step fix was dropped because the coordinator fixed it on dev in `2b68afcb`:
  - `a4fc5d56` countdown removed (nonzero dmask = fixed) · `89511f67` GPU `dbc`/`ndbc` removed
  - `59696b7d` per-node L_TRANSLATE · `31ff9a75` GPU contact filters skip Dirichlet nodes (C++ parity)
  - `0d1c2c33` half-open windows · `60977742` CPU closed form with anchors · `3cd1294c` CPU prescribed velocity
  - `85b2bb88` L_TRANSLATE migration script · `123aa1ff` Dump writes the `kinematics` group (format v1, see
    `TransformLibrary.export_kinematic_constraints`)
  - `fd498944` `gpu/kinematics` (`KinematicConstraint`, `KinematicConstraintSet`, loader)
  - `bb3bb070` FEM evaluates it each step · `34a16c28` state/ui/simulate load and attach it
  - `28a65649` GUI panel · `441e97e6` docs
- 2026-10-18 · Validation:
  - Unit tests: `gpu.kinematics.constraints` (4: GPU vs CPU editor over 140 steps, max error 3e-7 in x
    and v; graph replay identical; release velocity and frozen time; mid-run anchors and toggling),
    `vbd.ui.utils.transform_library` (5), `gpu.elasticity.fem` (1). All pass.
  - Old vs new CPU path: the closed form equals the incremental code to 1.5e-14 (float64) for every kind
    and overlap pattern. With float32 positions, dev's incremental constraints drift 1.6e-5 after 140 steps
    (compressing beam); the closed form is exact.
  - Constrained-node trajectories: CPU and GPU are both exact against the closed form, with 0 error at every
    checked step of the compressing beam (3 s on the GPU). On octo-sling (GPU), held groups match to 3e-8
    until release, and the translated patch moves at the prescribed 1.000 units/s until it is let go.
  - Free nodes, qualitative: **the GPU VBD sags about 6× more than the CPU VBD on the same beam even with
    no motion** (held ends, mean z −0.54 vs −0.08 at 0.5 s), so this is a solver difference unrelated to
    kinematics. The CPU VBD also hits NaN at 1.43 s on the compressing beam: the 1e-6 differences from the
    closed form grow chaotically (dev's code survives to 1.6 s with its drifted constraints).
  - MediumLongBendingBeam (fixed BCs): bit-identical to dev until the first-step fix, which shifts every GPU
    run by one step. CubeFallingOnCube with VertexSdf: bit-identical to dev with the bottom cube held. OGC
    crashes once bodies touch, a pre-existing dev bug (coordination log).
  - Panel smoke-tested with Polyscope's mock backend; the interactive GUI is not tested.
- 2026-10-18 · Data (git-ignored, backups next to each file): migrated `cow_on_trampoline` (5 → 0.287, old
  per-node speeds 0.279–0.288), `octo-sling-1m-tilted` (1 → 0.036, old per-node speeds 0.025–0.045, i.e.
  inexact: its nodes don't share a radius) and `octopus-trampoline` (5 → 0.380). Added `kinematics` to
  the BeamFallingOnBeam, CubeFallingOnCube and trampoline Dumps. MediumLongBendingBeam has no scene, so it
  keeps the dmask fallback.
- (Superseded 2026-10-20 by Handoff §5.) Not done / known gaps: the GPU headless output keeps writing only the initial `dmask`; the offline CPU
  simulator's `-o` drive-letter bug is documented, not fixed; a mid-run `--start-from` resume on the CPU
  re-anchors at the resumed state (exact unless non-commuting motions overlap).

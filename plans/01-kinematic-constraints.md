# Task 1: Kinematic Dirichlet constraints on the GPU

Status: **not started** · Branch `feature/gpu-kinematic-constraints` · Worktree `C:\git\pbat-kinematics`
Read first: `MASTER.md` (rules, ownership), `python/tools/sim/CLAUDE.md`, `python/tools/vbd/CLAUDE.md`.

## Goal

Support every procedural Dirichlet constraint that the CPU editor supports, in the GPU simulator
(`ui.py` and `simulate.py`), with the motion evaluated on the GPU inside the captured CUDA graph (no
per-step CPU/GPU round trips). Replace the current "release after X ms" hack. Give the feature proper names
in the new simulator.

## What exists today

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

- Unit tests: each kind's kernel against a NumPy closed form; window boundaries; the release velocity.
- Graph-capture replay over many steps: positions follow the closed form (this catches frozen-time bugs).
- Parity: a scene with rotating and translating constraints, run with the CPU `tools/vbd/simulate.py` and
  with the GPU `simulate.py`, comparing constrained-node trajectories exactly and free-node behavior
  qualitatively. Expect the CPU's incremental application to differ from the closed form at rounding level.
- `MediumLongBendingBeam` (fixed BCs) behaves as before.

## Files

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
3. Should the GPU GUI edit constraints, or only display and toggle the ones from the Dump?
4. Keep reading the millisecond-countdown format from existing Dumps, or regenerate them?
5. At release, should a node keep its prescribed velocity or start at rest? *(discussing: maintainer leans
   toward zero velocity; Claude explained the trade-off, awaiting final answer)*
6. ~~L_TRANSLATE semantics.~~ **Answered (2026-10-17):** the maintainer let Claude decide the intent.
   Finding: the CPU code divides by the Frobenius norm of the whole `3 x n` group (`np.linalg.norm` without
   `axis`), so it is a uniform xy-scaling about `origin`, not a per-node translation. Decision: L_TRANSLATE /
   `RadialTranslate` moves each node away from the local `origin` in the xy-plane at `speed` units/s along its
   own direction (nodes on the axis don't move), in both simulators. Closed form: x + speed·τ·r̂(x_a).
7. ~~What to do with scenes tuned under the old L_TRANSLATE behavior?~~ **Answered (2026-10-17):** one-off
   migration script that rewrites `speed` to speed·mean|r_j|/ρ₀ (exact when the group's nodes share one radius,
   as on rims), backing up the originals. Affected: `cow_on_trampoline`, `octo-sling-1m-tilted`,
   `octopus-trampoline` (in `data/scenarios`, git-ignored).

## Progress

(update as you go: decisions taken, commits, test results)

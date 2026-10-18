---
name: sim-kinematics
description: Author and run kinematic constraints (procedurally animated Dirichlet boundary conditions — fixed, translating, rotating, radially translating node groups over time windows) with the CPU editor's transform library and the GPU/CPU simulators, including scenes with static mesh and plane colliders. Use when asked to pin, move, twist, stretch or release parts of a body, to author such a scene, or to debug constrained-node behaviour.
---

# Kinematic constraints

Constraints are procedurally animated Dirichlet nodes. They are authored in the CPU editor as the **transform library** and evaluated with the same semantics on both sides:
- GPU: `python/tools/sim/gpu/kinematics/constraints.py`.
- CPU: `python/tools/vbd/ui/utils/transform_library.py`.

The exact semantics are in those module docstrings and in `python/tools/sim/CLAUDE.md` ("Kinematic constraints"). For general run options, see the `sim-run` skill.

Run from the repo root with `/c/git/python-3.12/Scripts/python.exe` (Bash). `data/` exists only in the main checkout, so use absolute paths from worktrees.

## Commands

```bash
# GPU from a scene file (no Dump needed): transforms become constraints, colliders become contact geometry
python -m python.tools.sim.simulate --scene data/scenarios/prisms/two_het_cubes/compressing_beam_two_het-cubes.h5 \
    -o out.h5 --solver VBD --cd VertexSdf --duration 3

# GPU from an editor Dump (its kinematics group is read automatically)
python -m python.tools.sim.simulate --fem-elasto-dynamics dump.h5:fem -o out.h5 --duration 2   # e.g. a Dump taken mid-run to debug a crash

# GUI: the "Kinematic constraints" panel lists constraints, marks active ones, toggles them (no recapture), shows/highlights nodes
python -m python.tools.sim.ui --scene scene.h5

# CPU reference run (CPU parameter format: root attrs dt, bdf_scheme, fem_dynamics_init_strategy + Solver/VBD and Contact groups)
python -m python.tools.vbd.simulate --scene scene.h5 --simulation-params cpu_params.h5 -s vbd -d 3 -o out.h5:sim

# Tests
python -m unittest python.tools.sim.gpu.kinematics.constraints python.tools.vbd.ui.utils.transform_library python.tools.vbd.ui.plane_collider

# Migrate scenes authored with the old group-normalized L_TRANSLATE (before 2026-10-18)
python -m python.tools.vbd.migrate_l_translate scene.h5 [--dry-run]
```

## Authoring in the CPU editor

Open it with `python -m python.tools.vbd.editor`. Claude can't drive the editor, so give the user these steps:

1. **Scene > Transforms tab:** pick a type (FIXED, TRANSLATE, G_ROTATE, L_ROTATE, L_TRANSLATE), press **Add**, then set the name, `begin`, `duration` and parameters. Note the transform's id; the tree label reads "id - TYPE".
2. **Selection tab:** set Property to "Dirichlet Group", pick a Box or Cylinder selector, press **Add**, set the value to the transform id, and apply. A value of 0 removes nodes from their group.
3. **Move, then hold:** use a motion transform plus an overlapping FIXED transform with a longer window, on the same nodes.
4. **Plane colliders:** **Objects tab > Add Plane Collider**, then set `point`, `normal` (pointing to the free side) and `tangent` in the Plane Colliders tree, and choose either Infinite (the default, 20 scene diameters) or half extents. They're saved with the session in a `plane_colliders` group. Static mesh colliders are also supported.
5. **Saving:** **Session > Save session** gives a scene file that runs directly with `--scene`. Alternatively, simulate and press **Dump**, which writes a `kinematics` group with `t0` and the open anchors, so a Dump taken mid-simulation resumes exactly.

## Semantics

- **Windows** are half-open, `[begin, begin + duration)`, rounded to the step grid with a tolerance of 1e-3 step.
- **Closed form:** positions come from per-node anchors (each node's position when its constrained run starts). Overlapping transforms compose in library order, so nothing drifts.
- **Release:** constrained nodes get the analytic prescribed velocity, so a node released while moving keeps moving.
- **Kinds:**

  | Editor type | Constraint | Meaning |
  |---|---|---|
  | FIXED | Fixed | Hold in place |
  | TRANSLATE | Translate | Constant velocity |
  | G_ROTATE / L_ROTATE | Rotate | Angular velocity in rad/s, about the world origin (G) or the transform's `origin` (L). The editor resets an L origin to the nodes' centroid whenever it redraws the transform |
  | L_TRANSLATE | RadialTranslate | Each node moves radially in xy at `speed` |

- **Colliders:**
  - Static colliders become pinned, element-less FEM nodes held by a permanent "Static colliders" constraint (`python/tools/sim/colliders.py`).
  - Planes are detected analytically on the GPU and tessellated on the CPU.
  - Scenes without colliders are bit-identical to runs from before colliders existed.

## Pitfalls

- **dmask:** the constraint set owns `dmask` and rewrites it every step. Never edit the FEM's `dmask`; edit or toggle constraints instead. The output's `FemElastoDynamics/dmask` holds only the initial set.
- **Recapture:** `advance_time = False` (frozen time, used by static solves) needs a graph recapture. `set_enabled` and setting the step don't.
- **Reset:** `reset()` builds a new constraint set. Time restarts at `t0`, windows are re-gridded with the new `dt`, and positions restart at X with the loaded `v0`.
- **Old Dumps:** a Dump without a `kinematics` group treats nonzero `dmask` as fixed forever.
- **L_ROTATE origin:** the editor resets an L_ROTATE origin to its nodes' centroid whenever it redraws the transform.
- **Old L_TRANSLATE scenes:** L_TRANSLATE used to be group-normalized, so migrate scenes authored before 2026-10-18 with `migrate_l_translate`.
- **Finite planes** are plates of `thickness` (default 10% of the smaller half-extent, editable in the editor). A body can pass under or beside them, but impacts faster than the thickness per step can tunnel through. Infinite planes catch any depth.
- **Colliders need `--scene`:** plane and static mesh colliders only work from scene files, since Dumps don't carry them.
- **Hard impacts** currently need an adaptive penalty (`--set Contact.ContactDynamics.penalty_adaptivity=1 --set Contact.ContactDynamics.gamman=1`). The adaptive modes still diverge with VBD-family solvers on some contact scenes until task 3's fix lands, so check the result.
- **Comparing contact scenes:** they aren't bit-reproducible run to run, so compare constrained-node trajectories or contact sets.

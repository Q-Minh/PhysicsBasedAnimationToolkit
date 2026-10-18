"""
Kinematic constraints: prescribed (procedural) motion of node groups over time windows.

A `KinematicConstraint` prescribes the motion of a group of nodes during the time window
[begin, begin + duration). A `KinematicConstraintSet` holds all of them on the device and, at the start
of every time step, evaluates which nodes are constrained (writing the FEM's `dmask`) and their
prescribed positions and velocities at the end of the step. The evaluation is in closed form and reads
the simulation time from a device array, so it is drift-free and runs inside the captured CUDA graph.

This is the GPU counterpart of the CPU editor's `TransformLibrary.evaluate`
(`python/tools/vbd/ui/utils/transform_library.py`). Semantics, shared by both:

- Step n goes from t_n = n h to t_{n+1}. Constraint k is active during step n when
  n_begin_k <= n < n_end_k, where the window boundaries are rounded to the step grid
  (`window_steps`). A node is constrained while at least one enabled constraint affecting it is active.
- When a node starts a constrained run at step n_a, its position x_a at t_{n_a} becomes its anchor.
  Its prescribed position at t_{n+1} is the composition, in constraint order,

      x = T_m(tau_m) o ... o T_1(tau_1) (x_a),   tau_k = clamp(t_{n+1}, b_k, e_k) - clamp(t_{n_a}, b_k, e_k),

  with [b_k, e_k) = [n_begin_k h, n_end_k h). A motion that ends while another constraint still holds
  the node leaves it where the motion stopped.
- The prescribed velocity is the time derivative of that composition (left derivative at the end of
  a window), so a node released while moving keeps moving smoothly.

Kinds (T(tau) applied to y):
- `Fixed`: y.
- `Translate`: y + rate tau direction.
- `Rotate`: p + R(direction, rate tau)(y - p), with pivot p = point and rate in rad/s.
- `RadialTranslate`: y + rate tau rhat(y), with rhat the unit xy-component of y - point (the node stays
  put if it lies on the axis through point).

Where constraints come from (all through `state.load_problem`):
- an editor Dump's `kinematics` group (`read_kinematic_constraints`, written by
  `TransformLibrary.export_kinematic_constraints`), with the Dump time t0 and the anchors of open runs;
- a Dump without that group: its nonzero `dmask` nodes, fixed forever (`fixed_constraints_from_dmask`);
- a CPU scene file (`state.load_scene`): its transform library, plus a "Static colliders" constraint that
  pins collider nodes (see `colliders.py`).
Editor kinds map as FIXED -> Fixed, TRANSLATE -> Translate, G_ROTATE/L_ROTATE -> Rotate (pivot 0 or
origin, rate in rad/s) and L_TRANSLATE -> RadialTranslate.

Usage (what `state.SimulationState` and `FemElastoDynamics` do):

    kcs = KinematicConstraintSet(constraints, n_nodes, h=fem.bdf.h, t0=t0, anchors=anchors)
    fem.set_kinematic_constraints(kcs)  # setup_time_integration_optimization() calls kcs.update(...),
                                        # fem.step() calls kcs.advance()

Pitfalls:
- The set owns `dmask`: `update` rewrites it for every node each step, so edit constraints (or
  `set_enabled`) instead of writing `dmask`.
- Time is the device step counter, so a new set restarts at t0. `SimulationState.reset()` builds a new set
  (e.g. after changing dt), and the windows are converted to steps with that set's h.
- `advance_time` is read on the host when the graph is recorded: recapture after changing it.
  `set_enabled` and `step` writes act on device data and need no recapture.
- Windows are rounded to the step grid with a 1e-3-step tolerance, the same as the CPU editor, so
  float32-stored durations land on the grid.
"""

import dataclasses
import enum
import typing
import h5py
import numpy as np
import warp as wp


class KinematicConstraintKind(enum.Enum):
    """Kind of prescribed motion. Values are those of the Dump's `kinematics/kind` dataset."""

    Fixed = 0
    Translate = 1
    Rotate = 2
    RadialTranslate = 3


KIND_FIXED = wp.constant(KinematicConstraintKind.Fixed.value)
KIND_TRANSLATE = wp.constant(KinematicConstraintKind.Translate.value)
KIND_ROTATE = wp.constant(KinematicConstraintKind.Rotate.value)
KIND_RADIAL_TRANSLATE = wp.constant(KinematicConstraintKind.RadialTranslate.value)

_MAX_STEP = np.iinfo(np.int32).max
_WINDOW_TOLERANCE = 1e-3  # in steps


@dataclasses.dataclass
class KinematicConstraint:
    """Prescribed motion of a group of nodes during the time window [begin, begin + duration).

    `direction` is used by Translate (motion direction) and Rotate (axis), `point` by Rotate (pivot) and
    RadialTranslate (origin), and `rate` by every kind but Fixed. `enabled` is toggled through
    `KinematicConstraintSet.set_enabled`.
    """

    name: str
    kind: KinematicConstraintKind
    begin: float  # seconds
    duration: float  # seconds, may be inf
    nodes: np.ndarray  # (n,) global node indices
    direction: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3)
    )  # translation direction or rotation axis (unit length)
    point: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3)
    )  # rotation pivot or radial origin
    rate: float = 0.0  # speed (units/s), or angular velocity (rad/s) for Rotate
    enabled: bool = True

    def window_steps(self, h: float) -> tuple[int, int]:
        """Half-open range [n_begin, n_end) of active time steps for step size h.

        Boundaries are rounded to the step grid with a tolerance of 1e-3 steps, like the CPU editor's
        `PrimitiveTransform.window_steps`, so that float32-stored windows land on the grid.
        """
        n_begin = int(np.ceil(self.begin / h - _WINDOW_TOLERANCE))
        end = self.begin + self.duration
        n_end = (
            int(np.ceil(end / h - _WINDOW_TOLERANCE))
            if np.isfinite(end) and end / h < _MAX_STEP
            else _MAX_STEP
        )
        return max(n_begin, -_MAX_STEP), n_end


class KinematicAnchors(typing.NamedTuple):
    """Anchors of the nodes that are in a constrained run at the initial time."""

    nodes: np.ndarray  # (A,) global node indices
    positions: np.ndarray  # (A, 3) anchor positions
    times: np.ndarray  # (A,) times at which the runs started, in seconds


def read_kinematic_constraints(
    grp: h5py.Group,
) -> tuple[list[KinematicConstraint], float, KinematicAnchors]:
    """Read the `kinematics` group of an editor Dump.

    Returns:
        tuple[list[KinematicConstraint], float, KinematicAnchors]: The constraints in order, the
        simulation time t0 of the Dump in seconds, and the anchors of the runs open at t0.
    """
    version = int(grp.attrs["version"])
    if version != 1:
        raise ValueError(f"Unsupported kinematics group version {version}")
    offsets = grp["node_offsets"][()]
    nodes = grp["nodes"][()]
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in grp["name"][()]]
    constraints = [
        KinematicConstraint(
            name=names[k],
            kind=KinematicConstraintKind(int(grp["kind"][k])),
            begin=float(grp["begin"][k]),
            duration=float(grp["duration"][k]),
            nodes=nodes[offsets[k] : offsets[k + 1]].astype(np.int64),
            direction=np.asarray(grp["direction"][k], dtype=float),
            point=np.asarray(grp["point"][k], dtype=float),
            rate=float(grp["rate"][k]),
        )
        for k in range(offsets.shape[0] - 1)
    ]
    anchors = KinematicAnchors(
        grp["anchor_nodes"][()].astype(np.int64),
        grp["anchor_positions"][()].reshape(-1, 3),
        grp["anchor_times"][()],
    )
    return constraints, float(grp.attrs["t0"]), anchors


def fixed_constraints_from_dmask(dmask: np.ndarray) -> list[KinematicConstraint]:
    """Constraints of a Dump without a `kinematics` group: nonzero dmask means permanently fixed."""
    nodes = np.flatnonzero(np.asarray(dmask).ravel())
    if nodes.size == 0:
        return []
    return [
        KinematicConstraint(
            name="Dirichlet",
            kind=KinematicConstraintKind.Fixed,
            begin=0.0,
            duration=np.inf,
            nodes=nodes,
        )
    ]


@wp.struct
class KinematicConstraintSetData:
    """Device data of a KinematicConstraintSet, passed to its kernels."""

    # --- Per constraint (K,) ---
    kind: wp.array[wp.int32]
    enabled: wp.array[wp.int32]
    n_begin: wp.array[wp.int32]  # first active step
    n_end: wp.array[wp.int32]  # one past the last active step
    t_begin: wp.array[wp.float64]  # n_begin * h
    t_end: wp.array[wp.float64]  # n_end * h, or inf
    direction: wp.array[wp.vec3f]
    point: wp.array[wp.vec3f]
    rate: wp.array[wp.float32]
    # --- Node -> constraints, CSR over all N nodes, in constraint order ---
    node_ptr: wp.array[wp.int32]  # (N+1,)
    node_constraints: wp.array[wp.int32]  # (nnz,)
    # --- Per node (N,) ---
    anchor_x: wp.array[wp.vec3f]  # anchor positions
    anchor_step: wp.array[wp.int32]  # step at which the node's constrained run started, -1 if free
    # --- Time ---
    step: wp.array[wp.int32]  # (1,) index n of the current time step
    h: wp.float64  # time step size


@wp.func
def _rotate(axis: wp.vec3f, c: wp.float32, s: wp.float32, z: wp.vec3f) -> wp.vec3f:
    """Rodrigues' rotation of z about the unit axis, with c = cos(theta), s = sin(theta)."""
    return z * c + wp.cross(axis, z) * s + axis * (wp.dot(axis, z) * (wp.float32(1) - c))


@wp.kernel
def _update_kernel(
    ks: KinematicConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    xt: wp.array[wp.vec3f],
    x: wp.array[wp.vec3f],
    v: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
):
    i = wp.tid()
    n = ks.step[0]
    kbeg = ks.node_ptr[i]
    kend = ks.node_ptr[i + 1]
    active = int(0)
    for j in range(kbeg, kend):
        k = ks.node_constraints[j]
        if ks.enabled[k] != 0 and ks.n_begin[k] <= n and n < ks.n_end[k]:
            active = 1
    if active == 0:
        dmask[i] = 0
        ks.anchor_step[i] = -1
        return
    if ks.anchor_step[i] < 0:
        ks.anchor_step[i] = n
        ks.anchor_x[i] = xt[i]
    ta = wp.float64(ks.anchor_step[i]) * ks.h
    t1 = wp.float64(n + 1) * ks.h
    y = ks.anchor_x[i]
    dy = wp.vec3f(wp.float32(0))
    for j in range(kbeg, kend):
        k = ks.node_constraints[j]
        if ks.enabled[k] == 0:
            continue
        tb = ks.t_begin[k]
        te = ks.t_end[k]
        tau64 = wp.clamp(t1, tb, te) - wp.clamp(ta, tb, te)
        tau = wp.float32(tau64)
        dtau = wp.float32(0)
        if tb < t1 and t1 <= te:
            dtau = wp.float32(1)
        rate = ks.rate[k]
        kind = ks.kind[k]
        if kind == KIND_TRANSLATE:
            d = ks.direction[k]
            y = y + d * (rate * tau)
            dy = dy + d * (rate * dtau)
        elif kind == KIND_ROTATE:
            a = ks.direction[k]
            p = ks.point[k]
            theta = wp.float64(rate) * tau64
            c = wp.float32(wp.cos(theta))
            s = wp.float32(wp.sin(theta))
            ry = _rotate(a, c, s, y - p)
            y = p + ry
            dy = _rotate(a, c, s, dy) + wp.cross(a, ry) * (rate * dtau)
        elif kind == KIND_RADIAL_TRANSLATE:
            r = y - ks.point[k]
            r = wp.vec3f(r[0], r[1], wp.float32(0))
            rho = wp.length(r)
            if rho > wp.float32(0):
                rhat = r / rho
                pdy = wp.vec3f(dy[0], dy[1], wp.float32(0))
                drhat = (pdy - rhat * wp.dot(rhat, pdy)) / rho
                y = y + rhat * (rate * tau)
                dy = dy + drhat * (rate * tau) + rhat * (rate * dtau)
    dmask[i] = 1
    x[i] = y
    v[i] = dy


@wp.kernel
def _advance_kernel(step: wp.array[wp.int32]):
    step[0] = step[0] + 1


class KinematicConstraintSet:
    """All kinematic constraints of a simulation, evaluated on the device every time step.

    Usage, once per time step and before the solve (`FemElastoDynamics` does this):
        kcs.update(xt, x, v, dmask)  # dmask, x and v of constrained nodes at the end of the step
        ... solve, BDF step ...
        kcs.advance()  # next time step

    `update` and `advance` read and write device arrays only, so both can be captured in a CUDA graph.
    """

    advance_time: bool
    """Whether `advance` moves to the next time step. Set it to False to evaluate the constraints at
    one fixed time (e.g. for static solves). The flag is read on the host, so a captured graph must be
    recaptured after changing it."""

    def __init__(
        self,
        constraints: list[KinematicConstraint],
        n_nodes: int,
        h: float,
        t0: float = 0.0,
        anchors: KinematicAnchors | None = None,
    ):
        """
        Args:
            constraints (list[KinematicConstraint]): Constraints, in composition order. The list is
                kept, and `set_enabled` updates its entries.
            n_nodes (int): Number of FEM nodes.
            h (float): Time step size.
            t0 (float): Initial simulation time, rounded to the step grid.
            anchors (KinematicAnchors | None): Anchors of the runs open at t0.
        """
        self.constraints = constraints
        self.advance_time = True
        self._h = h
        K = len(constraints)
        windows = np.array([c.window_steps(h) for c in constraints], dtype=np.int64).reshape(
            K, 2
        )
        t_end = windows[:, 1] * h
        t_end[windows[:, 1] == _MAX_STEP] = np.inf
        d = KinematicConstraintSetData()
        d.kind = wp.array([c.kind.value for c in constraints], dtype=wp.int32)
        d.enabled = wp.array([int(c.enabled) for c in constraints], dtype=wp.int32)
        d.n_begin = wp.array(windows[:, 0], dtype=wp.int32)
        d.n_end = wp.array(windows[:, 1], dtype=wp.int32)
        d.t_begin = wp.array(windows[:, 0] * h, dtype=wp.float64)
        d.t_end = wp.array(t_end, dtype=wp.float64)
        d.direction = wp.array(
            np.array([c.direction for c in constraints]).reshape(K, 3), dtype=wp.vec3f
        )
        d.point = wp.array(
            np.array([c.point for c in constraints]).reshape(K, 3), dtype=wp.vec3f
        )
        d.rate = wp.array([c.rate for c in constraints], dtype=wp.float32)
        # Node -> constraints CSR, constraints in order within each node
        knodes = [np.asarray(c.nodes, dtype=np.int64) for c in constraints]
        nodes = np.concatenate(knodes) if K > 0 else np.zeros(0, dtype=np.int64)
        kids = (
            np.concatenate(
                [np.full(kn.size, k, dtype=np.int64) for k, kn in enumerate(knodes)]
            )
            if K > 0
            else np.zeros(0, dtype=np.int64)
        )
        order = np.lexsort((kids, nodes))
        counts = np.bincount(nodes, minlength=n_nodes)
        d.node_ptr = wp.array(
            np.concatenate(([0], np.cumsum(counts))), dtype=wp.int32
        )
        d.node_constraints = wp.array(kids[order], dtype=wp.int32)
        # Anchors and time
        n0 = int(round(t0 / h))
        anchor_x = np.zeros((n_nodes, 3), dtype=np.float32)
        anchor_step = np.full(n_nodes, -1, dtype=np.int32)
        if anchors is not None and anchors.nodes.size > 0:
            anchor_x[anchors.nodes] = anchors.positions
            anchor_step[anchors.nodes] = np.round(anchors.times / h).astype(np.int32)
        d.anchor_x = wp.array(anchor_x, dtype=wp.vec3f)
        d.anchor_step = wp.array(anchor_step, dtype=wp.int32)
        d.step = wp.array([n0], dtype=wp.int32)
        d.h = wp.float64(h)
        self._data = d

    @property
    def data(self) -> KinematicConstraintSetData:  # pyright: ignore[reportGeneralTypeIssues]
        """Device data for kernels."""
        return self._data

    @property
    def h(self) -> float:
        """Time step size the windows were converted with."""
        return self._h

    @property
    def step(self) -> int:
        """Index of the current time step (copies from the device)."""
        return int(self._data.step.numpy()[0])

    @step.setter
    def step(self, n: int):
        """Jump to time step n (device write, no recapture needed). Open anchors are kept."""
        self._data.step.fill_(n)

    def set_enabled(self, k: int, enabled: bool):
        """Enable or disable constraint k. Takes effect at the next `update`, without recapture."""
        self.constraints[k].enabled = enabled
        self._data.enabled[k : k + 1].fill_(int(enabled))

    def update(
        self,
        xt: wp.array[wp.vec3f],
        x: wp.array[wp.vec3f],
        v: wp.array[wp.vec3f],
        dmask: wp.array[wp.int32],
    ):
        """Evaluate the constraints for the current time step.

        Args:
            xt (wp.array[wp.vec3f]): (N,) positions at the start of the step (anchors are taken from it).
            x (wp.array[wp.vec3f]): (N,) positions, prescribed positions at the end of the step are
                written to constrained nodes.
            v (wp.array[wp.vec3f]): (N,) velocities, prescribed velocities are written to
                constrained nodes.
            dmask (wp.array[wp.int32]): (N,) Dirichlet mask, written for all nodes.
        """
        wp.launch(
            _update_kernel,
            dim=x.shape[0],
            inputs=[self._data, xt, x, v, dmask],
        )

    def advance(self):
        """Move to the next time step, unless `advance_time` is False."""
        if self.advance_time:
            wp.launch(_advance_kernel, dim=1, inputs=[self._data.step])


import unittest


class TestKinematicConstraintSet(unittest.TestCase):
    h = 0.01

    def _scene(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3)).astype(np.float32)
        a = np.arange(0, 20)
        b = np.arange(10, 30)
        constraints = [
            KinematicConstraint(
                "radial",
                KinematicConstraintKind.RadialTranslate,
                0.0,
                1.0,
                a,
                point=np.array([0.1, -0.2, 0.0]),
                rate=0.5,
            ),
            KinematicConstraint(
                "rotate",
                KinematicConstraintKind.Rotate,
                0.2,
                0.6,
                b,
                direction=np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
                point=np.array([0.5, 0.0, 0.0]),
                rate=np.deg2rad(-60.0),
            ),
            KinematicConstraint(
                "translate",
                KinematicConstraintKind.Translate,
                0.5,
                0.5,
                b,
                direction=np.array([0.0, 0.0, 1.0]),
                rate=0.7,
            ),
            KinematicConstraint(
                "hold", KinematicConstraintKind.Fixed, 0.0, 1.2000000476837158, a
            ),
        ]
        return X, constraints

    def _cpu_reference(self, X, constraints, n_steps):
        """Trajectory of the CPU editor's TransformLibrary for the same constraints."""
        from ....vbd.ui.utils import transform_library as tl

        lib = tl.TransformLibrary()
        lib.on_mesh_added("m")
        for c in constraints:
            if c.kind == KinematicConstraintKind.Fixed:
                tr = tl.FixedTransform(c.name, c.begin, c.duration)
            elif c.kind == KinematicConstraintKind.Translate:
                tr = tl.TranslateTransform(c.name, c.begin, c.duration, c.direction, c.rate)
            elif c.kind == KinematicConstraintKind.Rotate:
                tr = tl.LocalRotateTransform(
                    c.name, c.begin, c.duration, c.direction, c.point, np.rad2deg(c.rate)
                )
            else:
                tr = tl.LocalTranslateTransform(c.name, c.begin, c.duration, c.point, c.rate)
            lib.add_transform(tr)
            tr.on_dirichlet_nodes_added("m", c.nodes)
        V = X.T.astype(float)
        xs, vs, masks = [], [], []
        for t in range(n_steps):
            V, dV, active = lib.evaluate("m", V, t, self.h)
            xs.append(V.T.copy())
            vs.append(dV.T.copy())
            masks.append(active.copy())
        return xs, vs, masks

    def _run(self, kcs, X, n_steps, capture: bool):
        xt = wp.array(X, dtype=wp.vec3f)
        x = wp.array(X, dtype=wp.vec3f)
        v = wp.zeros_like(x)
        dmask = wp.zeros(X.shape[0], dtype=wp.int32)

        def step():
            kcs.update(xt, x, v, dmask)
            wp.copy(xt, x)  # free nodes don't move, the end-of-step state becomes the next start
            kcs.advance()

        graph = None
        if capture:
            with wp.ScopedCapture() as capture_ctx:
                step()
            graph = capture_ctx.graph
        xs, vs, masks = [], [], []
        for _ in range(n_steps):
            if graph is not None:
                wp.capture_launch(graph)
            else:
                step()
            xs.append(x.numpy().copy())
            vs.append(v.numpy().copy())
            masks.append(dmask.numpy() != 0)
        return xs, vs, masks

    def test_matches_cpu_editor(self):
        X, constraints = self._scene()
        n_steps = 140
        xr, vr, mr = self._cpu_reference(X, constraints, n_steps)
        kcs = KinematicConstraintSet(constraints, X.shape[0], self.h)
        xs, vs, ms = self._run(kcs, X, n_steps, capture=False)
        for t in range(n_steps):
            np.testing.assert_array_equal(ms[t], mr[t], err_msg=f"step {t}")
            m = mr[t]
            np.testing.assert_allclose(xs[t][m], xr[t][m], atol=2e-5, err_msg=f"step {t}")
            np.testing.assert_allclose(vs[t][m], vr[t][m], atol=2e-5, err_msg=f"step {t}")

    def test_graph_replay_advances_time(self):
        X, constraints = self._scene()
        n_steps = 140
        eager = self._run(
            KinematicConstraintSet(constraints, X.shape[0], self.h), X, n_steps, False
        )
        replay = self._run(
            KinematicConstraintSet(constraints, X.shape[0], self.h), X, n_steps, True
        )
        for t in range(n_steps):
            np.testing.assert_array_equal(eager[0][t], replay[0][t])
            np.testing.assert_array_equal(eager[2][t], replay[2][t])

    def test_release_velocity_and_freeze(self):
        X = np.zeros((1, 3), dtype=np.float32)
        c = KinematicConstraint(
            "t",
            KinematicConstraintKind.Translate,
            0.0,
            0.1,
            np.array([0]),
            direction=np.array([1.0, 0.0, 0.0]),
            rate=2.0,
        )
        kcs = KinematicConstraintSet([c], 1, self.h)
        xs, vs, ms = self._run(kcs, X, 11, capture=False)
        # The last constrained step ends exactly at the window end, with the motion's velocity
        self.assertTrue(ms[9][0])
        self.assertFalse(ms[10][0])
        np.testing.assert_allclose(xs[9][0], [0.2, 0, 0], atol=1e-6)
        np.testing.assert_allclose(vs[9][0], [2.0, 0, 0], atol=1e-6)
        # Frozen time evaluates the same step again
        kcs = KinematicConstraintSet([c], 1, self.h, t0=0.05)
        kcs.advance_time = False
        xs, _, _ = self._run(kcs, X, 3, capture=False)
        self.assertEqual(kcs.step, 5)

    def test_mid_run_anchors_and_toggle(self):
        X, constraints = self._scene()
        xr, _, mr = self._cpu_reference(X, constraints, 80)
        # Resume at t0 = 0.4 s from the reference state, with the anchors of the open runs. Free
        # nodes don't move here, so every anchor is a rest position. Runs start when the first
        # constraint on the node starts: at 0 for the radial group, at 0.2 s for the rest.
        n0 = 40
        x0 = xr[n0 - 1].astype(np.float32)
        open_nodes = np.flatnonzero(mr[n0 - 1])
        times = np.where(np.isin(open_nodes, constraints[0].nodes), 0.0, 0.2)
        anchors = KinematicAnchors(open_nodes, X[open_nodes], times)
        kcs = KinematicConstraintSet(
            constraints, X.shape[0], self.h, t0=n0 * self.h, anchors=anchors
        )
        xs, _, ms = self._run(kcs, x0, 80 - n0, capture=False)
        for t in range(n0, 80):
            m = mr[t]
            np.testing.assert_array_equal(ms[t - n0], m)
            np.testing.assert_allclose(xs[t - n0][m], xr[t][m], atol=2e-5)
        # Disabling every constraint frees all nodes at the next update
        for k in range(len(constraints)):
            kcs.set_enabled(k, False)
        _, _, ms = self._run(kcs, xs[-1], 1, capture=False)
        self.assertFalse(np.any(ms[0]))


if __name__ == "__main__":
    wp.init()
    unittest.main()

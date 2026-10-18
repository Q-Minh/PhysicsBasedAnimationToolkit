import warp as wp
import cupy as cp
import numpy as np
from .. import types
from ..integration.bdf import Bdf
from ..kinematics.constraints import KinematicConstraintSet
from ..common import profiling
from pbatoolkit import pbat

# --- Strategy constants ---
STRATEGY_POSITION = wp.constant(
    int(pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position.value)
)
STRATEGY_FREE_TRAJECTORY = wp.constant(
    int(pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.FreeTrajectory.value)
)
STRATEGY_TRAJECTORY_WITH_EXTERNAL_LOAD = wp.constant(
    int(
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.TrajectoryWithExternalLoad.value
    )
)

_STRATEGY_MAP = {
    pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position: STRATEGY_POSITION,
    pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.FreeTrajectory: STRATEGY_FREE_TRAJECTORY,
    pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.TrajectoryWithExternalLoad: STRATEGY_TRAJECTORY_WITH_EXTERNAL_LOAD,
}


@wp.func
def is_dirichlet_node(dmask: wp.array[wp.int32], node: int) -> bool:
    return dmask[node] != 0


@wp.func
def is_free_node(dmask: wp.array[wp.int32], node: int) -> bool:
    return dmask[node] == 0


@wp.kernel
def _setup_time_integration_optimization_kernel(
    x: wp.array[wp.vec3f],
    xtilde: wp.array[wp.vec3f],
    fext: wp.array[wp.vec3f],
    m: wp.array[wp.float32],
    dmask: wp.array[wp.int32],
    xtilde_bdf: wp.array[wp.vec3f],
    vtilde_bdf: wp.array[wp.vec3f],
    bt: float,
    bt2: float,
    strategy: int,
):
    i = wp.tid()
    aexti = fext[i] / m[i]
    # xtilde = -(xtilde_bdf + bt * vtilde_bdf) + bt^2 * aext
    xti = -(xtilde_bdf[i] + bt * vtilde_bdf[i]) + bt2 * aexti
    if is_dirichlet_node(dmask, i):  # pyright: ignore[reportArgumentType]
        # Dirichlet nodes: xtilde = their position at the end of the step, i.e. the prescribed
        # position (kinematic constraints) or the current BDF position (fixed nodes)
        xti = x[i]

    xtilde[i] = xti  # pyright: ignore[reportIndexIssue]
    # Initialize x based on strategy (only for free nodes)
    if is_free_node(dmask, i):  # pyright: ignore[reportArgumentType]
        if strategy == STRATEGY_FREE_TRAJECTORY:
            x[i] = -(  # pyright: ignore[reportIndexIssue]
                xtilde_bdf[i] + bt * vtilde_bdf[i]
            )
        elif strategy == STRATEGY_TRAJECTORY_WITH_EXTERNAL_LOAD:
            x[i] = xti  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _back_substitute_velocities_kernel(
    x: wp.array[wp.vec3f],
    v: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    xtilde_bdf: wp.array[wp.vec3f],
    bt: float,
):
    i = wp.tid()
    if is_free_node(dmask, i):  # pyright: ignore[reportArgumentType]
        v[i] = (x[i] + xtilde_bdf[i]) / bt  # pyright: ignore[reportIndexIssue]


@wp.struct
class FemElastoDynamicsData:
    """Elastodynamic IVP (initial value problem) using FEM spatial discretization and BDF time discretization."""

    # --- Mesh (linear) ---
    X: wp.array[wp.vec3f]  # (N,) rest positions
    E: wp.array[
        types.vec4i  # type: ignore
    ]  # (E,) element connectivity (4 nodes per tet)

    # --- Dynamic state ---
    x: wp.array[wp.vec3f]  # (N,) current positions
    v: wp.array[wp.vec3f]  # (N,) velocities

    # --- Time integration ---
    xt: wp.array[wp.vec3f]  # (N,) current positions (BDF)
    xtilde: wp.array[wp.vec3f]  # (N,) BDF inertial target

    # --- Mass ---
    m: wp.array[wp.float32]  # (N,) lumped mass per node

    # --- External forces ---
    fext: wp.array[wp.vec3f]  # (N,) external force per node

    # --- Problem kind ---
    is_static: wp.int32  # 1 = static equilibrium: objective Psi(x) - fext^T x, no inertia, no BDF

    # --- Elastic quadrature ---
    wg: wp.array[wp.float32]  # (Q,) quadrature weights
    GNeg: wp.array[
        types.mat4x3f  # type: ignore
    ]  # (Q,) shape function gradients (4x3 per quad pt) at quad pts
    mug: wp.array[wp.float32]  # (Q,) 1st Lame parameter at quad pts
    lambdag: wp.array[wp.float32]  # (Q,) 2nd Lame parameter at quad pts

    # --- Dirichlet BCs ---
    dmask: wp.array[wp.int32]  # (N,) 0 = free, nonzero = constrained


class FemElastoDynamics:
    """Elastodynamic IVP (initial value problem) using FEM spatial discretization and BDF time discretization.

    With :attr:`static` set, the same data describes the static equilibrium problem
    min_x Psi(x) - fext^T x instead (no kinetic term, no BDF). Masses are kept, so lumped mass
    weighted quantities (and gravity loads) are unaffected.

    Usage:
        fem = FemElastoDynamics(cpu_fem)
        fem.set_time_integration_scheme(dt=0.01, s=1) # for BDF1
        fem.set_time_integration_scheme(dt=0.01, s=2) # for BDF2
        fem.set_initial_conditions(x0, v0)
        fem.set_kinematic_constraints(kcs)  # optional, prescribes dmask, x and v of constrained nodes
        for t in range(100):
            fem.setup_time_integration_optimization()
            solve(fem)
            fem.back_substitute_velocities()
            fem.step()
    """

    _data: FemElastoDynamicsData  # pyright: ignore[reportGeneralTypeIssues]
    kinematics: KinematicConstraintSet | None

    def __init__(self, fem: pbat.sim.dynamics.FemElastoDynamics):
        self._data = FemElastoDynamicsData()
        self._data.X = wp.array(fem.X.T, dtype=wp.vec3f)
        self._data.E = wp.array(fem.E.T, dtype=wp.vec4i)
        self._data.x = wp.array(fem.x.T, dtype=wp.vec3f)
        self._data.v = wp.array(fem.v.T, dtype=wp.vec3f)
        self._data.xtilde = wp.array(fem.x.T, dtype=wp.vec3f)
        self._data.m = wp.array(fem.m, dtype=wp.float32)
        self._data.fext = wp.array(fem.fext.T, dtype=wp.vec3f)
        self._data.wg = wp.array(fem.wgU, dtype=wp.float32)
        self._data.GNeg = wp.array(
            fem.GNegU.reshape(4, -1, 3).transpose(1, 0, 2),
            dtype=types.mat4x3f,
        )
        self._data.mug = wp.array(fem.lamegU[0, :], dtype=wp.float32)
        self._data.lambdag = wp.array(fem.lamegU[1, :], dtype=wp.float32)
        self._data.is_static = wp.int32(0)
        self._data.dmask = wp.array(fem.dmask, dtype=wp.int32)

        # BDF integrator (operates on flat 3*N dof vectors)
        bdf: pbat.sim.integration.Bdf = fem.bdf
        self.bdf = Bdf(step=int(bdf.s), order=2, dt=float(bdf.h))
        self.bdf.set_initial_conditions(
            cp.asarray(self._data.x).ravel(), cp.asarray(self._data.v).ravel()
        )
        self._data.xt = self.xt
        self.kinematics = None

    @property
    def data(self) -> FemElastoDynamicsData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data

    @property
    def static(self) -> bool:
        """Whether this is a static equilibrium problem, min_x Psi(x) - fext^T x.

        In static mode, solvers drop the inertia term and use :attr:`h2` = 1,
        :meth:`setup_time_integration_optimization` leaves free nodes in place,
        and :meth:`back_substitute_velocities` and :meth:`step` do nothing. Since :meth:`step` doesn't
        advance the kinematic constraints either, they stay evaluated at a frozen time. Leaving static
        mode restarts the time integration from the current positions at rest.
        """
        return bool(self._data.is_static)

    @static.setter
    def static(self, value: bool):
        if bool(value) == self.static:
            return
        self._data.is_static = wp.int32(1 if value else 0)
        if not value:
            # Restart at rest from the current positions. Write the BDF history in place, since
            # contact detection holds handles to xt.
            self.bdf.xt[0, :] = cp.asarray(self._data.x).ravel()[cp.newaxis, :]
            self.bdf.xt[1, :] = 0
            self._data.v.zero_()

    @property
    def h2(self) -> float:
        """Scale of the elastic energy in the optimization objective.

        ``BetaTilde^2`` for dynamics (the objective is 1/2 |x - xtilde|_M^2 + h^2 Psi(x)), and 1 for
        static problems.
        """
        if self.static:
            return 1.0
        h = self.bdf.beta_tilde
        return h * h

    @property
    def xt(self) -> wp.array[wp.vec3f]:
        _xt = self.bdf.current_state(0).reshape(-1, 3)
        assert _xt.flags["OWNDATA"] == False
        return wp.array(data=_xt, dtype=wp.vec3f)

    def set_time_integration_scheme(self, dt: float = 0.01, s: int = 1):
        """Configure BDF step and time step size."""
        self.bdf = Bdf(step=s, order=2, dt=dt)

    def set_initial_conditions(
        self, x0: cp.ndarray | wp.array, v0: cp.ndarray | wp.array
    ):
        """Set initial conditions. x0, v0 are (N, 3) or (3*N,) arrays."""
        self.bdf.set_initial_conditions(cp.asarray(x0).ravel(), cp.asarray(v0).ravel())

    def set_kinematic_constraints(self, kinematics: KinematicConstraintSet | None):
        """Prescribe the motion of Dirichlet nodes. The set then owns dmask: every time step, it marks
        the constrained nodes and sets their positions and velocities at the end of the step."""
        self.kinematics = kinematics

    def setup_time_integration_optimization(
        self,
        strategy: pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization = pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position,
    ):
        """Compute xtilde and initialize x for the time step.

        Strategies: "position", "free_trajectory", "trajectory_with_external_load". Static problems
        always use "position", i.e. free nodes start from the current x.
        """
        if self.static:
            strategy = pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position
        if self.kinematics is not None:
            with profiling.scope("kinematics.update"):
                self.kinematics.update(
                    self._data.xt, self._data.x, self._data.v, self._data.dmask
                )
        self.bdf.construct_equations()
        bt = self.bdf.beta_tilde
        bt2 = bt * bt
        N = self._data.x.shape[0]
        xtilde_bdf = self.bdf.inertia(0).reshape(-1, 3)
        vtilde_bdf = self.bdf.inertia(1).reshape(-1, 3)
        wp.launch(
            _setup_time_integration_optimization_kernel,
            dim=N,
            inputs=[
                self._data.x,
                self._data.xtilde,
                self._data.fext,
                self._data.m,
                self._data.dmask,
                xtilde_bdf,
                vtilde_bdf,
                bt,
                bt2,
                _STRATEGY_MAP[strategy],
            ],
        )

    def back_substitute_velocities(self):
        """Recover v from integrated x on free dofs. Does nothing for static problems."""
        if self.static:
            return
        bt = self.bdf.beta_tilde
        N = self._data.x.shape[0]
        xtilde_bdf = self.bdf.inertia(0).reshape(-1, 3)
        wp.launch(
            _back_substitute_velocities_kernel,
            dim=N,
            inputs=[
                self._data.x,
                self._data.v,
                self._data.dmask,
                xtilde_bdf,
                bt,
            ],
        )

    def step(self):
        """Advance BDF by one time step using current x, v. Does nothing for static problems."""
        if self.static:
            return
        self.bdf.step(
            cp.asarray(self._data.x).ravel(), cp.asarray(self._data.v).ravel()
        )
        if self.kinematics is not None:
            self.kinematics.advance()


import unittest


class TestFemElastoDynamics(unittest.TestCase):
    def _make_cpu_fem(self):
        from pbatoolkit import pypbat

        V = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        C = np.array(
            [
                [0, 1, 3, 5],
                [3, 2, 0, 6],
                [5, 4, 6, 0],
                [6, 7, 5, 3],
                [0, 5, 3, 6],
            ]
        )
        fem_cpu = pbat.sim.dynamics.FemElastoDynamics(V.T, C.T)
        fem_cpu.set_mass_matrix(1e3)
        mu, llambda = pypbat.fem.lame_coefficients(1e6, 0.45)
        fem_cpu.set_elastic_energy(mu, llambda)  # pyright: ignore[reportArgumentType]
        fem_cpu.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
        fem_cpu.set_time_integration_scheme(dt=1e-2, s=1)
        fem_cpu.constrain(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        return fem_cpu

    def test_construction(self):
        fem_cpu = self._make_cpu_fem()
        fem = FemElastoDynamics(fem_cpu)
        self.assertEqual(fem.data.x.shape[0], 8)
        self.assertTrue(np.all(fem.data.X.numpy() == fem_cpu.X.T))
        self.assertTrue(np.all(fem.data.E.numpy() == fem_cpu.E.T))
        self.assertTrue(np.all(fem.data.x.numpy() == fem_cpu.x.T))
        self.assertTrue(np.all(fem.data.v.numpy() == fem_cpu.v.T))
        self.assertTrue(np.all(fem.data.m.numpy() == fem_cpu.m))
        self.assertTrue(np.all(fem.data.fext.numpy() == fem_cpu.fext.T))
        self.assertTrue(np.all(fem.data.wg.numpy() == fem_cpu.wgU))
        self.assertTrue(
            np.all(
                fem.data.GNeg.numpy().transpose(1, 0, 2).reshape(4, -1) == fem_cpu.GNegU
            )
        )
        self.assertTrue(np.all(fem.data.mug.numpy() == fem_cpu.lamegU[0, :]))
        self.assertTrue(np.all(fem.data.lambdag.numpy() == fem_cpu.lamegU[1, :]))
        self.assertTrue(np.all(fem.data.dmask.numpy() == fem_cpu.dmask))

    def test_static_mode(self):
        fem_cpu = self._make_cpu_fem()
        fem = FemElastoDynamics(fem_cpu)
        self.assertFalse(fem.static)
        self.assertAlmostEqual(fem.h2, 1e-4, places=9)
        fem.static = True
        self.assertTrue(fem.static)
        self.assertEqual(fem.h2, 1.0)
        x0 = fem.data.x.numpy().copy()
        xt0 = fem.xt.numpy().copy()
        # Free nodes stay in place even with a strategy that would move them, and neither the
        # velocities nor the BDF history change.
        fem.setup_time_integration_optimization(
            pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.TrajectoryWithExternalLoad
        )
        self.assertTrue(np.all(fem.data.x.numpy() == x0))
        self.assertTrue(np.all(np.isfinite(fem.data.xtilde.numpy())))
        fem.data.x.assign(x0 + 0.1)
        fem.back_substitute_velocities()
        fem.step()
        self.assertTrue(np.all(fem.data.v.numpy() == 0.0))
        self.assertTrue(np.all(fem.xt.numpy() == xt0))
        # Leaving static mode restarts the time integration at rest from the current positions,
        # without reallocating xt.
        xt_ptr = fem.xt.ptr
        fem.static = False
        self.assertEqual(fem.xt.ptr, xt_ptr)
        self.assertTrue(np.allclose(fem.xt.numpy(), x0 + 0.1))
        self.assertTrue(np.all(fem.data.v.numpy() == 0.0))


if __name__ == "__main__":
    wp.init()
    unittest.main()

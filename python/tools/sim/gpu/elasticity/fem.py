import warp as wp
import cupy as cp
import numpy as np
from .. import types
from ..integration.bdf import Bdf
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
    x_bdf_current: wp.array[wp.vec3f],
    bt: float,
    bt2: float,
    strategy: int,
):
    i = wp.tid()
    aexti = fext[i] / m[i]
    # xtilde = -(xtilde_bdf + bt * vtilde_bdf) + bt^2 * aext
    xti = -(xtilde_bdf[i] + bt * vtilde_bdf[i]) + bt2 * aexti
    if is_dirichlet_node(dmask, i):  # pyright: ignore[reportArgumentType]
        # Dirichlet nodes: xtilde = current BDF position
        xti = x_bdf_current[i]
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
    xtilde: wp.array[wp.vec3f]  # (N,) BDF inertial target

    # --- Mass ---
    m: wp.array[wp.float32]  # (N,) lumped mass per node

    # --- External forces ---
    fext: wp.array[wp.vec3f]  # (N,) external force per node

    # --- Elastic quadrature ---
    wg: wp.array[wp.float32]  # (Q,) quadrature weights
    GNeg: wp.array[
        types.mat4x3f  # type: ignore
    ]  # (Q,) shape function gradients (4x3 per quad pt) at quad pts
    mug: wp.array[wp.float32]  # (Q,) 1st Lame parameter at quad pts
    lambdag: wp.array[wp.float32]  # (Q,) 2nd Lame parameter at quad pts

    # --- Dirichlet BCs ---
    dmask: wp.array[wp.int32]  # (N,) 0 = free, 1+ = constrained
    ndbc: wp.int32  # number of dirichlet constrained nodes
    dbc: wp.array[
        wp.int32
    ]  # (N,) concatenated vector of Dirichlet unconstrained and constrained nodes, partitioned as [ dbc[0:N-ndbc], dbc[N-ndbc:] ]


class FemElastoDynamics:
    """Elastodynamic IVP (initial value problem) using FEM spatial discretization and BDF time discretization.

    Usage:
        fem = FemElastoDynamics(cpu_fem)
        fem.set_time_integration_scheme(dt=0.01, s=1) # for BDF1
        fem.set_time_integration_scheme(dt=0.01, s=2) # for BDF2
        fem.set_initial_conditions(x0, v0)
        for t in range(100):
            fem.setup_time_integration_optimization()
            solve(fem)
            fem.back_substitute_velocities()
            fem.step()
    """

    _data: FemElastoDynamicsData  # pyright: ignore[reportGeneralTypeIssues]

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
        self._data.dmask = wp.array(fem.dmask, dtype=wp.int32)
        self._data.ndbc = int(fem.ndbc)
        self._data.dbc = wp.array(fem.dbc, dtype=wp.int32)

        # BDF integrator (operates on flat 3*N dof vectors)
        bdf: pbat.sim.integration.Bdf = fem.bdf
        self.bdf = Bdf(step=int(bdf.s), order=2, dt=float(bdf.h))
        self.bdf.set_initial_conditions(
            cp.asarray(self._data.x).ravel(), cp.asarray(self._data.v).ravel()
        )

    @property
    def data(self) -> FemElastoDynamicsData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data
    
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

    def setup_time_integration_optimization(
        self,
        strategy: pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization = pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position,
    ):
        """Compute xtilde and initialize x for the time step.

        Strategies: "position", "free_trajectory", "trajectory_with_external_load"
        """
        self.bdf.construct_equations()
        bt = self.bdf.beta_tilde
        bt2 = bt * bt
        N = self._data.x.shape[0]
        xtilde_bdf = self.bdf.inertia(0).reshape(-1, 3)
        vtilde_bdf = self.bdf.inertia(1).reshape(-1, 3)
        x_bdf_current = self.bdf.current_state(0).reshape(-1, 3)
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
                x_bdf_current,
                bt,
                bt2,
                _STRATEGY_MAP[strategy],
            ],
        )

    def back_substitute_velocities(self):
        """Recover v from integrated x on free dofs."""
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
        """Advance BDF by one time step using current x, v."""
        self.bdf.step(
            cp.asarray(self._data.x).ravel(), cp.asarray(self._data.v).ravel()
        )


import unittest


class TestFemElastoDynamics(unittest.TestCase):
    def test_construction(self):
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
        self.assertTrue(np.all(fem.data.dbc.numpy() == fem_cpu.dbc))


if __name__ == "__main__":
    wp.init()
    unittest.main()

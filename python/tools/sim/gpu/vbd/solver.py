"""
VBD (Vertex Block Descent) solver for FEM elasto-dynamics.

Mirrors `pbat::sim::algorithm::vbd::Solve` from `source/pbat/sim/algorithm/vbd/Core.h`.
Contact constraints are left as placeholders for future implementation.
"""

from typing import Tuple
import warp as wp
import numpy as np
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from .params import (
    Params,
    ParamsData,
)
from .kernels import (
    local_elastic_derivatives,
    add_inertia_derivatives,
    integrate_positions,
)
from ..contact.ogc import Ogc


@wp.kernel
def _vertex_solve_kernel(
    pbegin: int,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    h2: float,
):
    """Process one vertex in the current color partition."""
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid / block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]
    i = params.Padj[
        pbegin + block_id  # pyright: ignore[reportOperatorIssue, reportIndexIssue]
    ]
    # Skip Dirichlet nodes
    if is_dirichlet_node(fem.dmask, i):  # pyright: ignore[reportArgumentType]
        return
    xi = fem.x[i]  # pyright: ignore[reportIndexIssue]
    xtildei = fem.xtilde[i]  # pyright: ignore[reportIndexIssue]
    mi = fem.m[i]  # pyright: ignore[reportIndexIssue]
    # Accumulate elastic energy derivatives
    gil, Hil = local_elastic_derivatives(
        i, fem, params, local_tid, block_dims  # pyright: ignore[reportArgumentType]
    )
    gs, Hs = (
        wp.tile(gil, preserve_type=True),  # pyright: ignore[reportArgumentType]
        wp.tile(Hil, preserve_type=True),  # pyright: ignore[reportArgumentType]
    )
    gi, Hi = (
        wp.tile_reduce(wp.add, gs)[0],  # pyright: ignore[reportIndexIssue]
        wp.tile_reduce(wp.add, Hs)[0],  # pyright: ignore[reportIndexIssue]
    )
    gi *= h2  # pyright: ignore[reportOperatorIssue]
    Hi *= h2  # pyright: ignore[reportOperatorIssue]
    # TODO: AccumulateContactEnergy(i, params.xb, contact, gi, Hi)
    if local_tid > 0:
        return
    # Add inertia derivatives (K = m, already in position space)
    gi, Hi = add_inertia_derivatives(
        mi, xtildei, xi, gi, Hi
    )  # pyright: ignore[reportArgumentType]
    # Solve local system: x -= H^{-1} g
    dxi = integrate_positions(
        gi,
        Hi,
        params.vls_solver,
        params.vls_max_iters,
        params.vls_eps,
        params.hess_zero,
    )  # pyright: ignore[reportArgumentType]
    fem.x[i] -= dxi  # pyright: ignore[reportIndexIssue]


def linearize_constraints(fem: FemElastoDynamics, params: Params):
    """TODO: Linearize contact constraints at current iterate."""
    pass


def check_convergence(fem: FemElastoDynamics, params: Params) -> bool:
    """TODO: Check gradient norm convergence (elastic + momentum + contact)."""
    return False


def prepare_subproblem(fem: FemElastoDynamics, params: Params):
    """TODO: Assemble block-diagonal Hessian, update penalty parameter."""
    pass


def finalize_subproblem(fem: FemElastoDynamics, params: Params):
    """TODO: Update dual variables, restore feasibility, update constraint set."""
    pass


def iterate(fem: FemElastoDynamics, params: Params):
    """One VBD Gauss-Seidel sweep over all color partitions."""
    h = fem.bdf.beta_tilde
    h2 = h * h
    # Copy current positions to buffer (for contact lagging)
    wp.copy(params.data.xb, fem.data.x)
    # Process each color partition sequentially
    Pptr = params.data.Pptr.numpy()
    # NOTE: Should be no-copy if params.data.Pptr is already on CPU.
    # assert type(Pptr) == np.ndarray and Pptr.flags["OWNDATA"] == False
    n_partitions = len(Pptr) - 1
    for p in range(n_partitions):
        p_begin = int(Pptr[p])
        p_end = int(Pptr[p + 1])
        n_verts_in_partition = p_end - p_begin
        if n_verts_in_partition > 0:
            block_dim = 32
            wp.launch(
                kernel=_vertex_solve_kernel,
                dim=n_verts_in_partition * block_dim,
                inputs=[p_begin, fem.data, params.data, h2],
                block_dim=block_dim,
            )


def solve_subproblem(
    fem: FemElastoDynamics,
    params: Params,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    # TODO: prepare_subproblem(fem, params)
    prepare_subproblem(fem, params)
    for kp in range(n_subproblem_max_iters):
        iterate(fem, params)
    # TODO: finalize_subproblem(fem, params)
    finalize_subproblem(fem, params)


def solve(
    fem: FemElastoDynamics,
    params: Params,
    ogc: Ogc,
) -> bool:
    """Solve the VBD minimization problem.
    Mimics `pbat::sim::algorithm::vbd::Solve`:
    """
    converged = False
    n_max_iters = params.data.n_max_iters
    for k in range(n_max_iters):
        # TODO: Replace these OGC calls with a proper
        # contact.MeshDynamics class that uses OGC internally
        ogc.prepare_for_execution()
        ogc.detect_contacts()
        ogc.update_displacement_bounds()
        # TODO: linearize_constraints(fem, params)
        linearize_constraints(fem, params)
        # TODO: if check_convergence(fem, params): break
        if check_convergence(fem, params):
            converged = True
            break
        # Solve linearized subproblem
        solve_subproblem(fem, params)
    fem.back_substitute_velocities()
    return converged


def integrate(fem: FemElastoDynamics, params: Params, ogc: Ogc):
    """Integrate one time step: setup + solve + step.

    Mimics `pbat::sim::algorithm::vbd::Integrate`.
    """
    fem.setup_time_integration_optimization()
    solve(fem, params, ogc)
    fem.step()


# --- Unit tests ---
import unittest


class TestVbdSolver(unittest.TestCase):
    def test_single_iterate(self):
        """Test that a single VBD iterate modifies free node positions."""
        from pbatoolkit import pbat, pypbat
        from ..contact.multimesh import MultiMesh as GpuMultiMesh

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
            ],
            dtype=np.float32,
        )
        C = np.array(
            [
                [0, 1, 3, 5],
                [3, 2, 0, 6],
                [5, 4, 6, 0],
                [6, 7, 5, 3],
                [0, 5, 3, 6],
            ],
            dtype=np.int64,
        )
        n_nodes = V.shape[0]
        E = C.T
        # Setup FEM
        fem_cpu = pbat.sim.dynamics.FemElastoDynamics(V.T, C.T)
        fem_cpu.set_mass_matrix(1e3)
        mu, llambda = pypbat.fem.lame_coefficients(1e6, 0.45)
        fem_cpu.set_elastic_energy(mu, llambda)  # pyright: ignore[reportArgumentType]
        fem_cpu.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
        fem_cpu.set_time_integration_scheme(dt=1e-2, s=1)
        # Bottom 4 nodes free, top 4 fixed
        fem_cpu.constrain(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        fem = FemElastoDynamics(fem_cpu)
        # Setup VBD params
        GVGp, GVGe, GVGilocal = pbat.sim.algorithm.vbd.vertex_element_adjacency_graph(
            E, n_nodes
        )
        GVVp, GVVadj, colors = pbat.sim.algorithm.vbd.vertex_colors(E, n_nodes)
        params_cpu = pbat.sim.algorithm.vbd.Params()
        params_cpu.with_vertex_element_adjacency_graph(GVGp, GVGe, GVGilocal)
        params_cpu.with_vertex_colors(GVVp, GVVadj, colors)
        params_cpu.construct()
        params = Params(params_cpu)
        # Setup time integration
        fem.setup_time_integration_optimization()
        # Record positions before iterate
        x_before = fem.data.x.numpy().copy()
        # Run one iterate
        iterate(fem, params)
        wp.synchronize()
        x_after = fem.data.x.numpy()
        # Constrained nodes should not move
        np.testing.assert_allclose(x_after[4:], x_before[4:], atol=1e-10)
        # Free nodes should have moved (gravity pulls them)
        self.assertFalse(np.allclose(x_after[:4], x_before[:4], atol=1e-10))

    def test_integrate_step(self):
        """Test that integrate advances the simulation by one time step."""
        from pbatoolkit import pbat, pypbat
        from ..contact.multimesh import MultiMesh as GpuMultiMesh
        from ..contact.ogc import Ogc

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
            ],
            dtype=np.float32,
        )
        C = np.array(
            [
                [0, 1, 3, 5],
                [3, 2, 0, 6],
                [5, 4, 6, 0],
                [6, 7, 5, 3],
                [0, 5, 3, 6],
            ],
            dtype=np.int64,
        )
        n_nodes = V.shape[0]
        E = C.T
        fem_cpu = pbat.sim.dynamics.FemElastoDynamics(V.T, C.T)
        fem_cpu.set_mass_matrix(1e3)
        mu, llambda = pypbat.fem.lame_coefficients(1e6, 0.45)
        fem_cpu.set_elastic_energy(mu, llambda)  # pyright: ignore[reportArgumentType]
        fem_cpu.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
        fem_cpu.set_time_integration_scheme(dt=1e-2, s=1)
        fem_cpu.constrain(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        fem = FemElastoDynamics(fem_cpu)
        GVGp, GVGe, GVGilocal = pbat.sim.algorithm.vbd.vertex_element_adjacency_graph(
            E, n_nodes
        )
        GVVp, GVVadj, colors = pbat.sim.algorithm.vbd.vertex_colors(E, n_nodes)
        params_cpu = pbat.sim.algorithm.vbd.Params()
        params_cpu.with_vertex_element_adjacency_graph(GVGp, GVGe, GVGilocal)
        params_cpu.with_vertex_colors(GVVp, GVVadj, colors)
        params_cpu.n_max_iters = 1
        params_cpu.n_subproblem_max_iters = 5
        params_cpu.construct()
        params = Params(params_cpu)
        # Setup OGC
        multimesh_cpu = pbat.sim.contact.MultiMesh()
        multimesh_cpu.construct_from_tetrahedral_mesh(
            fem_cpu.E, np.full(n_nodes, 0, dtype=np.int64), n_components=1
        )
        multimesh = GpuMultiMesh(multimesh_cpu)
        ogc = Ogc(fem.data.x, multimesh)
        x_before = fem.data.x.numpy().copy()
        # Run some number of steps
        for t in range(10):
            fem.setup_time_integration_optimization()
            solve(fem, params, ogc)
            fem.back_substitute_velocities()
        wp.synchronize()
        x_after = fem.data.x.numpy()
        # Free nodes should have moved
        self.assertFalse(np.allclose(x_after[:4], x_before[:4], atol=1e-10))


if __name__ == "__main__":
    wp.init()
    wp.config.mode = "debug"
    wp.config.verify_cuda = True
    wp.config.verify_fp = True
    unittest.main()

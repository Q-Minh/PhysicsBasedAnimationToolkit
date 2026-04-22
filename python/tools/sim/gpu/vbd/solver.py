"""
VBD (Vertex Block Descent) solver for FEM elasto-dynamics.

Mirrors `pbat::sim::algorithm::vbd::Solve` from `source/pbat/sim/algorithm/vbd/Core.h`.
Contact constraints are left as placeholders for future implementation.
"""

import warp as wp
import warp.fem.linalg
import numpy as np
from .. import types
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from ..elasticity.snh import snh_grad_and_hess
from ..elasticity.chain import gradient_segment_wrt_dofs, hessian_block_wrt_dofs
from .params import (
    Params,
    ParamsData,
    VLS_SOLVER_INVERSE,
    VLS_SOLVER_LLT,
    VLS_SOLVER_QR,
    VLS_SOLVER_EVD,
)


# --- Warp kernels ---


@wp.func
def elastic_derivatives(
    i: int,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    gi: wp.vec3f,
    Hi: wp.mat33f,
):
    """Accumulate elastic gradient and hessian for vertex i over adjacent elements."""
    begin = params.GVGp[i]  # pyright: ignore[reportIndexIssue]
    end = params.GVGp[i + 1]  # pyright: ignore[reportIndexIssue]
    for n in range(begin, end):
        e = params.GVGe[n]  # pyright: ignore[reportIndexIssue]
        nodes = fem.E[e]  # pyright: ignore[reportIndexIssue]
        ilocal = (
            wp.int32(i == nodes[1])
            * wp.int32(1)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[2])
            * wp.int32(2)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[3])
            * wp.int32(3)  # pyright: ignore[reportOperatorIssue]
        )
        wg = fem.wg[e]  # pyright: ignore[reportIndexIssue]
        GP = fem.GNeg[e]  # pyright: ignore[reportIndexIssue]
        mu = fem.mug[e]  # pyright: ignore[reportIndexIssue]
        llambda = fem.lambdag[e]  # pyright: ignore[reportIndexIssue]
        # Gather element positions -> compute F
        xe = types.mat3x4f()
        for j in range(4):
            xj = fem.x[nodes[j]]
            for d in range(3):
                xe[d, j] = xj[d]
        # xe = 3x4 matrix of element positions (columns are nodes)
        F = xe @ GP
        # SNH grad and hess w.r.t. vec(F)
        gF, HF = snh_grad_and_hess(F, mu, llambda)
        # Chain rule: accumulate into vertex gradient and hessian
        gi += wg * gradient_segment_wrt_dofs(gF, GP, ilocal)
        Hi += wg * hessian_block_wrt_dofs(HF, GP, ilocal, ilocal)
    return gi, Hi


@wp.func
def add_inertia_derivatives(
    m: wp.float32, xtilde: wp.vec3f, x: wp.vec3f, gi: wp.vec3f, Hi: wp.mat33f
):
    """Add kinetic energy derivatives: g += K*(x - xtilde), diag(H) += K, where K = m."""
    gi += m * (x - xtilde)  # pyright: ignore[reportOperatorIssue]
    Hi[0, 0] += m  # pyright: ignore[reportIndexIssue]
    Hi[1, 1] += m  # pyright: ignore[reportIndexIssue]
    Hi[2, 2] += m  # pyright: ignore[reportIndexIssue]
    return gi, Hi


@wp.func
def integrate_positions(
    gi: wp.vec3f,
    Hi: wp.mat33f,
    solver: wp.int32,
    max_iters: wp.int32,
    eps: wp.float32,
    hess_zero: wp.float32,
):
    """Solve x -= H^{-1} g (direct inverse for 3x3)."""
    dxi = wp.vec3f()
    if solver == VLS_SOLVER_INVERSE:
        det = wp.determinant(Hi)  # pyright: ignore[reportArgumentType]
        if wp.abs(det) > hess_zero:  # pyright: ignore[reportArgumentType]
            dxi = wp.inverse(Hi) * gi  # pyright: ignore
    elif solver == VLS_SOLVER_LLT:
        assert False
    elif solver == VLS_SOLVER_QR:
        Q, R = wp.mat33f(), wp.mat33f()
        wp.qr3(Hi, Q, R)  # pyright: ignore[reportArgumentType]
        dxi = warp.fem.linalg.solve_triangular(
            R,
            wp.transpose(Q)  # pyright: ignore[reportOperatorIssue,reportArgumentType]
            * gi,
        )
    elif solver == VLS_SOLVER_EVD:
        V, eigs = wp.eig3(Hi)  # pyright: ignore[reportArgumentType]
        dxi = wp.transpose(V) * gi  # pyright: ignore[reportOperatorIssue]
        for d in range(3):
            eigd = wp.abs(eigs[d])  # pyright: ignore[reportIndexIssue]
            dxi[d] = (dxi[d] / eigd) if (eigd > hess_zero) else 0.0
        dxi = V * dxi
    else:
        assert False
    return dxi


@wp.kernel
def _vertex_solve_kernel(
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    pbegin: int,
    h2: float,
):
    """Process one vertex in the current color partition."""
    k = wp.tid()
    i = params.Padj[
        pbegin + k  # pyright: ignore[reportOperatorIssue, reportIndexIssue]
    ]
    # Skip Dirichlet nodes
    if is_dirichlet_node(fem.dmask, i):  # pyright: ignore[reportArgumentType]
        return
    xi = fem.x[i]  # pyright: ignore[reportIndexIssue]
    xtildei = fem.xtilde[i]  # pyright: ignore[reportIndexIssue]
    mi = fem.m[i]  # pyright: ignore[reportIndexIssue]
    # Accumulate elastic energy derivatives
    gi = wp.vec3f()
    Hi = wp.mat33f()
    gi, Hi = elastic_derivatives(
        i, fem, params, gi, Hi
    )  # pyright: ignore[reportArgumentType]
    # Scale by h^2
    gi *= h2  # pyright: ignore[reportOperatorIssue]
    Hi *= h2  # pyright: ignore[reportOperatorIssue]
    # TODO: AccumulateContactEnergy(i, params.xb, contact, gi, Hi)
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


# --- Contact placeholders (to be implemented) ---


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


# --- Solver API ---


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
            wp.launch(
                _vertex_solve_kernel,
                dim=n_verts_in_partition,
                inputs=[fem.data, params.data, p_begin, h2],
            )


def solve(fem: FemElastoDynamics, params: Params) -> bool:
    """Solve the VBD minimization problem.

    Mimics `pbat::sim::algorithm::vbd::Solve`:
        for k in range(n_max_iters):
            linearize_constraints(fem, params)
            if check_convergence(fem, params): break
            prepare_subproblem(fem, params)
            for kp in range(n_subproblem_max_iters):
                iterate(fem, params)
            finalize_subproblem(fem, params)
        fem.back_substitute_velocities()
    """
    converged = False
    n_max_iters = params.data.n_max_iters
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    for k in range(n_max_iters):
        # TODO: linearize_constraints(fem, params)
        linearize_constraints(fem, params)
        # TODO: if check_convergence(fem, params): break
        if check_convergence(fem, params):
            converged = True
            break
        # TODO: prepare_subproblem(fem, params)
        prepare_subproblem(fem, params)
        for kp in range(n_subproblem_max_iters):
            iterate(fem, params)
        # TODO: finalize_subproblem(fem, params)
        finalize_subproblem(fem, params)
    fem.back_substitute_velocities()
    return converged


def integrate(fem: FemElastoDynamics, params: Params):
    """Integrate one time step: setup + solve + step.

    Mimics `pbat::sim::algorithm::vbd::Integrate`.
    """
    fem.setup_time_integration_optimization()
    solve(fem, params)
    fem.step()


# --- Unit tests ---
import unittest


class TestVbdSolver(unittest.TestCase):
    def test_single_iterate(self):
        """Test that a single VBD iterate modifies free node positions."""
        from pbatoolkit import pbat, pypbat

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
        x_before = fem.data.x.numpy().copy()
        # Run some number of steps
        for t in range(10):
            integrate(fem, params)
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

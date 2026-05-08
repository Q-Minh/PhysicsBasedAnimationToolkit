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
from ..contact.dynamics import MeshDynamics


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


def initialize_solve(
    fem: FemElastoDynamics,
    params: Params,
    contact: MeshDynamics,
):
    """Initialize the VBD solve by updating contact constraint set and restoring feasibility.

    Mirrors ``pbat::sim::algorithm::vbd::InitializeSolve``.
    Called once after :meth:`FemElastoDynamics.setup_time_integration_optimization`,
    before the first call to :func:`solve`.
    """
    contact.ogc.compute_query_radius()
    contact.update_constraint_set(fem.xt)
    # contact.restore_feasibility(fem.data.x)


def finalize_subproblem(
    fem: FemElastoDynamics,
    params: Params,
    contact: MeshDynamics,
):
    """Finalize the current linearized subproblem.

    Mirrors ``pbat::sim::algorithm::vbd::FinalizeSubproblem``:
      1. Full dual update (slack + decay + Lagrange multipliers).
      2. Restore feasibility.
      3. Update constraint set for the next subproblem.
    """
    contact.update_dual(
        fem.data.x,
        request_slack_update=True,
        request_decay_update=True,
        request_lagrange_multiplier_update=True,
    )
    # contact.restore_feasibility(fem.data.x)
    contact.update_constraint_set(fem.data.x)


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
    contact: MeshDynamics,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    prepare_subproblem(fem, params)
    for kp in range(n_subproblem_max_iters):
        contact.update_dual(
            fem.data.x,
            request_slack_update=True,
            request_decay_update=False,
            request_lagrange_multiplier_update=False,
        )
        iterate(fem, params)
    finalize_subproblem(fem, params, contact)


def solve(
    fem: FemElastoDynamics,
    params: Params,
    contact: MeshDynamics,
) -> bool:
    """Solve the VBD minimization problem.
    Mimics `pbat::sim::algorithm::vbd::Solve`:
    """
    converged = False
    n_max_iters = params.data.n_max_iters
    for k in range(n_max_iters):
        # TODO: linearize_constraints(fem, params)
        linearize_constraints(fem, params)
        # TODO: if check_convergence(fem, params): break
        if check_convergence(fem, params):
            converged = True
            break
        # Solve linearized subproblem
        solve_subproblem(fem, params, contact)
    fem.back_substitute_velocities()
    return converged


def integrate(fem: FemElastoDynamics, params: Params, contact: MeshDynamics):
    """Integrate one time step: setup + initialize_solve + solve + step.

    Mimics `pbat::sim::algorithm::vbd::Integrate`.
    """
    fem.setup_time_integration_optimization()
    initialize_solve(fem, params, contact)
    solve(fem, params, contact)
    fem.step()


class VbdSolver:

    def __init__(self):
        self._cuda_graph = None

    def solve(
        self, fem: FemElastoDynamics, params: Params, contact: MeshDynamics
    ) -> bool:
        initialize_solve(fem, params, contact)
        converged = False
        for k in range(params.data.n_max_iters):
            linearize_constraints(fem, params)
            if check_convergence(fem, params):
                converged = True
                break
            prepare_subproblem(fem, params)
            if self._cuda_graph is None:
                with wp.ScopedCapture() as capture:
                    solve_subproblem(fem, params, contact)
                self._cuda_graph = capture
            else:
                wp.capture_launch(self._cuda_graph.graph)  # type: ignore
            finalize_subproblem(fem, params, contact)
        fem.back_substitute_velocities()
        return converged


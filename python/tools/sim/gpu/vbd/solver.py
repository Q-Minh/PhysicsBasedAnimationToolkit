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
    local_contact_derivatives,
    add_inertia_derivatives,
    integrate_positions,
)
from ..contact.dynamics import (
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
)


@wp.kernel
def _vertex_solve_kernel(
    pbegin: int,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    contact: ContactDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
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
    gil *= h2  # type: ignore
    Hil *= h2  # type: ignore
    vi = contact.meshes.GXV[i]
    if vi >= 0:
        gil_c, Hil_c = local_contact_derivatives(
            i, vi, fem, contact, params, local_tid, block_dims  # type: ignore
        )
        gil += gil_c
        Hil += Hil_c
    gis, His = (
        wp.tile(gil, preserve_type=True),  # pyright: ignore[reportArgumentType]
        wp.tile(Hil, preserve_type=True),  # pyright: ignore[reportArgumentType]
    )
    gi, Hi = (
        wp.tile_reduce(wp.add, gis)[0],  # pyright: ignore[reportIndexIssue]
        wp.tile_reduce(wp.add, His)[0],  # pyright: ignore[reportIndexIssue]
    )
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


def linearize_constraints(
    fem: FemElastoDynamics, contact: ContactDynamics, params: Params
):
    """TODO: Linearize contact constraints at current iterate."""
    pass


def check_convergence(
    fem: FemElastoDynamics, contact: ContactDynamics, params: Params
) -> bool:
    """TODO: Check gradient norm convergence (elastic + momentum + contact)."""
    return False


def prepare_subproblem(
    fem: FemElastoDynamics, contact: ContactDynamics, params: Params
):
    """TODO: Assemble block-diagonal Hessian, update penalty parameter."""
    pass


def initialize_solve(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
):
    """Initialize the VBD solve by updating contact constraint set and restoring feasibility.

    Mirrors ``pbat::sim::algorithm::vbd::InitializeSolve``.
    Called once after :meth:`FemElastoDynamics.setup_time_integration_optimization`,
    before the first call to :func:`solve`.
    """
    contact.ogc.compute_query_radius()
    contact.update_constraint_set(fem.xt)
    contact.restore_feasibility(fem.data.x)


def finalize_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
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
    contact.restore_feasibility(fem.data.x)
    contact.update_constraint_set(fem.data.x)


def iterate(fem: FemElastoDynamics, contact: ContactDynamics, params: Params):
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
                inputs=[p_begin, fem.data, contact.data, params.data, h2],
                block_dim=block_dim,
            )


def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    prepare_subproblem(fem, contact, params)
    for kp in range(n_subproblem_max_iters):
        contact.update_dual(
            fem.data.x,
            request_slack_update=True,
            request_decay_update=False,
            request_lagrange_multiplier_update=False,
        )
        iterate(fem, contact, params)
    finalize_subproblem(fem, contact, params)


class VbdSolver:

    def __init__(self):
        self._cuda_graph = None

    def solve(
        self, fem: FemElastoDynamics, params: Params, contact: ContactDynamics
    ) -> bool:
        converged = False
        if self._cuda_graph is None:
            with wp.ScopedCapture() as capture:
                initialize_solve(fem, contact, params)
                for k in range(params.data.n_max_iters):
                    linearize_constraints(fem, contact, params)
                    if check_convergence(fem, contact, params):
                        converged = True
                        break
                    prepare_subproblem(fem, contact, params)
                    solve_subproblem(fem, contact, params)
                    finalize_subproblem(fem, contact, params)
                fem.back_substitute_velocities()
            self._cuda_graph = capture
        else:
            wp.capture_launch(self._cuda_graph.graph)  # type: ignore
        return converged

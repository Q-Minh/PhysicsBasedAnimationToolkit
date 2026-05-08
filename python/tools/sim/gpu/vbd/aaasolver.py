from typing import Tuple

import numpy as np
import warp as wp

from python import fem
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from .params import Params, ParamsData
from .kernels import (
    local_elastic_derivatives,
    add_inertia_derivatives,
    integrate_positions,
)
from .solver import (
    check_convergence,
    linearize_constraints,
    prepare_subproblem,
    finalize_subproblem,
)
from ..contact.dynamics import MeshDynamics


@wp.func
def adapt_stencil_gradient_acceleration_parameter(
    kp: wp.int32,
    i: wp.int32,
    xi: wp.vec3f,
    gi: wp.vec3f,
    Hi: wp.mat33f,
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    is_surface_node: bool,
    eps: wp.float32,
) -> wp.float32:
    # TODO: Check if the node is a surface node
    grp = wp.int32(1) if is_surface_node else wp.int32(0)
    betaG = params.betaG[i, grp]
    if kp > 0:
        rhohat = params.rhohat[grp]
        gammaup = params.gammaup[grp]
        gammadown = params.gammadown[grp]
        ngk = wp.norm_l2(gi)
        gkm1 = params.gk[i]  # Previous gradient
        ngkm1 = wp.norm_l2(gkm1)
        ndgkm1 = wp.norm_l2(gi - gkm1)
        xk = params.xk[i]  # Previous position
        ndxkm1 = wp.max(
            wp.norm_l2(xi - xk),
            eps,  # pyright: ignore[reportCallIssue, reportArgumentType]
        )
        L = params.Hnk[i] + ngk / ndxkm1
        rho = ndgkm1 / wp.max(L * ndxkm1, eps)
        if ngk > ngkm1:
            betaG *= gammadown
        elif rho > rhohat:
            betaG += (wp.float32(1) - betaG) * gammaup
        params.betaG[i, grp] = betaG
    params.gk[i] = gi  # Store current gradient
    params.xk[i] = xi  # Store current position
    params.Hnk[i] = wp.sqrt(
        wp.ddot(Hi, Hi)  # pyright: ignore[reportArgumentType]
    )  # Store Hessian norm
    return betaG


@wp.func
def compute_thread_local_stencil_gradient_augmentation(
    local_tid: wp.int32,
    block_dims: wp.int32,
    i: wp.int32,
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    is_surface_node: bool,
):
    """Compute thread-local stencil gradient augmentation for vertex i."""
    ai = wp.vec3f()
    nbegin = params.GVVp[i]
    nend = params.GVVp[i + 1]
    n_neighbours = nend - nbegin
    for jlocal in range(local_tid, n_neighbours, block_dims):
        j = params.GVVadj[nbegin + jlocal]
        # TODO: Handle surface node's surface stencil
        # if is_surface_node and params.bSurfaceStencilSurfaceNeighboursOnly and fem.dynamic_meshes[j] < 0:
        #     continue
        ai += params.gk[j]
    return ai


@wp.kernel
def _accelerated_vertex_solve_kernel(
    pbegin: wp.int32,
    k: wp.int32,
    kp: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    h2: float,
):
    """Process one vertex in the current color partition with acceleration."""
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
    gis, His = (
        wp.tile(gil, preserve_type=True),  # pyright: ignore[reportArgumentType]
        wp.tile(Hil, preserve_type=True),  # pyright: ignore[reportArgumentType]
    )
    gi, Hi = (
        wp.tile_reduce(wp.add, gis)[0],  # pyright: ignore[reportIndexIssue]
        wp.tile_reduce(wp.add, His)[0],  # pyright: ignore[reportIndexIssue]
    )
    gi *= h2  # pyright: ignore[reportOperatorIssue]
    Hi *= h2  # pyright: ignore[reportOperatorIssue]
    # TODO: AccumulateContactEnergy(i, params.xb, contact, gi, Hi)
    eps = wp.float32(1e-10)  # pyright: ignore[reportArgumentType]
    # Augment residual
    is_surface_node = False  # multi_mesh.GXV[i] >= 0
    ai = wp.vec3f()
    if k > 0 or kp > 0:  # type: ignore
        ail = compute_thread_local_stencil_gradient_augmentation(
            local_tid=local_tid,  # pyright: ignore[reportArgumentType]
            block_dims=block_dims,  # pyright: ignore[reportArgumentType]
            i=i,
            params=params,
            is_surface_node=is_surface_node,
        )
        ais = wp.tile(ail, preserve_type=True)  # pyright: ignore[reportArgumentType]
        ai = wp.tile_sum(ais)[0]  # pyright: ignore[reportIndexIssue]
    if local_tid == 0:
        # Add inertia derivatives (K = m, already in position space)
        gi, Hi = add_inertia_derivatives(
            mi, xtildei, xi, gi, Hi
        )  # pyright: ignore[reportArgumentType]
        # Adapt stencil gradient acceleration parameter using the total gradient
        betaG = adapt_stencil_gradient_acceleration_parameter(
            kp=kp,
            i=i,
            xi=xi,
            gi=gi,
            Hi=Hi,
            params=params,
            eps=eps,
            is_surface_node=is_surface_node,
        )
        lam = (
            betaG
            * wp.dot(gi, ai)  # pyright: ignore[reportArgumentType, reportCallIssue]
            / wp.max(
                wp.dot(ai, ai),  # pyright: ignore[reportCallIssue, reportArgumentType]
                eps,
            )
        )
        lam = wp.max(lam, wp.float32(0))
        gi += lam * ai  # pyright: ignore[reportOperatorIssue]
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


def iterate(
    k: int,
    kp: int,
    fem: FemElastoDynamics,
    params: Params,
):
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
                kernel=_accelerated_vertex_solve_kernel,
                dim=n_verts_in_partition * block_dim,
                inputs=[p_begin, k, kp, fem.data, params.data, h2],
                block_dim=block_dim,
            )


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


def solve_subproblem(
    k: int,
    fem: FemElastoDynamics,
    params: Params,
    contact: MeshDynamics,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    for kp in range(n_subproblem_max_iters):
        contact.update_dual(
            fem.data.x,
            request_slack_update=True,
            request_decay_update=False,
            request_lagrange_multiplier_update=False,
        )
        iterate(k, kp, fem, params)


class AaaVbdSolver:

    def __init__(self):
        self._cuda_graph = None
        self._k = wp.array(
            [0], dtype=wp.int32
        )  # Iteration counter for acceleration schedule

    def initialize_solve(
        self, fem: FemElastoDynamics, params: Params, contact: MeshDynamics
    ):
        self.initialize_solve(fem, params, contact)
        self._k.fill_(0)

    def solve(
        self, fem: FemElastoDynamics, params: Params, contact: MeshDynamics
    ) -> bool:
        converged = False
        if self._cuda_graph is None:
            with wp.ScopedCapture() as capture:
                initialize_solve(fem, params, contact)
                for k in range(params.data.n_max_iters):
                    linearize_constraints(fem, params)
                    if check_convergence(fem, params):
                        converged = True
                        break
                    prepare_subproblem(fem, params)
                    solve_subproblem(k, fem, params, contact)
                    finalize_subproblem(fem, params, contact)
            self._cuda_graph = capture
        else:
            wp.capture_launch(self._cuda_graph.graph)  # type: ignore
        fem.back_substitute_velocities()
        return converged

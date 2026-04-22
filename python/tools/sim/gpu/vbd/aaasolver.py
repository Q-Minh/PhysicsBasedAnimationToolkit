from typing import Tuple

import warp as wp
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


@wp.func
def compute_stencil_gradient_augmentation(
    k: wp.int32,
    kp: wp.int32,
    i: wp.int32,
    xi: wp.vec3f,
    gi: wp.vec3f,
    Hi: wp.mat33f,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
):
    """Compute stencil gradient augmentation for vertex i."""
    # TODO: Check if the node is a surface node
    is_surface_node = False  # fem.dynamic_meshes[i] >= 0
    ii = 1 if is_surface_node else 0
    betaG = params.betaG[i, ii]  # Stencil gradient coefficient
    eps = 1e-10

    # Adapt stencil gradient acceleration parameter using the total gradient
    if kp > 0:
        rhohat = params.rhohat[ii]
        gammaup = params.gammaup[ii]
        gammadown = params.gammadown[ii]
        ngk = wp.norm_l2(gi)
        gkm1 = params.gk[i]  # Previous gradient
        ngkm1 = wp.norm_l2(gkm1)
        ndgkm1 = wp.norm_l2(gi - gkm1)

        xk = params.xk[i]  # Previous position
        ndxkm1 = wp.max(wp.norm_l2(xi - xk), eps)

        L = params.Hnk[i] + ngk / ndxkm1
        rho = ndgkm1 / wp.max(L * ndxkm1, eps)

        if ngk > ngkm1:
            betaG *= gammadown
        elif rho > rhohat:
            betaG += (wp.float32(1) - betaG) * gammaup

    params.betaG[i, ii] = betaG  # Update the parameter
    params.gk[i] = gi  # Store current gradient
    params.xk[i] = xi  # Store current position
    params.Hnk[i] = wp.sqrt(
        wp.ddot(Hi, Hi)  # pyright: ignore[reportArgumentType]
    )  # Store Hessian norm

    # Compute weighted stencil gradient augmentation
    ai = wp.vec3f()
    if kp == 0 and k == 0:
        return ai

    nbegin = params.GVVp[i]
    nend = params.GVVp[i + 1]
    for n in range(nbegin, nend):
        j = params.GVVadj[n]
        # TODO: Handle surface node's surface stencil
        # if is_surface_node and params.bSurfaceStencilSurfaceNeighboursOnly and fem.dynamic_meshes[j] < 0:
        #     continue
        ai += params.gk[j]

    # Add augmentation
    lam = (
        betaG
        * wp.dot(gi, ai)  # pyright: ignore[reportArgumentType, reportCallIssue]
        / wp.max(
            wp.dot(ai, ai), eps  # pyright: ignore[reportArgumentType, reportCallIssue]
        )
    )
    lam = wp.max(lam, wp.float32(0))
    return lam * ai  # pyright: ignore[reportOperatorIssue]


@wp.kernel
def _accelerated_vertex_solve_kernel(
    pbegin: int,
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
    gil *= h2  # pyright: ignore[reportOperatorIssue]
    Hil *= h2  # pyright: ignore[reportOperatorIssue]
    gs, Hs = (
        wp.tile(gil, preserve_type=True),  # pyright: ignore[reportArgumentType]
        wp.tile(Hil, preserve_type=True),  # pyright: ignore[reportArgumentType]
    )
    gi, Hi = (
        wp.tile_reduce(wp.add, gs)[0],  # pyright: ignore[reportIndexIssue]
        wp.tile_reduce(wp.add, Hs)[0],  # pyright: ignore[reportIndexIssue]
    )
    # TODO: AccumulateContactEnergy(i, params.xb, contact, gi, Hi)
    if local_tid > 0:
        return
    # Add inertia derivatives (K = m, already in position space)
    gi, Hi = add_inertia_derivatives(
        mi, xtildei, xi, gi, Hi
    )  # pyright: ignore[reportArgumentType]
    gi += compute_stencil_gradient_augmentation(
        k, kp, i, xi, gi, Hi, fem, params
    )  # pyright: ignore[reportOperatorIssue]
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


def initialize_solve(
    fem: FemElastoDynamics,
    params: Params,
):
    # TODO: Support resetting betaG to betaG0 depending on a warm start mask (see "vbd/Core.h")
    pass


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


def solve_subproblem(
    k: int,
    fem: FemElastoDynamics,
    params: Params,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    # TODO: prepare_subproblem(fem, params)
    prepare_subproblem(fem, params)
    for kp in range(n_subproblem_max_iters):
        iterate(k, kp, fem, params)
    # TODO: finalize_subproblem(fem, params)
    finalize_subproblem(fem, params)


def solve(
    fem: FemElastoDynamics,
    params: Params,
    capture: wp.ScopedCapture | None = None,
    request_capture: bool = False,
) -> Tuple[bool, wp.ScopedCapture]:
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
        if not request_capture:
            solve_subproblem(k, fem, params)
        elif capture is None:
            with wp.ScopedCapture() as scap:
                solve_subproblem(k, fem, params)
            capture = scap
        else:
            wp.capture_launch(capture.graph)  # pyright: ignore[reportArgumentType]
    fem.back_substitute_velocities()
    return converged, capture  # pyright: ignore[reportReturnType]

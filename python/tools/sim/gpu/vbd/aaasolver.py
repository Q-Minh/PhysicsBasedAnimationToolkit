from typing import Tuple

import numpy as np
import warp as wp
from ..common import profiling

from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from ..contact.dynamics import (
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
)
from ..contact.mesh.cd import ContactDetection
from .params import Params, ParamsData
from .kernels import (
    local_elastic_derivatives,
    local_contact_derivatives,
    add_inertia_or_external_load_derivatives,
    integrate_positions,
)
from .solver import (
    IterationSchedule,
    contact_partners_current,
    initialize_solve,
    solve_outer_loop,
    linearize_constraints,
    prepare_subproblem,
    finalize_subproblem,
)


@wp.func
def adapt_stencil_gradient_acceleration_parameter(
    kp: wp.int32,
    i: wp.int32,
    xi: wp.vec3f,
    gi: wp.vec3f,
    Hi: wp.mat33f,
    xk: wp.array[wp.vec3f],
    gk: wp.array[wp.vec3f],
    Hnk: wp.array[wp.float32],
    betaG: wp.array2d[wp.float32],
    xki: wp.vec3f,
    gkm1: wp.vec3f,
    Hnki: wp.float32,
    betaGi: wp.float32,
    rhohat: wp.vec2f,
    gammaup: wp.vec2f,
    gammadown: wp.vec2f,
    betaup: wp.float32,
    lipschitz_normalization: wp.int32,
    grp: wp.int32,
    eps: wp.float32,
) -> wp.float32:
    """Adapt vertex i's acceleration parameter and store its gradient, position and Hessian norm.

    ``xki``, ``gkm1``, ``Hnki`` and ``betaGi`` are vertex i's stored state, which the caller loads early so
    the loads overlap the derivative gathers; ``grp`` selects the interior (0) or surface (1) parameters.

    Beta shrinks by ``gammadown`` when the gradient grows, and otherwise grows toward ``betaup`` by
    ``gammaup`` when ``rho > rhohat``. ``rho = |g_k - g_{k-1}| / (L |x_k - x_{k-1}|)`` with the Lipschitz
    estimate ``L = |H_i|_F + |g_k| / |x_k - x_{k-1}|``, or ``|g_k - g_{k-1}| / |x_k - x_{k-1}|`` without
    ``lipschitz_normalization`` (then ``rhohat`` has stiffness units). C++ ``vbd::Iterate`` has
    ``betaup = 1`` and always normalizes.
    """
    if kp > 0:
        rhohati = rhohat[grp]  # type: ignore
        gammaupi = gammaup[grp]  # type: ignore
        gammadowni = gammadown[grp]  # type: ignore
        ngk = wp.norm_l2(gi)
        ngkm1 = wp.norm_l2(gkm1)
        ndgkm1 = wp.norm_l2(gi - gkm1)
        ndxkm1 = wp.max(
            wp.norm_l2(xi - xki),
            eps,  # pyright: ignore[reportCallIssue, reportArgumentType]
        )
        rho = ndgkm1 / ndxkm1
        if lipschitz_normalization != 0:
            L = Hnki + ngk / wp.max(ndxkm1, eps)
            rho = ndgkm1 / wp.max(L * ndxkm1, eps)
        if ngk > ngkm1:
            betaGi *= gammadowni
        elif rho > rhohati:  # type: ignore
            betaGi += (betaup - betaGi) * gammaupi
        betaG[i, grp] = betaGi  # type: ignore
    gk[i] = gi  # Store current gradient # type: ignore
    xk[i] = xi  # Store current position # type: ignore
    Hnk[i] = wp.sqrt(wp.ddot(Hi, Hi))  # Store Hessian norm # type: ignore
    return betaGi  # type: ignore


@wp.func
def compute_thread_local_stencil_gradient_augmentation(
    local_tid: wp.int32,
    block_dims: wp.int32,
    i: wp.int32,
    gk: wp.array[wp.vec3f],
    GVVp: wp.array[wp.int32],
    GVVadj: wp.array[wp.int32],
    GXV: wp.array[wp.int32],
    is_surface_node: bool,
    surface_neighbours_only: wp.int32,
):
    """Compute thread-local stencil gradient augmentation for vertex i: the sum of its neighbours'
    gradients ``gk[j]``. With ``surface_neighbours_only`` (``Params.surface_stencil_surface_neighbours_only``),
    surface vertices sum only their surface neighbours, as in C++ ``vbd::Iterate``."""
    ai = wp.vec3f()
    nbegin = GVVp[i]
    nend = GVVp[i + 1]
    n_neighbours = nend - nbegin
    for jlocal in range(local_tid, n_neighbours, block_dims):  # type: ignore
        j = GVVadj[nbegin + jlocal]
        # Load gk[j] unconditionally, so it doesn't wait for the GXV[j] load that decides whether it
        # counts.
        gkj = gk[j]
        skip = is_surface_node and surface_neighbours_only != 0 and GXV[j] < 0  # type: ignore
        ai += wp.where(skip, wp.vec3f(), gkj)  # type: ignore
    return ai


@wp.kernel
def _accelerated_vertex_solve_kernel(
    pbegin: wp.int32,
    k: wp.int32,
    kp: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    contact: ContactDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    h2: float,
    color: wp.int32,
    current_contacts: wp.int32,
):
    """Process one vertex in the current color partition (``color``) with acceleration.

    ``current_contacts``: see ``solver._vertex_solve_kernel``.
    """
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid // block_dims  # pyright: ignore[reportOperatorIssue]
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
    vi = contact.meshes.GXV[i]
    is_surface_node = vi >= 0
    # Load the vertex's acceleration state now: the tile reductions below are barriers, so loads issued
    # after them by thread 0 would wait on memory with no other work to hide it.
    grp = wp.int32(1) if is_surface_node else wp.int32(0)
    betaGi = params.betaG[i, grp]  # type: ignore
    gkm1 = params.gk[i]  # pyright: ignore[reportIndexIssue]
    xki = params.xk[i]  # pyright: ignore[reportIndexIssue]
    Hnki = params.Hnk[i]  # pyright: ignore[reportIndexIssue]
    # Accumulate elastic energy derivatives
    gil, Hil = local_elastic_derivatives(
        i,
        fem.x,
        fem.E,
        fem.wg,
        fem.GNeg,
        fem.mug,
        fem.lambdag,
        params.GVGp,
        params.GVGadj,
        local_tid,  # pyright: ignore[reportArgumentType]
        block_dims,  # pyright: ignore[reportArgumentType]
    )
    gil *= h2  # type: ignore
    Hil *= h2  # type: ignore
    # Only surface vertices with contact pairs have contact terms (MeshDynamicsData.in_contact).
    gather_contacts = False
    if is_surface_node:
        gather_contacts = contact.in_contact[vi] != 0
    if gather_contacts:
        gil_c, Hil_c = local_contact_derivatives(
            i,
            vi,
            fem.xt,
            params.xb,
            fem.x,
            params.colors,
            color,
            current_contacts,
            contact,
            local_tid,  # type: ignore
            block_dims,  # type: ignore
        )
        gil += gil_c  # type: ignore
        Hil += Hil_c  # type: ignore
    gis, His = (
        wp.tile(gil, preserve_type=True),  # pyright: ignore[reportArgumentType]
        wp.tile(Hil, preserve_type=True),  # pyright: ignore[reportArgumentType]
    )
    gi, Hi = (
        wp.tile_reduce(wp.add, gis)[0],  # pyright: ignore[reportIndexIssue]
        wp.tile_reduce(wp.add, His)[0],  # pyright: ignore[reportIndexIssue]
    )
    eps = wp.float32(1e-10)  # pyright: ignore[reportArgumentType]
    # Augment residual
    ai = wp.vec3f()
    if k > 0 or kp > 0:  # type: ignore
        ail = compute_thread_local_stencil_gradient_augmentation(
            local_tid=local_tid,  # pyright: ignore[reportArgumentType]
            block_dims=block_dims,  # pyright: ignore[reportArgumentType]
            i=i,
            gk=params.gk,
            GVVp=params.GVVp,
            GVVadj=params.GVVadj,
            GXV=contact.meshes.GXV,
            is_surface_node=is_surface_node,
            surface_neighbours_only=params.surface_stencil_surface_neighbours_only,
        )
        ais = wp.tile(ail, preserve_type=True)  # pyright: ignore[reportArgumentType]
        ai = wp.tile_sum(ais)[0]  # pyright: ignore[reportIndexIssue]
    if local_tid == 0:
        # Add inertia (K = m, already in position space) or, for static problems, external load derivatives
        gi, Hi = add_inertia_or_external_load_derivatives(
            fem.is_static, mi, xtildei, fem.fext[i], xi, gi, Hi
        )  # pyright: ignore[reportArgumentType]
        # Adapt stencil gradient acceleration parameter using the total gradient
        betaG = adapt_stencil_gradient_acceleration_parameter(
            kp=kp,
            i=i,
            xi=xi,
            gi=gi,
            Hi=Hi,
            xk=params.xk,
            gk=params.gk,
            Hnk=params.Hnk,
            betaG=params.betaG,
            xki=xki,
            gkm1=gkm1,
            Hnki=Hnki,
            betaGi=betaGi,
            rhohat=params.rhohat,
            gammaup=params.gammaup,
            gammadown=params.gammadown,
            betaup=params.betaup,
            lipschitz_normalization=params.lipschitz_normalization,
            grp=grp,
            eps=eps,
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


@profiling.scoped("aaavbd.sweep")
def iterate(
    k: int,
    kp: int,
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
):
    """One VBD Gauss-Seidel sweep over all color partitions (see ``solver.iterate`` on contacts)."""
    h2 = fem.h2
    current_contacts = contact_partners_current(contact)
    # Copy current positions to buffer (for contact lagging)
    wp.copy(dest=params.data.xb, src=fem.data.x)
    # Process each color partition sequentially
    Pptr = params.Pptr
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
                inputs=[p_begin, k, kp, fem.data, contact.data, params.data, h2, p, current_contacts],
                block_dim=block_dim,
            )


def subproblem_iteration(
    k: int, fem: FemElastoDynamics, contact: ContactDynamics, params: Params, kp: int
):
    """Iteration ``kp`` of subproblem ``k``: slack dual update, then one accelerated sweep.

    The sweep only distinguishes ``k == 0 and kp == 0`` (no stencil gradient augmentation yet) and
    ``kp == 0`` (no acceleration parameter adaptation yet) from later iterations.
    """
    contact.update_dual(
        fem.data.x,
        fem.xt,
        request_slack_update=True,
        request_decay_update=False,
        request_lagrange_multiplier_update=False,
    )
    iterate(k, kp, fem, contact, params)


@profiling.scoped("aaavbd.subproblem")
def solve_subproblem(
    k: int, fem: FemElastoDynamics, contact: ContactDynamics, params: Params
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    for kp in range(n_subproblem_max_iters):
        subproblem_iteration(k, fem, contact, params, kp)


class AaaVbdSolver:

    def __init__(self):
        pass

    def solve(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        cd: ContactDetection,
        params: Params,
    ) -> bool:
        initialize_solve(fem, contact, cd, params)
        converged = solve_outer_loop(
            fem, contact, cd, params, self.schedule(fem, contact, params)
        )
        fem.back_substitute_velocities()
        cd.on_time_step_ended()
        return converged

    def schedule(
        self, fem: FemElastoDynamics, contact: ContactDynamics, params: Params
    ) -> IterationSchedule:
        """Only the first iteration of a subproblem differs (no acceleration adaptation yet)."""
        return IterationSchedule(
            lambda k, kp: subproblem_iteration(k, fem, contact, params, kp),
            prefix=1,
            scope="aaavbd.subproblem",
        )

    @property
    def supports_graph_capture(self):
        return True
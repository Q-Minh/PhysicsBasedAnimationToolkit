import warp as wp
import warp.fem.linalg
from .. import types
from ..elasticity.fem import FemElastoDynamicsData
from ..elasticity.snh import snh_grad_and_hess, snh_hess
from ..elasticity.chain import gradient_segment_wrt_dofs, hessian_block_wrt_dofs
from ..contact.dynamics import MeshDynamicsData as ContactDynamicsData
from ..contact.dynamics import ConstraintSetData
from ..contact.ogc import OgcData, ContactPairsData
from ..contact import halfedges
from .params import (
    ParamsData,
    VLS_SOLVER_INVERSE,
    VLS_SOLVER_LLT,
    VLS_SOLVER_QR,
    VLS_SOLVER_EVD,
)


@wp.func
def accumulate_contact_al(
    c_n: wp.float32,
    c_f: wp.vec2f,
    wi: wp.float32,
    gamma: wp.float32,
    n: wp.vec3f,
    t: wp.vec3f,
    b: wp.vec3f,
    lambda_n: wp.float32,
    lambda_f: wp.vec2f,
    sigma_n: wp.float32,
    sigma_f: wp.float32,
):
    """Accumulate augmented Lagrangian contact gradient and Hessian for a single vertex.

    Given the full contact constraint values ``c_n`` (normal, with slack already subtracted)
    and ``c_f`` (tangential), the stencil weight ``wi`` for the vertex being solved, and the
    AL parameters, accumulates:

    .. code-block::

        gi += gamma * wi * ((sigma_n*c_n - lambda_n)*n + sum_k((sigma_f*c_f - lambda_f)[k]*tb[k]))
        Hi += gamma * wi^2 * (sigma_n*outer(n,n) + sigma_f*(outer(t,t) + outer(b,b)))

    Args:
        c_n: Normal constraint value ``dot(d, n) - dmin - s``.
        c_f: Tangential constraint value ``[dot(u, t), dot(u, b)]``.
        wi: Stencil weight for this vertex (``+1`` for u-side, ``-bary`` for v-side).
        gamma: Constraint activity decay factor.
        n: Contact normal.
        t: Contact first tangent.
        b: Contact bitangent.
        lambda_n: Normal Lagrange multiplier estimate.
        lambda_f: Tangential Lagrange multiplier estimate.
        sigma_n: Normal AL penalty.
        sigma_f: Tangential AL penalty.

    Returns:
        ``(gi, Hi)`` gradient and Hessian contributions for this contact.
    """
    dEn = sigma_n * c_n - lambda_n
    dEf = sigma_f * c_f - lambda_f  # pyright: ignore[reportOperatorIssue]
    gi = gamma * wi * (dEn * n + dEf[0] * t + dEf[1] * b)
    Hi = (
        gamma
        * (wi * wi)
        * (  # type: ignore
            sigma_n * wp.outer(n, n) + sigma_f * (wp.outer(t, t) + wp.outer(b, b))  # type: ignore
        )
    )
    return gi, Hi


@wp.func
def edge_closest_point(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    i_he: wp.int32,
    j_he: wp.int32,
    bary: wp.float32,
):
    """Interpolate closest point and its previous-step position on a half-edge.

    Args:
        x: Current vertex positions.
        xt: Previous time-step vertex positions.
        i_he: Global index of the incoming (source) vertex of the half-edge.
        j_he: Global index of the outgoing (target) vertex of the half-edge.
        bary: Barycentric parameter ``t`` along the edge (``0`` = source, ``1`` = target).

    Returns:
        ``(xcp, xtcp)`` — closest point at current and previous step.
    """
    one_m = wp.float32(1) - bary
    xcp = one_m * x[i_he] + bary * x[j_he]
    xtcp = one_m * xt[i_he] + bary * xt[j_he]
    return xcp, xtcp


@wp.func
def tri_closest_point(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    finds: wp.vec3i,
    b0: wp.float32,
    b1: wp.float32,
):
    """Interpolate closest point and its previous-step position on a triangle.

    Args:
        x: Current vertex positions.
        xt: Previous time-step vertex positions.
        finds: Global vertex indices of the triangle (3-vector).
        b0: First barycentric coordinate.
        b1: Second barycentric coordinate (third = ``1 - b0 - b1``).

    Returns:
        ``(xcp, xtcp)`` — closest point at current and previous step.
    """
    b2 = wp.float32(1) - b0 - b1
    xcp = b0 * x[finds[0]] + b1 * x[finds[1]] + b2 * x[finds[2]]  # type: ignore
    xtcp = b0 * xt[finds[0]] + b1 * xt[finds[1]] + b2 * xt[finds[2]]  # type: ignore
    return xcp, xtcp


@wp.func
def contact_constraints(
    d: wp.vec3f,
    u: wp.vec3f,
    n: wp.vec3f,
    t: wp.vec3f,
    b: wp.vec3f,
    s: wp.float32,
    dmin: wp.float32,
):
    """Compute normal and tangential contact constraints from gap and relative displacement.

    Args:
        d: Gap vector ``x_u - x_v`` (current positions).
        u: Relative displacement ``(x_u - xt_u) - (x_v - xt_v)`` over the time step.
        n: Contact normal (pointing from v to u).
        t: First contact tangent.
        b: Contact bitangent.
        s: AL inequality slack.
        dmin: Minimum separation distance margin.

    Returns:
        ``(c_n, c_f)`` — scalar normal constraint and 2-vector tangential constraint.
    """
    c_n = wp.dot(d, n) - dmin - s  # type: ignore
    c_f = wp.vec2f(wp.dot(u, t), wp.dot(u, b))  # type: ignore
    return c_n, c_f


@wp.func
def _contact_vv_fwd(
    k: wp.int32,
    i: wp.int32,
    j: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    cvv: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    vv_bases: wp.array[wp.mat33f],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    dmin: wp.float32,
):
    """Gradient and Hessian contribution of a single forward vertex-vertex contact for vertex i."""
    ntb = vv_bases[k]
    n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
    s = cvv.s[k]
    gamma = cvv.gamma[k]
    lambda_n = cvv.lambda_n[k]
    lambda_f = cvv.lambda_f[k]
    d = fem.x[i] - fem.x[j]
    u = (fem.x[i] - fem.xt[i]) - (fem.x[j] - fem.xt[j])
    c_n, c_f = contact_constraints(d, u, n, t, b, s, dmin)  # type: ignore
    return accumulate_contact_al(c_n, c_f, wp.float32(1), gamma, n, t, b, lambda_n, lambda_f, sigma_n, sigma_f)  # type: ignore


@wp.func
def _contact_vv_rev(
    k: wp.int32,
    i: wp.int32,
    j: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    cvv: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    vv_bases: wp.array[wp.mat33f],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    dmin: wp.float32,
):
    """Gradient and Hessian contribution of a single reverse vertex-vertex contact for vertex i (v-side)."""
    ntb = vv_bases[k]
    n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
    s = cvv.s[k]
    gamma = cvv.gamma[k]
    lambda_n = cvv.lambda_n[k]
    lambda_f = cvv.lambda_f[k]
    d = fem.x[j] - fem.x[i]  # forward gap: x_u - x_v = x_j - x_i (we are v)
    u = (fem.x[j] - fem.xt[j]) - (fem.x[i] - fem.xt[i])
    c_n, c_f = contact_constraints(d, u, n, t, b, s, dmin)  # type: ignore
    return accumulate_contact_al(c_n, c_f, wp.float32(-1), gamma, n, t, b, lambda_n, lambda_f, sigma_n, sigma_f)  # type: ignore


@wp.func
def _contact_ve_fwd(
    k: wp.int32,
    i: wp.int32,
    he: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    F: wp.array[wp.vec3i],
    cve: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    ve_bases: wp.array[wp.mat33f],
    ve_bary: wp.array[wp.float32],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    dmin: wp.float32,
):
    """Gradient and Hessian contribution of a single forward vertex-edge contact for vertex i (u-side)."""
    ntb = ve_bases[k]
    bary = ve_bary[k]  # type: ignore
    n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
    s = cve.s[k]
    gamma = cve.gamma[k]
    lambda_n = cve.lambda_n[k]
    lambda_f = cve.lambda_f[k]
    i_he = halfedges.incoming_vertex(F, he)
    j_he = halfedges.outgoing_vertex(F, he)
    xcp, xtcp = edge_closest_point(fem.x, fem.xt, i_he, j_he, bary)  # type: ignore
    d = fem.x[i] - xcp
    u = (fem.x[i] - fem.xt[i]) - (xcp - xtcp)
    c_n, c_f = contact_constraints(d, u, n, t, b, s, dmin)  # type: ignore
    return accumulate_contact_al(c_n, c_f, wp.float32(1), gamma, n, t, b, lambda_n, lambda_f, sigma_n, sigma_f)  # type: ignore


@wp.func
def _contact_ve_rev(
    k: wp.int32,
    i: wp.int32,
    i_he: wp.int32,
    j_he: wp.int32,
    j_global: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    cve: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    ve_bases: wp.array[wp.mat33f],
    ve_bary: wp.array[wp.float32],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    dmin: wp.float32,
):
    """Gradient and Hessian contribution of a single reverse vertex-edge contact for vertex i (v-side, on edge)."""
    ntb = ve_bases[k]
    bary = ve_bary[k]  # type: ignore
    n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
    s = cve.s[k]
    gamma = cve.gamma[k]
    lambda_n = cve.lambda_n[k]
    lambda_f = cve.lambda_f[k]
    xcp, xtcp = edge_closest_point(fem.x, fem.xt, i_he, j_he, bary)  # type: ignore
    d = fem.x[j_global] - xcp
    u = (fem.x[j_global] - fem.xt[j_global]) - (xcp - xtcp)
    c_n, c_f = contact_constraints(d, u, n, t, b, s, dmin)  # type: ignore
    one_m_bary = wp.float32(1) - bary  # type: ignore
    wi = -one_m_bary * wp.float32(i == i_he) - bary * wp.float32(i == j_he)  # type: ignore
    return accumulate_contact_al(c_n, c_f, wi, gamma, n, t, b, lambda_n, lambda_f, sigma_n, sigma_f)  # type: ignore


@wp.func
def _contact_vf_fwd(
    k: wp.int32,
    i: wp.int32,
    f: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    F: wp.array[wp.vec3i],
    cvf: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    vf_bases: wp.array[wp.mat33f],
    vf_bary: wp.array[wp.vec2f],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    dmin: wp.float32,
):
    """Gradient and Hessian contribution of a single forward vertex-triangle contact for vertex i (u-side)."""
    ntb = vf_bases[k]
    bary = vf_bary[k]  # type: ignore
    n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
    s = cvf.s[k]
    gamma = cvf.gamma[k]
    lambda_n = cvf.lambda_n[k]
    lambda_f = cvf.lambda_f[k]
    finds = F[f]
    xcp, xtcp = tri_closest_point(fem.x, fem.xt, finds, bary[0], bary[1])  # type: ignore
    d = fem.x[i] - xcp
    u = (fem.x[i] - fem.xt[i]) - (xcp - xtcp)
    c_n, c_f = contact_constraints(d, u, n, t, b, s, dmin)  # type: ignore
    return accumulate_contact_al(c_n, c_f, wp.float32(1), gamma, n, t, b, lambda_n, lambda_f, sigma_n, sigma_f)  # type: ignore


@wp.func
def _contact_vf_rev(
    c: wp.int32,
    i: wp.int32,
    f: wp.int32,
    j_global: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    F: wp.array[wp.vec3i],
    cvf: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    vf_bases: wp.array[wp.mat33f],
    vf_bary: wp.array[wp.vec2f],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    dmin: wp.float32,
):
    """Gradient and Hessian contribution of a single reverse vertex-triangle contact for vertex i (v-side, on triangle)."""
    ntb = vf_bases[c]
    bary = vf_bary[c]  # type: ignore
    n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
    s = cvf.s[c]
    gamma = cvf.gamma[c]
    lambda_n = cvf.lambda_n[c]
    lambda_f = cvf.lambda_f[c]
    b0 = bary[0]  # type: ignore
    b1 = bary[1]  # type: ignore
    b2 = wp.float32(1) - b0 - b1
    finds = F[f]
    xcp, xtcp = tri_closest_point(fem.x, fem.xt, finds, b0, b1)  # type: ignore
    d = fem.x[j_global] - xcp
    u = (fem.x[j_global] - fem.xt[j_global]) - (xcp - xtcp)
    c_n, c_f = contact_constraints(d, u, n, t, b, s, dmin)  # type: ignore
    wi = (
        -b0 * wp.float32(i == finds[0])  # type: ignore
        - b1 * wp.float32(i == finds[1])  # type: ignore
        - b2 * wp.float32(i == finds[2])  # type: ignore
    )
    return accumulate_contact_al(c_n, c_f, wi, gamma, n, t, b, lambda_n, lambda_f, sigma_n, sigma_f)  # type: ignore


@wp.func
def _contact_ee_fwd(
    l: wp.int32,
    i: wp.int32,
    i_he: wp.int32,
    j_he: wp.int32,
    he2: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    F: wp.array[wp.vec3i],
    cee: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    ee_bases: wp.array[wp.mat33f],
    ee_bary: wp.array[wp.vec2f],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    dmin: wp.float32,
):
    """Gradient and Hessian contribution of a single forward edge-edge contact for vertex i (u-side, on he)."""
    ntb = ee_bases[l]
    bary = ee_bary[l]  # type: ignore
    n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
    s_param = bary[0]  # type: ignore
    t_param = bary[1]  # type: ignore
    s_al = cee.s[l]
    gamma = cee.gamma[l]
    lambda_n = cee.lambda_n[l]
    lambda_f = cee.lambda_f[l]
    i_he2 = halfedges.incoming_vertex(F, he2)
    j_he2 = halfedges.outgoing_vertex(F, he2)
    xcp_u, xtcp_u = edge_closest_point(fem.x, fem.xt, i_he, j_he, s_param)  # type: ignore
    xcp_v, xtcp_v = edge_closest_point(fem.x, fem.xt, i_he2, j_he2, t_param)  # type: ignore
    d = xcp_u - xcp_v
    u = (xcp_u - xtcp_u) - (xcp_v - xtcp_v)
    c_n, c_f = contact_constraints(d, u, n, t, b, s_al, dmin)  # type: ignore
    one_m_s = wp.float32(1) - s_param  # type: ignore
    wi = one_m_s * wp.float32(i == i_he) + s_param * wp.float32(i == j_he)  # type: ignore
    return accumulate_contact_al(c_n, c_f, wi, gamma, n, t, b, lambda_n, lambda_f, sigma_n, sigma_f)  # type: ignore


@wp.func
def _contact_ee_rev(
    c: wp.int32,
    i: wp.int32,
    i_he: wp.int32,
    j_he: wp.int32,
    he2: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    F: wp.array[wp.vec3i],
    cee: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    ee_bases: wp.array[wp.mat33f],
    ee_bary: wp.array[wp.vec2f],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    dmin: wp.float32,
):
    """Gradient and Hessian contribution of a single reverse edge-edge contact for vertex i (v-side, on he)."""
    ntb = ee_bases[c]
    bary = ee_bary[c]  # type: ignore
    n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
    s_param = bary[0]  # type: ignore  # param on he2
    t_param = bary[1]  # type: ignore  # param on he
    s_al = cee.s[c]
    gamma = cee.gamma[c]
    lambda_n = cee.lambda_n[c]
    lambda_f = cee.lambda_f[c]
    i_he2 = halfedges.incoming_vertex(F, he2)
    j_he2 = halfedges.outgoing_vertex(F, he2)
    xcp_u, xtcp_u = edge_closest_point(fem.x, fem.xt, i_he2, j_he2, s_param)  # type: ignore
    xcp_v, xtcp_v = edge_closest_point(fem.x, fem.xt, i_he, j_he, t_param)  # type: ignore
    d = xcp_u - xcp_v  # same direction as the forward pair
    u = (xcp_u - xtcp_u) - (xcp_v - xtcp_v)
    c_n, c_f = contact_constraints(d, u, n, t, b, s_al, dmin)  # type: ignore
    one_m_t = wp.float32(1) - t_param  # type: ignore
    wi = -one_m_t * wp.float32(i == i_he) - t_param * wp.float32(i == j_he)  # type: ignore
    return accumulate_contact_al(c_n, c_f, wi, gamma, n, t, b, lambda_n, lambda_f, sigma_n, sigma_f)  # type: ignore


@wp.func
def local_elastic_derivatives(
    i: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    local_tid: wp.int32,
    block_dims: wp.int32,
):
    gi = wp.vec3f()
    Hi = wp.mat33f()
    GVGbegin = params.GVGp[i]
    n_adj_elems = params.GVGp[i + 1] - GVGbegin
    for elocal in range(local_tid, n_adj_elems, block_dims):
        e = params.GVGadj[GVGbegin + elocal]
        nodes = fem.E[e]
        ilocal = (
            wp.int32(i == nodes[1])
            * wp.int32(1)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[2])
            * wp.int32(2)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[3])
            * wp.int32(3)  # pyright: ignore[reportOperatorIssue]
        )
        wg = fem.wg[e]
        GP = fem.GNeg[e]
        mu = fem.mug[e]
        llambda = fem.lambdag[e]
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
def local_elastic_hessians(
    i: wp.int32,
    fem: FemElastoDynamicsData, # type: ignore
    params: ParamsData, # type: ignore
    local_tid: wp.int32,
    block_dims: wp.int32,
):
    Hi = wp.mat33f()
    GVGbegin = params.GVGp[i]
    n_adj_elems = params.GVGp[i + 1] - GVGbegin
    for elocal in range(local_tid, n_adj_elems, block_dims):
        e = params.GVGadj[GVGbegin + elocal]
        nodes = fem.E[e]
        ilocal = (
            wp.int32(i == nodes[1])
            * wp.int32(1)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[2])
            * wp.int32(2)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[3])
            * wp.int32(3)  # pyright: ignore[reportOperatorIssue]
        )
        wg = fem.wg[e]
        GP = fem.GNeg[e]
        mu = fem.mug[e]
        llambda = fem.lambdag[e]
        # Gather element positions -> compute F
        xe = types.mat3x4f()
        for j in range(4):
            xj = fem.x[nodes[j]]
            for d in range(3):
                xe[d, j] = xj[d]
        # xe = 3x4 matrix of element positions (columns are nodes)
        F = xe @ GP
        HF = snh_hess(F, mu, llambda)
        Hi += wg * hessian_block_wrt_dofs(HF, GP, ilocal, ilocal)
    return Hi


@wp.func
def local_contact_derivatives(
    i: wp.int32,
    vi: wp.int32,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    contact: ContactDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    local_tid: wp.int32,
    block_dims: wp.int32,
):
    gi = wp.vec3f()
    Hi = wp.mat33f()
    meshes = contact.meshes
    ogc = contact.ogc
    dmin = contact.dmin
    sigma_n = contact.sigma_n[0]
    sigma_f = contact.sigma_f[0]
    cvv, vv_bases = contact.cvv, ogc.vv_bases
    cve, ve_bases, ve_bary = contact.cve, ogc.ve_bases, ogc.ve_bary
    cvf, vf_bases, vf_bary = contact.cvf, ogc.vf_bases, ogc.vf_bary
    cee, ee_bases, ee_bary = contact.cee, ogc.ee_bases, ogc.ee_bary

    # 1a. Vertex-vertex contacts (forward)
    for k in range(ogc.vv.prefix[vi] + local_tid, ogc.vv.prefix[vi + 1], block_dims):
        gic, Hic = _contact_vv_fwd(k, i, meshes.V[ogc.vv.v[k]], fem, cvv, vv_bases, sigma_n, sigma_f, dmin)  # type: ignore
        gi += gic
        Hi += Hic
    # 1b. Vertex-vertex contacts (reverse)
    for k in range(ogc.rvv.prefix[vi] + local_tid, ogc.rvv.prefix[vi + 1], block_dims):
        gic, Hic = _contact_vv_rev(ogc.rvv2vv[k], i, meshes.V[ogc.rvv.v[k]], fem, cvv, vv_bases, sigma_n, sigma_f, dmin)  # type: ignore
        gi += gic
        Hi += Hic
    # 2. Vertex-halfedge contacts (forward)
    for k in range(ogc.ve.prefix[vi] + local_tid, ogc.ve.prefix[vi + 1], block_dims):
        hei = ogc.ve.v[k]
        hej = halfedges.opposite_half_edge(meshes.F, hei, meshes.GHEF)
        he = wp.max(hei, hej)
        gic, Hic = _contact_ve_fwd(k, i, he, fem, meshes.F, cve, ve_bases, ve_bary, sigma_n, sigma_f, dmin)  # type: ignore
        gi += gic
        Hi += Hic
    # 3. Vertex-triangle contacts (forward)
    for k in range(ogc.vf.prefix[vi] + local_tid, ogc.vf.prefix[vi + 1], block_dims):
        f = ogc.vf.v[k]
        gic, Hic = _contact_vf_fwd(k, i, f, fem, meshes.F, cvf, vf_bases, vf_bary, sigma_n, sigma_f, dmin)  # type: ignore
        gi += gic
        Hi += Hic
    # 4 & 5. Per-incident-halfedge loops (EE forward/reverse, VE/VF/EE reverse)
    for k in range(meshes.GVHEp[i], meshes.GVHEp[i + 1]):
        hei = meshes.GVHEadj[k]
        hej = halfedges.opposite_half_edge(meshes.F, hei, meshes.GHEF)
        he = wp.max(hei, hej)
        i_he = halfedges.incoming_vertex(meshes.F, he)
        j_he = halfedges.outgoing_vertex(meshes.F, he)
        # 4. EE contacts (forward): he is u-side
        for l in range(
            ogc.ee.prefix[he] + local_tid, ogc.ee.prefix[he + 1], block_dims
        ):
            hei2 = ogc.ee.v[l]
            hej2 = halfedges.opposite_half_edge(meshes.F, hei2, meshes.GHEF)
            he2 = wp.max(hei2, hej2)
            gic, Hic = _contact_ee_fwd(l, i, i_he, j_he, he2, fem, meshes.F, cee, ee_bases, ee_bary, sigma_n, sigma_f, dmin)  # type: ignore
            gi += gic
            Hi += Hic
        # 5.a VE contacts (reverse): he is v-side
        for l in range(
            ogc.rve.prefix[he] + local_tid, ogc.rve.prefix[he + 1], block_dims
        ):
            gic, Hic = _contact_ve_rev(ogc.rve2ve[l], i, i_he, j_he, meshes.V[ogc.rve.v[l]], fem, cve, ve_bases, ve_bary, sigma_n, sigma_f, dmin)  # type: ignore
            gi += gic
            Hi += Hic
        # 5.b VF contacts (reverse): face of hei contains vertex i
        f = halfedges.face_of_half_edge(hei)
        for l in range(
            ogc.rvf.prefix[f] + local_tid, ogc.rvf.prefix[f + 1], block_dims
        ):
            gic, Hic = _contact_vf_rev(ogc.rvf2vf[l], i, f, meshes.V[ogc.rvf.v[l]], fem, meshes.F, cvf, vf_bases, vf_bary, sigma_n, sigma_f, dmin)  # type: ignore
            gi += gic
            Hi += Hic
        # 5.c EE contacts (reverse): he is v-side
        for l in range(
            ogc.ree.prefix[he] + local_tid, ogc.ree.prefix[he + 1], block_dims
        ):
            hei2 = ogc.ree.v[l]
            hej2 = halfedges.opposite_half_edge(meshes.F, hei2, meshes.GHEF)
            he = wp.max(hei2, hej2)
            gic, Hic = _contact_ee_rev(ogc.ree2ee[l], i, i_he, j_he, he, fem, meshes.F, cee, ee_bases, ee_bary, sigma_n, sigma_f, dmin)  # type: ignore
            gi += gic
            Hi += Hic

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

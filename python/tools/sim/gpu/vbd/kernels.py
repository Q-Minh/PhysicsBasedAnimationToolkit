from typing import Tuple
import warp as wp
import warp.fem.linalg
from .. import types
from ..elasticity.snh import snh_grad_and_hess, snh_hess
from ..elasticity.chain import gradient_segment_wrt_dofs, hessian_block_wrt_dofs
from ..contact.dynamics import MeshDynamicsData as ContactDynamicsData
from ..contact.mesh.pairs import ContactBasesData
from ..contact import halfedges
from .params import (
    VLS_SOLVER_INVERSE,
    VLS_SOLVER_LLT,
    VLS_SOLVER_QR,
    VLS_SOLVER_EVD,
)


@wp.func
def local_elastic_derivatives(
    i: wp.int32,
    x: wp.array[wp.vec3f],
    E: wp.array[wp.vec4i],
    wg: wp.array[wp.float32],
    GNeg: wp.array[types.mat4x3f],  # type: ignore
    mug: wp.array[wp.float32],
    lambdag: wp.array[wp.float32],
    GVGp: wp.array[wp.int32],
    GVGadj: wp.array[wp.int32],
    local_tid: wp.int32,
    block_dims: wp.int32,
):
    gi = wp.vec3f()
    Hi = wp.mat33f()
    GVGbegin = GVGp[i]
    n_adj_elems = GVGp[i + 1] - GVGbegin
    for elocal in range(local_tid, n_adj_elems, block_dims):  # type: ignore
        e = GVGadj[GVGbegin + elocal]
        nodes = E[e]
        ilocal = (
            wp.int32(i == nodes[1])  # type: ignore
            * wp.int32(1)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[2])  # type: ignore
            * wp.int32(2)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[3])  # type: ignore
            * wp.int32(3)  # pyright: ignore[reportOperatorIssue]
        )
        wge = wg[e]  # type: ignore
        GP = GNeg[e]
        mu = mug[e]
        llambda = lambdag[e]
        # Gather element positions -> compute F
        xe = types.mat3x4f()
        for j in range(4):
            xj = x[nodes[j]]  # type: ignore
            for d in range(3):
                xe[d, j] = xj[d]  # type: ignore
        # xe = 3x4 matrix of element positions (columns are nodes)
        F = xe @ GP
        # SNH grad and hess w.r.t. vec(F)
        gF, HF = snh_grad_and_hess(F, mu, llambda)  # type: ignore
        # Chain rule: accumulate into vertex gradient and hessian
        gi += wge * gradient_segment_wrt_dofs(gF, GP, ilocal)
        Hi += wge * hessian_block_wrt_dofs(HF, GP, ilocal, ilocal)
    return gi, Hi


@wp.func
def local_elastic_hessians(
    i: wp.int32,
    x: wp.array[wp.vec3f],
    E: wp.array[wp.vec4i],
    wg: wp.array[wp.float32],
    GNeg: wp.array[types.mat4x3f],  # type: ignore
    mug: wp.array[wp.float32],
    lambdag: wp.array[wp.float32],
    GVGp: wp.array[wp.int32],
    GVGadj: wp.array[wp.int32],
    local_tid: wp.int32,
    block_dims: wp.int32,
):
    Hi = wp.mat33f()
    GVGbegin = GVGp[i]
    n_adj_elems = GVGp[i + 1] - GVGbegin
    for elocal in range(local_tid, n_adj_elems, block_dims):  # type: ignore
        e = GVGadj[GVGbegin + elocal]
        nodes = E[e]
        ilocal = (
            wp.int32(i == nodes[1])  # type: ignore
            * wp.int32(1)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[2])  # type: ignore
            * wp.int32(2)  # pyright: ignore[reportOperatorIssue]
            + wp.int32(i == nodes[3])  # type: ignore
            * wp.int32(3)  # pyright: ignore[reportOperatorIssue]
        )
        wge = wg[e]  # type: ignore
        GP = GNeg[e]
        mu = mug[e]
        llambda = lambdag[e]  # type: ignore
        # Gather element positions -> compute F
        xe = types.mat3x4f()
        for j in range(4):
            xj = x[nodes[j]]  # type: ignore
            for d in range(3):
                xe[d, j] = xj[d]  # type: ignore
        # xe = 3x4 matrix of element positions (columns are nodes)
        F = xe @ GP
        HF = snh_hess(F, mu, llambda)  # type: ignore
        Hi += wge * hessian_block_wrt_dofs(HF, GP, ilocal, ilocal)
    return Hi


@wp.func
def _contact_derivatives(
    c: wp.int32,
    bases: ContactBasesData,  # type: ignore
    s: wp.array[wp.float32],
    gamma: wp.array[wp.float32],
    lambda_n: wp.array[wp.float32],
    lambda_f: wp.array[wp.vec2f],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    xcp1: wp.vec3f,
    xtcp1: wp.vec3f,
    xcp2: wp.vec3f,
    xtcp2: wp.vec3f,
    dmin: wp.float32,
    wi: wp.float32,
):
    n, t, b = bases.n[c], bases.t[c], bases.b[c]  # type: ignore
    sk = s[c]
    gammak = gamma[c]
    dx = xcp1 - xcp2  # type: ignore
    du = (xcp1 - xtcp1) - (xcp2 - xtcp2)  # type: ignore
    c_n = wp.dot(dx, n) - dmin - sk  # type: ignore
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    dEn = sigma_n * c_n - lambda_n[c]
    dEf = sigma_f * c_f - lambda_f[c]
    gi = (gammak * wi) * (dEn * n + dEf[0] * t + dEf[1] * b)
    Hi = (gammak * wi * wi) * (
        sigma_n * wp.outer(n, n) + sigma_f * (wp.outer(t, t) + wp.outer(b, b))  # type: ignore
    )
    return gi, Hi


@wp.func
def local_contact_derivatives(
    i: wp.int32,
    vi: wp.int32,
    xt: wp.array[wp.vec3f],
    x: wp.array[wp.vec3f],
    contact: ContactDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    local_tid: wp.int32,
    block_dims: wp.int32,
):
    gi = wp.vec3f()
    Hi = wp.mat33f()
    meshes = contact.meshes
    fcontacts = contact.contacts
    rcontacts = contact.rcontacts
    dmin = contact.dmin
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    cvv, vv_bases = contact.cvv, fcontacts.vv_bases
    cve, ve_bases, ve_bary = contact.cve, fcontacts.ve_bases, fcontacts.ve_bary
    cvf, vf_bases, vf_bary = contact.cvf, fcontacts.vf_bases, fcontacts.vf_bary
    cee, ee_bases, ee_bary = contact.cee, fcontacts.ee_bases, fcontacts.ee_bary
    xi = x[i]
    xti = xt[i]

    # 1a. Vertex-vertex contacts (forward)
    for c in range(
        wp.int32(fcontacts.vv.prefix[vi]) + local_tid,
        wp.int32(fcontacts.vv.prefix[vi + 1]),
        block_dims,
    ):
        vj = fcontacts.vv.v[c]
        j = meshes.V[vj]
        xcp1 = xi
        xcp2 = x[j]
        xtcp1 = xti
        xtcp2 = xt[j]
        wi = wp.float32(1)
        gic, Hic = _contact_derivatives(
            c,  # type: ignore
            vv_bases,
            cvv.s,
            cvv.gamma,
            cvv.lambda_n,
            cvv.lambda_f,
            sigma_n,
            sigma_f,
            xcp1,  # type: ignore
            xtcp1,  # type: ignore
            xcp2,  # type: ignore
            xtcp2,  # type: ignore
            dmin,
            wi,
        )
        gi += gic
        Hi += Hic

    # 1b. Vertex-vertex contacts (reverse)
    for k in range(
        wp.int32(rcontacts.rvv.prefix[vi]) + local_tid,
        wp.int32(rcontacts.rvv.prefix[vi + 1]),
        block_dims,
    ):
        c = rcontacts.rvv2vv[k]
        vj = rcontacts.rvv.v[k]
        j = meshes.V[vj]
        xcp1 = x[j]
        xtcp1 = xt[j]
        xcp2 = xi
        xtcp2 = xti
        wi = wp.float32(-1)
        gic, Hic = _contact_derivatives(
            c,  # type: ignore
            vv_bases,
            cvv.s,
            cvv.gamma,
            cvv.lambda_n,
            cvv.lambda_f,
            sigma_n,
            sigma_f,
            xcp1,  # type: ignore
            xtcp1,  # type: ignore
            xcp2,  # type: ignore
            xtcp2,  # type: ignore
            dmin,
            wi,
        )
        gi += gic
        Hi += Hic

    # 2. Vertex-halfedge contacts (forward)
    for c in range(
        wp.int32(fcontacts.ve.prefix[vi]) + local_tid,
        wp.int32(fcontacts.ve.prefix[vi + 1]),
        block_dims,
    ):
        he = fcontacts.ve.v[c]
        a = halfedges.incoming_vertex(meshes.F, he)
        b = halfedges.outgoing_vertex(meshes.F, he)
        b1 = ve_bary[c]
        b0 = wp.float32(1) - b1
        xcp1 = xi
        xtcp1 = xti
        xcp2 = b0 * x[a] + b1 * x[b]
        xtcp2 = b0 * xt[a] + b1 * xt[b]
        wi = wp.float32(1)
        gic, Hic = _contact_derivatives(
            c,  # type: ignore
            ve_bases,
            cve.s,
            cve.gamma,
            cve.lambda_n,
            cve.lambda_f,
            sigma_n,
            sigma_f,
            xcp1,  # type: ignore
            xtcp1,  # type: ignore
            xcp2,  # type: ignore
            xtcp2,  # type: ignore
            dmin,
            wi,
        )
        gi += gic
        Hi += Hic

    # 3. Vertex-triangle contacts (forward)
    for c in range(
        wp.int32(fcontacts.vf.prefix[vi]) + local_tid,
        wp.int32(fcontacts.vf.prefix[vi + 1]),
        block_dims,
    ):
        f = fcontacts.vf.v[c]
        finds = meshes.F[f]
        uv = vf_bary[c]
        b1 = uv[0]
        b2 = uv[1]
        b0 = wp.float32(1) - b1 - b2
        xcp1 = xi
        xtcp1 = xti
        xcp2 = b0 * x[finds[0]] + b1 * x[finds[1]] + b2 * x[finds[2]]
        xtcp2 = b0 * xt[finds[0]] + b1 * xt[finds[1]] + b2 * xt[finds[2]]
        wi = wp.float32(1)
        gic, Hic = _contact_derivatives(
            c,  # type: ignore
            vf_bases,
            cvf.s,
            cvf.gamma,
            cvf.lambda_n,
            cvf.lambda_f,
            sigma_n,
            sigma_f,
            xcp1,  # type: ignore
            xtcp1,  # type: ignore
            xcp2,  # type: ignore
            xtcp2,  # type: ignore
            dmin,
            wi,
        )
        gi += gic
        Hi += Hic

    # 4 & 5. Per-incident-halfedge loops (EE forward/reverse, VE/VF/EE reverse)
    for k in range(meshes.GVHEp[i], meshes.GVHEp[i + 1]):
        hei = meshes.GVHEadj[k]
        hej = halfedges.opposite_half_edge(meshes.F, hei, meshes.GHEF)
        he = wp.uint32(wp.max(hei, hej))  # type: ignore
        i_he = halfedges.incoming_vertex(meshes.F, he)  # type: ignore
        j_he = halfedges.outgoing_vertex(meshes.F, he)  # type: ignore
        # 4. EE contacts (forward): he is u-side
        for c in range(
            wp.int32(fcontacts.ee.prefix[he]) + local_tid,
            wp.int32(fcontacts.ee.prefix[he + wp.uint32(1)]),
            block_dims,
        ):
            he2 = fcontacts.ee.v[c]
            i_he2 = halfedges.incoming_vertex(meshes.F, he2)
            j_he2 = halfedges.outgoing_vertex(meshes.F, he2)
            st = ee_bary[c]
            b1 = st[0]
            b0 = wp.float32(1) - b1
            b3 = st[1]
            b2 = wp.float32(1) - b3
            xcp1 = b0 * x[i_he] + b1 * x[j_he]
            xtcp1 = b0 * xt[i_he] + b1 * xt[j_he]
            xcp2 = b2 * x[i_he2] + b3 * x[j_he2]
            xtcp2 = b2 * xt[i_he2] + b3 * xt[j_he2]
            wi = wp.float32(i_he == i) * b0 + wp.float32(j_he == i) * b1
            gic, Hic = _contact_derivatives(
                c,  # type: ignore
                ee_bases,
                cee.s,
                cee.gamma,
                cee.lambda_n,
                cee.lambda_f,
                sigma_n,
                sigma_f,
                xcp1,
                xtcp1,
                xcp2,
                xtcp2,
                dmin,
                wi,
            )
            gi += gic
            Hi += Hic

        # 5.a VE contacts (reverse): he is v-side
        for l in range(
            wp.int32(rcontacts.rve.prefix[he]) + local_tid,
            wp.int32(rcontacts.rve.prefix[he + wp.uint32(1)]),
            block_dims,
        ):
            c = rcontacts.rve2ve[l]
            _vi = rcontacts.rve.v[l]
            _i = meshes.V[_vi]
            b1 = ve_bary[c]
            b0 = wp.float32(1) - b1
            xcp1 = x[_i]
            xtcp1 = xt[_i]
            xcp2 = b0 * x[i_he] + b1 * x[j_he]
            xtcp2 = b0 * xt[i_he] + b1 * xt[j_he]
            wi = -(wp.float32(i_he == i) * b0 + wp.float32(j_he == i) * b1)
            gic, Hic = _contact_derivatives(
                c,
                ve_bases,
                cve.s,
                cve.gamma,
                cve.lambda_n,
                cve.lambda_f,
                sigma_n,
                sigma_f,
                xcp1,  # type: ignore
                xtcp1,  # type: ignore
                xcp2,
                xtcp2,
                dmin,
                wi,
            )
            gi += gic
            Hi += Hic

        # 5.b VF contacts (reverse): face of hei contains vertex i
        f = wp.uint32(halfedges.face_of_half_edge(hei))  # type: ignore
        finds = meshes.F[f]
        for l in range(
            wp.int32(rcontacts.rvf.prefix[f]) + local_tid,
            wp.int32(rcontacts.rvf.prefix[f + wp.uint32(1)]),
            block_dims,
        ):
            c = rcontacts.rvf2vf[l]
            _vi = rcontacts.rvf.v[l]
            _i = meshes.V[_vi]
            uv = vf_bary[c]
            b0 = uv[0]
            b1 = uv[1]
            b2 = wp.float32(1) - b0 - b1
            xcp1 = x[_i]
            xtcp1 = xt[_i]
            xcp2 = b0 * x[finds[0]] + b1 * x[finds[1]] + b2 * x[finds[2]]
            xtcp2 = b0 * xt[finds[0]] + b1 * xt[finds[1]] + b2 * xt[finds[2]]
            wi = -(
                wp.float32(finds[0] == i) * b0
                + wp.float32(finds[1] == i) * b1
                + wp.float32(finds[2] == i) * b2
            )
            gic, Hic = _contact_derivatives(
                c,
                vf_bases,
                cvf.s,
                cvf.gamma,
                cvf.lambda_n,
                cvf.lambda_f,
                sigma_n,
                sigma_f,
                xcp1,  # type: ignore
                xtcp1,  # type: ignore
                xcp2,
                xtcp2,
                dmin,
                wi,
            )
            gi += gic
            Hi += Hic

        # 5.c EE contacts (reverse): he is v-side
        for l in range(
            wp.int32(rcontacts.ree.prefix[he]) + local_tid,
            wp.int32(rcontacts.ree.prefix[he + wp.uint32(1)]),
            block_dims,
        ):
            c = rcontacts.ree2ee[l]
            he2 = rcontacts.ree.v[l]
            i_he2 = halfedges.incoming_vertex(meshes.F, he2)
            j_he2 = halfedges.outgoing_vertex(meshes.F, he2)
            st = ee_bary[c]
            b1 = st[0]
            b0 = wp.float32(1) - b1
            b3 = st[1]
            b2 = wp.float32(1) - b3
            xcp1 = b0 * x[i_he2] + b1 * x[j_he2]
            xtcp1 = b0 * xt[i_he2] + b1 * xt[j_he2]
            xcp2 = b2 * x[i_he] + b3 * x[j_he]
            xtcp2 = b2 * xt[i_he] + b3 * xt[j_he]
            wi = -(wp.float32(i_he == i) * b2 + wp.float32(j_he == i) * b3)
            gic, Hic = _contact_derivatives(
                c,  # type: ignore
                ee_bases,
                cee.s,
                cee.gamma,
                cee.lambda_n,
                cee.lambda_f,
                sigma_n,
                sigma_f,
                xcp1,
                xtcp1,
                xcp2,
                xtcp2,
                dmin,
                wi,
            )
            gi += gic
            Hi += Hic

    return gi, Hi


# @wp.func
# def _contact_ntb_rayleigh(
#     ntb: wp.mat33f,
#     Hi: wp.mat33f,
# ) -> Tuple[wp.float32, wp.float32]:
#     """Compute per-contact Rayleigh quotients Q = (g^T Hi g) / (g^T g) for normal, tangent, and bitangent directions.

#     Since g = wi * dir and wi^2 cancels in the ratio, and n/t/b are unit vectors so g^T g = 1,
#     the quotients reduce to simply ``dir^T Hi dir``.

#     Args:
#         ntb: Row matrix whose rows are the contact normal, first tangent, and bitangent.
#         Hi: Symmetric positive-(semi)definite 3x3 dynamics Hessian block for vertex i.

#     Returns:
#         ``(Qn, Qf)`` where ``Qn = n^T Hi n`` and ``Qf = max(t^T Hi t, b^T Hi b)``.
#     """
#     n, t, b = ntb[0, :], ntb[1, :], ntb[2, :]  # type: ignore
#     Qn = wp.dot(n, Hi @ n)
#     Qf = wp.max(wp.dot(t, Hi @ t), wp.dot(b, Hi @ b))
#     return Qn, Qf  # type: ignore


# @wp.func
# def local_contact_rayleigh_quotients(
#     i: wp.int32,
#     vi: wp.int32,
#     Hi: wp.mat33f,
#     contact: ContactDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
#     local_tid: wp.int32,
#     block_dims: wp.int32,
# ):
#     """Compute thread-local maximum normal and tangential Rayleigh quotients over all contacts incident on vertex i.

#     Mirrors the loop structure of :func:`local_contact_derivatives`, but instead of accumulating
#     gradient and Hessian contributions, computes per-contact Rayleigh quotients
#     ``Q = 1 / (dir^T Hi dir)`` for the contact normal, tangent, and bitangent directions,
#     tracking the thread-local maxima.

#     Args:
#         i: Global FEM vertex index.
#         vi: Surface/contact vertex index (indexes into OGC contact lists).
#         Hi: Full dynamics Hessian block for vertex i (elastic + mass).
#         contact: Contact dynamics data.
#         local_tid: Thread index within the block (0..block_dims-1).
#         block_dims: Number of threads in the block.

#     Returns:
#         ``(local_Qn, local_Qf)`` thread-local maximum normal and tangential Rayleigh quotients.
#     """
#     maxQn = wp.float32(0)
#     maxQf = wp.float32(0)
#     meshes = contact.meshes
#     ogc = contact.ogc
#     vv_bases = ogc.vv_bases
#     ve_bases = ogc.ve_bases
#     vf_bases = ogc.vf_bases
#     ee_bases = ogc.ee_bases

#     # 1a. Vertex-vertex contacts (forward)
#     for k in range(ogc.vv.prefix[vi] + local_tid, ogc.vv.prefix[vi + 1], block_dims):
#         vvQn, vvQf = _contact_ntb_rayleigh(vv_bases[k], Hi)
#         maxQn = wp.max(maxQn, vvQn)
#         maxQf = wp.max(maxQf, vvQf)
#     # 1b. Vertex-vertex contacts (reverse)
#     for k in range(ogc.rvv.prefix[vi] + local_tid, ogc.rvv.prefix[vi + 1], block_dims):
#         rvvQn, rvvQf = _contact_ntb_rayleigh(vv_bases[ogc.rvv2vv[k]], Hi)  # type: ignore
#         maxQn = wp.max(maxQn, rvvQn)
#         maxQf = wp.max(maxQf, rvvQf)
#     # 2. Vertex-halfedge contacts (forward)
#     for k in range(ogc.ve.prefix[vi] + local_tid, ogc.ve.prefix[vi + 1], block_dims):
#         veQn, veQf = _contact_ntb_rayleigh(ve_bases[k], Hi)
#         maxQn = wp.max(maxQn, veQn)
#         maxQf = wp.max(maxQf, veQf)
#     # 3. Vertex-triangle contacts (forward)
#     for k in range(ogc.vf.prefix[vi] + local_tid, ogc.vf.prefix[vi + 1], block_dims):
#         vfQn, vfQf = _contact_ntb_rayleigh(vf_bases[k], Hi)
#         maxQn = wp.max(maxQn, vfQn)
#         maxQf = wp.max(maxQf, vfQf)
#     # 4 & 5. Per-incident-halfedge loops (EE forward/reverse, VE/VF/EE reverse)
#     for k in range(meshes.GVHEp[i], meshes.GVHEp[i + 1]):
#         hei = meshes.GVHEadj[k]
#         hej = halfedges.opposite_half_edge(meshes.F, hei, meshes.GHEF)
#         he = wp.max(hei, hej)
#         # 4. EE contacts (forward): he is u-side
#         for l in range(
#             ogc.ee.prefix[he] + local_tid, ogc.ee.prefix[he + 1], block_dims
#         ):
#             eeQn, eeQf = _contact_ntb_rayleigh(ee_bases[l], Hi)
#             maxQn = wp.max(maxQn, eeQn)
#             maxQf = wp.max(maxQf, eeQf)
#         # 5.a VE contacts (reverse): he is v-side
#         for l in range(
#             ogc.rve.prefix[he] + local_tid, ogc.rve.prefix[he + 1], block_dims
#         ):
#             rveQn, rveQf = _contact_ntb_rayleigh(ve_bases[ogc.rve2ve[l]], Hi)
#             maxQn = wp.max(maxQn, rveQn)
#             maxQf = wp.max(maxQf, rveQf)
#         # 5.b VF contacts (reverse): face of hei contains vertex i
#         f = halfedges.face_of_half_edge(hei)
#         for l in range(
#             ogc.rvf.prefix[f] + local_tid, ogc.rvf.prefix[f + 1], block_dims
#         ):
#             rvfQn, rvfQf = _contact_ntb_rayleigh(vf_bases[ogc.rvf2vf[l]], Hi)
#             maxQn = wp.max(maxQn, rvfQn)
#             maxQf = wp.max(maxQf, rvfQf)
#         # 5.c EE contacts (reverse): he is v-side
#         for l in range(
#             ogc.ree.prefix[he] + local_tid, ogc.ree.prefix[he + 1], block_dims
#         ):
#             reeQn, reeQf = _contact_ntb_rayleigh(ee_bases[ogc.ree2ee[l]], Hi)
#             maxQn = wp.max(maxQn, reeQn)
#             maxQf = wp.max(maxQf, reeQf)

#     return maxQn, maxQf


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
            if eigd > hess_zero:
                dxi[d] = dxi[d] / eigd
            else:
                dxi[d] = wp.float32(0)
        dxi = V * dxi
    else:
        assert False
    return dxi

from typing import Tuple
import warp as wp
import warp.fem.linalg
from .. import types
from ..elasticity.snh import snh_grad_and_hess, snh_hess
from ..elasticity.chain import gradient_segment_wrt_dofs, hessian_block_wrt_dofs
from ..contact.dynamics import (
    CONTACT_EE,
    CONTACT_VE,
    CONTACT_VF,
    CONTACT_VV,
    MeshDynamicsData as ContactDynamicsData,
    friction_penalty,
    normal_penalty,
)
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
    mu_f: wp.float32,
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
    # Coulomb limit: the AL friction force sigma_f c_f - lambda_f is capped at mu_f times the
    # contact's current normal force max(lambda_n - sigma_n c_n, 0), and its Hessian is scaled by
    # the same factor. Without it, friction acts as unbounded viscous drag within a step (only the
    # dual update projects lambda_f onto the cone).
    friction_limit = mu_f * wp.max(-dEn, wp.float32(0))
    friction_scale = wp.float32(1)
    friction_norm = wp.length(dEf)
    if friction_norm > friction_limit:
        friction_scale = friction_limit / friction_norm
        dEf = friction_scale * dEf
    gi = (gammak * wi) * (dEn * n + dEf[0] * t + dEf[1] * b)
    Hi = (gammak * wi * wi) * (
        sigma_n * wp.outer(n, n)  # type: ignore
        + (friction_scale * sigma_f) * (wp.outer(t, t) + wp.outer(b, b))  # type: ignore
    )
    return gi, Hi


@wp.func
def _contact_position(
    xb: wp.array[wp.vec3f],
    xc: wp.array[wp.vec3f],
    colors: wp.array[wp.int32],
    color: wp.int32,
    use_current: wp.int32,
    k: wp.int32,
):
    """Position of contact stencil node ``k`` in the vertex solve of a vertex of color ``color``.

    The current position if ``use_current`` is set and ``k`` has another color, else the sweep's
    snapshot (see :func:`local_contact_derivatives`).
    """
    if use_current != 0 and colors[k] != color:
        return xc[k]
    return xb[k]


@wp.func
def local_contact_derivatives(
    i: wp.int32,
    vi: wp.int32,
    xt: wp.array[wp.vec3f],
    x: wp.array[wp.vec3f],
    xc: wp.array[wp.vec3f],
    colors: wp.array[wp.int32],
    color: wp.int32,
    use_current: wp.int32,
    contact: ContactDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    local_tid: wp.int32,
    block_dims: wp.int32,
):
    """Contact gradient and Hessian block of vertex ``i`` (surface vertex ``vi``).

    Stencil positions are read from the sweep's snapshot ``x`` (``params.xb``), except, when
    ``use_current`` is set, for nodes of another color than the vertex's own (``color``): those are
    read from the current positions ``xc``. Earlier colors were written by an earlier, finished
    launch, so their current positions are final for this sweep (Gauss-Seidel), and later colors
    still hold their snapshot values. Same-color nodes may be written concurrently by this launch,
    so they keep the snapshot: reads never race, and results stay deterministic.
    """
    gi = wp.vec3f()
    Hi = wp.mat33f()
    meshes = contact.meshes
    fcontacts = contact.contacts
    rcontacts = contact.rcontacts
    dmin = contact.dmin
    mu_f = contact.mu_f
    cvv, vv_bases = contact.cvv, fcontacts.vv_bases
    cve, ve_bases, ve_bary = contact.cve, fcontacts.ve_bases, fcontacts.ve_bary
    cvf, vf_bases, vf_bary = contact.cvf, fcontacts.vf_bases, fcontacts.vf_bary
    cee, ee_bases, ee_bary = contact.cee, fcontacts.ee_bases, fcontacts.ee_bary
    xi = _contact_position(x, xc, colors, color, use_current, i)
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
        xcp2 = _contact_position(x, xc, colors, color, use_current, j)
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
            normal_penalty(contact, CONTACT_VV, wp.int32(c)),
            friction_penalty(contact, CONTACT_VV, wp.int32(c)),
            mu_f,
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
        xcp1 = _contact_position(x, xc, colors, color, use_current, j)
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
            normal_penalty(contact, CONTACT_VV, wp.int32(c)),
            friction_penalty(contact, CONTACT_VV, wp.int32(c)),
            mu_f,
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
        xcp2 = b0 * _contact_position(x, xc, colors, color, use_current, a) + b1 * _contact_position(x, xc, colors, color, use_current, b)
        xtcp2 = b0 * xt[a] + b1 * xt[b]
        wi = wp.float32(1)
        gic, Hic = _contact_derivatives(
            c,  # type: ignore
            ve_bases,
            cve.s,
            cve.gamma,
            cve.lambda_n,
            cve.lambda_f,
            normal_penalty(contact, CONTACT_VE, wp.int32(c)),
            friction_penalty(contact, CONTACT_VE, wp.int32(c)),
            mu_f,
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
        xcp2 = b0 * _contact_position(x, xc, colors, color, use_current, finds[0]) + b1 * _contact_position(x, xc, colors, color, use_current, finds[1]) + b2 * _contact_position(x, xc, colors, color, use_current, finds[2])
        xtcp2 = b0 * xt[finds[0]] + b1 * xt[finds[1]] + b2 * xt[finds[2]]
        wi = wp.float32(1)
        gic, Hic = _contact_derivatives(
            c,  # type: ignore
            vf_bases,
            cvf.s,
            cvf.gamma,
            cvf.lambda_n,
            cvf.lambda_f,
            normal_penalty(contact, CONTACT_VF, wp.int32(c)),
            friction_penalty(contact, CONTACT_VF, wp.int32(c)),
            mu_f,
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
            xcp1 = b0 * _contact_position(x, xc, colors, color, use_current, i_he) + b1 * _contact_position(x, xc, colors, color, use_current, j_he)
            xtcp1 = b0 * xt[i_he] + b1 * xt[j_he]
            xcp2 = b2 * _contact_position(x, xc, colors, color, use_current, i_he2) + b3 * _contact_position(x, xc, colors, color, use_current, j_he2)
            xtcp2 = b2 * xt[i_he2] + b3 * xt[j_he2]
            wi = wp.float32(i_he == i) * b0 + wp.float32(j_he == i) * b1
            gic, Hic = _contact_derivatives(
                c,  # type: ignore
                ee_bases,
                cee.s,
                cee.gamma,
                cee.lambda_n,
                cee.lambda_f,
                normal_penalty(contact, CONTACT_EE, wp.int32(c)),
                friction_penalty(contact, CONTACT_EE, wp.int32(c)),
                mu_f,
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
            xcp1 = _contact_position(x, xc, colors, color, use_current, _i)
            xtcp1 = xt[_i]
            xcp2 = b0 * _contact_position(x, xc, colors, color, use_current, i_he) + b1 * _contact_position(x, xc, colors, color, use_current, j_he)
            xtcp2 = b0 * xt[i_he] + b1 * xt[j_he]
            wi = -(wp.float32(i_he == i) * b0 + wp.float32(j_he == i) * b1)
            gic, Hic = _contact_derivatives(
                c,
                ve_bases,
                cve.s,
                cve.gamma,
                cve.lambda_n,
                cve.lambda_f,
                normal_penalty(contact, CONTACT_VE, wp.int32(c)),
                friction_penalty(contact, CONTACT_VE, wp.int32(c)),
                mu_f,
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
            b1 = uv[0]
            b2 = uv[1]
            b0 = wp.float32(1) - b1 - b2
            xcp1 = _contact_position(x, xc, colors, color, use_current, _i)
            xtcp1 = xt[_i]
            xcp2 = b0 * _contact_position(x, xc, colors, color, use_current, finds[0]) + b1 * _contact_position(x, xc, colors, color, use_current, finds[1]) + b2 * _contact_position(x, xc, colors, color, use_current, finds[2])
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
                normal_penalty(contact, CONTACT_VF, wp.int32(c)),
                friction_penalty(contact, CONTACT_VF, wp.int32(c)),
                mu_f,
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
            xcp1 = b0 * _contact_position(x, xc, colors, color, use_current, i_he2) + b1 * _contact_position(x, xc, colors, color, use_current, j_he2)
            xtcp1 = b0 * xt[i_he2] + b1 * xt[j_he2]
            xcp2 = b2 * _contact_position(x, xc, colors, color, use_current, i_he) + b3 * _contact_position(x, xc, colors, color, use_current, j_he)
            xtcp2 = b2 * xt[i_he] + b3 * xt[j_he]
            wi = -(wp.float32(i_he == i) * b2 + wp.float32(j_he == i) * b3)
            gic, Hic = _contact_derivatives(
                c,  # type: ignore
                ee_bases,
                cee.s,
                cee.gamma,
                cee.lambda_n,
                cee.lambda_f,
                normal_penalty(contact, CONTACT_EE, wp.int32(c)),
                friction_penalty(contact, CONTACT_EE, wp.int32(c)),
                mu_f,
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
def add_inertia_or_external_load_derivatives(
    is_static: wp.int32,
    m: wp.float32,
    xtilde: wp.vec3f,
    fext: wp.vec3f,
    x: wp.vec3f,
    gi: wp.vec3f,
    Hi: wp.mat33f,
):
    """Add the non-elastic, non-contact objective terms of a vertex.

    Dynamic problems add the kinetic energy derivatives (the external load is folded into
    ``xtilde``). Static problems, whose objective is Psi(x) - fext^T x, add only g -= fext.
    """
    if is_static != 0:
        gi -= fext  # pyright: ignore[reportOperatorIssue]
    else:
        gi, Hi = add_inertia_derivatives(m, xtilde, x, gi, Hi)
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
    """Return the step ``dx = H^{-1} g`` of the vertex solve ``x -= dx``.

    ``hess_zero`` is a pseudo-singularity tolerance relative to the scale of ``H``, so that it holds
    for any mass and time step (at small ``h``, ``H ~ m I`` and an absolute threshold on ``det H ~ m^3``
    would skip light vertices). Inverse returns a zero step when ``|det H| <= hess_zero (|H|_F / sqrt(3))^3``
    (equal to ``|tr(H)/3|^3`` for ``H = m I``, and robust to indefinite ``H``, whose trace can vanish);
    EVD zeroes the eigen-components with ``|lambda| <= hess_zero max|lambda|``. QR has no check.
    """
    dxi = wp.vec3f()
    if solver == VLS_SOLVER_INVERSE:
        det = wp.determinant(Hi)  # pyright: ignore[reportArgumentType]
        scale = wp.sqrt(wp.ddot(Hi, Hi) / wp.float32(3))  # |H|_F / sqrt(3)  # pyright: ignore[reportArgumentType]
        if wp.abs(det) > hess_zero * scale * scale * scale:  # pyright: ignore[reportArgumentType]
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
        eig_zero = hess_zero * wp.max(wp.abs(eigs))  # pyright: ignore[reportArgumentType]
        for d in range(3):
            eigd = wp.abs(eigs[d])  # pyright: ignore[reportIndexIssue]
            if eigd > eig_zero:
                dxi[d] = dxi[d] / eigd
            else:
                dxi[d] = wp.float32(0)
        dxi = V * dxi
    else:
        assert False
    return dxi


import unittest
import numpy as np


@wp.kernel
def _integrate_positions_kernel(
    H: wp.array[wp.mat33f],
    g: wp.array[wp.vec3f],
    solver: wp.int32,
    hess_zero: wp.float32,
    dx: wp.array[wp.vec3f],
):
    i = wp.tid()
    dx[i] = integrate_positions(g[i], H[i], solver, 25, wp.float32(1.19e-7), hess_zero)


class TestIntegratePositions(unittest.TestCase):
    def _solve(self, H: np.ndarray, g: np.ndarray, solver: int) -> np.ndarray:
        dx = wp.zeros(len(H), dtype=wp.vec3f)
        wp.launch(
            _integrate_positions_kernel,
            dim=len(H),
            inputs=[
                wp.array(H, dtype=wp.mat33f),
                wp.array(g, dtype=wp.vec3f),
                solver,
                1.19e-7,
            ],
            outputs=[dx],
        )
        return dx.numpy()

    def test_singularity_check_is_relative_to_the_hessian_scale(self):
        # H = m I at small time steps: light vertices must still move (det H = m^3 < hess_zero).
        masses = np.array([1.3e-5, 4.3e-2, 1.0, 1e3], dtype=np.float32)
        H = np.stack([m * np.eye(3, dtype=np.float32) for m in masses])
        g = np.ones((len(H), 3), dtype=np.float32)
        for solver in (VLS_SOLVER_INVERSE, VLS_SOLVER_EVD):
            dx = self._solve(H, g, solver)
            np.testing.assert_allclose(dx, g / masses[:, None], rtol=1e-5)
        # Pseudo-singular at any scale: Inverse skips the vertex, EVD drops the null direction.
        for m in (1e-4, 1.0, 1e4):
            Hs = (m * np.diag([1.0, 1.0, 1e-9])).astype(np.float32)[None]
            gs = np.ones((1, 3), dtype=np.float32)
            np.testing.assert_array_equal(self._solve(Hs, gs, VLS_SOLVER_INVERSE), 0.0)
            dx = self._solve(Hs, gs, VLS_SOLVER_EVD)[0]
            np.testing.assert_allclose(dx, [1.0 / m, 1.0 / m, 0.0], rtol=1e-5)


if __name__ == "__main__":
    wp.init()
    unittest.main()

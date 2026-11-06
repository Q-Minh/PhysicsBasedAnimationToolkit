import unittest

import warp as wp
from ...common import profiling
import cupy as cp
import numpy as np

from . import pairs
from ..multimesh import MultiMesh, MultiMeshData
from .. import halfedges
from .. import queries
from .cd import ContactDetection
from ...common import reduce, lower_bound
from ....common.fields import DocField, SerializableMixin

# Kinds of meshes, by how a vertex's signed distance to them is computed (see _mesh_kind)
MESH_CLOSED = wp.constant(0)  # ray-parity sign
MESH_OPEN = wp.constant(1)  # sign of the closest face's normal
MESH_PLANE = wp.constant(2)  # planar and open (plane colliders): analytic plane distance, unbounded range

MAX_VV_PER_THREAD = wp.constant(2)
MAX_VE_PER_THREAD = wp.constant(2)
MAX_VF_PER_THREAD = wp.constant(4)
# MAX_EE_PER_THREAD = wp.constant(8)
tvvlist = wp.types.vector(length=MAX_VV_PER_THREAD, dtype=wp.int32)
tvelist = wp.types.vector(length=MAX_VE_PER_THREAD, dtype=wp.int32)
tvflist = wp.types.vector(length=MAX_VF_PER_THREAD, dtype=wp.int32)
# teelist = wp.types.vector(length=MAX_EE_PER_THREAD, dtype=wp.int32)

MAX_BLOCK_DIM = 256

# Contact kinds of a working-set contact (Params.freeze_within_solve)
KIND_NONE = wp.constant(-1)
KIND_VV = wp.constant(0)
KIND_VE = wp.constant(1)
KIND_VF = wp.constant(2)


@wp.func
def _working_contact(W: pairs.ContactPairsData, meshes: MultiMeshData, v: wp.int32, m: wp.int32):  # type: ignore
    """The working set W's contact of surface vertex v with mesh m (at most one): (kind, feature, slot), kind
    KIND_NONE if there is none. The feature is a face (VF), a half-edge (VE) or a surface vertex (VV) index."""
    wk, wfeat, wslot = wp.int32(KIND_NONE), wp.int32(0), wp.int32(0)
    fb, fe = meshes.FP[m], meshes.FP[m + 1]
    for s in range(wp.int32(W.vf.prefix[v]), wp.int32(W.vf.prefix[v + 1])):
        f = wp.int32(W.vf.v[s])
        if f >= fb and f < fe:
            wk, wfeat, wslot = KIND_VF, f, s
    for s in range(wp.int32(W.ve.prefix[v]), wp.int32(W.ve.prefix[v + 1])):
        he = wp.int32(W.ve.v[s])
        f = halfedges.face_of_half_edge(he)
        if f >= fb and f < fe:
            wk, wfeat, wslot = KIND_VE, he, s
    vb, ve = meshes.VP[m], meshes.VP[m + 1]
    for s in range(wp.int32(W.vv.prefix[v]), wp.int32(W.vv.prefix[v + 1])):
        w = wp.int32(W.vv.v[s])
        if w >= vb and w < ve:
            wk, wfeat, wslot = KIND_VV, w, s
    return wk, wfeat, wslot


@wp.func
def _working_gap(
    W: pairs.ContactPairsData,  # type: ignore
    meshes: MultiMeshData,  # type: ignore
    x: wp.array[wp.vec3f],
    xi: wp.vec3f,
    kind: wp.int32,
    feat: wp.int32,
    slot: wp.int32,
) -> wp.float32:
    """The working-set contact's linearized gap at x, n . (x_i - c), with its stored normal n and its closest point c
    at its stored barycentrics: the constraint the solver evaluates (plus the margin)."""
    n = wp.vec3f(0.0)
    c = wp.vec3f(0.0)
    if kind == KIND_VF:
        n = W.vf_bases.n[slot]
        uv = W.vf_bary[slot]
        fi = meshes.F[feat]
        c = (wp.float32(1) - uv[0] - uv[1]) * x[fi[0]] + uv[0] * x[fi[1]] + uv[1] * x[fi[2]]
    elif kind == KIND_VE:
        n = W.ve_bases.n[slot]
        t = W.ve_bary[slot]
        c = (wp.float32(1) - t) * x[halfedges.incoming_vertex(meshes.F, feat)] + t * x[
            halfedges.outgoing_vertex(meshes.F, feat)
        ]
    else:
        n = W.vv_bases.n[slot]
        c = x[meshes.V[feat]]
    return wp.dot(n, xi - c)


@wp.func
def _restore_kept_stencil(
    k: wp.int32,
    n_u: wp.int32,
    P: pairs.PairsData,  # type: ignore
    B: pairs.ContactBasesData,  # type: ignore
    WP: pairs.PairsData,  # type: ignore
    WB: pairs.ContactBasesData,  # type: ignore
    kept: wp.array[wp.int32],
) -> wp.int32:
    """Pair slot k (of one contact type) found in the working set with its keep flag set gets the working set's
    basis; returns the working slot, or -1 (barycentrics are the caller's, by type)."""
    if wp.uint64(k) >= P.prefix[n_u]:
        return -1
    u, f = P.u[k], P.v[k]
    for s in range(wp.int32(WP.prefix[wp.int32(u)]), wp.int32(WP.prefix[wp.int32(u) + 1])):
        if WP.v[s] == f and kept[s] != 0:
            B.n[k] = WB.n[s]
            B.t[k] = WB.t[s]
            B.b[k] = WB.b[s]
            return s
    return -1


@wp.kernel
def _restore_kept_stencils(
    C: pairs.ContactPairsData,  # type: ignore
    W: pairs.ContactPairsData,  # type: ignore
    kept_vv: wp.array[wp.int32],
    kept_ve: wp.array[wp.int32],
    kept_vf: wp.array[wp.int32],
    n_u: wp.int32,
):
    """Contacts that detection kept from the working set (Params.freeze_within_solve) get their stored stencil back
    (basis and barycentrics), in place of the one assemble_contacts and the normal kernels computed at x: a kept
    feature re-linearized at x could be clamped off its region (a VF stencil off its face, with the face normal).
    Thread k over every type's slots."""
    k = wp.tid()
    _restore_kept_stencil(k, n_u, C.vv, C.vv_bases, W.vv, W.vv_bases, kept_vv)
    s = _restore_kept_stencil(k, n_u, C.ve, C.ve_bases, W.ve, W.ve_bases, kept_ve)
    if s >= 0:
        C.ve_bary[k] = W.ve_bary[s]
    s = _restore_kept_stencil(k, n_u, C.vf, C.vf_bases, W.vf, W.vf_bases, kept_vf)
    if s >= 0:
        C.vf_bary[k] = W.vf_bary[s]


def detection_block_dim(n_meshes: int) -> int:
    """Threads per block (one block per contact vertex) of _detect_vertex_mesh_contacts for n_meshes contact
    components: the fewest whole warps that give every partner its own thread, at most MAX_BLOCK_DIM.

    Each thread queries ceil(n_meshes / block_dim) partners, and with one pair per partner its pairs of one type
    must fit the per-thread lists (MAX_*_PER_THREAD). A fixed 256 left most threads idle: DragonImpact has 9 meshes,
    so 8 threads queried while the whole block summed and sorted its 256-wide tiles.

    Raises:
        ValueError: If a thread would have more partners than the smallest per-thread list holds.
    """
    block_dim = min(MAX_BLOCK_DIM, 32 * max(1, -(-n_meshes // 32)))
    per_thread = -(-n_meshes // block_dim)
    capacity = min(int(MAX_VV_PER_THREAD), int(MAX_VE_PER_THREAD), int(MAX_VF_PER_THREAD))
    if per_thread > capacity:
        raise ValueError(
            f"{n_meshes} contact components need {per_thread} partners per thread at block_dim {block_dim}, more "
            f"than the per-thread pair lists hold ({capacity}): raise MAX_*_PER_THREAD or MAX_BLOCK_DIM"
        )
    return block_dim


@wp.kernel
def _detect_vertex_mesh_contacts(
    mesh_ids: wp.array[wp.uint64],
    mesh_kind: wp.array[wp.int32],
    mesh_depth: wp.array[wp.float32],
    max_dist: wp.float32,
    threshold: wp.float32,
    widen: wp.int32,
    x0: wp.array[wp.vec3f],
    motion: wp.array[wp.float32],
    sds: wp.array[wp.float32],
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    contacts: pairs.ContactPairsData,  # type: ignore
    freeze: wp.int32,
    working: pairs.ContactPairsData,  # type: ignore
    kept_vv: wp.array[wp.int32],
    kept_ve: wp.array[wp.int32],
    kept_vf: wp.array[wp.int32],
):
    """One block per contact vertex v: its pairs with every other contact component (at most one per component).

    With freeze != 0 (Params.freeze_within_solve, a re-detection within a solve), a pair (v, m) that the working set
    (the previous detection's pairs) has keeps its contact while the stored linearization's gap at x is below the
    threshold: its key is emitted without a query, and kept_*[slot] marks the working slot for
    _restore_kept_stencils. Otherwise (a kept_* entry of 0, or no working contact) the pair is detected as usual.

    Notation:

    - i = meshes.V[v] is the vertex's node; x[i] its position where detection runs (xt_i at a time step's first
      detection, widen != 0, else the current iterate), and x0_i = x0[i] the step's starting iterate (the init
      strategy's predictor for the VBD family, xt for Newton's Position).
    - For each partner component m: c is the vertex's closest point on m at x, on the face with nodes j, k, l and
      barycentrics b0, b1, b2 (c = b0 x_j + b1 x_k + b2 x_l); sd is the signed distance to it (or to the plane, for
      plane colliders) and n the plane's normal.
    - delta = (x0_i - xt_i) - sum_a b_a (x0_a - xt_a), a over j, k, l: the vertex's motion relative to c from xt
      to x0.
    - gap0 = sd - |delta| on meshes, a lower bound of the distance from x0_i to c's position at x0, and
      sd + n . delta on planes, the exact distance at x0. The detection gap is gap = sd, or min(sd, gap0) with
      widen; a pair is detected when gap < threshold (dmin + band).
    - The query radius of the pair (v, m) is max(max_dist, threshold + |x0_i - xt_i| + motion[m]) with widen, else
      max_dist: motion[m] = max |x0_a - xt_a| over m's vertices, so it reaches every face the relative motion can
      bring within the threshold.
    """
    tid = wp.tid()
    block_dim = wp.block_dim()
    block_id = tid // block_dim  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dim  # pyright: ignore[reportOperatorIssue]
    v = block_id
    b = lower_bound(meshes.VP, meshes.VP.shape[0], v + 1) - 1  # type: ignore
    i = meshes.V[v]
    xi = x[i]
    n_verts, n_half_edges, n_tris = (
        meshes.V.shape[0],
        3 * meshes.F.shape[0],
        meshes.F.shape[0],
    )
    tnvv = wp.int32(0)
    tnve = wp.int32(0)
    tnvf = wp.int32(0)
    tvv = tvvlist(n_verts)
    tve = tvelist(n_half_edges)
    tvf = tvflist(n_tris)

    # 1. Detect thread local contacts
    tsdmin = max_dist
    # With widen (a time step's first detection, at x = xt), pairs are also detected when the step's starting
    # iterate x0 may bring them within the threshold: the vertex's own motion x0 - xt (the init strategy's
    # predictor for the VBD family, none for Newton's Position) relative to its partner's closest point.
    di = wp.vec3f(0.0)
    if widen != 0:
        di = x0[i] - xi
    # The loop variable m must not be reassigned in the body (k below is a node index).
    for m in range(local_tid, mesh_ids.shape[0], block_dim):
        if m == b:
            continue
        if freeze != 0:
            # Within a solve, the working set's contact with m keeps its feature and stencil while its linearized gap
            # stays below the threshold (and, on plates, above their depth); a pair that leaves is detected afresh
            wk, wfeat, wslot = _working_contact(working, meshes, v, m)
            if wk != KIND_NONE:
                wgap = _working_gap(working, meshes, x, xi, wk, wfeat, wslot)
                keep = wgap < threshold
                if mesh_kind[m] == MESH_PLANE and wgap < -mesh_depth[m]:
                    keep = False
                if wk == KIND_VF:
                    kept_vf[wslot] = wp.int32(keep)
                elif wk == KIND_VE:
                    kept_ve[wslot] = wp.int32(keep)
                else:
                    kept_vv[wslot] = wp.int32(keep)
                if keep:
                    tsdmin = wp.min(tsdmin, wgap)
                    if wk == KIND_VF:
                        assert tnvf < MAX_VF_PER_THREAD
                        tvf[tnvf] = wfeat
                        tnvf += 1
                    elif wk == KIND_VE:
                        assert tnve < MAX_VE_PER_THREAD
                        tve[tnve] = wfeat
                        tnve += 1
                    else:
                        assert tnvv < MAX_VV_PER_THREAD
                        tvv[tnvv] = wfeat
                        tnvv += 1
                    continue
        # The query reaches every face that the relative motion can bring within the threshold: the vertex's own
        # motion plus the partner's largest (motion[m]), as OGC's adaptive query radius
        query_dist = max_dist
        if widen != 0:
            query_dist = wp.max(max_dist, threshold + wp.norm_l2(di) + motion[m])
        # Closed meshes sign the distance by ray parity. Open ones can't, so they use the closest face's
        # normal. Planes are queried at any distance, so that fast vertices can't tunnel through them.
        kind = mesh_kind[m]
        if kind == MESH_CLOSED:
            qp = wp.mesh_query_point(mesh_ids[m], xi, query_dist)  # type: ignore
        elif kind == MESH_OPEN:
            qp = wp.mesh_query_point_sign_normal(mesh_ids[m], xi, query_dist)  # type: ignore
        else:
            qp = wp.mesh_query_point_no_sign(mesh_ids[m], xi, wp.float32(1e6))  # type: ignore
        if not qp.result:  # type: ignore
            continue
        f = meshes.FP[m] + qp.face  # type: ignore
        finds = meshes.F[f]
        j, k, l = finds[0], finds[1], finds[2]
        xj, xk, xl = x[j], x[k], x[l]
        # Warp's convention: the closest point is u * x_j + v * x_k + (1 - u - v) * x_l
        b0, b1, b2 = qp.u, qp.v, (wp.float32(1) - qp.u - qp.v)  # type: ignore
        xc = b0 * xj + b1 * xk + b2 * xl  # type: ignore
        sd = qp.sign * wp.norm_l2(xi - xc)  # type: ignore
        n = wp.vec3f(0.0)
        if kind == MESH_PLANE:
            # Signed distance to the plane, for vertices over the plane's extent only (the closest
            # point is then the vertex's projection)
            n = wp.normalize(wp.cross(xk - xj, xl - xj))
            sd = wp.dot(xi - xc, n)
            # The tolerance scales with the face, since xc's float32 error grows with its vertices'
            # distance (infinite planes are large quads)
            tangential = wp.norm_l2(xi - xc - sd * n)
            tol = wp.float32(1e-6) * (wp.norm_l2(xk - xj) + wp.norm_l2(xl - xj)) + wp.float32(1e-5) * wp.abs(sd)
            if tangential > tol:
                continue
            # Finite planes are plates: vertices deeper than their thickness passed under or beside them
            if sd < -mesh_depth[m]:
                continue
        # The gap that decides detection. With widen, also the gap at x0, estimated from the vertex's motion relative
        # to its closest point, delta: sd - |delta| (a lower bound of its distance to that point at x0) for meshes,
        # and the exact sd + n . delta for (flat) planes, so that motion along a plane (twisted bodies' ends next to
        # one) doesn't widen. Relative motion, not the fastest body's (a global widening, as OGC's query radius):
        # on CapsulesIntoBowl's checkpoints, the global one overflowed the VE capacity about 2x (2541 and 2998 pairs
        # for 1513 at steps 100 and 250), and it brought ArmadilloSpaghetti.Fall's floor and ceiling planes into
        # range; this one missed none of the pairs detected at x0 and found 1-3x as many VE and VF pairs (both
        # scenes' checkpoints, surface vertices, CPU, 2026-10-22).
        gap = sd
        if widen != 0:
            delta = di - (b0 * (x0[j] - xj) + b1 * (x0[k] - xk) + b2 * (x0[l] - xl))  # type: ignore
            if kind == MESH_PLANE:
                gap = wp.min(sd, sd + wp.dot(n, delta))
            else:
                gap = sd - wp.norm_l2(delta)
        # For the step filter: with widen, the estimated gap at x0, as when detection was at x0
        tsdmin = wp.min(tsdmin, gap)
        # TODO: Runs aren't reproducible when vertices sit exactly over a partner's vertices: two unit cubes
        # stacked 2 mm apart with aligned grids (checkpoint.py's test scene without its xy offset) gave VBD and
        # AAAVBD runs differing from the second step, by up to 1 cm after 10 steps (2026-10-21). Suspected: the
        # closest point is then a shared vertex or edge, so which face the query returns, and so the feature
        # below (VF, VE or VV, from exact zero barycentrics), is a tie. Offsetting the cube made runs
        # bit-identical. Not the feature: since it comes from the closest point's region (the same whichever
        # face shares the closest edge or vertex), one of two repeat runs still differed from the first step
        # (VBD 7.8 mm, AAAVBD 11.5 mm after 10 steps, 2026-10-22). Left to check: the signed distance's last
        # bits (from the returned face's barycentrics), and the order-dependent work after detection.
        if gap < threshold:
            # The closest feature (face interior, edge or vertex) comes from the closest point's Voronoi region
            # of the face, whose barycentrics are exactly zero off the interior, as C++ OGC's
            # (UvwPointInTriangle) and the second pass's (pairs._compute_vf_contact_data). Warp's barycentrics
            # aren't: w = 1 - u - v is ~1e-8 on the edge jk, which made such contacts VF, with the face normal.
            # Behind a sharp edge (faces' normals more than 90 deg apart) the vertex is then behind that face's
            # plane though outside the body, a false penetration of up to the distance (ArmadilloSpaghetti.Fall).
            # The feature no longer depends on which of the faces sharing the closest edge or vertex the query
            # returns.
            uvw = queries.closest_point_triangle(xi, xj, xk, xl)
            is_j_zero = wp.int32(uvw[0] == wp.float32(0))
            is_k_zero = wp.int32(uvw[1] == wp.float32(0))
            is_l_zero = wp.int32(uvw[2] == wp.float32(0))
            nz = is_j_zero + is_k_zero + is_l_zero
            if nz == 0:
                # Triangle contact
                assert tnvf < MAX_VF_PER_THREAD
                tvf[tnvf] = f
                tnvf += 1
            elif nz == 1:
                # Edge contact
                assert tnve < MAX_VE_PER_THREAD
                helocal = is_k_zero * wp.int32(1) + is_l_zero * wp.int32(2)  # type: ignore
                he_candidate = halfedges.next_half_edge(
                    halfedges.half_edge_of_face(f, helocal)
                )
                he = wp.max(
                    he_candidate,
                    halfedges.opposite_half_edge(meshes.F, he_candidate, meshes.GHEF),
                )
                tve[tnve] = he
                tnve += 1
            else:
                # Vertex contact
                assert tnvv < MAX_VV_PER_THREAD
                jnode = wp.int32(not is_j_zero) * j + wp.int32(not is_k_zero) * k + wp.int32(not is_l_zero) * l  # type: ignore
                tvv[tnvv] = meshes.GXV[jnode]  # type: ignore
                tnvv += 1

    # 2. Use block-level primitives to avoid atomic contention on global memory when writing contact pairs.
    # Contacts that don't fit the pair storage's capacity are dropped (raise the n_*_contact_capacity params).
    # NOTE:
    # 1. All contact pairs should be unique here, because this block's vertex only detects 1 contact
    # per every other mesh.
    # 2. We still need to sort the contacts by their partner indices, this is a precondition to ContactPairs.assemble_contacts.
    tnvv, tnve, tnvf = (
        wp.tile_extract(wp.tile_sum(wp.tile(tnvv)), 0),  # type: ignore
        wp.tile_extract(wp.tile_sum(wp.tile(tnve)), 0),  # type: ignore
        wp.tile_extract(wp.tile_sum(wp.tile(tnvf)), 0),  # type: ignore
    )
    has_vv_contacts, has_ve_contacts, has_vf_contacts = tnvv > 0, tnve > 0, tnvf > 0
    if has_vv_contacts:
        bvv = wp.tile(tvv)  # type: ignore
        wp.tile_sort(keys=bvv, values=bvv)
        vv_offset = wp.uint64(0)
        if local_tid == 0:
            vv_offset = wp.atomic_add(contacts.vv.prefix, n_verts, wp.uint64(tnvv))
        vv_offset = wp.tile_extract(wp.tile_from_thread(shape=1, value=vv_offset, thread_idx=0), 0)  # type: ignore
        for block_row in range(MAX_VV_PER_THREAD):
            c = vv_offset + wp.uint64(block_row * block_dim + local_tid)
            if bvv[block_row, local_tid] < n_verts and c < wp.uint64(contacts.vv.u.shape[0]):
                contacts.vv.u[c] = wp.uint32(v)
                contacts.vv.v[c] = wp.uint32(bvv[block_row, local_tid])
    if has_ve_contacts:
        bve = wp.tile(tve)  # type: ignore
        wp.tile_sort(keys=bve, values=bve)
        ve_offset = wp.uint64(0)
        if local_tid == 0:
            ve_offset = wp.atomic_add(contacts.ve.prefix, n_verts, wp.uint64(tnve))
        ve_offset = wp.tile_extract(wp.tile_from_thread(shape=1, value=ve_offset, thread_idx=0), 0)  # type: ignore
        for block_row in range(MAX_VE_PER_THREAD):
            c = ve_offset + wp.uint64(block_row * block_dim + local_tid)
            if bve[block_row, local_tid] < n_half_edges and c < wp.uint64(contacts.ve.u.shape[0]):
                contacts.ve.u[c] = wp.uint32(v)
                contacts.ve.v[c] = wp.uint32(bve[block_row, local_tid])
    if has_vf_contacts:
        bvf = wp.tile(tvf)  # type: ignore
        wp.tile_sort(keys=bvf, values=bvf)
        vf_offset = wp.uint64(0)
        if local_tid == 0:
            vf_offset = wp.atomic_add(contacts.vf.prefix, n_verts, wp.uint64(tnvf))
        vf_offset = wp.tile_extract(wp.tile_from_thread(shape=1, value=vf_offset, thread_idx=0), 0)  # type: ignore
        for block_row in range(MAX_VF_PER_THREAD):
            c = vf_offset + wp.uint64(block_row * block_dim + local_tid)
            if bvf[block_row, local_tid] < n_tris and c < wp.uint64(contacts.vf.u.shape[0]):
                contacts.vf.u[c] = wp.uint32(v)
                contacts.vf.v[c] = wp.uint32(bvf[block_row, local_tid])

    # 3. Update signed distances
    bsdmin = wp.tile(tsdmin)  # type: ignore
    sdmin = wp.tile_extract(wp.tile_min(bsdmin), 0)  # type: ignore
    sds[v] = sdmin  # type: ignore


@wp.kernel
def _component_motion(
    xt: wp.array[wp.vec3f],
    x0: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    motion: wp.array[wp.float32],
):
    """motion[b] = max |x0_i - xt_i| over the vertices i of contact component b (motion zeroed before)."""
    v = wp.tid()
    b = lower_bound(meshes.VP, meshes.VP.shape[0], v + 1) - 1  # type: ignore
    i = meshes.V[v]
    wp.atomic_max(motion, b, wp.norm_l2(x0[i] - xt[i]))


@wp.func
def _face_normal(x: wp.array[wp.vec3f], F: wp.array[wp.vec3i], f: wp.int32) -> wp.vec3f:
    """Unnormalized normal of face f, of length twice its area."""
    finds = F[f]
    xj = x[finds[0]]
    return wp.cross(x[finds[1]] - xj, x[finds[2]] - xj)


@wp.func
def _unit_face_normal(x: wp.array[wp.vec3f], F: wp.array[wp.vec3i], f: wp.int32) -> wp.vec3f:
    return wp.normalize(_face_normal(x, F, f))


@wp.func
def _oriented_normal(
    xi: wp.vec3f, d: wp.vec3f, pseudonormal: wp.vec3f, mesh_id: wp.uint64, kind: wp.int32
) -> wp.vec3f:
    """The contact normal: d = xi - c (from the partner's closest point c to the vertex xi) normalized and signed
    to point outside the partner. For a vertex on the surface (d ~ 0), the pseudonormal itself.

    On closed partners the sign is the detection's own inside/outside test (Warp's ray vote, which signed the
    distance that detected the pair); on open ones, the partner's pseudonormal (as Warp's sign_normal query).
    The pseudonormal's sign is exact only on an embedded surface: where a partner's surface folds over itself (an
    inverted surface tet under impact, so one face's normal is opposite its neighbours'), the two faces of the fold
    edge nearly cancel and an outside vertex over that edge got a normal pointing into the partner, a gap of
    -|d| (DragonImpact step 482, docs/solver-accuracy-notes.md entry 8)."""
    if wp.length(d) <= wp.float32(1e-7) * (wp.float32(1) + wp.length(pseudonormal)):
        return wp.normalize(pseudonormal)
    n = wp.normalize(d)
    outside = wp.dot(d, pseudonormal) >= wp.float32(0)
    if kind == MESH_CLOSED:
        # The sign doesn't depend on the radius, which only needs to reach c
        qp = wp.mesh_query_point(mesh_id, xi, wp.float32(2) * wp.length(d) + wp.float32(1e-6))  # type: ignore
        if qp.result:  # type: ignore
            outside = qp.sign >= wp.float32(0)  # type: ignore
    if not outside:
        n = -n
    return n


@wp.kernel
def _orient_vv_contact_normals(
    mesh_ids: wp.array[wp.uint64],
    mesh_kind: wp.array[wp.int32],
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    cpairs: pairs.PairsData,  # type: ignore
    bases: pairs.ContactBasesData,  # type: ignore
    n_u: wp.int32,
):
    k = wp.tid()
    if wp.uint64(k) >= cpairs.prefix[n_u]:  # type: ignore
        return
    vj = wp.int32(cpairs.v[k])
    i, j = meshes.V[wp.int32(cpairs.u[k])], meshes.V[vj]
    m = lower_bound(meshes.VP, meshes.VP.shape[0], vj + 1) - 1  # type: ignore
    # Angle-weighted vertex pseudonormal of j (the sign on open partners and for a vertex on the surface)
    pn = wp.vec3f(0.0)
    for a in range(meshes.GVHEp[j], meshes.GVHEp[j + 1]):
        f = halfedges.face_of_half_edge(meshes.GVHEadj[a])
        finds = meshes.F[f]
        p, q = finds[1], finds[2]
        if finds[1] == j:
            p, q = finds[2], finds[0]
        elif finds[2] == j:
            p, q = finds[0], finds[1]
        e1, e2 = wp.normalize(x[p] - x[j]), wp.normalize(x[q] - x[j])
        angle = wp.acos(wp.clamp(wp.dot(e1, e2), wp.float32(-1), wp.float32(1)))
        pn += angle * _unit_face_normal(x, meshes.F, f)
    n = _oriented_normal(x[i], x[i] - x[j], pn, mesh_ids[m], mesh_kind[m])
    t, b = pairs._build_contact_basis(n)
    bases.n[k] = n
    bases.t[k] = t
    bases.b[k] = b


@wp.kernel
def _orient_ve_contact_normals(
    mesh_ids: wp.array[wp.uint64],
    mesh_kind: wp.array[wp.int32],
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    cpairs: pairs.PairsData,  # type: ignore
    bases: pairs.ContactBasesData,  # type: ignore
    bary: wp.array[wp.float32],
    n_u: wp.int32,
):
    k = wp.tid()
    if wp.uint64(k) >= cpairs.prefix[n_u]:  # type: ignore
        return
    i = meshes.V[wp.int32(cpairs.u[k])]
    he = wp.int32(cpairs.v[k])
    f = halfedges.face_of_half_edge(he)
    m = lower_bound(meshes.FP, meshes.FP.shape[0], f + 1) - 1  # type: ignore
    xa = x[halfedges.incoming_vertex(meshes.F, he)]
    xb = x[halfedges.outgoing_vertex(meshes.F, he)]
    xc = (wp.float32(1) - bary[k]) * xa + bary[k] * xb
    # Edge pseudonormal: the sum of its faces' unit normals (the sign on open partners and for a vertex on the
    # surface)
    pn = _unit_face_normal(x, meshes.F, f)
    fo = meshes.GHEF[he][1]
    if fo >= 0:
        pn += _unit_face_normal(x, meshes.F, fo)
    n = _oriented_normal(x[i], x[i] - xc, pn, mesh_ids[m], mesh_kind[m])
    t, b = pairs._build_contact_basis(n)
    bases.n[k] = n
    bases.t[k] = t
    bases.b[k] = b


@wp.kernel
def _orient_vf_contact_normals(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    cpairs: pairs.PairsData,  # type: ignore
    bases: pairs.ContactBasesData,  # type: ignore
    n_u: wp.int32,
):
    k = wp.tid()
    if wp.uint64(k) >= cpairs.prefix[n_u]:  # type: ignore
        return
    # The closest point is inside the face, so the face's (outward) normal is the contact normal on both
    # sides of the surface
    n = _unit_face_normal(x, meshes.F, wp.int32(cpairs.v[k]))
    t, b = pairs._build_contact_basis(n)
    bases.n[k] = n
    bases.t[k] = t
    bases.b[k] = b


@wp.kernel
def _filter_initial_step(
    sds: wp.array[wp.float32],
    meshes: MultiMeshData,  # type: ignore
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    x: wp.array[wp.vec3f],
):
    tid = wp.tid()
    v = tid
    i = meshes.V[v]
    if dmask[i] != 0:
        return
    if sds[v] <= wp.float32(0):
        x[i] = xt[i]  # type: ignore


def _mesh_kind(F: np.ndarray, X: np.ndarray) -> int:
    """Kind of the triangle mesh F (|# triangles| x 3) over points X (|# points| x 3).

    MESH_CLOSED if every edge is shared by exactly 2 triangles, else MESH_PLANE if all its vertices lie in
    the plane of its first triangle (plane colliders), else MESH_OPEN.
    """
    E = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(E, axis=0, return_counts=True)
    if np.all(counts == 2):
        return int(MESH_CLOSED)
    P = X[np.unique(F)]
    n = np.cross(X[F[0, 1]] - X[F[0, 0]], X[F[0, 2]] - X[F[0, 0]])
    n /= np.linalg.norm(n)
    diameter = np.linalg.norm(P.max(axis=0) - P.min(axis=0))
    planar = np.all(np.abs((P - X[F[0, 0]]) @ n) <= 1e-6 * diameter)
    return int(MESH_PLANE) if planar else int(MESH_OPEN)


class Params(SerializableMixin):
    max_dist = DocField(
        0.05,
        "Maximum distance for closest point computations. It must exceed the detection threshold "
        "Contact.ContactDynamics.dmin + band, and bounds the signed distances that are tracked.",
    )
    band = DocField(
        1e-3,
        "Contacts are created for vertices closer than Contact.ContactDynamics.dmin + band to another mesh "
        "(or inside it). The band above the AL margin dmin keeps resting contacts, which the AL pushes to dmin, "
        "detected, so their multipliers persist. (Replaces the creation threshold `dmin`, an absolute signed "
        "distance whose default -1e-4 detected penetrating vertices only.)",
    )
    use_step_filter = DocField(
        False,
        "Whether to apply initial step filter.",
    )
    deactivate = DocField(False, "Whether to deactivate contact detection.")
    # TODO: Other within-solve rules were measured and rejected. From the investigate/stable-contact-set report
    # (2026-11-06): freezing without the drop (each pair keeps its step-start contact all solve, re-detection only
    # adds) kept the 6k-14k pairs that the step-start widening finds on BasketOfFruit as decayed proximal Hessian
    # blocks: AAAVBD +27% and VBD +10% median outer ratio at 5 x (3 x 50), +5% step time. Hysteresis alone (keep a
    # pair's contact while the re-detected gap is within 0.05 dmin and its normal within 5 deg) cut DragonImpact's
    # feature changes 7x, but ~160 large switches kept the period-2 cycle (13 um per outer iteration). Dropping a
    # kept pair as soon as the full dual update zeroes its lambda_n (unpressed: removed from that re-detection,
    # instead of when its gap leaves dmin + band) was much worse (2026-11-06, scratch experiment): DragonImpact
    # AAAVBD 10 x 100 from 7 s dropped 17k of 30k pairs at the first re-detection and re-added 900-10,000 dropped
    # pairs per re-detection, ending at ratio 2534 (band rule 10.3), 217 um per outer iteration (0.67) and
    # feasibility 2.16 (0.12); BasketOfFruit 5 x (3 x 50) max outer ratio AAAVBD 4601 (1283), VBD 1807 (509),
    # feasibility max 2.40 / 1.18 (0.85 / 1.44); on the basket's VBD 3 x 50 solves at step 51 it removed 24k-29k
    # pairs at the first re-detection and re-added 14k-20k at the next (band rule: 30-1,100 re-added), with
    # penetration up to 6.9 x the tolerance mid-solve (0.82). Unpressed pairs in the band exert no force, but
    # dropping them throws away pairs that the next subproblem presses again; the band is the better test that a
    # pair has separated (maintainer's expectation, confirmed). Untried next
    # steps: re-linearize only once the AL converged on the frozen set (an outer-outer loop), or evaluate each kept
    # pair's exact point-triangle distance in the vertex solves instead of its frozen plane.
    freeze_within_solve = DocField(
        True,
        "Keep each pair's linearization within a solve (one substep's outer loop). The solve's first detection (at "
        "xt) is full. At each re-detection, a (surface vertex, other mesh) pair that the previous detection has keeps "
        "its feature (face, edge or vertex), barycentrics and basis while its stored linearized gap is below the "
        "detection threshold (dmin + band); a pair whose gap leaves it is detected afresh (dropped, or replaced by "
        "a new feature), and new pairs are added. The solve's last detection is full, so the next solve warm-starts "
        "from keys detected at its xt: at a budget of 1 outer iteration this equals the default. A fixed constraint "
        "set per solve, which the augmented Lagrangian assumes: re-linearizing at every outer iteration made "
        "ArmadilloSpaghetti.DragonImpact's contact set cycle with period 2. VertexSdf only. On by default "
        "(maintainer, 2026-11-06); off re-linearizes at every re-detection, as before.",
    )


class Sd(ContactDetection):
    """Vertex-(Mesh)SDF based contact detection."""

    # TODO: CubeFallingOnCube's cubes pass through each other, before and after the detection band (2026-10-20).
    # Detection is vertex-only (no edge-edge pairs), which misses edges crossing with no vertex inside; edge-edge
    # pairs are the likely fix.

    _wp_meshes: list[wp.Mesh]
    _mesh_ids: wp.array[wp.uint64]
    _block_dim: int  # threads per contact vertex in detection (detection_block_dim)
    _mesh_kind: wp.array[wp.int32]  # (# meshes,) MESH_CLOSED, MESH_OPEN or MESH_PLANE
    _mesh_depth: wp.array[wp.float32]  # (# meshes,) contact depth of plane meshes (inf: half-space)
    _sds: wp.array[wp.float32]  # (# verts,) signed distances
    _mesh_streams: list[wp.Stream]
    _normal_streams: list[wp.Stream]
    # Params.freeze_within_solve, allocated while it is on: the working set (a copy of the last detection's forward
    # VV/VE/VF pairs, bases and barycentrics), per working slot whether the last detection kept it, and a (1,) flag
    # set while the last detection kept frozen stencils (cleared by a full one)
    _working = None  # pairs.ContactPairsData
    _kept: tuple[wp.array, wp.array, wp.array] | None = None
    _frozen: wp.array | None = None

    def __init__(self, params: Params | None):
        self.params = params or Params()
        self._contact_dmin = 0.0

    def set_contact_margin(self, dmin: float):
        """Set the AL contact margin (Contact.ContactDynamics.dmin): contacts are detected below dmin + band."""
        self._contact_dmin = float(dmin)

    def register_handles(
        self,
        xt: wp.array[wp.vec3f],
        xk: wp.array[wp.vec3f],
        x: wp.array[wp.vec3f],
        xtilde: wp.array[wp.vec3f],
        meshes: MultiMesh,
        contacts: pairs.ContactPairs,
        dmask: wp.array[wp.int32] | None = None,
    ):
        super().register_handles(xt, xk, x, xtilde, meshes, contacts, dmask)

        FP = self._meshes.data.FP.numpy()
        Fb = [
            cp.asarray(self._meshes.data.F)[FP[b] : FP[b + 1], :].ravel()
            for b in range(FP.shape[0] - 1)
        ]
        self._wp_meshes = [
            wp.Mesh(
                self._x,
                wp.array(
                    data=Fb[b],
                    dtype=wp.int32,
                ),
            )
            for b in range(FP.shape[0] - 1)
        ]
        self._mesh_ids = wp.array([m.id for m in self._wp_meshes], dtype=wp.uint64)
        self._block_dim = detection_block_dim(len(self._wp_meshes))
        # The same meshes over xt, for a time step's first detection (from_xt)
        self._wp_meshes_xt = [wp.Mesh(self._xt, wp.array(data=Fb[b], dtype=wp.int32)) for b in range(FP.shape[0] - 1)]
        self._mesh_ids_xt = wp.array([m.id for m in self._wp_meshes_xt], dtype=wp.uint64)
        # (# meshes,) each component's largest motion |x0 - xt| from xt to the time step's starting iterate x0
        self._motion = wp.zeros(FP.shape[0] - 1, dtype=wp.float32)
        X = self._x.numpy()
        self._mesh_kind = wp.array(
            [_mesh_kind(Fb[b].get().reshape(-1, 3), X) for b in range(len(Fb))],
            dtype=wp.int32,
        )
        self._mesh_depth = wp.full(len(Fb), float("inf"), dtype=wp.float32)
        self._sds = wp.full(
            (self._meshes.n_verts,), self.params.max_dist, dtype=wp.float32
        )
        self._mesh_streams = [wp.Stream() for _ in range(len(self._wp_meshes))]
        self._normal_streams = [wp.Stream() for _ in range(3)]  # (vv, ve, vf)
        self._working, self._kept, self._frozen = None, None, None
        if self.params.freeze_within_solve:
            self._allocate_working_set()

    def _allocate_working_set(self):
        """Allocate the working set of Params.freeze_within_solve: buffers like the contact storage's forward
        VV/VE/VF pairs."""
        C = self._contacts.read_data[0]

        def pairs_like(P):
            Q = pairs.PairsData()
            Q.prefix, Q.u, Q.v = wp.zeros_like(P.prefix), wp.zeros_like(P.u), wp.zeros_like(P.v)
            return Q

        def bases_like(B):
            Q = pairs.ContactBasesData()
            Q.n, Q.t, Q.b = wp.zeros_like(B.n), wp.zeros_like(B.t), wp.zeros_like(B.b)
            return Q

        W = pairs.ContactPairsData()
        W.vv, W.ve, W.vf = pairs_like(C.vv), pairs_like(C.ve), pairs_like(C.vf)
        W.vv_bases, W.ve_bases, W.vf_bases = bases_like(C.vv_bases), bases_like(C.ve_bases), bases_like(C.vf_bases)
        W.ve_bary, W.vf_bary = wp.zeros_like(C.ve_bary), wp.zeros_like(C.vf_bary)
        W.ee, W.ee_bases, W.ee_bary = C.ee, C.ee_bases, C.ee_bary  # unused: vertex SDF has no edge-edge pairs
        self._working = W
        self._kept = tuple(wp.zeros(P.u.shape[0], dtype=wp.int32) for P in (C.vv, C.ve, C.vf))
        self._frozen = wp.zeros(1, dtype=wp.int32)

    def _store_working_set(self):
        """Copy the detection's forward VV/VE/VF pairs, bases and barycentrics into the working set (whole buffers:
        the counts are on the device)."""
        C, W = self._contacts.read_data[0], self._working
        for P, Q in ((C.vv, W.vv), (C.ve, W.ve), (C.vf, W.vf)):
            wp.copy(Q.prefix, P.prefix)
            wp.copy(Q.u, P.u)
            wp.copy(Q.v, P.v)
        for B, Q in ((C.vv_bases, W.vv_bases), (C.ve_bases, W.ve_bases), (C.vf_bases, W.vf_bases)):
            wp.copy(Q.n, B.n)
            wp.copy(Q.t, B.t)
            wp.copy(Q.b, B.b)
        wp.copy(W.ve_bary, C.ve_bary)
        wp.copy(W.vf_bary, C.vf_bary)

    @property
    def freezes_within_solve(self) -> bool:
        return bool(self.params.freeze_within_solve) and not self.inactive

    @property
    def last_detection_frozen(self) -> wp.array | None:
        return self._frozen if self.freezes_within_solve else None

    def set_plane_depths(self, depths: np.ndarray):
        """Set how far below each plane mesh a vertex is still in contact with it.

        Args:
            depths (np.ndarray): (# meshes,) depth per contact mesh component, used for MESH_PLANE meshes
                only. inf (the default) makes a plane a half-space, as for infinite plane colliders.
                Finite plane colliders use their thickness.
        """
        self._mesh_depth.assign(np.asarray(depths, dtype=np.float32))

    contact_free = False
    """Set for contact surfaces with a single component (one body, no colliders): vertex SDF detection
    has no self-contact, so such scenes can't have contacts and detection is skipped."""

    @property
    def inactive(self) -> bool:
        """Whether detection, refit and the step filter are skipped (``deactivate`` or contact-free)."""
        return bool(self.params.deactivate) or self.contact_free

    def on_time_step_started(self):
        self.request_step_filter = True
        if self.inactive:
            return
        # x is the step's starting iterate: the solvers call this after the time integration setup
        self._motion.zero_()
        wp.launch(
            _component_motion,
            dim=self._meshes.n_verts,
            inputs=[self._xt, self._x, self._meshes.data, self._motion],  # type: ignore
        )

    @profiling.scoped("contact.detect")
    def prepare_detection(self):
        """Refit the meshes' BVHs. wp.Mesh.refit allocates scan temporaries, so it can't run in a
        conditional graph body."""
        if self.inactive:
            return
        self._refit()

    def _refit(self, meshes: list | None = None):
        main_stream = wp.get_stream()
        for mesh, stream in zip(meshes or self._wp_meshes, self._mesh_streams):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                mesh.refit()
            main_stream.wait_stream(stream)

    @profiling.scoped("contact.detect")
    def detect_contacts(self, from_xt: bool = False, prepared: bool = False, fresh: bool = False):
        if self.inactive:
            return
        # Params.freeze_within_solve: re-detections within a solve keep the working set's linearizations; a solve's
        # first detection (from_xt) and its last (fresh) are full
        freeze = bool(self.params.freeze_within_solve)
        if freeze and self._working is None:
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "Contact.CDType.VertexSdf.freeze_within_solve was turned on after the detector was built: reset "
                    "the simulation"
                )
            self._allocate_working_set()
        frozen = freeze and not (from_xt or fresh)
        # A time step's first detection (the solvers' initialize_solve, from_xt) is at xt, as C++
        # vbd::InitializeSolve builds the constraint set at xt and OGC honors from_xt, and also finds the pairs
        # that the step's starting iterate x (the init strategy's) may bring within the threshold (see the
        # kernel). Detecting at the predictor x instead (until 2026-10-22) started fast bodies deep inside their
        # partners, where the closest point of a thin part can be on its far side: ArmadilloSpaghetti.Fall's
        # first detection of step 58 had 923 penetrating VF contacts up to 25 mm deep (pillars 5 cm in radius),
        # and step 60 diverged; CubeFallingOnCube's top cube fell through the bottom one (aligned grids: the
        # predictor put vertices on the bottom cube's vertical edges, with horizontal normals).
        x = self._xt if from_xt else self._x
        if not prepared:
            # unscoped, so the refit doesn't nest as contact.detect/contact.detect
            self._refit(self._wp_meshes_xt if from_xt else self._wp_meshes)
        main_stream = wp.get_stream()
        self._contacts.clear()
        n_verts = self._meshes.data.V.shape[0]
        block_dim = self._block_dim
        wp.launch(
            _detect_vertex_mesh_contacts,
            dim=n_verts * block_dim,
            inputs=[
                self._mesh_ids_xt if from_xt else self._mesh_ids,
                self._mesh_kind,
                self._mesh_depth,
                self.params.max_dist,  # max_dist
                self._contact_dmin + self.params.band,  # detection threshold
                int(from_xt),  # widen
                self._x,  # x0: the iterate
                self._motion,
                self._sds,
                x,
                self._meshes.data,  # type: ignore
                self._contacts.write_data,  # type: ignore
                int(frozen),
                # Not read without freeze
                self._working if frozen else self._contacts.read_data[0],
                *(self._kept if frozen else (None, None, None)),
            ],
            block_dim=block_dim,
            stream=main_stream,
        )
        self._contacts.assemble_contacts(x, with_reverse_contacts=True)
        # Normals point from the partner surface to the vertex, on both sides of the surface: assemble_contacts
        # computes them as from outside (OGC's convention). VV and VE sign them with the detection's inside/outside
        # test, on the meshes detection queried.
        contact_data = self._contacts.read_data[0]
        signs = [self._mesh_ids_xt if from_xt else self._mesh_ids, self._mesh_kind]
        for kernel, capacity, prefix, extra, stream in zip(
            (_orient_vv_contact_normals, _orient_ve_contact_normals, _orient_vf_contact_normals),
            self._contacts.capacity[:3],
            (signs, signs, []),
            (
                [contact_data.vv, contact_data.vv_bases],
                [contact_data.ve, contact_data.ve_bases, contact_data.ve_bary],
                [contact_data.vf, contact_data.vf_bases],
            ),
            self._normal_streams,
        ):  # (vv, ve, vf)
            stream.wait_stream(main_stream)
            wp.launch(
                kernel=kernel,
                dim=capacity,
                inputs=prefix + [x, self._meshes.data] + extra + [self._meshes.n_verts],
                stream=stream,
            )
            main_stream.wait_stream(stream)
        if frozen:
            wp.launch(
                _restore_kept_stencils,
                dim=max(self._contacts.capacity[:3]),
                inputs=[contact_data, self._working, *self._kept, self._meshes.n_verts],
                stream=main_stream,
            )
        if freeze:
            self._store_working_set()
            self._frozen.fill_(int(frozen))

    @profiling.scoped("contact.filter")
    def filter_step(self):
        if self.inactive:
            return
        if self.request_step_filter and self.params.use_step_filter:
            n_verts = self._meshes.n_verts
            wp.launch(
                kernel=_filter_initial_step,
                dim=n_verts,
                inputs=[self._sds, self._meshes.data, self._xt, self._dmask, self._x],
            )
        self.request_step_filter = False

    def on_time_step_ended(self):
        pass

    @property
    def supports_conditional_capture(self) -> bool:
        # Only the refit allocates, and it lives in prepare_detection
        return True


@wp.kernel
def _edge_region_barycentrics(
    mesh: wp.uint64,
    x: wp.array[wp.vec3f],
    P: wp.array[wp.vec3f],
    warp_w: wp.array[wp.float32],
    uvw: wp.array[wp.vec3f],
):
    t = wp.tid()
    qp = wp.mesh_query_point_no_sign(mesh, x[t], wp.float32(10))
    warp_w[t] = wp.float32(1) - qp.u - qp.v  # type: ignore
    uvw[t] = queries.closest_point_triangle(x[t], P[0], P[1], P[2])


class TestFeatureClassification(unittest.TestCase):
    """The closest feature of a detected vertex (VF, VE or VV) comes from exact zero barycentrics of the
    closest point's region, not from Warp's mesh query."""

    def test_edge_region_is_exact(self):
        """Points beyond edge jk of a triangle: Warp's w = 1 - u - v is often not exactly 0 there (298 of
        these 1000 points, |w| up to 3e-8, Warp 1.17), the region-based barycentrics are."""
        rng = np.random.default_rng(0)
        P = np.array([[0.1, 0.0, 0.0], [0.9, 0.2, 0.0], [0.3, 0.8, 0.05]], dtype=np.float32)
        e = P[1] - P[0]
        normal = np.cross(e, P[2] - P[0])
        normal /= np.linalg.norm(normal)
        out = np.cross(e, normal)
        out *= -np.sign(np.dot(out, P[2] - P[0])) / np.linalg.norm(out)  # in plane, away from P[2]
        n = 1000
        X = (
            P[0]
            + rng.uniform(0.05, 0.95, (n, 1)) * e
            + rng.uniform(1e-3, 1e-2, (n, 1)) * out
            + rng.uniform(-1e-2, 1e-2, (n, 1)) * normal
        ).astype(np.float32)
        mesh = wp.Mesh(wp.array(P, dtype=wp.vec3f), wp.array([0, 1, 2], dtype=wp.int32))
        warp_w = wp.zeros(n, dtype=wp.float32)
        uvw = wp.zeros(n, dtype=wp.vec3f)
        wp.launch(
            _edge_region_barycentrics,
            dim=n,
            inputs=[mesh.id, wp.array(X, dtype=wp.vec3f), wp.array(P, dtype=wp.vec3f), warp_w, uvw],
        )
        warp_w, uvw = warp_w.numpy(), uvw.numpy()
        self.assertGreater(int(np.count_nonzero(warp_w)), 0, "no inexact Warp barycentric: the case isn't covered")
        np.testing.assert_array_equal(uvw[:, 2], 0)
        self.assertTrue(np.all((uvw[:, 0] > 0) & (uvw[:, 1] > 0)))

    def test_sharp_edge(self):
        """Vertices 2.5 mm outside a sharp wedge's apex edge (22.6 deg), in the edge's region and behind one
        face's plane: each is a VE contact at its true distance, with no penetration. With Warp's
        barycentrics, one of them was a VF contact with the face normal, at gap -2.3 mm."""
        from pbatoolkit import pbat, pypbat
        from ....state import SimulationState, load_params

        # The wedge: cross-section (x, z) apex (0, 0), (-1, 0.2), (-1, -0.2), extruded along y over [0, 1]
        W = np.array([[x, y, z] for y in (0.0, 1.0) for x, z in ((0, 0), (-1, 0.2), (-1, -0.2))])
        C = [np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])]
        n_top = np.array([0.2, 0.0, 1.0]) / np.hypot(0.2, 1.0)
        n_bottom = np.array([0.2, 0.0, -1.0]) / np.hypot(0.2, 1.0)
        distance = 2.5e-3
        X, probes = [W], []
        for q, y in enumerate(np.linspace(0.15, 0.9, 6)):
            # Mostly along one face's normal: behind the other face's plane
            d = 0.1 * n_top + n_bottom if q % 2 == 0 else n_top + 0.1 * n_bottom
            p = np.array([0.0, y, 0.0]) + distance * d / np.linalg.norm(d)
            probes.append(len(W) + 4 * q)
            X.append([p, p + [0.05, 0, 0.02], p + [0.05, 0.02, -0.02], p + [0.05, -0.02, -0.02]])
            C.append(np.array([[0, 1, 2, 3]]) + probes[-1])
        X, C = np.vstack(X), np.vstack(C)
        for e in C:  # positive volumes
            a, b, c, d = X[e]
            if np.dot(np.cross(b - a, c - a), d - a) < 0:
                e[[2, 3]] = e[[3, 2]]
        fem = pbat.sim.dynamics.FemElastoDynamics(X.T, C.T)
        fem.set_mass_matrix(1e3)
        fem.set_elastic_energy(*pypbat.fem.lame_coefficients(1e6, 0.45))
        fem.set_time_integration_scheme(dt=1e-2, s=1)
        state = SimulationState(fem, load_params(fem), None)
        for k in ("vv", "ve", "vf"):  # a slot per vertex
            setattr(state.contact_storage_params, f"n_{k}_contact_capacity", 1.0)
        state.reset()
        dmin = float(state.contact.params.dmin)
        self.assertLess(dmin, distance)
        state.detector.detect_contacts()
        state.contact.update_constraint_set()
        contacts = state.detector._contacts
        n_vv, n_ve, n_vf, _ = (int(c) for c in contacts.num_contacts)
        self.assertEqual((n_vv, n_ve, n_vf), (0, len(probes), 0))
        V, F = state.multimesh.data.V.numpy(), state.multimesh.data.F.numpy()
        x = state.fem.data.x.numpy()
        u, he = contacts.ve_contacts
        data = contacts.read_data[0]
        t, normals = data.ve_bary.numpy(), data.ve_bases.n.numpy()
        self.assertEqual(sorted(int(V[i]) for i in u[:n_ve]), probes)
        for c in range(n_ve):
            f, h = divmod(int(he[c]), 3)
            xc = (1 - t[c]) * x[F[f][h]] + t[c] * x[F[f][(h + 1) % 3]]
            self.assertAlmostEqual(float(np.dot(x[V[u[c]]] - xc, normals[c])), distance, delta=1e-6)
        state.convergence._measure_penetration()
        self.assertEqual(float(state.convergence.penetration.numpy()[0]), 0.0)


class TestDetectionBlockDim(unittest.TestCase):
    def test_partners_fit_the_thread_lists(self):
        """Whole warps, one partner per thread up to MAX_BLOCK_DIM meshes, and an error past the lists' size."""
        self.assertEqual(detection_block_dim(1), 32)
        self.assertEqual(detection_block_dim(9), 32)  # DragonImpact
        self.assertEqual(detection_block_dim(33), 64)
        self.assertEqual(detection_block_dim(MAX_BLOCK_DIM), MAX_BLOCK_DIM)
        self.assertEqual(detection_block_dim(MAX_BLOCK_DIM + 1), MAX_BLOCK_DIM)  # 2 partners per thread
        capacity = min(int(MAX_VV_PER_THREAD), int(MAX_VE_PER_THREAD), int(MAX_VF_PER_THREAD))
        self.assertEqual(detection_block_dim(capacity * MAX_BLOCK_DIM), MAX_BLOCK_DIM)
        with self.assertRaises(ValueError):
            detection_block_dim(capacity * MAX_BLOCK_DIM + 1)


class TestNormalOrientation(unittest.TestCase):
    def test_folded_partner(self):
        """An outside vertex over the fold edge of a partner whose surface tet inverted gets a normal pointing to
        it (DragonImpact step 482: a pillar vertex 1 cm over an armadillo edge whose faces' normals had cosine
        -0.9998 got a gap of -dmin, entry 8 of docs/solver-accuracy-notes.md).

        The partner is a triangular bipyramid (equator a, b, c, apices s, t; three tets around st). Apex t is moved
        just across the plane of face abs, on s's side of edge ab, so that only the tet abst inverts and face abt
        folds back over abs (cosine -0.9998, as on the armadillo). A probe vertex 2.5 mm over ab and 0.5 mm beyond
        it is outside the partner (winding number 0) and closest to the edge, where the edge pseudonormal's sign
        says inside: the pseudonormal-signed normal gave n . d = -|d|."""
        from pbatoolkit import pbat, pypbat
        from ....state import SimulationState, load_params

        a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.5, -1.0, 0])
        s, t = np.array([0.5, -0.3, 0.5]), np.array([0.5, -0.3, -0.5])
        nabs = np.cross(s - a, b - a)  # outward normal of face abs
        nabs /= np.linalg.norm(nabs)
        w = (s - a) - np.dot(s - a, b - a) * (b - a)  # in face abs, from ab toward s
        w /= np.linalg.norm(w)
        phi = 0.02
        t_folded = np.array([0.5, 0, 0]) + 0.3 * (np.cos(phi) * w + np.sin(phi) * nabs)
        p = np.array([0.4, 0, 0]) + 2.5e-3 * nabs - 5e-4 * w
        distance = np.hypot(2.5e-3, 5e-4)  # to its closest point (0.4, 0, 0) on ab
        X = np.vstack([a, b, c, s, t, p, p + [0.0, 0.05, 0.02], p + [0.02, 0.05, -0.02], p + [-0.02, 0.05, -0.02]])
        C = np.array([[0, 1, 3, 4], [0, 3, 4, 2], [1, 3, 4, 2], [5, 6, 7, 8]])
        for e in C:  # positive volumes at rest
            q0, q1, q2, q3 = X[e]
            if np.dot(np.cross(q1 - q0, q2 - q0), q3 - q0) < 0:
                e[[2, 3]] = e[[3, 2]]
        fem = pbat.sim.dynamics.FemElastoDynamics(X.T, C.T)
        fem.set_mass_matrix(1e3)
        fem.set_elastic_energy(*pypbat.fem.lame_coefficients(1e6, 0.45))
        fem.set_time_integration_scheme(dt=1e-2, s=1)
        state = SimulationState(fem, load_params(fem), None)
        for k in ("vv", "ve", "vf"):  # a slot per vertex
            setattr(state.contact_storage_params, f"n_{k}_contact_capacity", 1.0)
        state.reset()
        x = state.fem.data.x.numpy()
        x[4] = t_folded
        state.fem.data.x.assign(x)
        x = x.astype(np.float64)
        volumes = [np.dot(np.cross(x[e[1]] - x[e[0]], x[e[2]] - x[e[0]]), x[e[3]] - x[e[0]]) for e in C[:3]]
        self.assertEqual([v < 0 for v in volumes], [True, False, False], "only the tet abst must invert")
        dmin = float(state.contact.params.dmin)
        self.assertLess(dmin, distance)
        state.detector.detect_contacts()
        state.contact.update_constraint_set()
        contacts = state.detector._contacts
        self.assertEqual(tuple(int(k) for k in contacts.num_contacts)[:3], (0, 1, 0))
        V, F = state.multimesh.data.V.numpy(), state.multimesh.data.F.numpy()
        u, he = contacts.ve_contacts
        self.assertEqual(int(V[u[0]]), 5)
        f, h = divmod(int(he[0]), 3)
        self.assertEqual(sorted((int(F[f][h]), int(F[f][(h + 1) % 3]))), [0, 1])
        data = contacts.read_data[0]
        bary, n = float(data.ve_bary.numpy()[0]), data.ve_bases.n.numpy()[0]
        xc = (1 - bary) * x[F[f][h]] + bary * x[F[f][(h + 1) % 3]]
        self.assertAlmostEqual(float(np.dot(x[5] - xc, n)), distance, delta=1e-6)
        state.convergence._measure_penetration()
        self.assertEqual(float(state.convergence.penetration.numpy()[0]), 0.0)


class TestSurfaceVertices(unittest.TestCase):
    def test_interior_nodes_have_no_pairs(self):
        """Scenes with colliders build their contact surface from a triangle mesh over every node
        (colliders.add_colliders): its vertices must be the surface ones only, or interior nodes run detection and
        get pairs (158 of ArmadilloSpaghetti.Fall's 1647 pairs at step 50, CPU count, 2026-10-22). Two cubes with
        a center node each, 1 mm apart, with a detection threshold that reaches the centers."""
        from pbatoolkit import pbat, pypbat
        from ....colliders import add_colliders
        from ....state import CDType, SimulationState, load_params

        corners = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=np.float64)
        faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        C = [[q[0], q[k], q[k + 1], 8] for q in faces for k in (1, 2)]  # 2 tets per face, to the center node
        X = np.vstack([corners, [0.5, 0.5, 0.5], corners + [0.0, 0.0, 1.001], [0.5, 0.5, 1.501]])
        E = np.vstack([C, np.array(C) + 9])
        for e in E:  # positive volumes
            a, b, c, d = X[e]
            if np.dot(np.cross(b - a, c - a), d - a) < 0:
                e[[2, 3]] = e[[3, 2]]
        fem = pbat.sim.dynamics.FemElastoDynamics(X.T, E.T)
        fem.set_mass_matrix(1e3)
        fem.set_elastic_energy(*pypbat.fem.lame_coefficients(1e6, 0.45))
        fem.set_time_integration_scheme(dt=1e-2, s=1)
        _, geometry = add_colliders(fem, np.array([0, 9, 18]), [])
        self.assertEqual(np.asarray(geometry.mesh.V).tolist(), [i for i in range(18) if i not in (8, 17)])
        state = SimulationState(fem, load_params(fem), None, geometry)
        state.cd_params[CDType.VertexSdf].band = 0.6
        state.cd_params[CDType.VertexSdf].max_dist = 1.0
        for k in ("vv", "ve", "vf"):
            setattr(state.contact_storage_params, f"n_{k}_contact_capacity", 1.0)
        state.reset()
        self.assertGreater(float(state.contact.params.dmin) + 0.6, 0.501, "the threshold must reach the centers")
        state.detector.detect_contacts()
        contacts = state.detector._contacts
        V = state.multimesh.data.V.numpy()
        u = np.concatenate([contacts.vv_contacts[0], contacts.ve_contacts[0], contacts.vf_contacts[0]])
        self.assertGreater(len(u), 0)
        self.assertFalse(np.isin(V[u], [8, 17]).any(), "an interior node has a pair")


class TestFirstDetection(unittest.TestCase):
    """A time step's first detection is at xt and reaches the pairs the starting iterate brings within the
    threshold: two aligned unit cubes, the top one 5 mm above the fixed bottom one and falling at 3 m/s, so that
    its predictor ends 2.5 cm inside (CubeFallingOnCube's tunneling: detected at the predictor, the vertices were
    on the bottom cube's vertical edges, closest to its side faces, with horizontal normals)."""

    gap = 5e-3

    def _state(self):
        from pbatoolkit import pbat, pypbat
        from ....state import SimulationState, load_params

        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        C = np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [6, 7, 5, 3], [0, 5, 3, 6]])
        X = np.vstack([V, V + [0.0, 0.0, 1.0 + self.gap]]).astype(np.float64)  # aligned grids
        fem = pbat.sim.dynamics.FemElastoDynamics(X.T, np.vstack([C, C + 8]).T)
        fem.set_mass_matrix(1e3)
        fem.set_elastic_energy(*pypbat.fem.lame_coefficients(1e6, 0.45))
        fem.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
        fem.set_time_integration_scheme(dt=1e-2, s=1)
        fem.constrain(np.repeat([1, 0], 8))
        v = np.zeros((3, 16))
        v[2, 8:] = -3.0
        fem.set_initial_conditions(fem.x, v)
        state = SimulationState(fem, load_params(fem), None)
        for k in ("vv", "ve", "vf"):  # a slot per vertex
            setattr(state.contact_storage_params, f"n_{k}_contact_capacity", 1.0)
        state.reset()
        return state

    def test_detects_at_xt(self):
        state = self._state()
        threshold = float(state.contact.params.dmin) + float(state.detector.params.band)
        self.assertLess(threshold, self.gap, "the pairs must be beyond the threshold at xt")
        state.fem.setup_time_integration_optimization(state.step_init_strategy())
        self.assertLess(float(state.fem.data.x.numpy()[8:, 2].min()), 1.0 - 0.02, "the predictor must penetrate")
        state.detector.on_time_step_started()
        state.detector.detect_contacts(from_xt=True)
        contacts = state.detector._contacts
        self.assertEqual(tuple(int(c) for c in contacts.num_contacts)[:3], (8, 0, 0))
        V = state.multimesh.data.V.numpy()
        xt = state.fem.xt.numpy()
        u, w = contacts.vv_contacts
        normals = contacts.read_data[0].vv_bases.n.numpy()[: len(u)]
        for c in range(len(u)):
            self.assertGreater(abs(float(normals[c][2])), 0.99)  # the cubes' facing vertices
            self.assertAlmostEqual(float(np.dot(xt[V[u[c]]] - xt[V[w[c]]], normals[c])), self.gap, delta=1e-5)


def _vf_pairs(state) -> dict:
    """The detected VF pairs as {surface vertex: (face, barycentrics, normal)}."""
    contacts = state.detector._contacts
    u, f = contacts.vf_contacts
    data = contacts.read_data[0]
    bary, n = data.vf_bary.numpy(), data.vf_bases.n.numpy()
    return {int(u[c]): (int(f[c]), bary[c].copy(), n[c].copy()) for c in range(len(u))}


class TestFreezeWithinSolve(unittest.TestCase):
    """Params.freeze_within_solve: a fixed unit cube and three probe tets whose lowest vertices hover over its top
    face (two triangles; the probes start inside the triangle at the origin's x-axis edge, x > y and x + y < 1, so
    that the test holds for either diagonal)."""

    # Lowest vertex of each probe tet (x, y) and its heights over the face, in units of the detection threshold:
    # before and after the move. A slides to the face's other triangle (kept), B rises out of the threshold
    # (dropped), C comes within it (added).
    probes = {"A": ((0.25, 0.1), (0.75, 0.9)), "B": ((0.45, 0.2), None), "C": ((0.6, 0.3), None)}
    heights = {"A": (0.5, 0.6), "B": (0.5, 2.0), "C": (3.0, 0.5)}

    def _state(self, freeze: bool):
        from pbatoolkit import pbat, pypbat
        from ....state import CDType, SimulationState, load_params

        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        C = [np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [6, 7, 5, 3], [0, 5, 3, 6]])]
        X = [V.astype(np.float64)]
        self.node = {}
        for name, ((x, y), _) in self.probes.items():
            p = np.array([x, y, 1.0])
            self.node[name] = 8 + 4 * len(self.node)
            X.append([p, p + [0.05, 0, 0.05], p + [0, 0.05, 0.05], p + [0.03, 0.03, 0.1]])
            C.append(np.array([[0, 1, 2, 3]]) + self.node[name])
        X, C = np.vstack(X), np.vstack(C)
        for e in C:  # positive volumes
            a, b, c, d = X[e]
            if np.dot(np.cross(b - a, c - a), d - a) < 0:
                e[[2, 3]] = e[[3, 2]]
        fem = pbat.sim.dynamics.FemElastoDynamics(X.T, C.T)
        fem.set_mass_matrix(1e3)
        fem.set_elastic_energy(*pypbat.fem.lame_coefficients(1e6, 0.45))
        fem.set_time_integration_scheme(dt=1e-2, s=1)
        state = SimulationState(fem, load_params(fem), None)
        state.cd_params[CDType.VertexSdf].freeze_within_solve = freeze
        for k in ("vv", "ve", "vf"):  # a slot per vertex
            setattr(state.contact_storage_params, f"n_{k}_contact_capacity", 1.0)
        state.reset()
        self.threshold = float(state.contact.params.dmin) + float(state.detector.params.band)
        self.vertex = {name: int(np.flatnonzero(state.multimesh.data.V.numpy() == i)[0]) for name, i in self.node.items()}
        return state

    def _place(self, state, after: bool):
        """Translate each probe tet so that its lowest vertex is at its position and height."""
        x = state.fem.data.x.numpy()
        for name, (xy0, xy1) in self.probes.items():
            xy = xy1 if after and xy1 is not None else xy0
            h = self.heights[name][int(after)] * self.threshold
            i = self.node[name]
            x[i : i + 4] += np.array([xy[0], xy[1], 1.0 + h]) - x[i]
        state.fem.data.x.assign(x)

    def _detect(self, state, after: bool, **kwargs) -> dict:
        self._place(state, after)
        state.detector.detect_contacts(**kwargs)
        return _vf_pairs(state)

    def test_kept_dropped_added(self):
        state = self._state(freeze=True)
        A, B, C = (self.vertex[k] for k in "ABC")
        first = self._detect(state, after=False, fresh=True)
        self.assertEqual(sorted(first), sorted([A, B]))
        self.assertEqual(int(state.detector.last_detection_frozen.numpy()[0]), 0)
        frozen = self._detect(state, after=True)
        self.assertEqual(int(state.detector.last_detection_frozen.numpy()[0]), 1)
        # (a) A keeps its face, barycentrics and normal, though it is now over the other triangle
        self.assertEqual(sorted(frozen), sorted([A, C]), "B must be dropped, C added")
        self.assertEqual(frozen[A][0], first[A][0])
        np.testing.assert_array_equal(frozen[A][1], first[A][1])
        np.testing.assert_array_equal(frozen[A][2], first[A][2])
        # (b) C is new: its stencil is the fresh one, at its current position
        np.testing.assert_allclose(frozen[C][2], [0, 0, 1], atol=1e-6)
        # (c) A full detection re-linearizes A on the triangle it is over
        fresh = self._detect(state, after=True, fresh=True)
        self.assertEqual(int(state.detector.last_detection_frozen.numpy()[0]), 0)
        self.assertEqual(sorted(fresh), sorted([A, C]))
        self.assertNotEqual(fresh[A][0], first[A][0])
        # The same as without the option
        off = self._state(freeze=False)
        self._detect(off, after=False)
        reference = self._detect(off, after=True)
        self.assertIsNone(off.detector.last_detection_frozen)
        for v in (A, C):
            self.assertEqual(fresh[v][0], reference[v][0])
            np.testing.assert_array_equal(fresh[v][1], reference[v][1])
            np.testing.assert_array_equal(fresh[v][2], reference[v][2])

    def test_dropped_pair_is_detected_afresh(self):
        """A pair whose stored gap leaves the threshold isn't kept: detected afresh (here: A's gap, measured
        along its face's normal, leaves the threshold, and so does its distance)."""
        state = self._state(freeze=True)
        A = self.vertex["A"]
        self._detect(state, after=False, fresh=True)
        self.heights = dict(self.heights, A=(0.5, 1.5))
        self.assertNotIn(A, self._detect(state, after=True))


class TestFreezeLastDetection(unittest.TestCase):
    """With freeze_within_solve, a solve ends on a full detection, also when its outer loop exits early (decided on
    the device), eager and under graph capture: two cubes resting in contact (VBD, 3 outer iterations, early exit
    forced after the first by clearing the outer test's keep flag)."""

    @staticmethod
    def _pairs(state):
        contacts = state.detector._contacts
        data = contacts.read_data[0]
        out = []
        for kind in ("vv", "ve", "vf"):
            u, v = getattr(contacts, f"{kind}_contacts")
            n = getattr(data, f"{kind}_bases").n.numpy()[: len(u)]
            bary = {"vv": None, "ve": data.ve_bary, "vf": data.vf_bary}[kind]
            b = bary.numpy()[: len(u)] if bary is not None else np.zeros(len(u))
            o = np.lexsort((v, u))
            out.append((u[o], v[o], n[o], b[o]))
        return out

    def _assert_fresh(self, state, expect: bool):
        before = self._pairs(state)
        state.detector.detect_contacts(fresh=True)
        after = self._pairs(state)
        same = all(
            np.array_equal(a, b) for pa, pb in zip(before, after) for a, b in zip(pa, pb)
        )
        self.assertEqual(same, expect)

    def _run(self, capture: bool, refresh: bool = True):
        from ...vbd.convergence import _two_resting_cubes
        from ....state import CDType

        state = _two_resting_cubes("VBD")
        state.cd_params[CDType.VertexSdf].freeze_within_solve = True
        state.detector = state._make_contact_detector(state.contact.contacts)
        state.params[state.solver].data.n_max_iters = 3
        state.convergence_params.early_exit = True
        check = state.convergence
        end_outer = check.end_outer

        def stop_after_first(*args, **kwargs):
            end_outer(*args, **kwargs)
            check.keep.zero_()

        check.end_outer = stop_after_first
        original = Sd.last_detection_frozen
        if not refresh:
            Sd.last_detection_frozen = property(lambda self: None)
        try:
            if capture:
                with wp.ScopedCapture() as c:
                    state._advance()
                wp.capture_launch(c.graph)
            else:
                state._advance()
            wp.synchronize()
        finally:
            Sd.last_detection_frozen = original
        return state

    def test_early_exit_ends_fresh(self):
        for capture in (False, True):
            with self.subTest(capture=capture):
                state = self._run(capture)
                self.assertEqual(int(state.detector._frozen.numpy()[0]), 0)
                self._assert_fresh(state, True)

    def test_without_refresh_the_last_detection_is_frozen(self):
        """The check above has teeth: without the refresh after the loop, the early exit's detection is frozen."""
        state = self._run(capture=False, refresh=False)
        self.assertEqual(int(state.detector._frozen.numpy()[0]), 1)
        self._assert_fresh(state, False)


if __name__ == "__main__":
    wp.init()
    unittest.main()

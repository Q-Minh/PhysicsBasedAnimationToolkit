from typing import Tuple

import numpy as np
import cupy as cp
import cuda.compute
import warp as wp
import math

from .multimesh import MultiMesh, MultiMeshData
from . import halfedges
from . import queries
from .. import common
from ...common.fields import DocField
from ...gpu import common


class OgcParams:
    """Parameters for Offset Geometric Contact detection."""

    r = DocField(0.003, "Contact radius. Pairs closer than r are in contact.")
    arq = DocField(
        1.0, "Query radius multiplier for broad-phase BVH traversal (should be >= 0)."
    )
    gammap = DocField(
        0.45, "Relaxation factor for displacement bounds (0 < gammap < 0.5)."
    )
    n_vv_contact_capacity = DocField(
        1.0, "Vertex-vertex capacity multiplier (x num_vertices)."
    )
    n_ve_contact_capacity = DocField(
        1.0, "Vertex-edge capacity multiplier (x num_vertices)."
    )
    n_vf_contact_capacity = DocField(
        1.0, "Vertex-face capacity multiplier (x num_vertices)."
    )
    n_ee_contact_capacity = DocField(
        1.0, "Edge-edge capacity multiplier (x num_edges)."
    )


VF_E_FACE_TRIANGLE = wp.constant(0)
VF_E_FACE_EDGE = wp.constant(1)
VF_E_FACE_VERTEX = wp.constant(2)
MAX_VV_PER_THREAD = wp.constant(4)
MAX_VE_PER_THREAD = wp.constant(4)
MAX_VF_PER_THREAD = wp.constant(8)
MAX_EE_PER_THREAD = wp.constant(8)
_FUSED_CONTACT_DETECTION_BLOCK_SIZE = wp.constant(64)
tvvlist = wp.types.vector(length=MAX_VV_PER_THREAD, dtype=wp.int32)
tvelist = wp.types.vector(length=MAX_VE_PER_THREAD, dtype=wp.int32)
tvflist = wp.types.vector(length=MAX_VF_PER_THREAD, dtype=wp.int32)
teelist = wp.types.vector(length=MAX_EE_PER_THREAD, dtype=wp.int32)


@wp.struct
class ContactPairsData:
    """Data structure for contact pairs."""

    counts: wp.array[wp.int32]  # (# u + 1,) contact counts per u primitive
    prefix: wp.array[
        wp.int32
    ]  # (# u + 1,) contact prefix sums per u primitive, count is in last prefix element
    u: wp.array[wp.int32]  # (2*capacity,) u indices
    v: wp.array[wp.int32]  # (2*capacity,) v indices


class ContactPairs:
    """Data structure for contact pairs."""

    nu: int
    nv: int
    capacity: int
    data: ContactPairsData  # type: ignore

    def __init__(self, nu: int, nv: int, capacity: int):
        self.nu = nu
        self.nv = nv
        self.capacity = capacity
        self.data = ContactPairsData()
        self.data.counts = wp.zeros((nu + 1,), dtype=wp.int32)
        self.data.prefix = wp.zeros((nu + 1,), dtype=wp.int32)
        self.data.u = wp.full(shape=(2 * capacity,), value=nu, dtype=wp.int32)
        self.data.v = wp.full(shape=(2 * capacity,), value=nv, dtype=wp.int32)

    def clear(self):
        self.data.counts.fill_(wp.int32(0))
        self.data.u.fill_(self.nu)
        self.data.v.fill_(self.nv)

    def uv(self):
        """Get the pairs (u,v) on CPU

        Returns:
            Tuple[np.ndarray, np.ndarray]: (u, v) pairs
        """
        nuv = cp.asarray(self.data.prefix)[-1].get()
        u, v = self.data.u.numpy()[:nuv], self.data.v.numpy()[:nuv]
        return u, v

    def size(self):
        """Get the number of contact pairs."""
        return cp.asarray(self.data.prefix)[-1].get()


@wp.struct
class OgcData:
    """Data structure for OGC."""

    xk: wp.array[wp.vec3f]  # (N,) cached vertex positions at step k

    e_bvh_id: wp.uint64  # Edge BVH ID
    f_bvh_id: wp.uint64  # Triangle BVH ID
    e_lowers: wp.array[wp.vec3f]  # (# edges,) edge AABB lower bounds
    e_uppers: wp.array[wp.vec3f]  # (# edges,) edge AABB upper bounds
    f_lowers: wp.array[wp.vec3f]  # (# triangles,) triangle AABB lower bounds
    f_uppers: wp.array[wp.vec3f]  # (# triangles,) triangle AABB upper bounds

    arq: wp.float32  # OGC query radius multiplier
    r: wp.float32  # OGC contact radius
    rq: wp.array[wp.float32]  # (1,) OGC query radius
    gammap: (
        wp.float32
    )  # Relaxation parameter for vertex displacement bound, must satisfy `0 < gammap < 0.5`

    dminv: wp.array[wp.float32]  # (# vertices,) vertex minimum displacement bounds
    dmine: wp.array[wp.float32]  # (# half-edges,) half-edge minimum displacement bounds
    dminf: wp.array[wp.float32]  # (# triangles,) face minimum displacement bounds

    vv: ContactPairsData  # vertex-vertex contact pairs # type: ignore
    ve: ContactPairsData  # vertex-edge contact pairs # type: ignore
    vf: ContactPairsData  # vertex-face contact pairs # type: ignore
    ee: ContactPairsData  # edge-edge contact pairs # type: ignore

    rvv: ContactPairsData  # vertex-vertex reverse contact pairs # type: ignore
    rve: ContactPairsData  # vertex-edge reverse contact pairs # type: ignore
    rvf: ContactPairsData  # vertex-face reverse contact pairs # type: ignore
    ree: ContactPairsData  # edge-edge reverse contact pairs # type: ignore

    rvv2vv: wp.array[
        wp.int32
    ]  # (vv_capacity,) reverse-vv index -> forward-vv index, of size # vertex-vertex contacts
    rve2ve: wp.array[
        wp.int32
    ]  # (ve_capacity,) reverse-ve index -> forward-ve index, of size # vertex-edge contacts
    rvf2vf: wp.array[
        wp.int32
    ]  # (vf_capacity,) reverse-vf index -> forward-vf index, of size # vertex-face contacts
    ree2ee: wp.array[
        wp.int32
    ]  # (ee_capacity,) reverse-ee index -> forward-ee index, of size # edge-edge contacts

    # Contact bases and closest-point coordinates (computed in a second pass)
    vv_bases: wp.array[
        wp.mat33f
    ]  # (vv_capacity,) orthonormal contact frame per VV pair

    ve_bases: wp.array[
        wp.mat33f
    ]  # (ve_capacity,) orthonormal contact frame per VE pair
    ve_bary: wp.array[
        wp.float32
    ]  # (ve_capacity,) parameter t of closest point on the edge

    vf_bases: wp.array[
        wp.mat33f
    ]  # (vf_capacity,) orthonormal contact frame per VF pair
    vf_bary: wp.array[
        wp.vec2f
    ]  # (vf_capacity,) barycentric uv (with w = 1-u-v) of closest point on triangle

    ee_bases: wp.array[
        wp.mat33f
    ]  # (ee_capacity,) orthonormal contact frame per EE pair
    ee_bary: wp.array[wp.vec2f]  # (ee_capacity,) parameters (s,t) on edge1 and edge2

    # Planar DAT plane offsets
    vv_lambda: wp.array[
        wp.float32
    ]  # (vv_capacity,) vertex-vertex contact plane offsets
    ve_lambda: wp.array[wp.float32]  # (ve_capacity,) vertex-edge contact plane offsets
    vf_lambda: wp.array[wp.float32]  # (vf_capacity,) vertex-face contact plane offsets
    ee_lambda: wp.array[wp.float32]  # (ee_capacity,) edge-edge contact plane offsets

    tv: wp.array[wp.float32]  # (# verts,) planar DAT displacement scales


@wp.func
def _compute_edge_bounding_volume(
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    e: wp.int32,
    xi: wp.vec3f,
    xj: wp.vec3f,
    rq: wp.float32,
):
    xmid = float(0.5) * (xi + xj)  # type: ignore
    hlen = float(0.5) * wp.norm_l2(xj - xi)  # type: ignore
    radius = hlen + rq
    ogc.e_lowers[e] = xmid - wp.vec3f(radius)
    ogc.e_uppers[e] = xmid + wp.vec3f(radius)


@wp.func
def _compute_triangle_bounding_volume(
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    f: wp.int32,
    xi: wp.vec3f,
    xj: wp.vec3f,
    xk: wp.vec3f,
    rq: wp.float32,
):
    xmin = wp.min(
        xi, wp.min(xj, xk)  # pyright: ignore[reportArgumentType, reportCallIssue]
    )
    xmax = wp.max(
        xi, wp.max(xj, xk)  # pyright: ignore[reportArgumentType, reportCallIssue]
    )
    ogc.f_lowers[f] = xmin - wp.vec3f(rq)
    ogc.f_uppers[f] = xmax + wp.vec3f(rq)


@wp.kernel
def _compute_bounding_volumes(
    xk: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    tid = wp.tid()
    n_verts = meshes.V.shape[0]
    n_edges = meshes.E.shape[0]
    n_triangles = meshes.F.shape[0]
    rq = ogc.r + ogc.arq * ogc.rq[0]
    if tid < n_verts:
        v = tid
        # ogc.dminv[v] = rq
    if tid < n_edges:
        e = tid
        he = meshes.EHE[e]
        einds = meshes.E[e]
        # ogc.dmine[he[0]] = rq
        # if he[1] >= 0:
        #     ogc.dmine[he[1]] = rq
        _compute_edge_bounding_volume(ogc, e, xk[einds[0]], xk[einds[1]], rq)  # type: ignore
    if tid < n_triangles:
        f = tid
        finds = meshes.F[f]
        # ogc.dminf[f] = rq
        _compute_triangle_bounding_volume(ogc, f, xk[finds[0]], xk[finds[1]], xk[finds[2]], rq)  # type: ignore


@wp.func
def _closest_face_point_triangle(uvw: wp.vec3f) -> Tuple[wp.int32, wp.int32]:
    fzero = wp.float32(0)
    fone = wp.float32(1)
    one = wp.int32(1)
    two = wp.int32(2)
    u, v, w = uvw[0], uvw[1], uvw[2]  # type: ignore
    n_zeros = int(u == fzero) + int(v == fzero) + int(w == fzero)
    is_vertex, is_edge = (n_zeros == two), (n_zeros == one)
    e_face = (wp.int32(is_edge) * 1) + (wp.int32(is_vertex) * 2)  # type: ignore
    a_local = wp.int32(is_vertex) * (
        wp.int32(v == fone) * one + (wp.int32(w == fone) * two)  # type: ignore
    ) + wp.int32(is_edge) * (
        wp.int32(u == fzero) * one + wp.int32(v == fzero) * two
    )  # type: ignore
    return a_local, e_face


@wp.func
def _vertex_triangle_contact_face_index(
    F: wp.array[wp.vec3i],
    f: wp.int32,
    a_local: wp.int32,
    e_face: wp.int32,
) -> wp.int32:
    """Determine the contact face index for a point-triangle contact based on the barycentric coordinates `uvw`.

    Returns:
        (wp.int32): There are 3 cases based on the contact type determined by `e_face`:
            - VF_E_FACE_TRIANGLE: return the face index
            - VF_E_FACE_EDGE: return the half-edge index
            - VF_E_FACE_VERTEX: return the vertex's global point index
    """
    return (
        # face contact, return face index
        wp.int32(e_face == VF_E_FACE_TRIANGLE) * f
        # edge contact, return half-edge index
        + wp.int32(e_face == VF_E_FACE_EDGE) * (wp.int32(3) * f + a_local)
        # vertex contact, return global vertex point index
        + wp.int32(e_face == VF_E_FACE_VERTEX) * F[f][a_local]  # type: ignore
    )


@wp.func
def is_vertex_feasible(
    x: wp.array[wp.vec3f],  # (N,) points
    F: wp.array[wp.vec3i],  # (M,) surface (triangle) indices into points
    GVHEp: wp.array[wp.int32],  # `|# points + 1| x 1` point to half-edge prefix
    GVHEadj: wp.array[wp.int32],  # `|# half edges| x 1` point to half-edge adjacency
    i: wp.int32,  # point index
    y: wp.vec3f,  # query point
):
    in_vertex_feasible_region = wp.bool(True)
    xi = x[i]
    for k in range(GVHEp[i], GVHEp[i + 1]):  # type: ignore
        he = GVHEadj[k]
        xj = x[halfedges.outgoing_vertex(F, he)]  # type: ignore
        in_vertex_feasible_region = in_vertex_feasible_region and (
            wp.dot(y - xi, xi - xj) >= wp.float32(0)  # type: ignore
        )
    return in_vertex_feasible_region


@wp.func
def is_edge_feasible(
    x: wp.array[wp.vec3f],  # (N,) points
    F: wp.array[wp.vec3i],  # (M,) surface (triangle) indices into points
    GHEF: wp.array[wp.vec2i],  # `2 x |# half edges|` half-edge to face adjacency
    he: wp.int32,  # half-edge index
    y: wp.vec3f,  # query point
    check_adjacent_facets: wp.bool = wp.bool(True),
):
    fzero = wp.float32(0)
    zero = wp.int32(0)
    i, j = (
        halfedges.incoming_vertex(F, he),
        halfedges.outgoing_vertex(F, he),
    )
    xi, xj = x[i], x[j]
    in_edge_feasible_region = wp.bool(True)
    in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xi, xj - xi) >= fzero)  # type: ignore
    in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xj, xi - xj) >= fzero)  # type: ignore
    if check_adjacent_facets:
        hef = GHEF[he]
        k = halfedges.next_vertex(F, he, wp.int32(2))
        fi = hef[0]  # type: ignore
        fj = hef[1]  # type: ignore
        # Handle boundary edge case: no adjacent face (i.e. fj == -1)
        fj = wp.int32(fj < zero) * fi + wp.int32(fj >= zero) * fj  # type: ignore
        # Get the third vertex l of triangle fj that is not part of undirected edge (i,j).
        # NOTE: Whenever fj == fi (i.e. boundary edge), l == k.
        fjinds = F[fj]
        l = (
            wp.int32(fjinds[0] != i and fjinds[0] != j) * fjinds[0]  # type: ignore
            + wp.int32(fjinds[1] != i and fjinds[1] != j) * fjinds[1]  # type: ignore
            + wp.int32(fjinds[2] != i and fjinds[2] != j) * fjinds[2]  # type: ignore
        )
        xk, xl = x[k], x[l]
        xij = xj - xi
        xijn2 = wp.dot(xij, xij)  # type: ignore
        # Tangent to the plane spanned by triangle fi, perpendicular to edge (i,j)
        pin = (xi - xk) + (wp.dot(xk - xi, xij) / xijn2) * xij  # type: ignore
        # Tangent to the plane spanned by triangle fj, perpendicular to edge (i,j)
        # NOTE: whenever fj == fi (i.e. boundary edge), pjn == pin
        pjn = (xi - xl) + (wp.dot(xl - xi, xij) / xijn2) * xij  # type: ignore
        in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xi, pin) >= fzero)  # type: ignore
        in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xi, pjn) >= fzero)  # type: ignore
    return in_edge_feasible_region


@wp.func
def is_triangle_feasible(
    xi: wp.vec3f,  # triangle vertex position 1
    xj: wp.vec3f,  # triangle vertex position 2
    xk: wp.vec3f,  # triangle vertex position 3
    y: wp.vec3f,  # query point
):
    n = wp.cross(xj - xi, xk - xi)  # type: ignore
    in_face_feasible_region = wp.dot(y - xi, n) >= wp.float32(0)  # type: ignore
    return in_face_feasible_region


@wp.func
def _classify_vertex_facet_contacts(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    v: wp.int32,
    i: wp.int32,
    xi: wp.vec3f,
    tvf: tvflist,  # type: ignore
    n_verts: wp.int32,
    n_half_edges: wp.int32,
    n_tris: wp.int32,
) -> Tuple[tvvlist, tvelist, tvflist, wp.int32, wp.int32, wp.int32]:  # type: ignore
    tvv, tve = tvvlist(n_verts), tvelist(n_half_edges)
    n_vv, n_ve, n_vf = wp.int32(0), wp.int32(0), wp.int32(0)
    # tdmin = ogc.dminv[v]  # thread local vertex minimum distance
    for brow in range(MAX_VF_PER_THREAD):
        f = tvf[brow]
        if f >= n_tris:
            break
        finds = meshes.F[f]
        are_adjacent = (i == finds[0]) or (i == finds[1]) or (i == finds[2])
        if are_adjacent:
            tvf[brow] = n_tris
            continue
        xj, xk, xl = x[finds[0]], x[finds[1]], x[finds[2]]
        uvw = queries.closest_point_triangle(
            xi, xj, xk, xl  # pyright: ignore[reportArgumentType]
        )
        xc = uvw[0] * xj + uvw[1] * xk + uvw[2] * xl  # type: ignore
        d = wp.norm_l2(xi - xc)
        # tdmin = wp.min(tdmin, d)
        # wp.atomic_min(ogc.dminf, f, d)
        if d > ogc.r:
            tvf[brow] = n_tris
            continue
        a_local, e_face = _closest_face_point_triangle(uvw)
        a = _vertex_triangle_contact_face_index(meshes.F, f, a_local, e_face)
        if e_face == VF_E_FACE_VERTEX:
            if is_vertex_feasible(x, meshes.F, meshes.GVHEp, meshes.GVHEadj, a, xi):
                if n_vv < MAX_VV_PER_THREAD:
                    tvv[n_vv] = meshes.GXV[a]
                    n_vv += wp.int32(1)
                else:
                    assert False
            tvf[brow] = n_tris
        elif e_face == VF_E_FACE_EDGE:
            if is_edge_feasible(
                x, meshes.F, meshes.GHEF, a, xi, check_adjacent_facets=wp.bool(True)
            ):  # type: ignore
                if n_ve < MAX_VE_PER_THREAD:
                    hei, hej = a, halfedges.opposite_half_edge(meshes.F, a, meshes.GHEF)
                    tve[n_ve] = wp.max(hei, hej)
                    n_ve += wp.int32(1)
                else:
                    assert False
            tvf[brow] = n_tris
        else:  # VF_E_FACE_TRIANGLE
            if is_triangle_feasible(xj, xk, xl, xi):  # type: ignore
                n_vf += wp.int32(1)
            else:
                tvf[brow] = n_tris
    return tvv, tve, tvf, n_vv, n_ve, n_vf  # , tdmin  # type: ignore


@wp.func
def _classify_edge_edge_contacts(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    e1: wp.int32,
    einds1: wp.vec2i,
    xi1: wp.vec3f,
    xj1: wp.vec3f,
    hei1: wp.int32,
    tee: teelist,  # type: ignore
    n_half_edges: wp.int32,
    n_edges: wp.int32,
) -> Tuple[teelist, wp.int32]:  # type: ignore
    fzero = wp.float32(0)
    fone = wp.float32(1)
    n_ee = wp.int32(0)
    # tdmin = ogc.dmine[hei1]  # thread local edge minimum distance
    for brow in range(MAX_EE_PER_THREAD):
        e2 = tee[brow]  # pyright: ignore[reportIndexIssue]
        if e2 >= n_edges:
            tee[brow] = n_half_edges
            continue
        ehe2 = meshes.EHE[e2]
        hei2, hej2 = ehe2[0], ehe2[1]
        he2 = wp.max(hei2, hej2)
        einds2 = meshes.E[e2]
        xi2, xj2 = x[einds2[0]], x[einds2[1]]
        tee[brow] = n_half_edges
        are_adjacent = (
            (einds1[0] == einds2[0])  # type: ignore
            or (einds1[0] == einds2[1])  # type: ignore
            or (einds1[1] == einds2[0])  # type: ignore
            or (einds1[1] == einds2[1])  # type: ignore
        )
        if are_adjacent:
            continue
        st = queries.closest_points_line_segments(xi1, xj1, xi2, xj2)  # type: ignore
        xc1 = (fone - st[0]) * xi1 + st[0] * xj1  # type: ignore
        xc2 = (fone - st[1]) * xi2 + st[1] * xj2  # type: ignore
        d = wp.norm_l2(xc1 - xc2)
        # tdmin = wp.min(tdmin, d)
        # Only store each unordered pair once (deduplication guard)
        if e1 >= e2:
            continue
        if d > ogc.r:
            continue
        is_xc1_vertex = st[0] == fzero or st[0] == fone  # type: ignore
        is_xc2_vertex = st[1] == fzero or st[1] == fone  # type: ignore
        if is_xc1_vertex or is_xc2_vertex:
            continue
        if is_edge_feasible(
            x, meshes.F, meshes.GHEF, hei1, xc2, check_adjacent_facets=wp.bool(True)  # type: ignore
        ) and is_edge_feasible(
            x, meshes.F, meshes.GHEF, he2, xc1, check_adjacent_facets=wp.bool(True)  # type: ignore
        ):  # type: ignore
            if n_ee < MAX_EE_PER_THREAD:
                tee[brow] = he2
                n_ee += wp.int32(1)
            else:
                assert False
    return tee, n_ee  # tdmin  # type: ignore


@wp.kernel(launch_bounds=_FUSED_CONTACT_DETECTION_BLOCK_SIZE)
def _fused_contact_detection(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    """Use block-parallelism to compute vv,ve,vf,ee contact pairs"""
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid // block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]
    n_verts = meshes.V.shape[0]
    n_tris = meshes.F.shape[0]
    n_edges = meshes.E.shape[0]
    n_half_edges = n_tris * wp.int32(3)

    # Location of this thread within the block for cooperative execution
    bcol = local_tid
    next_col = (local_tid + 1) % block_dims
    last_col = block_dims - wp.int32(1)

    # VV,VE,VF contact detection
    if block_id < n_verts:
        v = block_id
        i = meshes.V[v]
        xi = x[i]
        # 1. Query all nearby faces
        tvf = tvflist(n_tris)
        query = wp.tile_bvh_query_aabb(
            ogc.f_bvh_id, xi, xi  # pyright: ignore[reportArgumentType]
        )
        for brow in range(MAX_VF_PER_THREAD):
            if not wp.tile_query_valid(query):
                break
            candidates = wp.tile_bvh_query_next(query)
            f = wp.untile(candidates)
            if f >= 0:
                tvf[brow] = f
        # 2. Classify and store vv,ve,vf contacts
        tvv, tve, tvf, tnvv, tnve, tnvf = _classify_vertex_facet_contacts(
            x, meshes, ogc, v, i, xi, tvf, n_verts, n_half_edges, n_tris  # type: ignore
        )
        # 2.a Reduce dminv across the block and write from the last thread.
        # dminv = wp.tile_min(wp.tile(tdmin))[0]  # type: ignore
        # if local_tid == last_col:
        #     ogc.dminv[v] = dminv  # type: ignore
        # 2.b Count contacts (including duplicates) for early exit opportunity
        tnvv, tnve, tnvf = (
            wp.tile_sum(wp.tile(tnvv))[0],  # type: ignore
            wp.tile_sum(wp.tile(tnve))[0],  # type: ignore
            wp.tile_sum(wp.tile(tnvf))[0],  # type: ignore
        )
        has_vv_contacts, has_ve_contacts, has_vf_contacts = (
            tnvv > int(0),
            tnve > int(0),
            tnvf > int(0),
        )
        if has_vv_contacts:
            # 1. Sort contacts
            bvv = wp.tile(tvv)  # type: ignore
            wp.tile_sort(keys=bvv, values=bvv)
            last_row = MAX_VV_PER_THREAD - wp.int32(1)
            assert bvv[last_row, last_col] == n_verts
            # 2. Compute adjacent differences
            tvv_adj_diff = tvvlist()
            for brow in range(MAX_VV_PER_THREAD):
                is_neighbour_on_next_row = local_tid == last_col
                next_row = (
                    brow + wp.int32(is_neighbour_on_next_row)
                ) % MAX_VV_PER_THREAD
                are_different = bvv[brow, bcol] != bvv[next_row, next_col]
                tvv_adj_diff[brow] = wp.int32(are_different)
            bvv_adj_diff = wp.tile(tvv_adj_diff)  # type: ignore
            # 3. Compute exclusive prefix sum (unique contact count is last element of prefix sum)
            bvv_prefix = wp.tile_scan_exclusive(bvv_adj_diff)
            # 4. Determine global write offset via atomic add
            tvv_offset = wp.int32(0)
            tnvv = bvv_prefix[last_row, last_col]  # type: ignore
            if local_tid == last_col:
                tvv_offset = wp.atomic_add(ogc.vv.counts, n_verts, tnvv)  # type: ignore
                assert tvv_offset + tnvv <= ogc.vv.u.shape[0] // wp.int32(2)
                # 4.a. Write unique contact counts to global count arrays
                ogc.vv.counts[v] = tnvv  # type: ignore
            bvv_offset = wp.tile_from_thread(
                shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
                value=tvv_offset,
                thread_idx=last_col,
            )  # type: ignore
            # 5. Write unique contacts to global contact list
            for brow in range(MAX_VV_PER_THREAD):
                is_marked_unique = bvv_adj_diff[brow, bcol] == wp.int32(1)
                is_last_element = (brow == last_row) and (bcol == last_col)
                if is_marked_unique and not is_last_element:
                    k = bvv_offset[bcol] + bvv_prefix[brow, bcol]  # type: ignore
                    ogc.vv.u[k] = v
                    ogc.vv.v[k] = bvv[brow, bcol]

        if has_ve_contacts:
            # 1. Sort contacts
            bve = wp.tile(tve)  # type: ignore
            wp.tile_sort(keys=bve, values=bve)
            last_row = MAX_VE_PER_THREAD - wp.int32(1)
            assert bve[last_row, last_col] == n_half_edges
            # 2. Compute adjacent differences
            tve_adj_diff = tvelist()
            for brow in range(MAX_VE_PER_THREAD):
                is_neighbour_on_next_row = local_tid == last_col
                next_row = (
                    brow + wp.int32(is_neighbour_on_next_row)
                ) % MAX_VE_PER_THREAD
                are_different = bve[brow, bcol] != bve[next_row, next_col]
                tve_adj_diff[brow] = wp.int32(are_different)
            bve_adj_diff = wp.tile(tve_adj_diff)  # type: ignore
            # 3. Compute exclusive prefix sum (unique contact count is last element of prefix sum)
            bve_prefix = wp.tile_scan_exclusive(bve_adj_diff)
            # 4. Determine global write offset via atomic add
            tve_offset = wp.int32(0)
            tnve = bve_prefix[last_row, last_col]  # type: ignore
            if local_tid == last_col:
                tve_offset = wp.atomic_add(ogc.ve.counts, n_verts, tnve)  # type: ignore
                assert tve_offset + tnve <= ogc.ve.u.shape[0] // wp.int32(2)
                # 4.a. Write unique contact counts to global count arrays
                ogc.ve.counts[v] = tnve  # type: ignore
            bve_offset = wp.tile_from_thread(
                shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
                value=tve_offset,
                thread_idx=last_col,
            )  # type: ignore
            # 5. Write unique contacts to global contact list
            for brow in range(MAX_VE_PER_THREAD):
                is_marked_unique = bve_adj_diff[brow, bcol] == wp.int32(1)
                is_last_element = (brow == last_row) and (bcol == last_col)
                if is_marked_unique and not is_last_element:
                    k = bve_offset[bcol] + bve_prefix[brow, bcol]  # type: ignore
                    ogc.ve.u[k] = v
                    ogc.ve.v[k] = bve[brow, bcol]

        if has_vf_contacts:
            # 1. Sort contacts
            bvf = wp.tile(tvf)  # type: ignore
            wp.tile_sort(keys=bvf, values=bvf)
            last_row = MAX_VF_PER_THREAD - wp.int32(1)
            assert bvf[last_row, last_col] == n_tris
            # 2. Determine global write offset (vf contacts are already unique, count=tnvf)
            tvf_offset = wp.int32(0)
            if local_tid == last_col:
                tvf_offset = wp.atomic_add(ogc.vf.counts, n_verts, tnvf)  # type: ignore
                assert tvf_offset + tnvf <= ogc.vf.u.shape[0] // wp.int32(2)
                # 2.a. Write unique contact counts to global count arrays
                ogc.vf.counts[v] = tnvf  # type: ignore
            bvf_offset = wp.tile_from_thread(
                shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
                value=tvf_offset,
                thread_idx=last_col,
            )  # type: ignore
            # 3. Write unique contacts to global contact list
            for brow in range(MAX_VF_PER_THREAD):
                if bvf[brow, bcol] < n_tris:
                    k = bvf_offset[bcol] + brow * block_dims + bcol
                    ogc.vf.u[k] = v
                    ogc.vf.v[k] = bvf[brow, bcol]

    # EE contact detection
    if block_id < n_edges:
        e = block_id
        hei, hej = meshes.EHE[e][0], meshes.EHE[e][1]
        # Store the larger half-edge index to handle boundary edges
        he_max = wp.max(hei, hej)
        e1 = e
        einds1 = meshes.E[e1]
        xi1, xj1 = x[einds1[0]], x[einds1[1]]
        # 1. Query all nearby edges
        tee = teelist(n_edges)
        query = wp.tile_bvh_query_aabb(
            ogc.e_bvh_id,
            ogc.e_lowers[e],
            ogc.e_uppers[e],
        )
        for brow in range(MAX_EE_PER_THREAD):
            if not wp.tile_query_valid(query):
                break
            candidates = wp.tile_bvh_query_next(query)
            e2 = wp.untile(candidates)
            if e2 >= 0:
                tee[brow] = e2
        # 2. Classify and store ee contacts
        tee, tnee = _classify_edge_edge_contacts(
            x, meshes, ogc, e1, einds1, xi1, xj1, hei, tee, n_half_edges, n_edges  # type: ignore
        )
        # 2.a Reduce dmine across the block and write from the last thread.
        # dmine = wp.tile_min(wp.tile(tdmine))[0]  # type: ignore
        # if local_tid == last_col:
        #     ogc.dmine[hei] = dmine
        #     if hej >= wp.int32(0):
        #         ogc.dmine[hej] = dmine
        # 3. Count contacts
        tnee = wp.tile_sum(wp.tile(tnee))[0]  # type: ignore
        has_ee_contacts = tnee > wp.int32(0)
        if has_ee_contacts:
            # 1. Sort contacts (already unique)
            bee = wp.tile(tee)  # type: ignore
            wp.tile_sort(keys=bee, values=bee)
            last_row = MAX_EE_PER_THREAD - wp.int32(1)
            assert bee[last_row, last_col] == n_half_edges
            # 2. Determine global write offset via atomic add
            tee_offset = wp.int32(0)
            if local_tid == last_col:
                tee_offset = wp.atomic_add(ogc.ee.counts, n_half_edges, tnee)  # type: ignore
                assert tee_offset + tnee <= ogc.ee.u.shape[0] // wp.int32(2)
                # 2.a. Write unique contact counts to global count arrays
                ogc.ee.counts[he_max] = tnee  # type: ignore
            bee_offset = wp.tile_from_thread(
                shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
                value=tee_offset,
                thread_idx=last_col,
            )  # type: ignore
            # 3. Write contacts to global contact set
            for brow in range(MAX_EE_PER_THREAD):
                if bee[brow, bcol] < n_half_edges:
                    k = bee_offset[bcol] + brow * block_dims + bcol  # type: ignore
                    ogc.ee.u[k] = he_max
                    ogc.ee.v[k] = bee[brow, bcol]


@wp.kernel
def _update_displacement_bounds(
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    v = wp.tid()
    i = meshes.V[v]
    for k in range(meshes.GVHEp[i], meshes.GVHEp[i + 1]):  # type: ignore
        he = meshes.GVHEadj[k]
        f = halfedges.face_of_half_edge(he)
        ogc.dminv[v] = wp.min(ogc.dminv[v], ogc.dminf[f])
        ogc.dminv[v] = wp.min(ogc.dminv[v], ogc.dmine[he])
    ogc.dminv[v] *= ogc.gammap


@wp.kernel
def _truncate_displacements(
    xk: wp.array[
        wp.vec3f
    ],  # (N,) reference positions cached at last prepare_for_execution
    dminv: wp.array[wp.float32],  # (n_verts,) per-vertex-primitive displacement bounds
    V: wp.array[wp.int32],  # (n_verts,) vertex primitive -> global point index
    x: wp.array[wp.vec3f],  # (N,) positions to truncate in-place
):
    """Truncate the displacement of vertex primitive v to remain within its bound dminv[v].

    For each vertex primitive v with global point index i = V[v]:
      d = x[i] - xk[i]
      if |d| > dminv[v]: x[i] = xk[i] + d * (dminv[v] / |d|)
    """
    v = wp.tid()
    i = V[v]
    b = dminv[v]
    d = x[i] - xk[i]
    dnorm = wp.norm_l2(d)
    if dnorm > b:  # type: ignore
        x[i] = xk[i] + (b / dnorm) * d  # type: ignore


@wp.kernel
def _compute_reverse_contact_counts(ogc: OgcData):  # type: ignore
    """Compute reverse contact counts for each contact pair type.

    Args:
        ogc (OgcData): The contact data after contact detection kernel invocation.
    """
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid // block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]
    n_verts = ogc.vv.counts.shape[0] - wp.int32(1)
    n_half_edges = ogc.rve.counts.shape[0] - wp.int32(1)
    n_tris = ogc.rvf.counts.shape[0] - wp.int32(1)
    nvv = ogc.vv.counts[n_verts]
    nve = ogc.ve.counts[n_verts]
    nvf = ogc.vf.counts[n_verts]
    nee = ogc.ee.counts[n_half_edges]
    is_last_col = local_tid == block_dims - wp.int32(1)
    base_idx = block_id * block_dims

    # --- Vertex-vertex: reverse array sorted by v (stored as rvv.u) ---
    if base_idx < n_verts:
        v = tid
        # Each thread finds lower_bound of its primitive index in rvv.u
        if v < n_verts - wp.int32(1):
            lb_vv = common.lower_bound(ogc.rvv.u, nvv, v)  # type: ignore
        else:
            lb_vv = nvv
        # Share lower bounds across the block so each thread can compute its count
        # as lb[local_tid+1] - lb[local_tid].
        blb_vv = wp.tile(lb_vv)  # type: ignore
        if is_last_col:
            if v >= n_verts - wp.int32(1):
                lb_vv_next = nvv
            else:
                lb_vv_next = common.lower_bound(ogc.rvv.u, nvv, v + wp.int32(1))  # type: ignore
        else:
            lb_vv_next = blb_vv[
                local_tid + 1
            ]  # Get next lower bound from right neighbour
        if v < n_verts:
            ogc.rvv.counts[v] = lb_vv_next - lb_vv

    # --- (Half-)edge-vertex: reverse array sorted by he (stored as rve.u) ---
    if base_idx < n_half_edges:
        he = tid
        if he < n_half_edges - wp.int32(1):
            lb_rve = common.lower_bound(ogc.rve.u, nve, he)  # type: ignore
        else:
            lb_rve = nve
        blb_rve = wp.tile(lb_rve)  # type: ignore
        if is_last_col:
            if he >= n_half_edges - wp.int32(1):
                lb_rve_next = nve
            else:
                lb_rve_next = common.lower_bound(ogc.rve.u, nve, he + wp.int32(1))  # type: ignore
        else:
            lb_rve_next = blb_rve[
                local_tid + 1
            ]  # Get next lower bound from right neighbour
        if he < n_half_edges:
            ogc.rve.counts[he] = lb_rve_next - lb_rve

    # --- Triangle-vertex: reverse array sorted by f (stored as rvf.u) ---
    if base_idx < n_tris:
        f = tid
        if f < n_tris - wp.int32(1):
            lb_rvf = common.lower_bound(ogc.rvf.u, nvf, f)  # type: ignore
        else:
            lb_rvf = nvf
        blb_rvf = wp.tile(lb_rvf)  # type: ignore
        if is_last_col:
            if f >= n_tris - wp.int32(1):
                lb_rvf_next = nvf
            else:
                lb_rvf_next = common.lower_bound(ogc.rvf.u, nvf, f + wp.int32(1))  # type: ignore
        else:
            lb_rvf_next = blb_rvf[
                local_tid + 1
            ]  # Get next lower bound from right neighbour
        if f < n_tris:
            ogc.rvf.counts[f] = lb_rvf_next - lb_rvf  # type: ignore

    # --- (Half-)edge-(half-)edge: reverse array sorted by he (stored as ree.u) ---
    if base_idx < n_half_edges:
        he = tid
        if he < n_half_edges - wp.int32(1):
            lb_ree = common.lower_bound(ogc.ree.u, nee, he)  # type: ignore
        else:
            lb_ree = nee
        blb_ree = wp.tile(lb_ree)  # type: ignore
        if is_last_col:
            if he >= n_half_edges - wp.int32(1):
                lb_ree_next = nee
            else:
                lb_ree_next = common.lower_bound(ogc.ree.u, nee, he + wp.int32(1))  # type: ignore
        else:
            lb_ree_next = blb_ree[
                local_tid + 1
            ]  # Get next lower bound from right neighbour
        if he < n_half_edges:
            ogc.ree.counts[he] = lb_ree_next - lb_ree  # type: ignore


@wp.kernel
def _build_reverse_to_forward_map(
    fwd: ContactPairsData,  # pyright: ignore[reportGeneralTypeIssues]
    rev: ContactPairsData,  # pyright: ignore[reportGeneralTypeIssues]
    n_fwd_key: wp.int32,  # index into fwd.prefix for the total forward count
    rx2x: wp.array[wp.int32],
):
    """Map each reverse contact index k to its corresponding forward contact index.

    For the k-th reverse contact (rev.u[k], rev.v[k]) = (r_u, r_v), the matching
    forward contact is the pair (r_v, r_u) in the sorted forward list. It is located
    via a binary search using the pair-keyed lower_bound.
    """
    k = wp.tid()
    n = fwd.prefix[n_fwd_key]
    if k >= n:
        return
    ru = rev.u[k]
    rv = rev.v[k]
    l = common.lower_bound(fwd.u, fwd.v, n, rv, ru)  # type: ignore
    rx2x[k] = l  # type: ignore


@wp.func
def _build_contact_basis(n: wp.vec3f) -> wp.mat33f:
    """Build an orthonormal contact frame from a unit normal vector.

    The returned mat33f has the three basis vectors as rows:
      row 0 = n  (contact normal)
      row 1 = t1 (first tangent)
      row 2 = t2 (second tangent)
    """
    right = wp.vec3f(wp.float32(1), wp.float32(0), wp.float32(0))
    if wp.abs(n[0]) > wp.float32(0.9):  # type: ignore
        right = wp.vec3f(wp.float32(0), wp.float32(1), wp.float32(0))
    t1 = wp.normalize(wp.cross(n, right))  # type: ignore
    t2 = wp.cross(n, t1)  # type: ignore
    return wp.mat33f(
        n[0],  # type: ignore
        n[1],  # type: ignore
        n[2],  # type: ignore
        t1[0],  # type: ignore
        t1[1],  # type: ignore
        t1[2],  # type: ignore
        t2[0],  # type: ignore
        t2[1],  # type: ignore
        t2[2],  # type: ignore
    )


@wp.kernel
def _compute_vv_contact_data(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    """Second pass: compute contact basis for each vertex-vertex contact pair."""
    k = wp.tid()
    n_verts = meshes.V.shape[0]
    n_vv = ogc.vv.counts[n_verts]
    if k >= n_vv:
        return
    u, v = ogc.vv.u[k], ogc.vv.v[k]
    xi = x[meshes.V[u]]
    xj = x[meshes.V[v]]
    assert wp.norm_l2(xi - xj) > wp.float32(1e-10)  # type: ignore
    n = wp.normalize(xi - xj)  # type: ignore
    ogc.vv_bases[k] = _build_contact_basis(n)


@wp.kernel
def _compute_ve_contact_data(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    """Second pass: compute contact basis and edge parameter t for each vertex-edge contact pair."""
    k = wp.tid()
    n_verts = meshes.V.shape[0]
    n_ve = ogc.ve.counts[n_verts]
    if k >= n_ve:
        return
    v, he = ogc.ve.u[k], ogc.ve.v[k]
    xi = x[meshes.V[v]]
    xa = x[halfedges.incoming_vertex(meshes.F, he)]  # type: ignore
    xb = x[halfedges.outgoing_vertex(meshes.F, he)]  # type: ignore
    uv = queries.closest_point_on_line_segment(xi, xa, xb)  # type: ignore
    assert wp.norm_l2(xi - (uv[0] * xa + uv[1] * xb)) > wp.float32(1e-10)  # type: ignore
    n = wp.normalize(xi - (uv[0] * xa + uv[1] * xb))  # type: ignore
    ogc.ve_bases[k] = _build_contact_basis(n)
    ogc.ve_bary[k] = uv[1]  # type: ignore


@wp.kernel
def _compute_vf_contact_data(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    """Second pass: compute contact basis and barycentric uvw for each vertex-face contact pair."""
    k = wp.tid()
    n_verts = meshes.V.shape[0]
    n_vf = ogc.vf.counts[n_verts]
    if k >= n_vf:
        return
    v, f = ogc.vf.u[k], ogc.vf.v[k]
    xi = x[meshes.V[v]]
    finds = meshes.F[f]
    xa = x[finds[0]]
    xb = x[finds[1]]
    xc = x[finds[2]]
    uvw = queries.closest_point_triangle(xi, xa, xb, xc)  # type: ignore
    xc = uvw[0] * xa + uvw[1] * xb + uvw[2] * xc  # type: ignore
    assert wp.norm_l2(xi - xc) > wp.float32(1e-10)  # type: ignore
    n = wp.normalize(xi - xc)
    ogc.vf_bases[k] = _build_contact_basis(n)
    ogc.vf_bary[k] = wp.vec2f(uvw[0], uvw[1])  # type: ignore


@wp.kernel
def _compute_ee_contact_data(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    """Second pass: compute contact basis and parameters (s,t) for each edge-edge contact pair."""
    k = wp.tid()
    n_half_edges = ogc.ee.counts.shape[0] - wp.int32(1)
    n_ee = ogc.ee.counts[n_half_edges]
    if k >= n_ee:
        return
    he1 = ogc.ee.u[k]
    he2 = ogc.ee.v[k]
    xi1 = x[halfedges.incoming_vertex(meshes.F, he1)]
    xj1 = x[halfedges.outgoing_vertex(meshes.F, he1)]
    xi2 = x[halfedges.incoming_vertex(meshes.F, he2)]
    xj2 = x[halfedges.outgoing_vertex(meshes.F, he2)]
    st = queries.closest_points_line_segments(xi1, xj1, xi2, xj2)  # type: ignore
    xc1 = (wp.float32(1.0) - st[0]) * xi1 + st[0] * xj1  # type: ignore
    xc2 = (wp.float32(1.0) - st[1]) * xi2 + st[1] * xj2  # type: ignore
    assert wp.norm_l2(xc1 - xc2) > wp.float32(1e-10)  # type: ignore
    n = wp.normalize(xc1 - xc2)
    ogc.ee_bases[k] = _build_contact_basis(n)
    ogc.ee_bary[k] = st


@wp.func
def _planar_dat_truncate_one(
    xk: wp.vec3f,
    x: wp.vec3f,
    dx: wp.vec3f,
    n: wp.vec3f,
    xc1: wp.vec3f,
    xc2: wp.vec3f,
    lambda_c: wp.float32,
    gamma: wp.float32,
) -> wp.float32:
    t = wp.float32(1)
    p = (wp.float32(1) - lambda_c) * xc1 + lambda_c * xc2
    den = wp.dot(dx, n)  # type: ignore
    # Assert that if the denominator is near zero, i.e. the vertex
    # is moving parallel to the plane, then the vertex is on the
    # correct side of the plane (i.e. non penetrating).
    eps = wp.float32(1e-10)  # type: ignore
    parallel = wp.abs(den) <= eps
    # assert not parallel or wp.dot(x - p, n) > wp.float32(0)  # type: ignore
    if not parallel:
        tk = gamma * (wp.dot(p - xk, n) / den)  # type: ignore
        if tk > wp.float32(0) and tk < t:
            t = tk
    return t


@wp.func
def _vertex_triangle_planar_dat(
    x: wp.array[wp.vec3f],
    xk: wp.array[wp.vec3f],
    i: wp.int32,
    finds: wp.vec3i,
    xi: wp.vec3f,
    xki: wp.vec3f,
    dxi: wp.vec3f,
    gamma: wp.float32,
) -> Tuple[wp.float32, wp.float32, wp.float32, wp.float32]:  # type: ignore
    tv = wp.float32(1)
    ta = wp.float32(1)
    tb = wp.float32(1)
    tc = wp.float32(1)
    zero = wp.float32(1e-10)  # type: ignore
    a = finds[0]  # type: ignore
    b = finds[1]  # type: ignore
    c = finds[2]  # type: ignore
    are_adjacent = (a == i) or (b == i) or (c == i)
    if are_adjacent:
        return tv, ta, tb, tc
    xa = x[a]
    xb = x[b]
    xc = x[c]
    xka = xk[a]
    xkb = xk[b]
    xkc = xk[c]
    # Compute separating plane
    uvw = queries.closest_point_triangle(xki, xka, xkb, xkc)  # type: ignore
    xkc2 = uvw[0] * xka + uvw[1] * xkb + uvw[2] * xkc  # type: ignore
    xkc1 = xki
    n = xkc1 - xkc2
    dxin = wp.max(wp.dot(dxi, -n), wp.float32(0))  # type: ignore
    dxfn = wp.max(
        wp.max(
            wp.max(
                wp.dot(xa - xka, n),  # type: ignore
                wp.dot(xb - xkb, n),  # type: ignore
            ),
            wp.dot(xc - xkc, n),  # type: ignore
        ),
        wp.float32(0),
    )
    den = dxin + dxfn
    if dxin <= zero and dxfn <= zero:
        llambda = wp.float32(0.5)  # type: ignore
    else:
        llambda = dxin / den
    # Perform ray-plane intersection and atomic min truncation
    tv = _planar_dat_truncate_one(
        xki,  # type: ignore
        xi,  # type: ignore
        dxi,  # type: ignore
        n,
        xkc1,  # type: ignore
        xkc2,
        llambda,
        gamma,
    )
    ta = _planar_dat_truncate_one(
        xka,  # type: ignore
        xa,  # type: ignore
        xa - xka,  # type: ignore
        n,
        xkc1,  # type: ignore
        xkc2,
        llambda,
        gamma,
    )
    tb = _planar_dat_truncate_one(
        xkb,  # type: ignore
        xb,  # type: ignore
        xb - xkb,  # type: ignore
        n,
        xkc1,  # type: ignore
        xkc2,
        llambda,
        gamma,
    )
    tc = _planar_dat_truncate_one(
        xkc,  # type: ignore
        xc,  # type: ignore
        xc - xkc,  # type: ignore
        n,
        xkc1,  # type: ignore
        xkc2,
        llambda,
        gamma,
    )
    return tv, ta, tb, tc


@wp.func
def _edge_edge_planar_dat(
    x: wp.array[wp.vec3f],
    xk: wp.array[wp.vec3f],
    einds1: wp.vec2i,
    einds2: wp.vec2i,
    xi1: wp.vec3f,
    xki1: wp.vec3f,
    dxi1: wp.vec3f,
    xj1: wp.vec3f,
    xkj1: wp.vec3f,
    dxj1: wp.vec3f,
    gamma: wp.float32,
) -> Tuple[wp.float32, wp.float32, wp.float32, wp.float32]:  # type: ignore
    tvi1 = wp.float32(1)
    tvj1 = wp.float32(1)
    tvi2 = wp.float32(1)
    tvj2 = wp.float32(1)
    are_adjacent = (
        (einds1[0] == einds2[0])  # type: ignore
        or (einds1[0] == einds2[1])  # type: ignore
        or (einds1[1] == einds2[0])  # type: ignore
        or (einds1[1] == einds2[1])  # type: ignore
    )
    if are_adjacent:
        return tvi1, tvj1, tvi2, tvj2
    xi2, xj2 = x[einds2[0]], x[einds2[1]]  # type: ignore
    xki2, xkj2 = xk[einds2[0]], xk[einds2[1]]  # type: ignore
    st = queries.closest_points_line_segments(xki1, xkj1, xki2, xkj2)  # type: ignore
    xc1 = (wp.float32(1) - st[0]) * xki1 + st[0] * xkj1  # type: ignore
    xc2 = (wp.float32(1) - st[1]) * xki2 + st[1] * xkj2  # type: ignore
    n = wp.normalize(xc1 - xc2)
    dxi2 = xi2 - xki2
    dxj2 = xj2 - xkj2
    dxe1n = wp.max(
        wp.max(wp.dot(dxi1, -n), wp.dot(dxj1, -n)),  # type: ignore
        wp.float32(0),
    )
    dxe2n = wp.max(
        wp.max(wp.dot(dxi2, n), wp.dot(dxj2, n)),  # type: ignore
        wp.float32(0),
    )
    den = dxe1n + dxe2n
    if den <= wp.float32(1e-10):  # type: ignore
        llambda = wp.float32(0.5)  # type: ignore
    else:
        llambda = dxe1n / den
    # Perform ray-plane intersection and atomic min truncation for edge 1
    tvi1 = _planar_dat_truncate_one(
        xki1, xi1, dxi1, -n, xc1, xc2, llambda, gamma  # type: ignore
    )
    tvj1 = _planar_dat_truncate_one(
        xkj1, xj1, dxj1, -n, xc1, xc2, llambda, gamma  # type: ignore
    )
    tvi2 = _planar_dat_truncate_one(
        xki2,  # type: ignore
        xi2,  # type: ignore
        dxi2,  # type: ignore
        n,
        xc1,
        xc2,
        llambda,
        gamma,
    )
    tvj2 = _planar_dat_truncate_one(
        xkj2,  # type: ignore
        xj2,  # type: ignore
        dxj2,  # type: ignore
        n,
        xc1,
        xc2,
        llambda,
        gamma,
    )
    return tvi1, tvj1, tvi2, tvj2


@wp.kernel
def _planar_dat(
    xk: wp.array[
        wp.vec3f
    ],  # (N,) reference positions cached at last prepare_for_execution
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid // block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]

    n_verts = meshes.V.shape[0]
    n_edges = meshes.E.shape[0]
    gamma = wp.float32(2) * ogc.gammap

    # Loop over all vertex-triangle broad phase pairs and planar DAT on
    # each side using atomic mins on ray-plane intersections
    if block_id < n_verts:
        v = block_id
        i = meshes.V[v]
        xi = x[i]
        xki = xk[i]
        dxi = xi - xki
        tv = wp.float32(1)
        # Visit each candidate face
        query = wp.tile_bvh_query_aabb(
            ogc.f_bvh_id, xki, xki  # pyright: ignore[reportArgumentType]
        )
        # Store all candidates in thread local candidate list
        tvf = tvflist(wp.int32(-1))
        for brow in range(MAX_VF_PER_THREAD):
            if not wp.tile_query_valid(query):
                break
            candidates = wp.tile_bvh_query_next(query)
            f = wp.untile(candidates)  # type: ignore
            tvf[brow] = f
        # Visit each candidate pair
        n_tris = meshes.F.shape[0]
        for brow in range(MAX_VF_PER_THREAD):
            f = tvf[brow]
            if f >= wp.int32(0) and f < n_tris:
                assert f < n_tris
                finds = meshes.F[f]
                tv_candidate = wp.float32(1)
                ta = wp.float32(1)
                tb = wp.float32(1)
                tc = wp.float32(1)
                tv_candidate, ta, tb, tc = _vertex_triangle_planar_dat(
                    x, xk, i, finds, xi, xki, dxi, gamma  # type: ignore
                )
                tv = wp.min(tv, tv_candidate)
                va = meshes.GXV[finds[0]]
                vb = meshes.GXV[finds[1]]
                vc = meshes.GXV[finds[2]]
                # TODO: For some reason, these scattered atomic mins cause issues,
                # probably also with stability.
                # wp.atomic_min(ogc.tv, va, ta)
                # wp.atomic_min(ogc.tv, vb, tb)
                # wp.atomic_min(ogc.tv, vc, tc)

        # Global write
        # TODO: Truncation yields really weird artifacts!!
        # tvs = wp.tile(tv)  # type: ignore
        # tvs_min = wp.tile_min(tvs)
        # tv_min = wp.tile_extract(tvs_min, wp.int32(0))  # type: ignore
        # if local_tid == 0:
        #     wp.atomic_min(ogc.tv, v, tv_min)  # type: ignore

    # Loop over all edge-edge broad phase (one-sided) pairs and planar DAT on each side using atomic mins on ray-plane intersections
    if block_id < n_edges:
        e = block_id
        e1 = e
        einds1 = meshes.E[e1]
        xi1, xj1 = x[einds1[0]], x[einds1[1]]
        xki1, xkj1 = xk[einds1[0]], xk[einds1[1]]
        dxi1 = xi1 - xki1
        dxj1 = xj1 - xkj1
        tvi1 = wp.float32(1)
        tvj1 = wp.float32(1)
        # Query all nearby edges
        query = wp.tile_bvh_query_aabb(
            ogc.e_bvh_id,
            ogc.e_lowers[e],
            ogc.e_uppers[e],
        )
        # Store all candidates in thread local candidate list
        tee = teelist(wp.int32(-1))
        for brow in range(MAX_EE_PER_THREAD):
            if not wp.tile_query_valid(query):
                break
            candidates = wp.tile_bvh_query_next(query)
            e2 = wp.untile(candidates)
            tee[brow] = e2
        # Visit each candidate pair (e1, e2) where e1 < e2 to avoid double counting
        for brow in range(MAX_EE_PER_THREAD):
            e2 = tee[brow]
            # Only visit unique pairs (e1, e2) where e1 < e2
            if e2 > e1:  # type: ignore
                einds2 = meshes.E[e2]
                tvi1_candidate, tvj1_candidate, tvi2, tvj2 = _edge_edge_planar_dat(
                    x,
                    xk,
                    einds1,
                    einds2,
                    xi1,  # type: ignore
                    xki1,  # type: ignore
                    dxi1,  # type: ignore
                    xj1,  # type: ignore
                    xkj1,  # type: ignore
                    dxj1,  # type: ignore
                    gamma,
                )
                tvi1 = wp.min(tvi1, tvi1_candidate)
                tvj1 = wp.min(tvj1, tvj1_candidate)
                # TODO: For some reason, these scattered atomic mins cause issues,
                # probably also with stability.
                # wp.atomic_min(ogc.tv, meshes.GXV[einds2[0]], tvi2)  # type: ignore
                # wp.atomic_min(ogc.tv, meshes.GXV[einds2[1]], tvj2)  # type: ignore

        # TODO: Investigate why these edge-edge planar DATs make the sim super unstable.
        # Global write
        # tvi1s = wp.tile(tvi1)  # type: ignore
        # tvj1s = wp.tile(tvj1)  # type: ignore
        # tvi1s_min = wp.tile_min(tvi1s)
        # tvj1s_min = wp.tile_min(tvj1s)
        # tvi1_min = wp.tile_extract(tvi1s_min, wp.int32(0))  # type: ignore
        # tvj1_min = wp.tile_extract(tvj1s_min, wp.int32(0))  # type: ignore
        # if local_tid == 0:
        #     vi1 = meshes.GXV[einds1[0]]
        #     vj1 = meshes.GXV[einds1[1]]
        #     wp.atomic_min(ogc.tv, vi1, tvi1_min)  # type: ignore
        #     wp.atomic_min(ogc.tv, vj1, tvj1_min)  # type: ignore


@wp.kernel
def _planar_dat_truncate(
    xk: wp.array[
        wp.vec3f
    ],  # (N,) reference positions cached at last prepare_for_execution
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    tid = wp.tid()
    vi = tid
    i = meshes.V[vi]
    xki = xk[i]
    xi = x[i]
    t = ogc.tv[vi]
    xi = xki + t * (xi - xki)
    dxi = xi - xki
    dxinorm = wp.norm_l2(dxi)
    rq = ogc.r + ogc.arq * ogc.rq[0]
    if dxinorm > wp.float32(0):
        dxi = wp.min(wp.float32(1), rq / dxinorm) * dxi
    x[i] = xki + dxi  # type: ignore


class Ogc:
    """Offset Geometric Contact"""

    _vv: ContactPairs  # Vertex-Vertex contact pair list
    _rvv: ContactPairs  # Vertex-Vertex reverse contact pair list
    _ve: ContactPairs  # Vertex-Edge contact pair list
    _rve: ContactPairs  # Vertex-Edge reverse contact pair list
    _vf: ContactPairs  # Vertex-Face contact pair list
    _rvf: ContactPairs  # Vertex-Face reverse contact pair list
    _ee: ContactPairs  # Edge-Edge contact pair list
    _ree: ContactPairs  # Edge-Edge reverse contact pair list
    _streams: list[wp.Stream]  # Stream list

    _ogc: OgcData  # pyright: ignore[reportGeneralTypeIssues]
    _e_bvh: wp.Bvh  # BVH over edges
    _f_bvh: wp.Bvh  # BVH over faces
    _xk: wp.array[wp.vec3f]  # (N,) cached vertex positions at step k
    _meshes: MultiMesh  # Meshes # type: ignore

    def __init__(
        self,
        points: wp.array[wp.vec3f],
        meshes: MultiMesh,  # type: ignore
        params: OgcParams = OgcParams(),
    ):
        """Construct ogc data

        Args:
            points (wp.array[wp.vec3f]): (N,) points
            meshes (MultiMesh): Multi-body mesh
            params (OgcParams, optional): Contact detection parameters. Defaults to OgcParams().
        """
        self._xk = wp.empty_like(points)
        wp.copy(dest=self._xk, src=points)
        self._meshes = meshes
        self._ogc = OgcData()
        self._ogc.e_lowers = wp.zeros((meshes.n_edges,), dtype=wp.vec3f)
        self._ogc.e_uppers = wp.zeros((meshes.n_edges,), dtype=wp.vec3f)
        self._ogc.f_lowers = wp.zeros((meshes.n_triangles,), dtype=wp.vec3f)
        self._ogc.f_uppers = wp.zeros((meshes.n_triangles,), dtype=wp.vec3f)
        self._ogc.arq = params.arq
        self._ogc.r = params.r
        self._ogc.gammap = params.gammap
        self._ogc.dminv = wp.zeros((meshes.n_verts,), dtype=wp.float32)
        self._ogc.dmine = wp.zeros((meshes.n_half_edges,), dtype=wp.float32)
        self._ogc.dminf = wp.zeros((meshes.n_triangles,), dtype=wp.float32)
        self._ogc.rq = wp.zeros((1,), dtype=wp.float32)  # (1,) OGC query radius

        vv_capacity = int(params.n_vv_contact_capacity * meshes.n_verts)  # type: ignore
        ve_capacity = int(params.n_ve_contact_capacity * meshes.n_verts)  # type: ignore
        vf_capacity = int(params.n_vf_contact_capacity * meshes.n_verts)  # type: ignore
        ee_capacity = int(params.n_ee_contact_capacity * meshes.n_edges)  # type: ignore

        self._vv, self._rvv = ContactPairs(
            meshes.n_verts, meshes.n_verts, vv_capacity
        ), ContactPairs(meshes.n_verts, meshes.n_verts, vv_capacity)
        self._ogc.vv, self._ogc.rvv = self._vv.data, self._rvv.data

        self._ve, self._rve = ContactPairs(
            meshes.n_verts, meshes.n_half_edges, ve_capacity
        ), ContactPairs(meshes.n_half_edges, meshes.n_verts, ve_capacity)
        self._ogc.ve, self._ogc.rve = self._ve.data, self._rve.data

        self._vf, self._rvf = ContactPairs(
            meshes.n_verts, meshes.n_triangles, vf_capacity
        ), ContactPairs(meshes.n_triangles, meshes.n_verts, vf_capacity)
        self._ogc.vf, self._ogc.rvf = self._vf.data, self._rvf.data

        self._ee, self._ree = ContactPairs(
            meshes.n_half_edges, meshes.n_half_edges, ee_capacity
        ), ContactPairs(meshes.n_half_edges, meshes.n_half_edges, ee_capacity)
        self._ogc.ee, self._ogc.ree = self._ee.data, self._ree.data

        self._ogc.rvv2vv = wp.empty((vv_capacity,), dtype=wp.int32)  # type: ignore
        self._ogc.rve2ve = wp.empty((ve_capacity,), dtype=wp.int32)  # type: ignore
        self._ogc.rvf2vf = wp.empty((vf_capacity,), dtype=wp.int32)  # type: ignore
        self._ogc.ree2ee = wp.empty((ee_capacity,), dtype=wp.int32)  # type: ignore

        self._ogc.vv_bases = wp.empty((vv_capacity,), dtype=wp.mat33f)  # type: ignore
        self._ogc.ve_bases = wp.empty((ve_capacity,), dtype=wp.mat33f)  # type: ignore
        self._ogc.ve_bary = wp.empty((ve_capacity,), dtype=wp.float32)  # type: ignore
        self._ogc.vf_bases = wp.empty((vf_capacity,), dtype=wp.mat33f)  # type: ignore
        self._ogc.vf_bary = wp.empty((vf_capacity,), dtype=wp.vec2f)  # type: ignore
        self._ogc.ee_bases = wp.empty((ee_capacity,), dtype=wp.mat33f)  # type: ignore
        self._ogc.ee_bary = wp.empty((ee_capacity,), dtype=wp.vec2f)  # type: ignore

        self._ogc.vv_lambda = wp.empty((vv_capacity,), dtype=wp.float32)  # type: ignore
        self._ogc.ve_lambda = wp.empty((ve_capacity,), dtype=wp.float32)  # type: ignore
        self._ogc.vf_lambda = wp.empty((vf_capacity,), dtype=wp.float32)  # type: ignore
        self._ogc.ee_lambda = wp.empty((ee_capacity,), dtype=wp.float32)  # type: ignore

        self._ogc.tv = wp.empty((meshes.n_verts,), dtype=wp.float32)  # type: ignore

        self._streams = [wp.Stream() for _ in range(10)]

        dim = max(meshes.n_verts, meshes.n_edges, meshes.n_triangles)
        wp.launch(
            kernel=_compute_bounding_volumes,
            dim=dim,
            inputs=[self._xk, self._meshes.data, self._ogc],
        )
        self._e_bvh, self._f_bvh = (
            wp.Bvh(
                self._ogc.e_lowers,
                self._ogc.e_uppers,
                constructor="lbvh",
                groups=None,
            ),
            wp.Bvh(
                self._ogc.f_lowers,
                self._ogc.f_uppers,
                constructor="lbvh",
                groups=None,
            ),
        )
        self._ogc.e_bvh_id, self._ogc.f_bvh_id = (
            self._e_bvh.id,
            self._f_bvh.id,
        )

    def enable_adaptive_query_radius(
        self, xt: wp.array[wp.vec3f], xtilde: wp.array[wp.vec3f]
    ):
        # Compute rq = r + beta * (xtilde - xt).colwise().norm().maxCoeff()
        # 1. Capture xt, xtilde as CuPy 3 x N arrays
        self._rq_xtc = cp.asarray(xt)
        self._rq_xtildec = cp.asarray(xtilde)
        # 2. Use ZipIterator(xtc[0,:], xtc[1,:], xtc[2,:], xtildec[0,:], xtildec[1,:], xtildec[2,:])
        self._rq_zip_it = cuda.compute.ZipIterator(
            self._rq_xtc[:, 0],
            self._rq_xtc[:, 1],
            self._rq_xtc[:, 2],
            self._rq_xtildec[:, 0],
            self._rq_xtildec[:, 1],
            self._rq_xtildec[:, 2],
        )
        # 3. Use TransformIterator on the ZipIterator as transform = lambda x: sqrt((x[3] - x[0])**2 + (x[4] - x[1])**2 + (x[5] - x[2])**2)
        self._rq_transform_it = cuda.compute.TransformIterator(
            self._rq_zip_it,
            lambda x: math.sqrt(
                (x[3] - x[0]) ** 2 + (x[4] - x[1]) ** 2 + (x[5] - x[2]) ** 2
            ),
        )
        # 4. Use cuda.compute reduce_into on the transform iterator and store into CuPy array view of self._ogc.rq
        self._rq_op = cuda.compute.OpKind.MAXIMUM
        self._rq_init = np.zeros(1, dtype=np.float32)
        self._rq_d_out = cp.asarray(self._ogc.rq)
        self._rq_reductor = cuda.compute.make_reduce_into(
            d_in=self._rq_transform_it,
            d_out=self._rq_d_out,
            op=self._rq_op,
            h_init=self._rq_init,
        )
        rq_storage_size = self._rq_reductor(
            temp_storage=None,
            d_in=self._rq_transform_it,
            d_out=self._rq_d_out,
            num_items=self._rq_xtc.shape[0],
            op=self._rq_op,
            h_init=self._rq_init,
        )
        self._rq_storage = cp.empty((rq_storage_size,), dtype=np.uint8)

    def compute_query_radius(self):
        # 0. Use cuda.compute and CuPy, using a stream wrapper that
        main_stream = wp.get_stream()
        self._rq_reductor(
            temp_storage=self._rq_storage,
            d_in=self._rq_transform_it,
            d_out=self._rq_d_out,
            num_items=self._rq_xtc.shape[0],
            op=self._rq_op,
            h_init=self._rq_init,
            stream=common.Stream(main_stream),
        )

    def prepare_for_execution(
        self,
        xk: wp.array[wp.vec3f],
        request_rebuild: bool = True,
    ):
        main_stream = wp.get_stream()
        wp.copy(dest=self._xk, src=xk, stream=main_stream)
        # Reset contact pairs
        for contacts, stream in zip(
            [
                self._vv,
                self._rvv,
                self._ve,
                self._rve,
                self._vf,
                self._rvf,
                self._ee,
                self._ree,
            ],
            self._streams[2:10],
        ):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream, sync_enter=False):
                contacts.clear()
        # Recompute BVH
        wp.launch(
            _compute_bounding_volumes,
            dim=max(
                self._meshes.n_verts,
                self._meshes.n_edges,
                self._meshes.n_triangles,
            ),
            inputs=[self._xk, self._meshes.data, self._ogc],
            stream=main_stream,
        )
        for bvh, stream in zip([self._e_bvh, self._f_bvh], self._streams[:2]):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream, sync_enter=False):
                if request_rebuild:
                    bvh.rebuild()
                else:
                    bvh.refit()
        # Fence
        for stream in self._streams[:10]:
            main_stream.wait_stream(stream)

    def detect_contacts(self):  # type: ignore
        n_verts, n_edges, n_half_edges, n_tris = (
            self._meshes.n_verts,
            self._meshes.n_edges,
            self._meshes.n_half_edges,
            self._meshes.n_triangles,
        )
        main_stream = wp.get_stream()
        # 1. Compute contact set
        wp.launch(
            _fused_contact_detection,
            dim=max(n_verts, n_edges) * _FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            inputs=[
                self._xk,
                self._meshes.data,
                self._ogc,
            ],
            block_dim=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            stream=main_stream,
        )
        # 2. Construct CSR representation of forward contacts
        # Each pair (u,v) for a given u is stored contiguously and sorted by v after
        # the fused contact detection. We only need to (stable-)sort by u. The counts
        # for each u are stored in counts, so the CSR prefix is an exclusive scan.
        for stream in self._streams[:8]:
            stream.wait_stream(main_stream)
        vv_capacity, ve_capacity, vf_capacity, ee_capacity = self.capacity
        with wp.ScopedStream(self._streams[0], sync_enter=False):
            wp.utils.radix_sort_pairs(
                keys=self._vv.data.u, values=self._vv.data.v, count=vv_capacity
            )
            wp.launch(
                _compute_vv_contact_data,
                dim=vv_capacity,
                inputs=[self._xk, self._meshes.data, self._ogc],
            )
        with wp.ScopedStream(self._streams[1], sync_enter=False):
            wp.utils.radix_sort_pairs(
                keys=self._ve.data.u, values=self._ve.data.v, count=ve_capacity
            )
            wp.launch(
                _compute_ve_contact_data,
                dim=ve_capacity,
                inputs=[self._xk, self._meshes.data, self._ogc],
            )
        with wp.ScopedStream(self._streams[2], sync_enter=False):
            wp.utils.radix_sort_pairs(
                keys=self._vf.data.u, values=self._vf.data.v, count=vf_capacity
            )
            wp.launch(
                _compute_vf_contact_data,
                dim=vf_capacity,
                inputs=[self._xk, self._meshes.data, self._ogc],
            )
        with wp.ScopedStream(self._streams[3], sync_enter=False):
            wp.utils.radix_sort_pairs(
                keys=self._ee.data.u, values=self._ee.data.v, count=ee_capacity
            )
            wp.launch(
                _compute_ee_contact_data,
                dim=ee_capacity,
                inputs=[self._xk, self._meshes.data, self._ogc],
            )
        with wp.ScopedStream(self._streams[4], sync_enter=False):
            wp.utils.array_scan(
                self._vv.data.counts, self._vv.data.prefix, inclusive=False
            )
        with wp.ScopedStream(self._streams[5], sync_enter=False):
            wp.utils.array_scan(
                self._ve.data.counts, self._ve.data.prefix, inclusive=False
            )
        with wp.ScopedStream(self._streams[6], sync_enter=False):
            wp.utils.array_scan(
                self._vf.data.counts, self._vf.data.prefix, inclusive=False
            )
        with wp.ScopedStream(self._streams[7], sync_enter=False):
            wp.utils.array_scan(
                self._ee.data.counts, self._ee.data.prefix, inclusive=False
            )
        # 3. Construct CSR representation of backward contacts
        # We need to store pairs (v,u) for each (u,v) for reverse contacts via mem copy.
        # Then, we sort by v (named u in reverse ContactPairs).
        wp.copy(
            dest=self._rvv.data.u,
            src=self._vv.data.v,
            count=vv_capacity,
            stream=self._streams[4],
        )
        wp.copy(
            dest=self._rvv.data.v,
            src=self._vv.data.u,
            count=vv_capacity,
            stream=self._streams[4],
        )
        wp.copy(
            dest=self._rve.data.u,
            src=self._ve.data.v,
            count=ve_capacity,
            stream=self._streams[5],
        )
        wp.copy(
            dest=self._rve.data.v,
            src=self._ve.data.u,
            count=ve_capacity,
            stream=self._streams[5],
        )
        wp.copy(
            dest=self._rvf.data.u,
            src=self._vf.data.v,
            count=vf_capacity,
            stream=self._streams[6],
        )
        wp.copy(
            dest=self._rvf.data.v,
            src=self._vf.data.u,
            count=vf_capacity,
            stream=self._streams[6],
        )
        wp.copy(
            dest=self._ree.data.u,
            src=self._ee.data.v,
            count=ee_capacity,
            stream=self._streams[7],
        )
        wp.copy(
            dest=self._ree.data.v,
            src=self._ee.data.u,
            count=ee_capacity,
            stream=self._streams[7],
        )
        with wp.ScopedStream(self._streams[4], sync_enter=False):
            wp.utils.radix_sort_pairs(
                keys=self._rvv.data.u, values=self._rvv.data.v, count=vv_capacity
            )
        with wp.ScopedStream(self._streams[5], sync_enter=False):
            wp.utils.radix_sort_pairs(
                keys=self._rve.data.u, values=self._rve.data.v, count=ve_capacity
            )
        with wp.ScopedStream(self._streams[6], sync_enter=False):
            wp.utils.radix_sort_pairs(
                keys=self._rvf.data.u, values=self._rvf.data.v, count=vf_capacity
            )
        with wp.ScopedStream(self._streams[7], sync_enter=False):
            wp.utils.radix_sort_pairs(
                keys=self._ree.data.u, values=self._ree.data.v, count=ee_capacity
            )
        # Fence
        for stream in self._streams:
            main_stream.wait_stream(stream)
        block_dim = 256
        n_threads = max(n_verts, n_half_edges, n_tris)
        # Round up to a multiple of block_dim so that wp.tile() is always called by
        # exactly block_dim threads per block. If dim is not a multiple of block_dim,
        # Warp auto-skips threads with tid >= dim, causing wp.tile() to be called by
        # fewer than block_dim threads in the last block, which produces garbage values.
        n_threads_padded = (n_threads + block_dim - 1) // block_dim * block_dim
        wp.launch(
            kernel=_compute_reverse_contact_counts,
            dim=n_threads_padded,
            inputs=[self._ogc],
            block_dim=block_dim,
            stream=main_stream,
        )
        for stream in self._streams[:4]:
            stream.wait_stream(main_stream)
        with wp.ScopedStream(self._streams[0], sync_enter=False):
            wp.utils.array_scan(
                self._rvv.data.counts, self._rvv.data.prefix, inclusive=False
            )
        with wp.ScopedStream(self._streams[1], sync_enter=False):
            wp.utils.array_scan(
                self._rve.data.counts, self._rve.data.prefix, inclusive=False
            )
        with wp.ScopedStream(self._streams[2], sync_enter=False):
            wp.utils.array_scan(
                self._rvf.data.counts, self._rvf.data.prefix, inclusive=False
            )
        with wp.ScopedStream(self._streams[3], sync_enter=False):
            wp.utils.array_scan(
                self._ree.data.counts, self._ree.data.prefix, inclusive=False
            )
        for stream in self._streams[:4]:
            main_stream.wait_stream(stream)
        # 5. Build reverse-to-forward contact index maps in parallel across contact types.
        # Each reverse contact (r_u, r_v) at position k must find its forward counterpart
        # (r_v, r_u) in the sorted forward list via binary search.
        for stream in self._streams[:4]:
            stream.wait_stream(main_stream)
        with wp.ScopedStream(self._streams[0], sync_enter=False):
            wp.launch(
                _build_reverse_to_forward_map,
                dim=vv_capacity,
                inputs=[
                    self._ogc.vv,
                    self._ogc.rvv,
                    n_verts,
                    self._ogc.rvv2vv,
                ],
            )
        with wp.ScopedStream(self._streams[1], sync_enter=False):
            wp.launch(
                _build_reverse_to_forward_map,
                dim=ve_capacity,
                inputs=[
                    self._ogc.ve,
                    self._ogc.rve,
                    n_verts,
                    self._ogc.rve2ve,
                ],
            )
        with wp.ScopedStream(self._streams[2], sync_enter=False):
            wp.launch(
                _build_reverse_to_forward_map,
                dim=vf_capacity,
                inputs=[self._ogc.vf, self._ogc.rvf, n_verts, self._ogc.rvf2vf],
            )
        with wp.ScopedStream(self._streams[3], sync_enter=False):
            wp.launch(
                _build_reverse_to_forward_map,
                dim=ee_capacity,
                inputs=[
                    self._ogc.ee,
                    self._ogc.ree,
                    n_half_edges,
                    self._ogc.ree2ee,
                ],
            )
        for stream in self._streams[:4]:
            main_stream.wait_stream(stream)

    def update_displacement_bounds(self):
        pass
        # wp.launch(
        #     _update_displacement_bounds,
        #     dim=self._meshes.n_verts,
        #     inputs=[self._meshes.data, self._ogc],
        # )

    def truncate(self, x: wp.array):
        """Truncate per-vertex displacements in-place to stay within OGC displacement bounds.

        Displacement is measured from the reference positions cached at the last
        ``prepare_for_execution`` call (``self._xk``).  If vertex ``v``'s displacement
        ``|x[V[v]] - xk[V[v]]|`` exceeds ``dminv[v]``, it is scaled back to the bound.

        Args:
            x: Current vertex positions to truncate in-place
               (``wp.array[wp.vec3f]``, global-point indexed, shape ``(N,)``).
        """
        block_dim = 64
        n_verts, n_edges = self._meshes.n_verts, self._meshes.n_edges
        self._ogc.tv.fill_(wp.float32(1))
        wp.launch(
            kernel=_planar_dat,
            dim=max(n_verts, n_edges) * block_dim,
            inputs=[self._xk, x, self._meshes.data, self._ogc],
            block_dim=block_dim,
        )
        wp.launch(
            kernel=_planar_dat_truncate,
            dim=n_verts,
            inputs=[self._xk, x, self._meshes.data, self._ogc],
        )
        # NOTE: This is OGC truncation.
        # wp.launch(
        #     _truncate_displacements,
        #     dim=self._meshes.n_verts,
        #     inputs=[self._xk, self._ogc.dminv, self._meshes.data.V, x],
        # )

    @property
    def data(self) -> OgcData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._ogc

    @property
    def meshes(self) -> MultiMesh:
        return self._meshes

    @property
    def capacity(self) -> Tuple[int, int, int, int]:
        return (
            self._vv.capacity,
            self._ve.capacity,
            self._vf.capacity,
            self._ee.capacity,
        )

    @property
    def n_primitives(self) -> Tuple[int, int, int, int]:
        """
        Returns:
            Tuple[int, int, int, int]: Number of vertices, edges, half-edges, and triangles in the input meshes, which define the primitive counts for contact pair types. Note that edge-edge contacts are defined over half-edges, so the primitive count for edge-edge contacts is the number of half-edges, not edges.
        """
        return (
            self._meshes.n_verts,
            self._meshes.n_edges,
            self._meshes.n_half_edges,
            self._meshes.n_triangles,
        )

    @property
    def num_contacts(self) -> Tuple[int, int, int, int]:
        return (
            self._vv.size(),
            self._ve.size(),
            self._vf.size(),
            self._ee.size(),
        )

    @property
    def vv_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._vv.uv()

    @property
    def ve_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._ve.uv()

    @property
    def vf_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._vf.uv()

    @property
    def ee_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._ee.uv()

    @property
    def rvv_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._rvv.uv()

    @property
    def rve_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._rve.uv()

    @property
    def rvf_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._rvf.uv()

    @property
    def ree_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._ree.uv()

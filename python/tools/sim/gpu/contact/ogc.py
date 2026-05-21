from typing import Tuple

import numpy as np
import cupy as cp
import cuda.compute
import warp as wp
import math
from enum import Enum

from .multimesh import MultiMesh, MultiMeshData
from . import halfedges
from . import queries
from .. import common
from ...common.fields import DocField
from ...gpu import common
from .mesh import pairs
from .mesh.cd import ContactDetection
from ..common import reduce


class TruncationStrategy(Enum):
    NoTruncation = 0
    Distance = 1
    PlanarDAT = 2


TRUNCATION_NONE = wp.constant(TruncationStrategy.NoTruncation.value)
TRUNCATION_DISTANCE = wp.constant(TruncationStrategy.Distance.value)
TRUNCATION_PLANARDAT = wp.constant(TruncationStrategy.PlanarDAT.value)


class OgcParams:
    """Parameters for Offset Geometric Contact detection."""

    r = DocField(0.003, "Contact radius. Pairs closer than r are in contact.")
    arq = DocField(
        1.0, "Query radius multiplier for broad-phase BVH traversal (should be >= 0)."
    )
    gammap = DocField(
        0.45, "Relaxation factor for displacement bounds (0 < gammap < 0.5)."
    )
    truncation_strategy = DocField(
        TruncationStrategy.NoTruncation, "Truncation strategy for contact queries."
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

    tv: wp.array[wp.float32]  # (# verts,) planar DAT displacement scales
    truncation_strategy: wp.int32  # Truncation strategy for contact queries


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
        if ogc.truncation_strategy == TRUNCATION_DISTANCE:
            ogc.dminv[v] = rq
    if tid < n_edges:
        e = tid
        he = meshes.EHE[e]
        einds = meshes.E[e]
        if ogc.truncation_strategy == TRUNCATION_DISTANCE:
            ogc.dmine[he[0]] = rq
            if he[1] >= 0:
                ogc.dmine[he[1]] = rq
        _compute_edge_bounding_volume(ogc, e, xk[einds[0]], xk[einds[1]], rq)  # type: ignore
    if tid < n_triangles:
        f = tid
        finds = meshes.F[f]
        if ogc.truncation_strategy == TRUNCATION_DISTANCE:
            ogc.dminf[f] = rq
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
        halfedges.incoming_vertex(F, he),  # type: ignore
        halfedges.outgoing_vertex(F, he),  # type: ignore
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
    r: wp.float32,
    tvf: tvflist,  # type: ignore
    n_verts: wp.int32,
    n_half_edges: wp.int32,
    n_tris: wp.int32,
) -> Tuple[tvvlist, tvelist, tvflist, wp.int32, wp.int32, wp.int32, wp.float32]:  # type: ignore
    tvv, tve = tvvlist(n_verts), tvelist(n_half_edges)
    n_vv, n_ve, n_vf = wp.int32(0), wp.int32(0), wp.int32(0)
    tdmin = wp.float32(0)
    if ogc.truncation_strategy == TRUNCATION_DISTANCE:
        tdmin = ogc.dminv[v]  # thread local vertex minimum distance
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
        if ogc.truncation_strategy == TRUNCATION_DISTANCE:
            tdmin = wp.min(tdmin, d)
            wp.atomic_min(ogc.dminf, f, d)
        if d > r:
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
    return tvv, tve, tvf, n_vv, n_ve, n_vf, tdmin  # type: ignore


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
    r: wp.float32,
    tee: teelist,  # type: ignore
    n_half_edges: wp.int32,
    n_edges: wp.int32,
) -> Tuple[teelist, wp.int32, wp.float32]:  # type: ignore
    fzero = wp.float32(0)
    fone = wp.float32(1)
    n_ee = wp.int32(0)
    tdmin = wp.float32(0)
    if ogc.truncation_strategy == TRUNCATION_DISTANCE:
        tdmin = ogc.dmine[hei1]  # thread local edge minimum distance
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
        if ogc.truncation_strategy == TRUNCATION_DISTANCE:
            tdmin = wp.min(tdmin, d)
        # Only store each unordered pair once (deduplication guard)
        if e1 >= e2:
            continue
        if d > r:
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
    return tee, n_ee, tdmin  # type: ignore


@wp.kernel(launch_bounds=_FUSED_CONTACT_DETECTION_BLOCK_SIZE)
def _fused_contact_detection(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: pairs.ContactPairsData,  # type: ignore
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
    rq = ogc.r + ogc.arq * ogc.rq[0]
    r = ogc.r
    if ogc.truncation_strategy == TRUNCATION_NONE:
        r = rq

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
        tvv, tve, tvf, tnvv, tnve, tnvf, tdmin = _classify_vertex_facet_contacts(
            x, meshes, ogc, v, i, xi, r, tvf, n_verts, n_half_edges, n_tris  # type: ignore
        )
        # 2.a Reduce dminv across the block and write from the last thread.
        if ogc.truncation_strategy == TRUNCATION_DISTANCE:
            dminv = wp.tile_min(wp.tile(tdmin))[0]  # type: ignore
            if local_tid == last_col:
                ogc.dminv[v] = dminv  # type: ignore
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
            tvv_offset = wp.uint64(0)
            tnvv = bvv_prefix[last_row, last_col]  # type: ignore
            if local_tid == last_col:
                tvv_offset = wp.atomic_add(contacts.vv.prefix, n_verts, wp.uint64(tnvv))  # type: ignore
                # assert wp.int32(tvv_offset) + tnvv <= contacts.vv.u.shape[
                #     0
                # ] // wp.int32(2)
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
                    k = bvv_offset[bcol] + wp.uint64(bvv_prefix[brow, bcol])  # type: ignore
                    contacts.vv.u[k] = wp.uint32(v)
                    contacts.vv.v[k] = wp.uint32(bvv[brow, bcol])

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
            tve_offset = wp.uint64(0)
            tnve = bve_prefix[last_row, last_col]  # type: ignore
            if local_tid == last_col:
                tve_offset = wp.atomic_add(contacts.ve.prefix, n_verts, wp.uint64(tnve))  # type: ignore
                # assert wp.int32(tve_offset) + tnve <= contacts.ve.u.shape[
                #     0
                # ] // wp.int32(2)
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
                    k = bve_offset[bcol] + wp.uint64(bve_prefix[brow, bcol])  # type: ignore
                    contacts.ve.u[k] = wp.uint32(v)
                    contacts.ve.v[k] = wp.uint32(bve[brow, bcol])

        if has_vf_contacts:
            # 1. Sort contacts (already unique, count=tnvf)
            bvf = wp.tile(tvf)  # type: ignore
            wp.tile_sort(keys=bvf, values=bvf)
            last_row = MAX_VF_PER_THREAD - wp.int32(1)
            assert bvf[last_row, last_col] == n_tris
            # 2. Determine global write offset (vf contacts are already unique, count=tnvf)
            tvf_offset = wp.uint64(0)
            if local_tid == last_col:
                tvf_offset = wp.atomic_add(contacts.vf.prefix, n_verts, wp.uint64(tnvf))  # type: ignore
                # assert wp.int32(tvf_offset) + tnvf <= contacts.vf.u.shape[
                #     0
                # ] // wp.int32(2)
            bvf_offset = wp.tile_from_thread(
                shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
                value=tvf_offset,
                thread_idx=last_col,
            )  # type: ignore
            # 3. Write unique contacts to global contact list
            for brow in range(MAX_VF_PER_THREAD):
                if bvf[brow, bcol] < n_tris:
                    k = bvf_offset[bcol] + wp.uint64(brow * block_dims + bcol)
                    contacts.vf.u[k] = wp.uint32(v)
                    contacts.vf.v[k] = wp.uint32(bvf[brow, bcol])

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
        tee, tnee, tdmin = _classify_edge_edge_contacts(
            x, meshes, ogc, e1, einds1, xi1, xj1, hei, r, tee, n_half_edges, n_edges  # type: ignore
        )
        # 2.a Reduce dmine across the block and write from the last thread.
        if ogc.truncation_strategy == TRUNCATION_DISTANCE:
            dminv = wp.tile_min(wp.tile(tdmin))[0]  # type: ignore
            if local_tid == last_col:
                ogc.dmine[hei] = tdmin
                if hej >= wp.int32(0):
                    ogc.dmine[hej] = tdmin
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
            tee_offset = wp.uint64(0)
            if local_tid == last_col:
                tee_offset = wp.atomic_add(contacts.ee.prefix, n_half_edges, wp.uint64(tnee))  # type: ignore
                # assert wp.int32(tee_offset) + tnee <= contacts.ee.u.shape[
                #     0
                # ] // wp.int32(2)
            bee_offset = wp.tile_from_thread(
                shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
                value=tee_offset,
                thread_idx=last_col,
            )  # type: ignore
            # 3. Write contacts to global contact set
            for brow in range(MAX_EE_PER_THREAD):
                if bee[brow, bcol] < n_half_edges:
                    k = bee_offset[bcol] + wp.uint64(brow * block_dims + bcol)  # type: ignore
                    contacts.ee.u[k] = wp.uint32(he_max)  # type: ignore
                    contacts.ee.v[k] = wp.uint32(bee[brow, bcol])


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


class Ogc(ContactDetection):
    """Offset Geometric Contact"""

    _ogc: OgcData  # pyright: ignore[reportGeneralTypeIssues]
    _e_bvh: wp.Bvh  # BVH over edges
    _f_bvh: wp.Bvh  # BVH over faces
    _meshes: MultiMesh  # Meshes # type: ignore
    _truncation_strategy: TruncationStrategy
    _streams: list[wp.Stream]  # Stream list
    _query_radius_reduction: reduce.Reduce
    _params: OgcParams

    def __init__(
        self,
        params: OgcParams = OgcParams(),
    ):
        self._params = params

    def enable_adaptive_query_radius(
        self, xt: wp.array[wp.vec3f], xtilde: wp.array[wp.vec3f]
    ):
        # Compute rq = r + beta * (xtilde - xt).colwise().norm().maxCoeff()
        # 1. Capture xt, xtilde as CuPy 3 x N arrays
        xtc = cp.asarray(xt)
        xtildec = cp.asarray(xtilde)
        # 2. Use ZipIterator(xtc[0,:], xtc[1,:], xtc[2,:], xtildec[0,:], xtildec[1,:], xtildec[2,:])
        zip_it = cuda.compute.ZipIterator(
            xtc[:, 0],
            xtc[:, 1],
            xtc[:, 2],
            xtildec[:, 0],
            xtildec[:, 1],
            xtildec[:, 2],
        )
        # 3. Use TransformIterator on the ZipIterator as transform = lambda x: sqrt((x[3] - x[0])**2 + (x[4] - x[1])**2 + (x[5] - x[2])**2)
        transform_it = cuda.compute.TransformIterator(
            zip_it,
            lambda x: math.sqrt(
                (x[3] - x[0]) ** 2 + (x[4] - x[1]) ** 2 + (x[5] - x[2]) ** 2
            ),
        )
        # 4. Use cuda.compute reduce_into on the transform iterator and store into CuPy array view of self._ogc.rq
        self._query_radius_reduction = reduce.Reduce(
            d_in=transform_it,
            d_out=cp.asarray(self._ogc.rq),
            num_items=xt.shape[0],
            op=cuda.compute.OpKind.MAXIMUM,
        )

    def prepare_for_execution(
        self,
        request_rebuild: bool = True,
    ):
        main_stream = wp.get_stream()
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
        for stream in self._streams[:2]:
            main_stream.wait_stream(stream)

    def _detect_contacts(self, contacts: pairs.ContactPairs):  # type: ignore
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
                contacts.write_data,
            ],
            block_dim=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            stream=main_stream,
        )
        # 2. Update displacement bounds
        wp.launch(
            kernel=_update_displacement_bounds,
            dim=n_verts,
            inputs=[
                self._meshes.data,
                self._ogc,
            ],
        )

    def truncate(self, x: wp.array):
        """Truncate per-vertex displacements in-place to stay within OGC displacement bounds.

        Displacement is measured from the reference positions cached at the last
        ``prepare_for_execution`` call (``self._xk``).  If vertex ``v``'s displacement
        ``|x[V[v]] - xk[V[v]]|`` exceeds ``dminv[v]``, it is scaled back to the bound.

        Args:
            x: Current vertex positions to truncate in-place
               (``wp.array[wp.vec3f]``, global-point indexed, shape ``(N,)``).
        """
        if self._ogc.truncation_strategy == TRUNCATION_DISTANCE:
            block_dim = 32
            wp.launch(
                _truncate_displacements,
                dim=self._meshes.n_verts,
                inputs=[self._xk, self._ogc.dminv, self._meshes.data.V, x],
                block_dim=block_dim,
            )
        elif self._ogc.truncation_strategy == TRUNCATION_PLANARDAT:
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

    def register_handles(
        self,
        xt: wp.array[wp.vec3f],
        xk: wp.array[wp.vec3f],
        x: wp.array[wp.vec3f],
        xtilde: wp.array[wp.vec3f],
        meshes: MultiMesh, 
        contacts: pairs.ContactPairs,
    ):
        super().register_handles(xt, xk, x, xtilde, meshes, contacts)

        # Construct
        self._ogc = OgcData()
        self._ogc.truncation_strategy = int(self._params.truncation_strategy.value)  # type: ignore
        self._ogc.e_lowers = wp.zeros((meshes.n_edges,), dtype=wp.vec3f)
        self._ogc.e_uppers = wp.zeros((meshes.n_edges,), dtype=wp.vec3f)
        self._ogc.f_lowers = wp.zeros((meshes.n_triangles,), dtype=wp.vec3f)
        self._ogc.f_uppers = wp.zeros((meshes.n_triangles,), dtype=wp.vec3f)
        self._ogc.arq = self._params.arq
        self._ogc.r = self._params.r
        self._ogc.gammap = self._params.gammap
        self._ogc.dminv = wp.zeros((meshes.n_verts,), dtype=wp.float32)
        self._ogc.dmine = wp.zeros((meshes.n_half_edges,), dtype=wp.float32)
        self._ogc.dminf = wp.zeros((meshes.n_triangles,), dtype=wp.float32)
        self._ogc.rq = wp.zeros((1,), dtype=wp.float32)  # (1,) OGC query radius
        self._ogc.tv = wp.empty((meshes.n_verts,), dtype=wp.float32)  # type: ignore

        dim = max(meshes.n_verts, meshes.n_edges, meshes.n_triangles)
        wp.launch(
            kernel=_compute_bounding_volumes,
            dim=dim,
            inputs=[self._x, self._meshes.data, self._ogc],
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
        self._streams = [wp.Stream() for _ in range(2)]
        self.enable_adaptive_query_radius(xt, xtilde)

    def on_time_step_started(self):
        main_stream = wp.get_stream()
        self._query_radius_reduction(main_stream)

    def detect_contacts(self, from_xt: bool = False):
        main_stream = wp.get_stream()
        x = self._xt if from_xt else self._x
        wp.copy(dest=self._xk, src=x, stream=main_stream)
        self.prepare_for_execution(request_rebuild=True)
        self._contacts.clear()
        self._detect_contacts(self._contacts)
        self._contacts.assemble_contacts(self._xk, with_reverse_contacts=True)

    def filter_step(self):
        self.truncate(self._x)

    def on_time_step_ended(self):
        pass

    @property
    def data(self) -> OgcData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._ogc

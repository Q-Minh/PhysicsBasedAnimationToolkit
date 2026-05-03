from typing import Any, Tuple

import numpy as np
import cupy as cp

import warp as wp
import polyscope as ps
import polyscope.imgui as imgui

from .set import ContactSet, ContactSetData

from .multimesh import MultiMesh, MultiMeshData
from . import halfedges
from . import queries


VF_E_FACE_TRIANGLE = wp.constant(0)
VF_E_FACE_EDGE = wp.constant(1)
VF_E_FACE_VERTEX = wp.constant(2)
MAX_VV_PER_THREAD = wp.constant(2)
MAX_VE_PER_THREAD = wp.constant(2)
MAX_VF_PER_THREAD = wp.constant(4)
MAX_EE_PER_THREAD = wp.constant(4)
_FUSED_CONTACT_DETECTION_BLOCK_SIZE = wp.constant(64)
tvvlist = wp.types.vector(length=MAX_VV_PER_THREAD, dtype=wp.int32)
tvelist = wp.types.vector(length=MAX_VE_PER_THREAD, dtype=wp.int32)
tvflist = wp.types.vector(length=MAX_VF_PER_THREAD, dtype=wp.int32)
teelist = wp.types.vector(length=MAX_EE_PER_THREAD, dtype=wp.int32)


@wp.struct
class OgcData:
    """Data structure for OGC."""

    e_bvh_id: wp.uint64  # Edge BVH ID
    f_bvh_id: wp.uint64  # Triangle BVH ID
    e_lowers: wp.array[wp.vec3f]  # (# edges,) edge AABB lower bounds
    e_uppers: wp.array[wp.vec3f]  # (# edges,) edge AABB upper bounds
    f_lowers: wp.array[wp.vec3f]  # (# triangles,) triangle AABB lower bounds
    f_uppers: wp.array[wp.vec3f]  # (# triangles,) triangle AABB upper bounds

    rq: wp.float32  # OGC query radius
    r: wp.float32  # OGC contact radius
    gammap: (
        wp.float32
    )  # Relaxation parameter for vertex displacement bound, must satisfy `0 < gammap < 0.5`

    dminv: wp.array[wp.float32]  # (# vertices,) vertex minimum displacement bounds
    dmine: wp.array[wp.float32]  # (# half-edges,) half-edge minimum displacement bounds
    dminf: wp.array[wp.float32]  # (# triangles,) face minimum displacement bounds

    nvv: wp.array[
        wp.int32
    ]  # (# verts + 1,) array of vertex-vertex contact counts per vertex and the total count at the array's tail
    vv_u: wp.array[wp.int32]  # (# vertex-vertex contacts capacity,) u from pairs (u,v)
    vv_v: wp.array[wp.int32]  # (# vertex-vertex contacts capacity,) v from pairs (u,v)

    nve: wp.array[
        wp.int32
    ]  # (# verts + 1,) array of vertex-edge contact counts per vertex and the total count at the array's tail
    ve_u: wp.array[
        wp.int32
    ]  # (# vertex-(half-)edge contacts capacity,) u from pairs (u,v)
    ve_v: wp.array[
        wp.int32
    ]  # (# vertex-(half-)edge contacts capacity,) v from pairs (u,v)

    nvf: wp.array[
        wp.int32
    ]  # (# verts + 1,) array of vertex-triangle contact counts per vertex and the total count at the array's tail
    vf_u: wp.array[
        wp.int32
    ]  # (# vertex-triangle contacts capacity,) u from pairs (u,v)
    vf_v: wp.array[
        wp.int32
    ]  # (# vertex-triangle contacts capacity,) v from pairs (u,v)

    nee: wp.array[
        wp.int32
    ]  # (# half-edges + 1,) array of (half-)edge-(half-)edge contact counts per half-edge and the total count at the array's tail
    ee_u: wp.array[
        wp.int32
    ]  # (# (half-)edge-(half-)edge contacts capacity,) u from pairs (u,v)
    ee_v: wp.array[
        wp.int32
    ]  # (# (half-)edge-(half-)edge contacts capacity,) v from pairs (u,v)


@wp.func
def _compute_edge_bounding_volume(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    e: wp.int32,
):
    ogc.dmine[e] = ogc.rq
    einds = meshes.E[e]
    xi, xj = x[einds[0]], x[einds[1]]
    xmid = float(0.5) * (xi + xj)
    hlen = float(0.5) * wp.norm_l2(xj - xi)
    radius = hlen + ogc.rq
    ogc.e_lowers[e] = xmid - wp.vec3f(radius)
    ogc.e_uppers[e] = xmid + wp.vec3f(radius)


@wp.func
def _compute_triangle_bounding_volume(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    f: wp.int32,
):
    finds = meshes.F[f]
    xi, xj, xk = x[finds[0]], x[finds[1]], x[finds[2]]
    xmin = wp.min(
        xi, wp.min(xj, xk)  # pyright: ignore[reportArgumentType, reportCallIssue]
    )
    xmax = wp.max(
        xi, wp.max(xj, xk)  # pyright: ignore[reportArgumentType, reportCallIssue]
    )
    ogc.f_lowers[f] = xmin - wp.vec3f(ogc.rq)
    ogc.f_uppers[f] = xmax + wp.vec3f(ogc.rq)


@wp.kernel
def _compute_bounding_volumes_kernel(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    tid = wp.tid()
    n_verts = meshes.V.shape[0]
    n_edges = meshes.E.shape[0]
    n_triangles = meshes.F.shape[0]
    if tid < n_verts:
        v = tid
        ogc.dminv[v] = ogc.rq
    if tid < n_edges:
        e = tid
        he = meshes.EHE[e]
        ogc.dmine[he[0]] = ogc.rq
        if he[1] >= 0:
            ogc.dmine[he[1]] = ogc.rq
        _compute_edge_bounding_volume(x, meshes, ogc, e)  # type: ignore
    if tid < n_triangles:
        f = tid
        ogc.dminf[f] = ogc.rq
        _compute_triangle_bounding_volume(x, meshes, ogc, f)  # type: ignore


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
    GXV: wp.array[wp.int32],
    f: wp.int32,
    a_local: wp.int32,
    e_face: wp.int32,
) -> wp.int32:
    """Determine the contact face index for a point-triangle contact based on the barycentric coordinates `uvw`."""
    return (
        # face contact, return face index
        wp.int32(e_face == VF_E_FACE_TRIANGLE) * f
        # edge contact, return half-edge index
        + wp.int32(e_face == VF_E_FACE_EDGE) * (wp.int32(3) * f + a_local)
        # vertex contact, return global vertex index
        + wp.int32(e_face == VF_E_FACE_VERTEX) * GXV[F[f][a_local]]  # type: ignore
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
    fi: wp.int32,  # face index
    he: wp.int32,  # half-edge index
    y: wp.vec3f,  # query point
    check_adjacent_facets: wp.bool = wp.bool(True),
):
    fzero = wp.float32(0)
    zero = wp.int32(0)
    hef = GHEF[he]
    fj = hef[1]  # type: ignore
    # Handle boundary edge case: no adjacent face (i.e. fj == -1)
    fj = wp.int32(fj < zero) * fi + wp.int32(fj >= zero) * fj  # type: ignore
    i, j = (
        halfedges.incoming_vertex(F, he),
        halfedges.outgoing_vertex(F, he),
    )
    xi, xj = x[i], x[j]
    in_edge_feasible_region = wp.bool(True)
    in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xi, xj - xi) >= fzero)  # type: ignore
    in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xj, xi - xj) >= fzero)  # type: ignore
    if check_adjacent_facets:
        # Get the third vertex l of triangle fj that is not part of undirected edge (i,j).
        # NOTE: Whenever fj == fi (i.e. boundary edge), l == k.
        k = halfedges.next_vertex(F, he, wp.int32(2))
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
        ogc.dminv[v] = wp.min(ogc.dminv[v], d)
        wp.atomic_min(ogc.dminf, f, d)
        if d >= ogc.r:
            tvf[brow] = n_tris
            continue
        a_local, e_face = _closest_face_point_triangle(uvw)
        a = _vertex_triangle_contact_face_index(
            meshes.F, meshes.GXV, f, a_local, e_face
        )
        if e_face == VF_E_FACE_VERTEX:
            if is_vertex_feasible(x, meshes.F, meshes.GVHEp, meshes.GVHEadj, a, xi):
                tvv[brow] = a
                n_vv += wp.int32(1)
            tvf[brow] = n_tris
        elif e_face == VF_E_FACE_EDGE:
            if is_edge_feasible(
                x, meshes.F, meshes.GHEF, f, a, xi, check_adjacent_facets=wp.bool(True)
            ):  # type: ignore
                tve[brow] = (
                    a  # NOTE: Should we use the largest of opposite half-edge indices?
                )
                n_ve += wp.int32(1)
            tvf[brow] = n_tris
        else:  # VF_E_FACE_TRIANGLE
            n_vf += wp.int32(1)
    return tvv, tve, tvf, n_vv, n_ve, n_vf


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
    hej1: wp.int32,
    tee: teelist,  # type: ignore
    n_half_edges: wp.int32,
) -> Tuple[teelist, wp.int32]:  # type: ignore
    fzero = wp.float32(0)
    fone = wp.float32(1)
    n_ee = wp.int32(0)
    for brow in range(MAX_EE_PER_THREAD):
        e2 = tee[brow]  # pyright: ignore[reportIndexIssue]
        if e2 >= n_half_edges:
            break
        einds2 = meshes.E[e2]
        xi2, xj2 = x[einds2[0]], x[einds2[1]]
        are_adjacent = (
            (einds1[0] == einds2[0])  # type: ignore
            or (einds1[0] == einds2[1])  # type: ignore
            or (einds1[1] == einds2[0])  # type: ignore
            or (einds1[1] == einds2[1])  # type: ignore
        )
        if are_adjacent:
            tee[brow] = n_half_edges  # type: ignore
            continue
        st = queries.closest_points_line_segments(xi1, xj1, xi2, xj2)  # type: ignore
        xc1 = (fone - st[0]) * xi1 + st[0] * xj1  # type: ignore
        xc2 = (fone - st[1]) * xi2 + st[1] * xj2  # type: ignore
        d = wp.norm_l2(xc1 - xc2)
        # Update displacement bounds before the deduplication guard so that both edges in a pair
        # update their own bounds when they each encounter the symmetric (e1,e2)/(e2,e1) pair.
        ogc.dmine[hei1] = wp.min(ogc.dmine[hei1], d)
        if hej1 >= wp.int32(0):
            ogc.dmine[hej1] = wp.min(ogc.dmine[hej1], d)
        # Only store each unordered pair once (deduplication guard)
        if e1 >= e2:
            tee[brow] = n_half_edges
            continue
        if d >= ogc.r:
            tee[brow] = n_half_edges
            continue
        is_xc1_vertex = st[0] == fzero or st[0] == fone  # type: ignore
        is_xc2_vertex = st[1] == fzero or st[1] == fone  # type: ignore
        if is_xc1_vertex or is_xc2_vertex:
            tee[brow] = n_half_edges
            continue
        he2 = meshes.EHE[e2]
        hei2, hej2 = he2[0], he2[1]
        tee[n_ee] = wp.max(hei2, hej2)
        n_ee += wp.int32(1)
    return tee, n_ee


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
            candidates = wp.tile_bvh_query_next(query)
            f = candidates[local_tid]  # type: ignore
            if f >= 0:
                tvf[brow] = f
            no_more_candidates = candidates[last_col] < int(0)  # type: ignore
            if no_more_candidates:
                break
        # 2. Classify and store vv,ve,vf contacts
        tvv, tve, tvf, tnvv, tnve, tnvf = _classify_vertex_facet_contacts(
            x, meshes, ogc, v, i, xi, tvf, n_verts, n_half_edges, n_tris  # type: ignore
        )
        # 2.a Count contacts (including duplicates) for early exit opportunity
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
                tvv_offset = wp.atomic_add(ogc.nvv, n_verts, tnvv)  # type: ignore
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
                    ogc.vv_u[k] = v
                    ogc.vv_v[k] = bvv[brow, bcol]
            ogc.nvv[v] = tnvv  # type: ignore

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
                tve_offset = wp.atomic_add(ogc.nve, n_verts, tnve)  # type: ignore
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
                    ogc.ve_u[k] = v
                    ogc.ve_v[k] = bve[brow, bcol]
            ogc.nve[v] = tnve  # type: ignore

        if has_vf_contacts:
            # 1. Sort contacts
            bvf = wp.tile(tvf)  # type: ignore
            wp.tile_sort(keys=bvf, values=bvf)
            last_row = MAX_VF_PER_THREAD - wp.int32(1)
            assert bvf[last_row, last_col] == n_tris
            # 2. Determine global write offset (vf contacts are already unique, count=tnvf)
            tvf_offset = wp.int32(0)
            if local_tid == last_col:
                tvf_offset = wp.atomic_add(ogc.nvf, n_verts, tnvf)  # type: ignore
            bvf_offset = wp.tile_from_thread(
                shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
                value=tvf_offset,
                thread_idx=last_col,
            )  # type: ignore
            # 3. Write unique contacts to global contact list
            for brow in range(MAX_VF_PER_THREAD):
                if bvf[brow, bcol] < n_tris:
                    k = bvf_offset[bcol] + brow * block_dims + bcol
                    ogc.vf_u[k] = v
                    ogc.vf_v[k] = bvf[brow, bcol]
            ogc.nvf[v] = tnvf  # type: ignore

    # EE contact detection
    if block_id < n_edges:
        e = block_id
        hei, hej = meshes.EHE[e][0], meshes.EHE[e][1]
        # Store the larger half-edge index to handle boundary edges
        he_min, he_max = wp.min(hei, hej), wp.max(hei, hej)
        e1 = e
        einds1 = meshes.E[e1]
        xi1, xj1 = x[einds1[0]], x[einds1[1]]
        # 1. Query all nearby edges
        tee = teelist(n_half_edges)
        query = wp.tile_bvh_query_aabb(
            ogc.e_bvh_id,
            ogc.e_lowers[e],
            ogc.e_uppers[e],
        )
        for brow in range(MAX_EE_PER_THREAD):
            candidates = wp.tile_bvh_query_next(query)
            e2 = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
            if e2 >= 0:
                tee[brow] = e2
            no_more_candidates = candidates[last_col] < int(0)  # type: ignore
            if no_more_candidates:
                break
        # 2. Classify and store ee contacts
        tee, tnee = _classify_edge_edge_contacts(
            x, meshes, ogc, e1, einds1, xi1, xj1, hei, hej, tee, n_half_edges  # type: ignore
        )
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
                tee_offset = wp.atomic_add(ogc.nee, n_half_edges, tnee)  # type: ignore
            bee_offset = wp.tile_from_thread(
                shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
                value=tee_offset,
                thread_idx=last_col,
            )  # type: ignore
            # 3. Write contacts to global contact set
            for brow in range(MAX_EE_PER_THREAD):
                if bee[brow, bcol] < n_half_edges:
                    k = bee_offset[bcol] + brow * block_dims + bcol  # type: ignore
                    ogc.ee_u[k] = he_max
                    ogc.ee_v[k] = bee[brow, bcol]
            # 4. Write unique contact counts to global count arrays
            ogc.nee[he_max] = tnee  # type: ignore
            if he_min >= wp.int32(0):
                ogc.nee[he_min] = wp.int32(0)  # type: ignore


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


class Ogc:
    """Offset Geometric Contact"""

    _ogc: OgcData  # pyright: ignore[reportGeneralTypeIssues]
    _e_bvh: wp.Bvh  # BVH over edges
    _f_bvh: wp.Bvh  # BVH over faces
    _points: wp.array[wp.vec3f]  # (N,) vertex positions
    _meshes: MultiMesh  # Meshes # type: ignore

    def __init__(
        self,
        points: wp.array[wp.vec3f],
        meshes: MultiMesh,  # type: ignore
        r: float = 0.002,
        gammap: float = 0.45,
        n_vv_contact_capacity: float = float(1),
        n_ve_contact_capacity: float = float(1),
        n_vf_contact_capacity: float = float(1),
        n_ee_contact_capacity: float = float(1),
    ):
        """Construct ogc data

        Args:
            points (wp.array[wp.vec3f]): (N,) points
            meshes (MultiMesh): Multi-body mesh
            gammap (float, optional): Fraction of displacement (<0.5) for OGC bounds. Defaults to 0.45.
            n_vv_contact_capacity (int, optional): Multiplier s.t.
            # point-point contact capacity = n_vv_contact_capacity * n_verts. Defaults to 1.
            n_ve_contact_capacity (int, optional): Multiplier s.t.
            # point-edge contact capacity = n_ve_contact_capacity * n_verts. Defaults to 1.
            n_vf_contact_capacity (int, optional): Multiplier s.t.
            # point-face contact capacity = n_vf_contact_capacity * n_verts. Defaults to 1.
            n_ee_contact_capacity (int, optional): Multiplier s.t.
            # edge-edge contact capacity = n_ee_contact_capacity * n_edges. Defaults to 1.
        """
        self._points = points
        self._meshes = meshes
        self._ogc = OgcData()
        self._ogc.e_lowers = wp.zeros((meshes.n_edges,), dtype=wp.vec3f)
        self._ogc.e_uppers = wp.zeros((meshes.n_edges,), dtype=wp.vec3f)
        self._ogc.f_lowers = wp.zeros((meshes.n_triangles,), dtype=wp.vec3f)
        self._ogc.f_uppers = wp.zeros((meshes.n_triangles,), dtype=wp.vec3f)
        self._ogc.rq = 2 * r
        self._ogc.r = r
        self._ogc.gammap = gammap
        self._ogc.dminv = wp.zeros((meshes.n_verts,), dtype=wp.float32)
        self._ogc.dmine = wp.zeros((meshes.n_half_edges,), dtype=wp.float32)
        self._ogc.dminf = wp.zeros((meshes.n_triangles,), dtype=wp.float32)
        self._ogc.nvv = wp.zeros((meshes.n_verts + 1,), dtype=wp.int32)
        self._ogc.nve = wp.zeros((meshes.n_verts + 1,), dtype=wp.int32)
        self._ogc.nvf = wp.zeros((meshes.n_verts + 1,), dtype=wp.int32)
        self._ogc.nee = wp.zeros((meshes.n_half_edges + 1,), dtype=wp.int32)
        vv_capacity = int(n_vv_contact_capacity * meshes.n_verts)
        ve_capacity = int(n_ve_contact_capacity * meshes.n_verts)
        vf_capacity = int(n_vf_contact_capacity * meshes.n_verts)
        ee_capacity = int(n_ee_contact_capacity * meshes.n_edges)
        self._ogc.vv_u = wp.full(
            shape=(2 * vv_capacity,), value=meshes.n_verts, dtype=wp.int32
        )
        self._ogc.vv_v = wp.full(
            shape=(2 * vv_capacity,), value=meshes.n_verts, dtype=wp.int32
        )
        self._ogc.ve_u = wp.full(
            shape=(2 * ve_capacity,), value=meshes.n_verts, dtype=wp.int32
        )
        self._ogc.ve_v = wp.full(
            shape=(2 * ve_capacity,), value=meshes.n_half_edges, dtype=wp.int32
        )
        self._ogc.vf_u = wp.full(
            shape=(2 * vf_capacity,), value=meshes.n_verts, dtype=wp.int32
        )
        self._ogc.vf_v = wp.full(
            shape=(2 * vf_capacity,), value=meshes.n_triangles, dtype=wp.int32
        )
        self._ogc.ee_u = wp.full(
            shape=(2 * ee_capacity,), value=meshes.n_half_edges, dtype=wp.int32
        )
        self._ogc.ee_v = wp.full(
            shape=(2 * ee_capacity,), value=meshes.n_half_edges, dtype=wp.int32
        )
        dim = max(meshes.n_verts, meshes.n_edges, meshes.n_triangles)
        wp.launch(
            kernel=_compute_bounding_volumes_kernel,
            dim=dim,
            inputs=[self._points, self._meshes.data, self._ogc],
        )
        self._e_bvh, self._f_bvh = (
            wp.Bvh(
                self._ogc.e_lowers,
                self._ogc.e_uppers,
                constructor="lbvh",
                groups=None,
                leaf_size=4,
            ),
            wp.Bvh(
                self._ogc.f_lowers,
                self._ogc.f_uppers,
                constructor="lbvh",
                groups=None,
                leaf_size=4,
            ),
        )
        self._ogc.e_bvh_id, self._ogc.f_bvh_id = (
            self._e_bvh.id,
            self._f_bvh.id,
        )

    def prepare_for_execution(self, request_rebuild: bool = True):
        wp.launch(
            _compute_bounding_volumes_kernel,
            dim=max(
                self._meshes.n_verts,
                self._meshes.n_edges,
                self._meshes.n_triangles,
            ),
            inputs=[self._points, self._meshes.data, self._ogc],
        )
        for bvh in (self._e_bvh, self._f_bvh):
            if request_rebuild:
                bvh.rebuild()
            else:
                bvh.refit()
        # Reset contact pairs
        self._ogc.nvv.fill_(wp.int32(0))
        self._ogc.nve.fill_(wp.int32(0))
        self._ogc.nvf.fill_(wp.int32(0))
        self._ogc.nee.fill_(wp.int32(0))
        self._ogc.vv_u.fill_(self._meshes.n_verts)
        self._ogc.vv_v.fill_(self._meshes.n_verts)
        self._ogc.ve_u.fill_(self._meshes.n_verts)
        self._ogc.ve_v.fill_(self._meshes.n_half_edges)
        self._ogc.vf_u.fill_(self._meshes.n_verts)
        self._ogc.vf_v.fill_(self._meshes.n_triangles)
        self._ogc.ee_u.fill_(self._meshes.n_half_edges)
        self._ogc.ee_v.fill_(self._meshes.n_half_edges)

    def detect_contacts(self):  # type: ignore
        dim = max(self._meshes.n_verts, self._meshes.n_edges)
        wp.launch(
            _fused_contact_detection,
            dim=dim * _FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            inputs=[
                self._points,
                self._meshes.data,
                self._ogc,
            ],
            block_dim=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
        )
        vv_capacity, ve_capacity, vf_capacity, ee_capacity = self.capacity
        wp.utils.radix_sort_pairs(
            keys=self._ogc.vv_u, values=self._ogc.vv_v, count=vv_capacity
        )
        wp.utils.radix_sort_pairs(
            keys=self._ogc.ve_u, values=self._ogc.ve_v, count=ve_capacity
        )
        wp.utils.radix_sort_pairs(
            keys=self._ogc.vf_u, values=self._ogc.vf_v, count=vf_capacity
        )
        wp.utils.radix_sort_pairs(
            keys=self._ogc.ee_u, values=self._ogc.ee_v, count=ee_capacity
        )

    def update_displacement_bounds(self):
        wp.launch(
            _update_displacement_bounds,
            dim=self._meshes.n_verts,
            inputs=[self._meshes.data, self._ogc],
        )

    @property
    def data(self) -> OgcData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._ogc

    @property
    def capacity(self) -> Tuple[int, int, int, int]:
        return (
            self._ogc.vv_u.shape[0] // 2,
            self._ogc.ve_u.shape[0] // 2,
            self._ogc.vf_u.shape[0] // 2,
            self._ogc.ee_u.shape[0] // 2,
        )

    @property
    def vv_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        n_vv = cp.asarray(self._ogc.nvv)[-1].get()
        vv_u, vv_v = self._ogc.vv_u.numpy()[:n_vv], self._ogc.vv_v.numpy()[:n_vv]
        return vv_u, vv_v

    @property
    def ve_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        n_ve = cp.asarray(self._ogc.nve)[-1].get()
        ve_u, ve_v = self._ogc.ve_u.numpy()[:n_ve], self._ogc.ve_v.numpy()[:n_ve]
        return ve_u, ve_v

    @property
    def vf_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        n_vf = cp.asarray(self._ogc.nvf)[-1].get()
        vf_u, vf_v = self._ogc.vf_u.numpy()[:n_vf], self._ogc.vf_v.numpy()[:n_vf]
        return vf_u, vf_v

    @property
    def ee_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        n_ee = cp.asarray(self._ogc.nee)[-1].get()
        ee_u, ee_v = self._ogc.ee_u.numpy()[:n_ee], self._ogc.ee_v.numpy()[:n_ee]
        return ee_u, ee_v


class OgcContactBrowser:
    """Simple Polyscope/imgui browser for OGC contact pairs."""

    _CONTACT_KINDS = ["VV", "VE", "VF", "EE"]

    def __init__(self, ogc: Ogc):
        self._ogc = ogc
        self._x = ogc._points.numpy()
        self._V = ogc._meshes.data.V.numpy()
        self._F = ogc._meshes.data.F.numpy()
        self._kind_idx: int = 0
        self._contact_idx: int = 0
        self._stencil_pc = None
        self._stencil_cn = None
        self._stencil_sm = None
        self._last_visualized: tuple[int, int, int] | None = None

    def clear(self):
        if self._stencil_pc is not None:
            ps.remove_point_cloud(self._stencil_pc.get_name())
            self._stencil_pc = None
        if self._stencil_cn is not None:
            ps.remove_curve_network(self._stencil_cn.get_name())
            self._stencil_cn = None
        if self._stencil_sm is not None:
            ps.remove_surface_mesh(self._stencil_sm.get_name())
            self._stencil_sm = None

    def draw(self):
        imgui.PushID("OgcContactBrowser")  # type: ignore

        imgui.Text(f"# Vertex-Vertex Contacts: {len(self._ogc.vv_contacts[0])}")  # type: ignore
        imgui.Text(f"# Vertex-Edge Contacts: {len(self._ogc.ve_contacts[0])}")  # type: ignore
        imgui.Text(f"# Vertex-Triangle Contacts: {len(self._ogc.vf_contacts[0])}")  # type: ignore
        imgui.Text(f"# Edge-Edge Contacts: {len(self._ogc.ee_contacts[0])}")  # type: ignore

        _, self._kind_idx = imgui.Combo("Kind", self._kind_idx, self._CONTACT_KINDS)  # type: ignore

        contacts = self._get_selected_contacts()
        n = len(contacts[0])
        if n == 0:
            self._contact_idx = 0
            self.clear()
            self._last_visualized = None
            imgui.Text("No contacts of selected kind.")  # type: ignore
            imgui.PopID()  # type: ignore
            return

        self._contact_idx = max(0, min(self._contact_idx, n - 1))

        if imgui.Button("<##ogc_prev"):  # type: ignore
            self._contact_idx = max(0, self._contact_idx - 1)
        imgui.SameLine()  # type: ignore
        imgui.SetNextItemWidth(80)  # type: ignore
        _, self._contact_idx = imgui.InputInt("Index", self._contact_idx)  # type: ignore
        self._contact_idx = max(0, min(self._contact_idx, n - 1))
        imgui.SameLine()  # type: ignore
        if imgui.Button(">##ogc_next"):  # type: ignore
            self._contact_idx = min(n - 1, self._contact_idx + 1)
        imgui.SameLine()  # type: ignore
        imgui.Text(f"/ {n - 1}")  # type: ignore

        signature = (self._kind_idx, self._contact_idx, n)
        if signature != self._last_visualized:
            self._visualize_current_contact()
            self._last_visualized = signature

        u = int(contacts[0][self._contact_idx])
        v = int(contacts[1][self._contact_idx])
        imgui.Text(f"pair = ({u}, {v})")  # type: ignore

        imgui.PopID()  # type: ignore

    def _get_selected_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        if self._kind_idx == 0:
            return self._ogc.vv_contacts
        if self._kind_idx == 1:
            return self._ogc.ve_contacts
        if self._kind_idx == 2:
            return self._ogc.vf_contacts
        return self._ogc.ee_contacts

    def _visualize_current_contact(self):
        self.clear()

        contacts = self._get_selected_contacts()
        if len(contacts[0]) == 0:
            return

        x = self._x
        V = self._V
        F = self._F
        k = self._contact_idx

        if self._kind_idx == 0:
            u, v = int(contacts[0][k]), int(contacts[1][k])
            iu = int(V[u])
            iv = int(V[v])
            self._stencil_pc = ps.register_point_cloud("OGC VV Contact", x[[iu, iv], :])

        elif self._kind_idx == 1:
            u, he = int(contacts[0][k]), int(contacts[1][k])
            iu = int(V[u])
            f = he // 3
            e_local = he % 3
            i = int(F[f, e_local])
            j = int(F[f, (e_local + 1) % 3])
            self._stencil_pc = ps.register_point_cloud("OGC VE Vertex", x[[iu], :])
            self._stencil_cn = ps.register_curve_network(
                "OGC VE Edge",
                x[[i, j], :],
                np.array([[0, 1]], dtype=np.int32),
            )

        elif self._kind_idx == 2:
            u, f = int(contacts[0][k]), int(contacts[1][k])
            iu = int(V[u])
            tri = F[f, :]
            self._stencil_pc = ps.register_point_cloud("OGC VF Vertex", x[[iu], :])
            self._stencil_sm = ps.register_surface_mesh(
                "OGC VF Triangle",
                x[tri, :],
                np.array([[0, 1, 2]], dtype=np.int32),
            )

        elif self._kind_idx == 3:
            he0, he1 = int(contacts[0][k]), int(contacts[1][k])
            f0, e0 = he0 // 3, he0 % 3
            f1, e1 = he1 // 3, he1 % 3
            i0 = int(F[f0, e0])
            i1 = int(F[f0, (e0 + 1) % 3])
            j0 = int(F[f1, e1])
            j1 = int(F[f1, (e1 + 1) % 3])
            self._stencil_cn = ps.register_curve_network(
                "OGC EE Edges",
                x[[i0, i1, j0, j1], :],
                np.array([[0, 1], [2, 3]], dtype=np.int32),
            )

        else:
            raise NotImplementedError(f"Contact kind {self._kind_idx} not supported")

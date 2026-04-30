from typing import Any, Tuple

import warp as wp
import cuda.compute

from .set import ContactSet, ContactSetData

from .multimesh import MultiMesh, MultiMeshData
from . import halfedges
from . import queries
from .. import common

GEOMETRY_DYNAMIC = wp.constant(0)
GEOMETRY_STATIC = wp.constant(1)
GEOMETRY_COUNT = wp.constant(2)

GeometryPrefix = wp.types.vector(length=GEOMETRY_COUNT + 1, dtype=wp.uint32)
BvhIdArray = wp.types.vector(length=GEOMETRY_COUNT, dtype=wp.uint64)


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

    vv_u: wp.array[wp.int32]  # (# vertex-vertex contacts capacity,) u from pairs (u,v)
    vv_v: wp.array[wp.int32]  # (# vertex-vertex contacts capacity,) v from pairs (u,v)

    ve_u: wp.array[
        wp.int32
    ]  # (# vertex-(half-)edge contacts capacity,) u from pairs (u,v)
    ve_v: wp.array[
        wp.int32
    ]  # (# vertex-(half-)edge contacts capacity,) v from pairs (u,v)

    vf_u: wp.array[
        wp.int32
    ]  # (# vertex-triangle contacts capacity,) u from pairs (u,v)
    vf_v: wp.array[
        wp.int32
    ]  # (# vertex-triangle contacts capacity,) v from pairs (u,v)

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
    ogc.dminf[f] = ogc.rq
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
        f = tid
        _compute_edge_bounding_volume(x, meshes, ogc, f)  # type: ignore
    if tid < n_triangles:
        e = tid
        _compute_triangle_bounding_volume(x, meshes, ogc, e)  # type: ignore


VF_E_FACE_TRIANGLE = wp.constant(0)
VF_E_FACE_EDGE = wp.constant(1)
VF_E_FACE_VERTEX = wp.constant(2)


@wp.func
def _closest_face_point_triangle(uvw: wp.vec3f) -> Tuple[wp.int32, wp.int32]:
    fzero = wp.float32(0)
    fone = wp.float32(1)
    one = wp.int32(1)
    two = wp.int32(2)
    u, v, w = uvw[0], uvw[1], uvw[2]  # pyright: ignore[reportIndexIssue]
    n_zeros = (
        int(u == fzero)  # pyright: ignore[reportIndexIssue]
        + int(v == fzero)  # pyright: ignore[reportIndexIssue]
        + int(w == fzero)  # pyright: ignore[reportIndexIssue]
    )
    is_vertex, is_edge = (n_zeros == two), (n_zeros == one)
    e_face = (wp.int32(is_edge) * 1) + (
        wp.int32(is_vertex) * 2
    )  # pyright: ignore[reportOperatorIssue]
    a = wp.int32(is_vertex) * (
        wp.int32(v == fone) * one
        + (wp.int32(w == fone) * two)  # pyright: ignore[reportOperatorIssue]
    ) + wp.int32(is_edge) * (
        wp.int32(u == fzero) * one + wp.int32(v == fzero) * two
    )  # pyright: ignore[reportOperatorIssue]
    return a, e_face


@wp.func
def _point_triangle_contact_face_index(
    F: wp.array[wp.vec3i],
    f: wp.int32,
    a_local: wp.int32,
    e_face: wp.int32,
) -> wp.int32:
    """Determine the contact face index for a point-triangle contact based on the barycentric coordinates `uvw`."""
    three = wp.int32(3)
    return (
        # face contact, return face index
        wp.int32(e_face == VF_E_FACE_TRIANGLE) * f
        # edge contact, return half-edge index
        + wp.int32(e_face == VF_E_FACE_EDGE) * (three * f + a_local)
        # vertex contact, return global point index
        + wp.int32(e_face == VF_E_FACE_VERTEX)
        * F[f][a_local]  # pyright: ignore[reportIndexIssue]
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
    he_start = GVHEp[i]
    he_end = GVHEp[i + 1]
    xi = x[i]
    for he in range(he_start, he_end):  # pyright: ignore[reportArgumentType]
        xj = x[halfedges.outgoing_vertex(F, he)]  # pyright: ignore[reportArgumentType]
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
    check_adjacent_facets: bool = True,
):
    zero = wp.float32(0)
    fj = GHEF[he][1]  # pyright: ignore[reportIndexIssue]
    # Handle boundary edge case: no adjacent face (i.e. fj == -1)
    fj = (
        wp.int32(fj < zero) * fi + wp.int32(fj >= zero) * fj
    )  # pyright: ignore[reportOperatorIssue]
    i, j = (
        halfedges.incoming_vertex(F, he),
        halfedges.outgoing_vertex(F, he),
    )
    xi, xj = x[i], x[j]
    in_edge_feasible_region = wp.bool(True)
    in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xi, xj - xi) >= zero)  # type: ignore
    in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xj, xi - xj) >= zero)  # type: ignore
    if check_adjacent_facets:
        k = halfedges.next_vertex(F, he, wp.int32(1))
        # Get the third vertex l of triangle fj that is not part of undirected edge (i,j).
        # NOTE: Whenever fj == fi (i.e. boundary edge), l == k.
        l = (
            wp.int32(F[fj][0] != i and F[fj][0] != j) * F[fj][0]  # type: ignore
            + wp.int32(F[fj][1] != i and F[fj][1] != j) * F[fj][1]  # type: ignore
            + wp.int32(F[fj][2] != i and F[fj][2] != j) * F[fj][2]  # type: ignore
        )
        xk, xl = x[k], x[l]
        xij = xj - xi
        xijn2 = wp.dot(xij, xij)  # type: ignore
        # Tangent to the plane spanned by triangle fi, perpendicular to edge (i,j)
        pin = (xi - xk) + (wp.dot(xk - xi, xij) / xijn2) * xij  # type: ignore
        # Tangent to the plane spanned by triangle fj, perpendicular to edge (i,j)
        # NOTE: whenever fj == fi (i.e. boundary edge), pjn == pin
        pjn = (xi - xl) + (wp.dot(xl - xi, xij) / xijn2) * xij  # type: ignore
        in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xi, pin) >= zero)  # type: ignore
        in_edge_feasible_region = in_edge_feasible_region and (wp.dot(y - xi, pjn) >= zero)  # type: ignore
    return in_edge_feasible_region


MAX_VV_PER_THREAD = wp.constant(4)
MAX_VE_PER_THREAD = wp.constant(4)
MAX_VF_PER_THREAD = wp.constant(2)
MAX_EE_PER_THREAD = wp.constant(2)
_FUSED_CONTACT_DETECTION_BLOCK_SIZE = wp.constant(64)


@wp.func
def _vertex_facet_kernel(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    i: wp.int32,
    v: wp.int32,
    xi: wp.vec3f,
    f: wp.int32,
    vv: wp.tile[wp.int32, _VV_BLOCK_CAPACITY],  # type: ignore
    ve: wp.tile[wp.int32, _VE_BLOCK_CAPACITY],  # type: ignore
    vf: wp.tile[wp.int32, _VF_BLOCK_CAPACITY],  # type: ignore
    vv_offset: wp.int32,
    ve_offset: wp.int32,
    vf_offset: wp.int32,
    vv_count: wp.int32,
    ve_count: wp.int32,
    vf_count: wp.int32,
) -> Tuple[wp.int32, wp.int32, wp.int32]:
    finds = meshes.F[f]
    are_adjacent = (i == finds[0]) or (i == finds[1]) or (i == finds[2])
    if are_adjacent:
        return vv_count, ve_count, vf_count
    xj = x[finds[0]]
    xk = x[finds[1]]
    xl = x[finds[2]]
    # 1. Compute closest point projection and distance
    uvw = queries.closest_point_triangle(
        xi, xj, xk, xl  # pyright: ignore[reportArgumentType]
    )
    xc = (
        uvw[0] * xj  # pyright: ignore[reportIndexIssue]
        + uvw[1] * xk  # pyright: ignore[reportIndexIssue]
        + uvw[2] * xl  # pyright: ignore[reportIndexIssue]
    )
    d = wp.norm_l2(xi - xc)
    ogc.dminv[v] = wp.min(ogc.dminv[v], d)
    wp.atomic_min(ogc.dminf, f, d)
    # 2. If distance < ogc.r, update contact set
    if d < ogc.r:
        # a. Handle vv,ve,vf cases for vertex-triangle contact detection
        a_local, e_face = _closest_face_point_triangle(uvw)
        a = _point_triangle_contact_face_index(meshes.F, f, a_local, e_face)
        if e_face == VF_E_FACE_VERTEX:
            if is_vertex_feasible(x, meshes.F, meshes.GVHEp, meshes.GVHEadj, a, xi):  # type: ignore
                # Add VV contact
                vv[vv_offset + vv_count] = a  # type: ignore
                vv_count += wp.int32(1)
        elif e_face == VF_E_FACE_EDGE:
            if is_edge_feasible(
                x, meshes.F, meshes.GHEF, f, a, xi, check_adjacent_facets=True
            ):  # type: ignore
                # Add VE contact
                ve[ve_offset + ve_count] = a  # type: ignore
                ve_count += wp.int32(1)
        elif e_face == VF_E_FACE_TRIANGLE:
            # Add VF contact
            vf[vf_offset + vf_count] = a  # type: ignore
            vf_count += wp.int32(1)
        else:
            assert False
        # b. Handle ve cases for edge-edge contact detection
        for helocal in range(3):
            # Skip if this ve pair was already added from vertex-triangle contact detection
            if helocal == a_local and e_face == VF_E_FACE_EDGE:
                continue
            he = halfedges.half_edge_of_face(
                f, helocal  # pyright: ignore[reportArgumentType]
            )
            j, k = halfedges.incoming_vertex(meshes.F, he), halfedges.outgoing_vertex(
                meshes.F, he
            )
            xj = x[j]
            xk = x[k]
            uv = queries.closest_point_on_line_segment(xi, xj, xk)  # type: ignore
            xce = (wp.float32(1) - uv[1]) * xj + uv[1] * xk  # type: ignore
            de = wp.norm_l2(xi - xce)
            is_vertex = uv[0] == wp.float32(0) or uv[1] == wp.float32(0)  # type: ignore
            # If this is a vertex-edge pair that is within the contact radius
            if de < ogc.r and not is_vertex:
                if is_vertex_feasible(
                    x, meshes.F, meshes.GVHEp, meshes.GVHEadj, i, xce
                ):
                    # Create VE pair
                    ve[ve_offset + ve_count] = he  # type: ignore
                    ve_count += wp.int32(1)
    return vv_count, ve_count, vf_count


@wp.func
def _edge_edge_kernel(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    e1: wp.int32,
    e2: wp.int32,
    einds1: wp.vec2i,
    xi1: wp.vec3f,
    xj1: wp.vec3f,
    hei: wp.int32,
    hej: wp.int32,
    ee: wp.tile[wp.int32, _EE_BLOCK_CAPACITY],  # type: ignore
    ee_offset: wp.int32,
    ee_count: wp.int32,
) -> wp.int32:
    einds2 = meshes.E[e2]
    xi2, xj2 = x[einds2[0]], x[einds2[1]]
    are_adjacent = (
        (einds1[0] == einds2[0])  # type: ignore
        or (einds1[0] == einds2[1])  # type: ignore
        or (einds1[1] == einds2[0])  # type: ignore
        or (einds1[1] == einds2[1])  # type: ignore
    )
    if are_adjacent:
        return ee_count
    zero = wp.float32(0)
    one = wp.float32(1)
    # 1. Compute closest point projection and distance
    st = queries.closest_points_line_segments(
        xi1, xj1, xi2, xj2  # pyright: ignore[reportArgumentType]
    )
    xc1 = (one - st[0]) * xi1 + st[0] * xj1  # pyright: ignore[reportIndexIssue]
    xc2 = (one - st[1]) * xi2 + st[1] * xj2  # pyright: ignore[reportIndexIssue]
    d = wp.norm_l2(xc1 - xc2)
    ogc.dmine[hei] = wp.min(ogc.dmine[hei], d)
    if hej >= 0:
        ogc.dmine[hej] = wp.min(ogc.dmine[hej], d)
    # NOTE:
    # We only exit after updating the displacement bounds, because we launch a thread per edge, so that
    # this pair (e1,e2) will be encountered as (e2,e1), so we allow both e1 and e2 to update their displacement bounds
    # without requiring global synchronization (i.e. atomic min).
    if e1 >= e2:
        return ee_count  # Avoid duplicate edge-edge tests
    # 2. If distance < ogc.r, update contact set
    if d < ogc.r:
        is_xc1_vertex = (
            st[0] == zero or st[0] == one  # pyright: ignore[reportIndexIssue]
        )
        is_xc2_vertex = (
            st[1] == zero or st[1] == one  # pyright: ignore[reportIndexIssue]
        )
        if not (is_xc1_vertex or is_xc2_vertex):
            # Create EE pair
            ee[ee_offset + ee_count] = (e1, e2)  # type: ignore
            ee_count += wp.int32(1)
    return ee_count


tvvlist = wp.types.vector(length=MAX_VV_PER_THREAD, dtype=wp.int32)
tvelist = wp.types.vector(length=MAX_VE_PER_THREAD, dtype=wp.int32)
tvflist = wp.types.vector(length=MAX_VF_PER_THREAD, dtype=wp.int32)
teelist = wp.types.vector(length=MAX_EE_PER_THREAD, dtype=wp.int32)


@wp.kernel(launch_bounds=_FUSED_CONTACT_DETECTION_BLOCK_SIZE)
def _fused_contact_detection(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    nvv: wp.array[
        wp.int32
    ],  # (# verts + 1,) number of vertex-vertex contacts per vertex
    nve: wp.array[wp.int32],  # (# verts + 1,) number of vertex-edge contacts per vertex
    nvf: wp.array[wp.int32],  # (# verts + 1,) number of vertex-face contacts per vertex
    nee: wp.array[
        wp.int32
    ],  # (# half-edges + 1,) number of edge-edge contacts per half-edge
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
    is_last_column = local_tid == block_dims - wp.int32(1)

    # VV,VE,VF contact detection
    if block_id < n_verts:
        v = block_id
        i = meshes.V[v]
        xi = x[i]
        # 1. Query all nearby faces
        vf = tvflist(n_tris)
        query = wp.tile_bvh_query_aabb(
            ogc.f_bvh_id, xi, xi  # pyright: ignore[reportArgumentType]
        )
        for brow in range(MAX_VF_PER_THREAD):
            candidates = wp.tile_bvh_query_next(query)
            f = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
            if f < 0:
                break
            vf[brow] = f
        # 2. Classify and store vv,ve,vf contacts
        vv, ve, vf = _classify_vertex_facet_contacts(...)
        # 3. Keep unique contacts (duplicates exist for vv,ve) via
        # adjacent difference and (exclusive) prefix sum
        # 3.a Sort
        bvv, bve, bvf = wp.tile(vv), wp.tile(ve), wp.tile(vf)  # type: ignore
        wp.tile_sort(bvv, bvv), wp.tile_sort(bve, bve), wp.tile_sort(bvf, bvf)  # type: ignore
        # 3.b Adjacent diff/sum for vv,ve duplicates
        vv_adj_diff, ve_adj_diff = tvvlist(), tvelist()
        for brow in range(MAX_VV_PER_THREAD):
            next_row = (brow + wp.int32(is_last_column)) % MAX_VV_PER_THREAD
            are_different = bvv[brow, bcol] != bvv[next_row, next_col]
            vv_adj_diff[brow] = wp.int32(are_different)
        for brow in range(MAX_VE_PER_THREAD):
            next_row = (brow + wp.int32(is_last_column)) % MAX_VE_PER_THREAD
            are_different = bve[brow, bcol] != bve[next_row, next_col]
            ve_adj_diff[brow] = wp.int32(are_different)
        bvv_adj_diff, bve_adj_diff = wp.tile(vv_adj_diff), wp.tile(ve_adj_diff)  # type: ignore
        bvv_prefix, bve_prefix = wp.tile_scan_exclusive(bvv_adj_diff), wp.tile_scan_exclusive(bve_adj_diff)  # type: ignore
        # 3.c Count vf contacts
        tnvf = wp.int32(0)
        for brow in range(MAX_VF_PER_THREAD):
            if bvf[brow, bcol] < n_tris:
                tnvf = brow * block_dims + bcol + wp.int32(1)
        bnvf = wp.tile_max(wp.tile(tnvf))  # type: ignore
        # 4. Determine global write offset via atomic add
        tvv_offset, tve_offset, tvf_offset = wp.int32(0), wp.int32(0), wp.int32(0)
        if is_last_column:
            tvv_offset = wp.atomic_add(nvv, n_verts, bvv_adj_diff[MAX_VV_PER_THREAD - 1, bcol])  # type: ignore
            tve_offset = wp.atomic_add(nve, n_verts, bve_adj_diff[MAX_VE_PER_THREAD - 1, bcol])  # type: ignore
            tvf_offset = wp.atomic_add(nvf, n_verts, bnvf[0])  # type: ignore
        bvv_offset = wp.tile_from_thread(
            shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            value=tvv_offset,
            thread_idx=local_tid,
        )  # type: ignore
        bve_offset = wp.tile_from_thread(
            shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            value=tve_offset,
            thread_idx=local_tid,
        )  # type: ignore
        tvf_offset = wp.tile_from_thread(
            shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            value=tvf_offset,
            thread_idx=local_tid,
        )  # type: ignore
        # 5. Write unique contacts to global contact set
        for brow in range(MAX_VV_PER_THREAD):
            is_marked_unique = bvv_adj_diff[brow, bcol] == int(1)
            is_last_row = brow == MAX_VV_PER_THREAD - wp.int32(1)
            is_last_element = is_last_row and is_last_column
            if is_marked_unique and not is_last_element:
                k = bvv_offset[bcol] + bvv_prefix[brow, bcol]  # type: ignore
                ogc.vv_u[k] = v
                ogc.vv_v[k] = bvv[brow, bcol]
        for brow in range(MAX_VE_PER_THREAD):
            is_marked_unique = bve_adj_diff[brow, bcol] == int(1)
            is_last_row = brow == MAX_VE_PER_THREAD - wp.int32(1)
            is_last_element = is_last_row and is_last_column
            if is_marked_unique and not is_last_element:
                k = bve_offset[bcol] + bve_prefix[brow, bcol]  # type: ignore
                ogc.ve_u[k] = v
                ogc.ve_v[k] = bve[brow, bcol]
        for brow in range(MAX_VF_PER_THREAD):
            if bvf[brow, bcol] < n_tris:
                k = tvf_offset[bcol] + brow * block_dims + bcol
                ogc.vf_u[k] = v
                ogc.vf_v[k] = bvf[brow, bcol]

    # EE contact detection
    if block_id < n_edges:
        e = block_id
        hei, hej = meshes.EHE[e][0], meshes.EHE[e][1]
        e1 = e
        einds1 = meshes.E[e1]
        xi1, xj1 = x[einds1[0]], x[einds1[1]]
        # 1. Query all nearby edges
        ee = teelist(n_half_edges)
        query = wp.tile_bvh_query_aabb(
            ogc.e_bvh_id,
            ogc.e_lowers[e],
            ogc.e_uppers[e],
        )
        for brow in range(MAX_EE_PER_THREAD):
            candidates = wp.tile_bvh_query_next(query)
            e2 = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
            if e2 < 0:
                break
            ee[brow] = e2
        # 2. Classify and store ee contacts
        ee = _classify_edge_edge_contacts(...)  # type: ignore
        # 3. Sort contacts (already unique)
        bee = wp.tile(ee)  # type: ignore
        wp.tile_sort(bee, bee)
        # 3.a Count ee contacts
        tnee = wp.int32(0)
        for brow in range(MAX_EE_PER_THREAD):
            if bee[brow, bcol] < n_half_edges:
                tnee = brow * block_dims + bcol + wp.int32(1)
        bnee = wp.tile_max(wp.tile(tnee))  # type: ignore
        # 4. Determine global write offset via atomic add
        tee_offset = wp.int32(0)
        if is_last_column:
            tee_offset = wp.atomic_add(nee, n_half_edges, bnee[0])  # type: ignore
        bee_offset = wp.tile_from_thread(
            shape=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            value=tee_offset,
            thread_idx=local_tid,
        )  # type: ignore
        # 5. Write contacts to global contact set
        for brow in range(MAX_EE_PER_THREAD):
            if bee[brow, bcol] < n_half_edges:
                k = bee_offset[bcol] + brow * block_dims + bcol  # type: ignore
                ogc.ee_u[k] = wp.max(
                    hei, hej
                )  # Store the larger half-edge index to handle boundary edges
                ogc.ee_v[k] = bee[brow, bcol]


@wp.kernel
def _update_displacement_bounds(
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    v = wp.tid()
    for k in range(
        meshes.GVHEp[v], meshes.GVHEp[v + 1]  # pyright: ignore[reportOperatorIssue]
    ):
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
        n_vv_contact_capacity: int = 1,
        n_ve_contact_capacity: int = 1,
        n_vf_contact_capacity: int = 1,
        n_ee_contact_capacity: int = 1,
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
        n_verts, n_edges, n_tris, n_half_edges = self.n_primitives
        self._ogc.e_lowers = wp.zeros((n_edges,), dtype=wp.vec3f)
        self._ogc.e_uppers = wp.zeros((n_edges,), dtype=wp.vec3f)
        self._ogc.f_lowers = wp.zeros((n_tris,), dtype=wp.vec3f)
        self._ogc.f_uppers = wp.zeros((n_tris,), dtype=wp.vec3f)
        self._ogc.rq = r
        self._ogc.r = r
        self._ogc.gammap = gammap
        self._ogc.dminv = wp.zeros((n_verts,), dtype=wp.float32)
        self._ogc.dmine = wp.zeros((n_half_edges,), dtype=wp.float32)
        self._ogc.dminf = wp.zeros((n_tris,), dtype=wp.float32)
        self._ogc.vv_u = wp.zeros((n_vv_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.vv_v = wp.zeros((n_vv_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.ve_u = wp.zeros((n_ve_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.ve_v = wp.zeros((n_ve_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.vf_u = wp.zeros((n_vf_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.vf_v = wp.zeros((n_vf_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.ee_u = wp.zeros((n_ee_contact_capacity * n_edges,), dtype=wp.int32)
        self._ogc.ee_v = wp.zeros((n_ee_contact_capacity * n_edges,), dtype=wp.int32)
        dim = max(n_verts, n_edges, n_tris)
        wp.launch(
            kernel=_compute_bounding_volumes_kernel,
            dim=dim,
            inputs=[self._points, self._meshes.data, self._ogc],
        )
        # NOTE: We could use the groups to distinguish between bodies, which MultiMesh stores as prefix sums in VP, EP, FP.
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

    @property
    def n_primitives(self):
        """
        Returns:
            Tuple[int, int, int, int]: Number of vertices, edges, faces, and half-edges
        """
        return (
            self._meshes.data.V.shape[0],
            self._meshes.data.E.shape[0],
            self._meshes.data.F.shape[0],
            3 * self._meshes.data.F.shape[0],
        )

    def prepare_for_execution(self, request_rebuild: bool = True):
        n_verts, n_edges, n_tris, _ = self.n_primitives
        wp.launch(
            _compute_bounding_volumes_kernel,
            dim=max(
                n_verts,
                n_edges,
                n_tris,
            ),
            inputs=[self._points, self._meshes.data, self._ogc],
        )
        for bvh in (self._e_bvh, self._f_bvh):
            if request_rebuild:
                bvh.rebuild()
            else:
                bvh.refit()

    def detect_contacts(self, contacts: ContactSet):  # type: ignore
        n_verts, n_edges, n_tris, n_half_edges = self.n_primitives
        contacts.clear()
        dim = max(n_verts, n_edges)
        wp.launch(
            _fused_contact_detection,
            dim=dim * _FUSED_CONTACT_DETECTION_BLOCK_SIZE,
            inputs=[
                self._points,
                self._meshes.data,
                self._ogc,
                contacts.data.nxx,
                contacts.data.nxe,
                contacts.data.nxf,
                contacts.data.nee,
            ],
            block_dim=_FUSED_CONTACT_DETECTION_BLOCK_SIZE,
        )
        print(self._ogc.xxu)
        print(self._ogc.xxv)
        print(self._ogc.xeu)
        print(self._ogc.xev)
        print(self._ogc.xfu)
        print(self._ogc.xfv)
        print(self._ogc.eeu)
        print(self._ogc.eev)
        # TODO: Use cuda.compute to sort the contact pairs only w.r.t. the first 32 bits.

    def update_displacement_bounds(self):
        wp.launch(
            _update_displacement_bounds,
            dim=self._meshes.data.V.shape[0],
            inputs=[self._meshes.data, self._ogc],
        )

    @property
    def data(self) -> OgcData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._ogc

    @property
    def capacity(self) -> Tuple[int, int, int, int]:
        return (
            self._ogc.vv_u.shape[0],
            self._ogc.ve_u.shape[0],
            self._ogc.vf_u.shape[0],
            self._ogc.ee_u.shape[0],
        )

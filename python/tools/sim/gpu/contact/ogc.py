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

    xxu: wp.array[wp.int32]  # (# point-point contacts capacity,) u from pairs (u,v)
    xxv: wp.array[wp.int32]  # (# point-point contacts capacity,) v from pairs (u,v)
    xeu: wp.array[
        wp.int32
    ]  # (# point-(half-)edge contacts capacity,) u from pairs (u,v)
    xev: wp.array[
        wp.int32
    ]  # (# point-(half-)edge contacts capacity,) v from pairs (u,v)
    xfu: wp.array[wp.int32]  # (# point-triangle contacts capacity,) u from pairs (u,v)
    xfv: wp.array[wp.int32]  # (# point-triangle contacts capacity,) v from pairs (u,v)
    eeu: wp.array[
        wp.int32
    ]  # (# (half-)edge-(half-)edge contacts capacity,) u from pairs (u,v)
    eev: wp.array[
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
_VV_BLOCK_CAPACITY = (
    MAX_VV_PER_THREAD * _FUSED_CONTACT_DETECTION_BLOCK_SIZE + wp.constant(1)
)
_VE_BLOCK_CAPACITY = (
    MAX_VE_PER_THREAD * _FUSED_CONTACT_DETECTION_BLOCK_SIZE + wp.constant(1)
)
_VF_BLOCK_CAPACITY = (
    MAX_VF_PER_THREAD * _FUSED_CONTACT_DETECTION_BLOCK_SIZE + wp.constant(1)
)
_EE_BLOCK_CAPACITY = (
    MAX_EE_PER_THREAD * _FUSED_CONTACT_DETECTION_BLOCK_SIZE + wp.constant(1)
)


@wp.func
def _tile_adjacent_difference(
    values: wp.tile[wp.int32, _FUSED_CONTACT_DETECTION_BLOCK_SIZE + wp.constant(1)],  # type: ignore
    diffs: wp.tile[wp.int32, _FUSED_CONTACT_DETECTION_BLOCK_SIZE + wp.constant(1)],  # type: ignore
    rows: wp.int32,
    local_tid: wp.int32,
):
    for j in range(rows):
        idx = local_tid * rows + j  # pyright: ignore[reportOperatorIssue]
        diff = values[idx + 1] - values[idx]  # pyright: ignore[reportIndexIssue]
        diffs[idx] = wp.int32(diff == wp.int32(0))  # pyright: ignore[reportIndexIssue]


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


@wp.kernel(launch_bounds=_FUSED_CONTACT_DETECTION_BLOCK_SIZE)
def _fused_contact_detection(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    nxx: wp.array[wp.int32],  # (# points + 1,) number of point-point contacts per point
    nxe: wp.array[wp.int32],  # (# points + 1,) number of point-edge contacts per point
    nxf: wp.array[wp.int32],  # (# points + 1,) number of point-face contacts per point
    nee: wp.array[
        wp.int32
    ],  # (# half-edges + 1,) number of edge-edge contacts per half-edge
):
    """Use block-parallelism to compute vv,ve,vf,ee contact pairs
    1. We use 4 tiles to store v,e,f,e indices to (other) contacting elements,
    each with fixed capacity per thread, stored in columns of the tiles. A
    sentinel value of # points, # half-edges, or # faces is used to indicate
    an empty slot. We allocate a single extra slot at the end of each tile that
    will always have the sentinel value.
    2. We use 4 additional tiles of the same size to store the adjacent
    differences (of value 0 or 1 using uint16)

    Args:
        x (wp.array[wp.vec3f]): _description_
    """
    tid = wp.tid()
    block_dims = wp.block_dim()
    assert block_dims == _FUSED_CONTACT_DETECTION_BLOCK_SIZE
    block_id = tid // block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]
    n_points = x.shape[0]
    n_verts = meshes.V.shape[0]
    n_tris = meshes.F.shape[0]
    n_hedges = n_tris * wp.int32(3)
    zero = wp.int32(0)
    one = wp.int32(1)
    vv_capacity = _VV_BLOCK_CAPACITY
    ve_capacity = _VE_BLOCK_CAPACITY
    vf_capacity = _VF_BLOCK_CAPACITY
    ee_capacity = _EE_BLOCK_CAPACITY
    vv = wp.tile_full(
        (_VV_BLOCK_CAPACITY,),
        n_verts,
        dtype=wp.int32,
        storage="shared",
    )
    vvprefix = wp.tile_zeros(
        (_VV_BLOCK_CAPACITY,),
        dtype=wp.int32,
        storage="shared",
    )
    ve = wp.tile_full(
        (_VE_BLOCK_CAPACITY,),
        n_hedges,
        dtype=wp.int32,
        storage="shared",
    )
    veprefix = wp.tile_zeros(
        (_VE_BLOCK_CAPACITY,),
        dtype=wp.int32,
        storage="shared",
    )
    vf = wp.tile_full(
        (_VF_BLOCK_CAPACITY,),
        n_tris,
        dtype=wp.int32,
        storage="shared",
    )
    vfprefix = wp.tile_zeros(
        (_VF_BLOCK_CAPACITY,),
        dtype=wp.int32,
        storage="shared",
    )
    ee = wp.tile_full(
        (_EE_BLOCK_CAPACITY,),
        n_hedges,
        dtype=wp.int32,
        storage="shared",
    )
    eeprefix = wp.tile_zeros(
        (_EE_BLOCK_CAPACITY,),
        dtype=wp.int32,
        storage="shared",
    )

    # VV,VE,VF contact detection
    if block_id < meshes.V.shape[0]:
        v = block_id
        i = meshes.V[v]
        xi = x[i]
        vf_query = wp.tile_bvh_query_aabb(
            ogc.f_bvh_id, xi, xi  # pyright: ignore[reportArgumentType]
        )
        candidates = wp.tile_bvh_query_next(vf_query)
        vv_count, ve_count, vf_count = wp.int32(0), wp.int32(0), wp.int32(0)
        vv_offset, ve_offset, vf_offset = (
            local_tid * MAX_VV_PER_THREAD,
            local_tid * MAX_VE_PER_THREAD,
            local_tid * MAX_VF_PER_THREAD,
        )
        while candidates[local_tid] >= 0:  # pyright: ignore[reportIndexIssue]
            assert (
                vv_count <= MAX_VV_PER_THREAD
                and ve_count <= MAX_VE_PER_THREAD
                and vf_count <= MAX_VF_PER_THREAD
            )
            f = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
            candidates = wp.tile_bvh_query_next(vf_query)
            vv_count, ve_count, vf_count = _vertex_facet_kernel(
                x,
                meshes,
                ogc,
                i,
                v,  # type: ignore
                xi,  # type: ignore
                f,
                vv,
                ve,
                vf,
                vv_offset,
                ve_offset,
                vf_offset,
                vv_count,
                ve_count,
                vf_count,
            )
    # EE contact detection
    if block_id < meshes.E.shape[0]:
        e = block_id
        hei, hej = meshes.EHE[e][0], meshes.EHE[e][1]
        ee_query = wp.tile_bvh_query_aabb(
            ogc.e_bvh_id,
            ogc.e_lowers[e],
            ogc.e_uppers[e],
        )
        e1 = e
        einds1 = meshes.E[e1]
        xi1, xj1 = x[einds1[0]], x[einds1[1]]
        candidates = wp.tile_bvh_query_next(ee_query)
        ee_count, ee_offset = wp.int32(0), local_tid * MAX_EE_PER_THREAD
        while candidates[local_tid] >= 0:  # pyright: ignore[reportIndexIssue]
            assert ee_count <= MAX_EE_PER_THREAD
            e2 = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
            candidates = wp.tile_bvh_query_next(ee_query)
            ee_count = _edge_edge_kernel(
                x,
                meshes,
                ogc,
                e1,  # type: ignore
                e2,
                einds1,
                xi1,  # type: ignore
                xj1,  # type: ignore
                hei,
                hej,
                ee,
                ee_offset,
                ee_count,
            )

    # Sort and compute adjacent differences and then count total unique vv,ve,vf,ee pairs
    # using a prefix sum. After the prefix sum, the last element of the diff tile will
    # contain the total count of unique pairs in this block, which we can use
    # to compute global offsets for writing pairs to global memory.
    if block_id < meshes.V.shape[0]:
        wp.tile_sort(vv, vvprefix)  # type: ignore
        wp.tile_sort(ve, veprefix)  # type: ignore
        wp.tile_sort(vf, vfprefix)  # type: ignore
        _tile_adjacent_difference(vv, vvprefix, MAX_VV_PER_THREAD, local_tid)  # type: ignore
        _tile_adjacent_difference(ve, veprefix, MAX_VE_PER_THREAD, local_tid)  # type: ignore
        _tile_adjacent_difference(vf, vfprefix, MAX_VF_PER_THREAD, local_tid)  # type: ignore
        wp.tile_scan_exclusive(vvprefix)
        wp.tile_scan_exclusive(veprefix)
        wp.tile_scan_exclusive(vfprefix)
    if block_id < meshes.E.shape[0]:
        wp.tile_sort(ee, eeprefix)  # type: ignore
        _tile_adjacent_difference(ee, eeprefix, MAX_EE_PER_THREAD, local_tid)  # type: ignore
        wp.tile_scan_exclusive(eeprefix)

    # The first thread needs to update the global contact pair list counters,
    # and then set the global list begin offsets for this block in the last
    # element of the diff tiles. This step must execute before the following
    # memory write.
    if local_tid == zero:
        vv_count, ve_count, vf_count, ee_count = (
            vvprefix[vv_capacity],  # type: ignore
            veprefix[ve_capacity],  # type: ignore
            vfprefix[vf_capacity],  # type: ignore
            eeprefix[ee_capacity],  # type: ignore
        )
        if block_id < meshes.V.shape[0]:
            if vv_count > zero:  # type: ignore
                vvprefix[vv_capacity] = wp.atomic_add(nxx, n_points, vv_count)  # type: ignore
                nxx[i] = vv_count  # type: ignore
            if ve_count > zero:  # type: ignore
                veprefix[ve_capacity] = wp.atomic_add(nxe, n_points, ve_count)  # type: ignore
                nxe[i] = ve_count  # type: ignore
            if vf_count > zero:  # type: ignore
                vfprefix[vf_capacity] = wp.atomic_add(nxf, n_points, vf_count)  # type: ignore
                nxf[i] = vf_count  # type: ignore
        if block_id < meshes.E.shape[0]:
            if ee_count > zero:  # type: ignore
                eeprefix[ee_capacity] = wp.atomic_add(nee, n_hedges, ee_count)  # type: ignore
                nee[i] = ee_count  # type: ignore
    # Wait for the first thread to finish updating the global counters.
    common.barrier.sync_threads()
    # Write contact pairs to global list with global offsets in last
    # element of diff tiles.
    if block_id < meshes.V.shape[0]:
        vv_offset = vvprefix[vv_capacity]  # type: ignore
        ve_offset = veprefix[ve_capacity]  # type: ignore
        vf_offset = vfprefix[vf_capacity]  # type: ignore
        for row in range(MAX_VV_PER_THREAD):
            idx = local_tid * MAX_VV_PER_THREAD + row
            if vvprefix[idx + 1] > vvprefix[idx]:  # type: ignore
                ogc.xxu[vv_offset + vvprefix[idx]] = block_id  # type: ignore
                ogc.xxv[vv_offset + vvprefix[idx]] = vv[idx]  # type: ignore
        for row in range(MAX_VE_PER_THREAD):
            idx = local_tid * MAX_VE_PER_THREAD + row
            if veprefix[idx + 1] > veprefix[idx]:  # type: ignore
                ogc.xeu[ve_offset + veprefix[idx]] = block_id  # type: ignore
                ogc.xev[ve_offset + veprefix[idx]] = ve[idx]  # type: ignore
        for row in range(MAX_VF_PER_THREAD):
            idx = local_tid * MAX_VF_PER_THREAD + row
            if vfprefix[idx + 1] > vfprefix[idx]:  # type: ignore
                ogc.xfu[vf_offset + vfprefix[idx]] = block_id  # type: ignore
                ogc.xfv[vf_offset + vfprefix[idx]] = vf[idx]  # type: ignore
    if block_id < meshes.E.shape[0]:
        ee_offset = eeprefix[ee_capacity]  # type: ignore
        he = wp.max(hei, hej)  # Use max to ignore boundary half edge (-1)
        for row in range(MAX_EE_PER_THREAD):
            idx = local_tid * MAX_EE_PER_THREAD + row
            if eeprefix[idx + 1] > eeprefix[idx]:  # type: ignore
                ogc.eeu[ee_offset + eeprefix[idx]] = he  # type: ignore
                ogc.eev[ee_offset + eeprefix[idx]] = ee[idx]  # type: ignore


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
        self._ogc.xxu = wp.zeros((n_vv_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.xxv = wp.zeros((n_vv_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.xeu = wp.zeros((n_ve_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.xev = wp.zeros((n_ve_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.xfu = wp.zeros((n_vf_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.xfv = wp.zeros((n_vf_contact_capacity * n_verts,), dtype=wp.int32)
        self._ogc.eeu = wp.zeros((n_ee_contact_capacity * n_edges,), dtype=wp.int32)
        self._ogc.eev = wp.zeros((n_ee_contact_capacity * n_edges,), dtype=wp.int32)
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
            self._ogc.xxu.shape[0],
            self._ogc.xeu.shape[0],
            self._ogc.xfu.shape[0],
            self._ogc.eeu.shape[0],
        )

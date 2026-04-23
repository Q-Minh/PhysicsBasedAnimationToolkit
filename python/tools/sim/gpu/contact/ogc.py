from typing import Any, Tuple

import warp as wp
import cuda.compute

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

    # bvh_id: (
    #     BvhIdArray  # BVH ID for OGC queries # pyright: ignore[reportInvalidTypeForm]
    # )
    e_bvh_id: wp.uint64  # Edge BVH ID
    f_bvh_id: wp.uint64  # Triangle BVH ID
    e_lowers: wp.array[wp.vec3f]  # (# half-edges,) half-edge AABB lower bounds
    e_uppers: wp.array[wp.vec3f]  # (# half-edges,) half-edge AABB upper bounds
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

    # point_geom_prefix: GeometryPrefix  # geometry prefix for points # pyright: ignore[reportInvalidTypeForm]
    # half_edge_geom_prefix: GeometryPrefix  # geometry prefix for edges # pyright: ignore[reportInvalidTypeForm]
    # face_geom_prefix: GeometryPrefix  # geometry prefix for faces # pyright: ignore[reportInvalidTypeForm]

    xx: wp.array[
        wp.uint64
    ]  # (# point-point contacts,) pairs (i, j) stored as `i << 32 | j`
    rxx: wp.array[
        wp.uint64
    ]  # (# point-point contacts,) reverse pairs (j, i) stored as `j << 32 | i`
    xe: wp.array[
        wp.uint64
    ]  # (# point-(half-)edge contacts,) pairs (i, he) stored as `i << 32 | he`
    rxe: wp.array[
        wp.uint64
    ]  # (# point-(half-)edge contacts,) reverse pairs (he, i) stored as `he << 32 | i`
    xf: wp.array[
        wp.uint64
    ]  # (# point-face contacts,) pairs (i, f) stored as `i << 32 | f`
    rxf: wp.array[
        wp.uint64
    ]  # (# point-face contacts,) reverse pairs (f, i) stored as `f << 32 | i`
    ee: wp.array[
        wp.uint64
    ]  # (# (half-)edge-(half-)edge contacts,) pairs (hei, hej) stored as `hei << 32 | hej`
    ree: wp.array[
        wp.uint64
    ]  # (# (half-)edge-(half-)edge contacts,) reverse pairs (hej, hei) stored as `hej << 32 | hei`


@wp.kernel
def _compute_bounding_volumes(
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
        ogc.dminv[v] = ogc.rq**2
    if tid < n_edges:
        e = tid
        ogc.dmine[e] = ogc.rq**2
        inds = meshes.E[e]
        xi, xj = x[inds[0]], x[inds[1]]
        xmid = float(0.5) * (xi + xj)
        hlen = float(0.5) * wp.norm_l2(xj - xi)
        radius = hlen + ogc.rq
        ogc.e_lowers[e] = xmid - wp.vec3f(radius)
        ogc.e_uppers[e] = xmid + wp.vec3f(radius)
    if tid < n_triangles:
        f = tid
        ogc.dminf[f] = ogc.rq**2
        inds = meshes.F[f]
        xi, xj, xk = x[inds[0]], x[inds[1]], x[inds[2]]
        xmin = wp.min(
            xi, wp.min(xj, xk)  # pyright: ignore[reportArgumentType, reportCallIssue]
        )
        xmax = wp.max(
            xi, wp.max(xj, xk)  # pyright: ignore[reportArgumentType, reportCallIssue]
        )
        ogc.f_lowers[f] = xmin - wp.vec3f(ogc.rq)
        ogc.f_uppers[f] = xmax + wp.vec3f(ogc.rq)


VF_E_FACE_TRIANGLE = wp.constant(0)
VF_E_FACE_EDGE = wp.constant(1)
VF_E_FACE_VERTEX = wp.constant(2)


@wp.func
def _closest_face_point_triangle(uvw: wp.vec3f) -> Tuple[wp.int32, wp.int32]:
    zero = wp.float32(0)
    one = wp.float32(1)
    two = wp.float32(2)
    u, v, w = uvw[0], uvw[1], uvw[2]  # pyright: ignore[reportIndexIssue]
    n_zeros = (
        int(u == zero)  # pyright: ignore[reportIndexIssue]
        + int(v == zero)  # pyright: ignore[reportIndexIssue]
        + int(w == zero)  # pyright: ignore[reportIndexIssue]
    )
    is_vertex, is_edge = (n_zeros == two), (n_zeros == one)
    e_face = (wp.int32(is_edge) * 1) + (
        wp.int32(is_vertex) * 2
    )  # pyright: ignore[reportOperatorIssue]
    a = wp.int32(is_vertex) * (
        wp.int32(v == one) * one
        + (wp.int32(w == one) * two)  # pyright: ignore[reportOperatorIssue]
    ) + wp.int32(is_edge) * (
        wp.int32(u == zero) * one + wp.int32(v == zero) * two
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
        (e_face == VF_E_FACE_TRIANGLE) * f
        # edge contact, return half-edge index
        + (e_face == VF_E_FACE_EDGE) * (three * f + a_local)
        # vertex contact, return global point index
        + (e_face == VF_E_FACE_VERTEX)
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
    in_vertex_feasible_region = True
    he_start = GVHEp[i]
    he_end = GVHEp[i + 1]
    xi = x[i]
    for he in range(he_start, he_end):  # pyright: ignore[reportArgumentType]
        xj = x[halfedges.outgoing_vertex(F, he)]  # pyright: ignore[reportArgumentType]
        in_vertex_feasible_region &= wp.dot(  # pyright: ignore[reportCallIssue]
            y - xi, xi - xj  # pyright: ignore[reportArgumentType]
        ) >= wp.float32(0)
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
    in_edge_feasible_region = True
    in_edge_feasible_region &= wp.dot(y - xi, xj - xi) >= zero  # type: ignore
    in_edge_feasible_region &= wp.dot(y - xj, xi - xj) >= zero  # type: ignore
    if check_adjacent_facets:
        k = halfedges.next_vertex(F, he, wp.int16(1))
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
        in_edge_feasible_region &= wp.dot(y - xi, pin) >= zero  # type: ignore
        in_edge_feasible_region &= wp.dot(y - xi, pjn) >= zero  # type: ignore
    return in_edge_feasible_region


MAX_VV_PER_THREAD = wp.constant(4)
MAX_VE_PER_THREAD = wp.constant(4)
MAX_VF_PER_THREAD = wp.constant(2)
MAX_EE_PER_THREAD = wp.constant(2)
_FUSED_CONTACT_DETECTION_BLOCK_SIZE = wp.constant(32)


@wp.func
def _tile_adjacent_difference(
    values: wp.tile[wp.int32, _FUSED_CONTACT_DETECTION_BLOCK_SIZE],  # type: ignore
    diffs: wp.tile[wp.int16, _FUSED_CONTACT_DETECTION_BLOCK_SIZE],  # type: ignore
    rows: wp.int32,
    local_tid: wp.int32,
):
    for j in range(rows):
        idx = local_tid * rows + j  # pyright: ignore[reportOperatorIssue]
        diff = values[idx + 1] - values[idx]  # pyright: ignore[reportIndexIssue]
        diffs[idx] = wp.int16(diff == wp.int32(0))  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _fused_contact_detection(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
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
    block_id = tid // block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]

    vsentinel = meshes.V.shape[0]
    fsentinel = meshes.F.shape[0]
    hesentinel = fsentinel * wp.int32(3)
    zero = wp.int32(0)
    one = wp.int32(1)
    vv_capacity = block_dims * MAX_VV_PER_THREAD
    ve_capacity = block_dims * MAX_VE_PER_THREAD
    vf_capacity = block_dims * MAX_VF_PER_THREAD
    ee_capacity = block_dims * MAX_EE_PER_THREAD
    vv = wp.tile_full(
        (vv_capacity + one,),
        vsentinel,
        dtype=wp.int32,
        storage="shared",
    )
    vvprefix = wp.tile_zeros(
        (vv_capacity + one,),
        dtype=wp.int16,
        storage="shared",
    )
    ve = wp.tile_full(
        (ve_capacity + one,),
        hesentinel,
        dtype=wp.int32,
        storage="shared",
    )
    veprefix = wp.tile_zeros(
        (ve_capacity + one,),
        dtype=wp.int16,
        storage="shared",
    )
    vf = wp.tile_full(
        (vf_capacity + one,),
        fsentinel,
        dtype=wp.int32,
        storage="shared",
    )
    vfprefix = wp.tile_zeros(
        (vf_capacity + one,),
        dtype=wp.int16,
        storage="shared",
    )
    ee = wp.tile_full(
        (ee_capacity + one,),
        hesentinel,
        dtype=wp.int32,
        storage="shared",
    )
    eeprefix = wp.tile_zeros(
        (ee_capacity + one,),
        dtype=wp.int16,
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
        while candidates[local_tid] >= 0:  # pyright: ignore[reportIndexIssue]
            f = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
            candidates = wp.tile_bvh_query_next(vf_query)
            finds = meshes.F[f]
            are_adjacent = (i == finds[0]) or (i == finds[1]) or (i == finds[2])
            if are_adjacent:
                continue
            xj, xk, xl = x[finds[0]], x[finds[1]], x[finds[2]]
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
                is_feasible = False
                if e_face == VF_E_FACE_VERTEX:
                    if is_vertex_feasible(x, meshes.F, meshes.GVHEp, meshes.GVHEadj, a, xi):  # type: ignore
                        is_feasible = True
                        # TODO: Create VV pair
                        pass
                elif e_face == VF_E_FACE_EDGE:
                    if is_edge_feasible(x, meshes.F, meshes.GHEF, f, a, xi, check_adjacent_facets=True):  # type: ignore
                        is_feasible = True
                        # TODO: Create VE pair
                        pass
                elif e_face == VF_E_FACE_TRIANGLE:
                    # TODO: Create VF pair
                    pass
                else:
                    assert False
                # b. Handle ve cases for edge-edge contact detection
                for helocal in range(3):
                    # Skip if this ve pair was already added from vertex-triangle contact detection
                    if helocal == a_local and e_face == VF_E_FACE_EDGE and is_feasible:
                        continue
                    he = halfedges.half_edge_of_face(
                        f, helocal  # pyright: ignore[reportArgumentType]
                    )
                    j, k = halfedges.incoming_vertex(
                        meshes.F, he
                    ), halfedges.outgoing_vertex(meshes.F, he)
                    xj, xk = x[j], x[k]
                    uv = queries.closest_point_on_line_segment(xi, xj, xk)  # type: ignore
                    xce = (wp.float32(1) - uv[1]) * xj + uv[1] * xk  # type: ignore
                    de = wp.norm_l2(xi - xce)
                    is_vertex = uv[0] == wp.float32(0) or uv[1] == wp.float32(0)  # type: ignore
                    # If this is a vertex-edge pair that is within the contact radius
                    if de < ogc.r and not is_vertex:
                        if is_vertex_feasible(
                            x, meshes.F, meshes.GVHEp, meshes.GVHEadj, i, xce
                        ):
                            # TODO: Create VE pair
                            pass
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
        while candidates[local_tid] >= 0:  # pyright: ignore[reportIndexIssue]
            e2 = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
            candidates = wp.tile_bvh_query_next(ee_query)
            einds2 = meshes.E[e2]
            xi2, xj2 = x[einds2[0]], x[einds2[1]]
            are_adjacent = (
                (einds1[0] == einds2[0])
                or (einds1[0] == einds2[1])
                or (einds1[1] == einds2[0])
                or (einds1[1] == einds2[1])
            )
            if are_adjacent:
                continue
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
                continue  # Avoid duplicate edge-edge tests
            # 2. If distance < ogc.r, update contact set
            if d < ogc.r:
                is_xc1_vertex = (
                    st[0] == zero or st[0] == one  # pyright: ignore[reportIndexIssue]
                )
                is_xc2_vertex = (
                    st[1] == zero or st[1] == one  # pyright: ignore[reportIndexIssue]
                )
                if not (is_xc1_vertex or is_xc2_vertex):
                    # TODO: Create EE pair
                    pass

    # Sort and compute adjacent differences and then count total unique vv,ve,vf,ee pairs
    # using a prefix sum. After the prefix sum, the last element of the diff tile will
    # contain the total count of unique pairs in this block, which we can use
    # to compute global offsets for writing pairs to global memory.
    if block_id < meshes.V.shape[0]:
        wp.tile_sort(wp.tile_view(vv, 0, block_dims * MAX_VV_PER_THREAD), vvprefix)  # type: ignore
        wp.tile_sort(wp.tile_view(ve, 0, block_dims * MAX_VE_PER_THREAD), veprefix)  # type: ignore
        wp.tile_sort(wp.tile_view(vf, 0, block_dims * MAX_VF_PER_THREAD), vfprefix)  # type: ignore
        _tile_adjacent_difference(vv, vvprefix, MAX_VV_PER_THREAD, local_tid)  # type: ignore
        _tile_adjacent_difference(ve, veprefix, MAX_VE_PER_THREAD, local_tid)  # type: ignore
        _tile_adjacent_difference(vf, vfprefix, MAX_VF_PER_THREAD, local_tid)  # type: ignore
        wp.tile_scan_exclusive(vvprefix)
        wp.tile_scan_exclusive(veprefix)
        wp.tile_scan_exclusive(vfprefix)
    if block_id < meshes.E.shape[0]:
        wp.tile_sort(wp.tile_view(ee, 0, block_dims * MAX_EE_PER_THREAD), eeprefix)  # type: ignore
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
                vvprefix[vv_capacity] = wp.atomic_add(ogc.nxx, zero, vv_count)  # type: ignore
            if ve_count > zero:  # type: ignore
                veprefix[ve_capacity] = wp.atomic_add(ogc.nxe, zero, ve_count)  # type: ignore
            if vf_count > zero:  # type: ignore
                vfprefix[vf_capacity] = wp.atomic_add(ogc.nxf, zero, vf_count)  # type: ignore
        if block_id < meshes.E.shape[0]:
            if ee_count > zero:  # type: ignore
                eeprefix[ee_capacity] = wp.atomic_add(ogc.nee, zero, ee_count)  # type: ignore
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
                ogc.xx[vv_offset + vvprefix[idx]] = (  # type: ignore
                    (wp.uint64(i) << wp.uint64(32)) | wp.uint64(vv[idx])  # type: ignore
                )
        for row in range(MAX_VE_PER_THREAD):
            idx = local_tid * MAX_VE_PER_THREAD + row
            if veprefix[idx + 1] > veprefix[idx]:  # type: ignore
                ogc.xe[ve_offset + veprefix[idx]] = (  # type: ignore
                    (wp.uint64(i) << wp.uint64(32)) | wp.uint64(ve[idx])  # type: ignore
                )
        for row in range(MAX_VF_PER_THREAD):
            idx = local_tid * MAX_VF_PER_THREAD + row
            if vfprefix[idx + 1] > vfprefix[idx]:  # type: ignore
                ogc.xf[vf_offset + vfprefix[idx]] = (  # type: ignore
                    (wp.uint64(i) << wp.uint64(32)) | wp.uint64(vf[idx])  # type: ignore
                )
    if block_id < meshes.E.shape[0]:
        ee_offset = eeprefix[ee_capacity]  # type: ignore
        he = wp.max(hei, hej)  # Use max to ignore boundary half edge (-1)
        for row in range(MAX_EE_PER_THREAD):
            idx = local_tid * MAX_EE_PER_THREAD + row
            if eeprefix[idx + 1] > eeprefix[idx]:  # type: ignore
                ogc.ee[ee_offset + eeprefix[idx]] = (  # type: ignore
                    (wp.uint64(he) << wp.uint64(32)) | wp.uint64(ee[idx])  # type: ignore
                )


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
    # _bvhs: list[wp.Bvh]  # BVHs for different geometry types
    _e_bvh: wp.Bvh  # BVH over edges
    _f_bvh: wp.Bvh  # BVH over faces
    _points: wp.array[wp.vec3f]  # (N,) vertex positions
    _meshes: MultiMesh  # Meshes

    def __init__(
        self,
        points: wp.array[wp.vec3f],
        meshes: MultiMesh,
        r: float = 0.002,
        gammap: float = 0.45,
        n_max_contact_pairs: Tuple[int, int, int, int] = (10000, 10000, 10000, 10000),
    ):
        self._points = points
        self._meshes = meshes
        self._ogc = OgcData()
        n_pts, n_edges, n_tris, n_half_edges = self.n_primitives
        self._ogc.v_lowers = wp.zeros((n_pts,), dtype=wp.vec3f)
        self._ogc.v_uppers = wp.zeros((n_pts,), dtype=wp.vec3f)
        self._ogc.e_lowers = wp.zeros((n_edges,), dtype=wp.vec3f)
        self._ogc.e_uppers = wp.zeros((n_edges,), dtype=wp.vec3f)
        self._ogc.f_lowers = wp.zeros((n_tris,), dtype=wp.vec3f)
        self._ogc.f_uppers = wp.zeros((n_tris,), dtype=wp.vec3f)
        self._ogc.rq = r
        self._ogc.r = r
        self._ogc.gammap = gammap
        self._ogc.dminv = wp.zeros((n_pts,), dtype=wp.float32)
        self._ogc.dmine = wp.zeros((n_half_edges,), dtype=wp.float32)
        self._ogc.dminf = wp.zeros((n_tris,), dtype=wp.float32)
        self._ogc.xx = wp.zeros((n_max_contact_pairs[0],), dtype=wp.uint64)
        self._ogc.rxx = wp.zeros((n_max_contact_pairs[0],), dtype=wp.uint64)
        self._ogc.xe = wp.zeros((n_max_contact_pairs[1],), dtype=wp.uint64)
        self._ogc.rxe = wp.zeros((n_max_contact_pairs[1],), dtype=wp.uint64)
        self._ogc.xf = wp.zeros((n_max_contact_pairs[2],), dtype=wp.uint64)
        self._ogc.rxf = wp.zeros((n_max_contact_pairs[2],), dtype=wp.uint64)
        self._ogc.ee = wp.zeros((n_max_contact_pairs[3],), dtype=wp.uint64)
        self._ogc.ree = wp.zeros((n_max_contact_pairs[3],), dtype=wp.uint64)
        wp.launch(
            _compute_bounding_volumes,
            dim=max(
                n_pts,
                n_edges,
                n_tris,
            ),
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
        n_pts, n_edges, n_tris, _ = self.n_primitives
        wp.launch(
            _compute_bounding_volumes,
            dim=max(
                n_pts,
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

    def detect_contacts(self):
        wp.launch(
            _fused_contact_detection,
            dim=max(self._meshes.data.V.shape[0], self._meshes.data.E.shape[0]),
            inputs=[self._points, self._meshes.data, self._ogc],
        )
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

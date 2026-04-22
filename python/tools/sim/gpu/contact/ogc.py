from typing import Tuple

import warp as wp

from .multimesh import MultiMesh, MultiMeshData
from .halfedges import face_of_half_edge
from . import queries

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


@wp.kernel
def _vertex_facet_contact_detection(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid / block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]
    v = block_id
    i = meshes.V[v]
    xi = x[i]
    query = wp.tile_bvh_query_aabb(
        ogc.f_bvh_id, xi, xi  # pyright: ignore[reportArgumentType]
    )
    candidates = wp.tile_bvh_query_next(query)
    while candidates[local_tid] >= 0:  # pyright: ignore[reportIndexIssue]
        f = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
        candidates = wp.tile_bvh_query_next(query)
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
            uvw[0] * xj + uvw[1] * xk + uvw[2] * xl  # pyright: ignore[reportIndexIssue]
        )
        d = wp.norm_l2(xi - xc)
        ogc.dminv[v] = wp.min(ogc.dminv[v], d)
        wp.atomic_min(ogc.dminf, f, d)
        # 2. If distance < ogc.r, update contact set
        if d < ogc.r:
            a_local, e_face = _closest_face_point_triangle(uvw)
            a = _point_triangle_contact_face_index(meshes.F, f, a_local, e_face)
            # TODO: Update contact set
            if e_face == VF_E_FACE_TRIANGLE:
                pass
            elif e_face == VF_E_FACE_EDGE:
                pass
            elif e_face == VF_E_FACE_VERTEX:
                pass
            else:
                assert False


@wp.kernel
def _edge_edge_contact_detection(
    x: wp.array[wp.vec3f],  # (N,) points
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
):
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid / block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]
    e1 = block_id
    einds1 = meshes.E[e1]
    xi1, xj1 = x[einds1[0]], x[einds1[1]]
    ehe1 = meshes.EHE[e1]
    query = wp.tile_bvh_query_aabb(
        ogc.e_bvh_id,
        ogc.e_lowers[e1],
        ogc.e_uppers[e1],
    )
    candidates = wp.tile_bvh_query_next(query)
    while candidates[local_tid] >= 0:  # pyright: ignore[reportIndexIssue]
        e2 = candidates[local_tid]  # pyright: ignore[reportIndexIssue]
        candidates = wp.tile_bvh_query_next(query)
        if e1 >= e2:
            continue  # Avoid duplicate edge-edge tests
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
        # TODO:
        zero = wp.float32(0)
        one = wp.float32(1)
        # 1. Compute closest point projection and distance
        st = queries.closest_points_line_segments(
            xi1, xj1, xi2, xj2  # pyright: ignore[reportArgumentType]
        )
        xc1 = (one - st[0]) * xi1 + st[0] * xj1  # pyright: ignore[reportIndexIssue]
        xc2 = (one - st[1]) * xi2 + st[1] * xj2  # pyright: ignore[reportIndexIssue]
        d = wp.norm_l2(xc1 - xc2)
        ehe2 = meshes.EHE[e1]
        wp.atomic_min(ogc.dmine, ehe1[0], d)
        if ehe1[1] >= 0:
            wp.atomic_min(ogc.dmine, ehe1[1], d)
        wp.atomic_min(ogc.dmine, ehe2[0], d)
        if ehe2[1] >= 0:
            wp.atomic_min(ogc.dmine, ehe2[1], d)
        # 2. If distance < ogc.r, update contact set
        if d < ogc.r:
            is_xc1_vertex = (
                st[0] == zero or st[0] == one  # pyright: ignore[reportIndexIssue]
            )
            is_xc2_vertex = (
                st[1] == zero or st[1] == one  # pyright: ignore[reportIndexIssue]
            )
            # TODO: Update contact set
            if is_xc1_vertex and not is_xc2_vertex:
                pass
            elif not is_xc1_vertex and is_xc2_vertex:
                pass
            else:
                pass


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
        f = face_of_half_edge(he)
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

    def update_displacement_bounds(self):
        wp.launch(
            _update_displacement_bounds,
            dim=self._meshes.data.V.shape[0],
            inputs=[self._meshes.data, self._ogc],
        )

    @property
    def data(self) -> OgcData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._ogc

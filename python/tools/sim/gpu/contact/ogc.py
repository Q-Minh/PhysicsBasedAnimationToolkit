from typing import Tuple

import warp as wp

from .multimesh import MultiMesh, MultiMeshData

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
    bvh_id: wp.uint64  # BVH ID for OGC queries
    x_lowers: wp.array[wp.vec3f]  # (N,) point AABB lower bounds
    x_uppers: wp.array[wp.vec3f]  # (N,) point AABB upper bounds
    e_lowers: wp.array[wp.vec3f]  # (N,) edge AABB lower bounds
    e_uppers: wp.array[wp.vec3f]  # (N,) edge AABB upper bounds
    f_lowers: wp.array[wp.vec3f]  # (N,) triangle AABB lower bounds
    f_uppers: wp.array[wp.vec3f]  # (N,) triangle AABB upper bounds

    rq: wp.float32  # OGC query radius
    r: wp.float32  # OGC contact radius
    gammap: (
        wp.float32
    )  # Relaxation parameter for vertex displacement bound, must satisfy `0 < gammap < 0.5`

    dminv: wp.array[wp.float32]  # (N,) vertex minimum displacement bounds
    dmine: wp.array[wp.float32]  # (N,) edge minimum displacement bounds
    dminf: wp.array[wp.float32]  # (N,) face minimum displacement bounds

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
        xi = x[meshes.V[v]]
        ogc.x_lowers[v] = xi - wp.vec3f(ogc.rq)
        ogc.x_uppers[v] = xi + wp.vec3f(ogc.rq)
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


class Ogc:
    """Offset Geometric Contact"""

    _ogc: OgcData  # pyright: ignore[reportGeneralTypeIssues]
    # _bvhs: list[wp.Bvh]  # BVHs for different geometry types
    _v_bvh: wp.Bvh  # BVH over vertices
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
        n_pts, n_edges, n_tris = (
            meshes.data.V.shape[0],
            meshes.data.E.shape[0],
            meshes.data.F.shape[0],
        )
        self._ogc.x_lowers = wp.zeros((n_pts,), dtype=wp.vec3f)
        self._ogc.x_uppers = wp.zeros((n_pts,), dtype=wp.vec3f)
        self._ogc.e_lowers = wp.zeros((n_edges,), dtype=wp.vec3f)
        self._ogc.e_uppers = wp.zeros((n_edges,), dtype=wp.vec3f)
        self._ogc.f_lowers = wp.zeros((n_tris,), dtype=wp.vec3f)
        self._ogc.f_uppers = wp.zeros((n_tris,), dtype=wp.vec3f)
        self._ogc.rq = r
        self._ogc.r = r
        self._ogc.gammap = gammap
        self._ogc.dminv = wp.zeros((n_pts,), dtype=wp.float32)
        self._ogc.dmine = wp.zeros((n_edges,), dtype=wp.float32)
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
        self._v_bvh, self._e_bvh, self._f_bvh = (
            wp.Bvh(
                self._ogc.x_lowers,
                self._ogc.x_uppers,
                constructor="lbvh",
                groups=None,
                leaf_size=4,
            ),
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

    @property
    def n_primitives(self):
        return (
            self._meshes.data.V.shape[0],
            self._meshes.data.E.shape[0],
            self._meshes.data.F.shape[0],
        )

    def prepare_for_execution(self, request_rebuild: bool = True):
        n_pts, n_edges, n_tris = self.n_primitives
        wp.launch(
            _compute_bounding_volumes,
            dim=max(
                n_pts,
                n_edges,
                n_tris,
            ),
            inputs=[self._points, self._meshes.data, self._ogc],
        )
        for bvh in (self._v_bvh, self._e_bvh, self._f_bvh):
            if request_rebuild:
                bvh.rebuild()
            else:
                bvh.refit()

    @property
    def data(self) -> OgcData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._ogc

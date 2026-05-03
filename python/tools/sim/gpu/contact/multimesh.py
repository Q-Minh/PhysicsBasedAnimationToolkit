import warp as wp
from pbatoolkit import pbat


@wp.struct
class MultiMeshData:
    """Data structure for multi-mesh contact handling."""

    V: wp.array[wp.int32]  # (N,) surface vertex indices into points
    F: wp.array[wp.vec3i]  # (M,) surface (triangle) indices into points
    E: wp.array[wp.vec2i]  # (K,) surface (edge) indices into points
    VP: wp.array[wp.int32]  # `|# connected components + 1| x 1` vertex prefix
    FP: wp.array[wp.int32]  # `|# connected components + 1| x 1` face prefix
    EP: wp.array[wp.int32]  # `|# connected components + 1| x 1` edge prefix
    GVHEp: wp.array[wp.int32]  # `|# points + 1| x 1` point to half-edge prefix
    GVHEadj: wp.array[wp.int32]  # `|# half edges| x 1` point to half-edge adjacency
    GHEF: wp.array[wp.vec2i]  # `2 x |# half edges|` half-edge to face adjacency
    EHE: wp.array[
        wp.vec2i
    ]  # `2 x |# edges|` edge to half-edge adjacency, -1 indicates no half-edge
    GXV: wp.array[
        wp.int32
    ]  # `|# points| x 1` point to vertex mapping, with `-1` for non-vertices


class MultiMesh:
    """Wrapper for multi-mesh contact handling."""

    _data: MultiMeshData  # pyright: ignore[reportGeneralTypeIssues]


    def __init__(self, mesh: pbat.sim.contact.MultiMesh):
        self._data = MultiMeshData()
        self._data.V = wp.array(mesh.V, dtype=wp.int32)
        self._data.F = wp.array(mesh.F.T, dtype=wp.vec3i)
        self._data.E = wp.array(mesh.E.T, dtype=wp.vec2i)
        self._data.VP = wp.array(mesh.VP, dtype=wp.int32)
        self._data.FP = wp.array(mesh.FP, dtype=wp.int32)
        self._data.EP = wp.array(mesh.EP, dtype=wp.int32)
        self._data.GVHEp = wp.array(mesh.GVHEp, dtype=wp.int32)
        self._data.GVHEadj = wp.array(mesh.GVHEadj, dtype=wp.int32)
        self._data.GHEF = wp.array(mesh.GHEF.T, dtype=wp.vec2i)
        self._data.EHE = wp.array(mesh.EHE.T, dtype=wp.vec2i)
        self._data.GXV = wp.array(mesh.GXV, dtype=wp.int32)


    @property
    def data(self) -> MultiMeshData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data

    @property
    def n_verts(self) -> int:
        return self._data.V.shape[0]

    @property
    def n_edges(self) -> int:
        return self._data.E.shape[0]

    @property
    def n_triangles(self) -> int:
        return self._data.F.shape[0]

    @property
    def n_half_edges(self) -> int:
        return 3 * self.n_triangles


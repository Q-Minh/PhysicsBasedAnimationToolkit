import warp as wp
from pbatoolkit import pbat


@wp.struct
class MultiMeshData:
    """Data structure for multi-mesh contact handling."""

    V: wp.array[wp.uint32]  # (N,) surface vertex indices into points
    F: wp.array[wp.vec3ui]  # (M,) surface (triangle) indices into points
    E: wp.array[wp.vec2ui]  # (K,) surface (edge) indices into points
    VP: wp.array[wp.uint32]  # `|# connected components + 1| x 1` vertex prefix
    FP: wp.array[wp.uint32]  # `|# connected components + 1| x 1` face prefix
    EP: wp.array[wp.uint32]  # `|# connected components + 1| x 1` edge prefix
    GVHEp: wp.array[wp.uint32]  # `|# points + 1| x 1` point to half-edge prefix
    GVHEadj: wp.array[wp.uint32]  # `|# half edges| x 1` point to half-edge adjacency
    GHEF: wp.array[wp.vec2ui]  # `2 x |# half edges|` half-edge to face adjacency
    EHE: wp.array[wp.vec2ui]  # `2 x |# edges|` edge to half-edge adjacency
    GXV: wp.array[
        wp.uint32
    ]  # `|# points| x 1` point to vertex mapping, with `-1` for non-vertices


class MultiMesh:
    """Wrapper for multi-mesh contact handling."""

    _data: MultiMeshData  # pyright: ignore[reportGeneralTypeIssues]

    def __init__(self, mesh: pbat.sim.contact.MultiMesh):
        self._data.V = wp.array(mesh.V, dtype=wp.uint32)
        self._data.F = wp.array(mesh.F.T, dtype=wp.vec3ui)
        self._data.E = wp.array(mesh.E.T, dtype=wp.vec2ui)
        self._data.VP = wp.array(mesh.VP, dtype=wp.uint32)
        self._data.FP = wp.array(mesh.FP, dtype=wp.uint32)
        self._data.EP = wp.array(mesh.EP, dtype=wp.uint32)
        self._data.GVHEp = wp.array(mesh.GVHEp, dtype=wp.uint32)
        self._data.GVHEadj = wp.array(mesh.GVHEadj, dtype=wp.uint32)
        self._data.GHEF = wp.array(mesh.GHEF.T, dtype=wp.vec2ui)
        self._data.EHE = wp.array(mesh.EHE.T, dtype=wp.vec2ui)
        self._data.GXV = wp.array(mesh.GXV, dtype=wp.uint32)

    @property
    def data(self) -> MultiMeshData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data

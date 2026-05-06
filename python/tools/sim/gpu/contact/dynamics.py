import warp as wp
from .ogc import Ogc, OgcData
from .constraints import ConstraintSet, ConstraintSetData


@wp.struct
class MeshDynamicsData:

    ogc: OgcData  # type: ignore
    cvv: ConstraintSetData  # type: ignore
    cve: ConstraintSetData  # type: ignore
    cvf: ConstraintSetData  # type: ignore
    cee: ConstraintSetData  # type: ignore


class MeshDynamics:

    _data: MeshDynamicsData  # type: ignore

    def __init__(self, ogc: Ogc):
        self.ogc = ogc
        vv_capacity, ve_capacity, vf_capacity, ee_capacity = self.ogc.capacity
        n_verts, n_edges, n_half_edges, n_triangles = self.ogc.n_primitives
        self.cvv = ConstraintSet(n_verts, vv_capacity)
        self.cve = ConstraintSet(n_verts, ve_capacity)
        self.cvf = ConstraintSet(n_verts, vf_capacity)
        self.cee = ConstraintSet(n_half_edges, ee_capacity)
        self._data = MeshDynamicsData()
        self._data.ogc = self.ogc.data
        self._data.cvv = self.cvv.data
        self._data.cve = self.cve.data
        self._data.cvf = self.cvf.data
        self._data.cee = self.cee.data

    @property
    def data(self) -> MeshDynamicsData:  # type: ignore
        return self._data

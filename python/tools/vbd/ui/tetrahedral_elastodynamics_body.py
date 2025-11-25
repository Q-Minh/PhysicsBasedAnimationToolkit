# type: ignore
import numpy as np
import h5py as h5
import polyscope as ps
import polyscope.imgui as imgui


class TetrahedralElastodynamicsBody:
    _V: np.ndarray[float]  # Vertex positions
    _T: np.ndarray[float]  # Tetrahedral elements
    _Ye: np.ndarray[float]  # `|# tetrahedra|` array of Young's moduli
    _nue: np.ndarray[float]  # `|# tetrahedra|` array of Poisson's ratios
    _rhoe: np.ndarray[float]  # `|# tetrahedra|` array of mass densities
    _bext: np.ndarray[float]  # `|# vertices| x 3` array of external body forces
    _aext: np.ndarray[float]  # `3 x 1` external acceleration
    _v0: np.ndarray[float]  # `|# vertices| x 3` array of initial velocities
    _d_mask: np.ndarray[int]  # `|# vertices|` integer array for Dirichlet constraints
    _dirty: bool = False  # Flag indicating if the body has been modified
    _vm: ps.VolumeMesh = None  # Polyscope volume mesh for visualization
    _pc: ps.PointCloud = None  # Polyscope point cloud for visualization
    _name: str = None  # Name of the body

    def __init__(self):
        self._V = None
        self._T = None
        self._Ye = None
        self._nue = None
        self._rhoe = None
        self._bext = None
        self._aext = None
        self._v0 = None
        self._d_mask = None
        self._dirty = False

    def construct(
        self,
        V: np.ndarray[float],
        T: np.ndarray[float],
        Ye: np.ndarray[float],
        nue: np.ndarray[float],
        rhoe: np.ndarray[float],
        bext: np.ndarray[float],
        aext: np.ndarray[float],
        v0: np.ndarray[float],
        d_mask: np.ndarray[bool],
    ):
        if (
            V is None
            or T is None
            or Ye is None
            or nue is None
            or rhoe is None
            or bext is None
            or aext is None
            or v0 is None
            or d_mask is None
        ):
            raise ValueError("All parameters must be provided and non-None.")
        self._V = V
        self._T = T
        self._Ye = Ye
        self._nue = nue
        self._rhoe = rhoe
        self._bext = bext
        self._aext = aext
        self._v0 = v0
        self._d_mask = d_mask
        self._dirty = True
        self._throw_if_invalid_state()

    def draw(self, n_dirichlet_groups: int):
        if self._vm is None:
            return
        if self._dirty:
            self._vm.add_scalar_quantity(
                "Young's Modulus",
                np.log10(self._Ye + 1),
                defined_on="cells",
                enabled=True,
            )
            self._vm.add_scalar_quantity(
                "Poisson's Ratio",
                self._nue,
                defined_on="cells",
                enabled=False,
            )
            self._vm.add_scalar_quantity(
                "Mass Density",
                np.log10(self._rhoe + 1),
                defined_on="cells",
                enabled=False,
            )
            self._vm.add_vector_quantity(
                "External Load",
                self._bext,
                defined_on="cells",
                enabled=False,
            )
            self._vm.add_vector_quantity(
                "Initial Velocity",
                self._v0,
                defined_on="vertices",
                enabled=False,
            )
            d_nodes = np.where(self._d_mask > 0)[0]
            if d_nodes.shape[0] > 0:
                self._pc = ps.register_point_cloud(
                    "Dirichlet Nodes", self._V[d_nodes, :]
                )
                d_groups = self._d_mask[d_nodes]
                self._pc.add_scalar_quantity(
                    "Dirichlet Group",
                    d_groups,
                    defined_on="points",
                    cmap="turbo",
                    vminmax=(0, n_dirichlet_groups - 1),
                    enabled=True,
                )
            self._dirty = False
        _, aext = imgui.InputFloat3("External Acceleration", self._aext)
        self._aext = np.array(aext)

    def on_mesh_loaded(self, name: str, V: np.ndarray[float], T: np.ndarray[float]):
        self._Ye = np.full(T.shape[0], 1e6)
        self._nue = np.full(T.shape[0], 0.45)
        self._rhoe = np.full(T.shape[0], 1e3)
        self._bext = np.zeros((T.shape[0], 3))
        self._aext = np.array([0.0, -9.81, 0.0])
        self._v0 = np.zeros((V.shape[0], 3))
        self._d_mask = np.full(V.shape[0], False, dtype=bool)
        self._on_mesh_loaded(
            name,
            V,
            T,
            self._Ye,
            self._nue,
            self._rhoe,
            self._bext,
            self._aext,
            self._v0,
            self._d_mask,
        )

    def on_mesh_removed(self):
        if self._vm is not None:
            ps.remove_volume_mesh(self._vm.get_name())
        if self._pc is not None:
            ps.remove_point_cloud(self._pc.get_name())
        self._vm = None

    def set_young_modulus(self, Y: float, einds: np.ndarray[int] = None):
        if einds is None:
            einds = np.arange(self._T.shape[0])
        self._Ye[einds] = Y
        self._dirty = True

    def set_poisson_ratio(self, nu: float, einds: np.ndarray[int] = None):
        if einds is None:
            einds = np.arange(self._T.shape[0])
        self._nue[einds] = nu
        self._dirty = True

    def set_mass_density(self, rho: float, einds: np.ndarray[int] = None):
        if einds is None:
            einds = np.arange(self._T.shape[0])
        self._rhoe[einds] = rho
        self._dirty = True

    def set_external_load(self, bext: np.ndarray[float], einds: np.ndarray[int] = None):
        if bext.shape[0] != 3:
            raise ValueError("External load must be a 3-vector.")
        if einds is None:
            einds = np.arange(self._T.shape[0])
        self._bext[einds, :] = bext
        self._dirty = True

    def set_external_acceleration(self, aext: np.ndarray[float]):
        if aext.shape[0] != 3:
            raise ValueError("External acceleration must be a 3-vector.")
        self._aext = aext

    def set_initial_velocities(
        self, v0: np.ndarray[float], vinds: np.ndarray[int] = None
    ):
        if v0.shape[0] != 3:
            raise ValueError("Initial velocity v0 must be 3-vector.")
        if vinds is None:
            vinds = np.arange(self._V.shape[0])
        self._v0[vinds, :] = v0
        self._dirty = True

    def set_dirichlet_group(self, group: int, vinds: np.ndarray[int]):
        self._d_mask[vinds] = group
        self._dirty = True

    def serialize(self, grp: h5.Group):
        grp = grp["tools.vbd.ui.TetrahedralElastodynamicsBody"]
        grp["V"] = self._V
        grp["T"] = self._T
        grp["Ye"] = self._Ye
        grp["nue"] = self._nue
        grp["rhoe"] = self._rhoe
        grp["bext"] = self._bext
        grp["aext"] = self._aext
        grp["v0"] = self._v0
        grp["d_mask"] = self._d_mask
        grp.attrs["name"] = self._name

    def deserialize(self, grp: h5.Group):
        grp = grp["tools.vbd.ui.TetrahedralElastodynamicsBody"]
        self._V = grp["V"][:]
        self._T = grp["T"][:]
        self._Ye = grp["Ye"][:]
        self._nue = grp["nue"][:]
        self._rhoe = grp["rhoe"][:]
        self._bext = grp["bext"][:]
        self._aext = grp["aext"][:]
        self._v0 = grp["v0"][:]
        self._d_mask = grp["d_mask"][:]
        self._name = grp.attrs["name"]
        self._on_mesh_loaded(
            self._name,
            self._V,
            self._T,
            self._Ye,
            self._nue,
            self._rhoe,
            self._bext,
            self._aext,
            self._v0,
            self._d_mask,
        )

    @property
    def name(self):
        return self._name

    @property
    def T(self):
        return self._T

    @property
    def VT(self):
        if self._V is None or self._vm is None:
            return None
        T = self._vm.get_transform()
        VH = np.vstack([self._V.T, np.ones((1, self._V.shape[0]))])
        VT = (T @ VH).T[:, :3]
        return VT

    def _on_mesh_loaded(
        self,
        name: str,
        V: np.ndarray[float],
        T: np.ndarray[float],
        Ye: np.ndarray[float],
        nue: np.ndarray[float],
        rhoe: np.ndarray[float],
        bext: np.ndarray[float],
        aext: np.ndarray[float],
        v0: np.ndarray[float],
        d_mask: np.ndarray[bool],
    ):
        self._V = V
        self._T = T
        self._Ye = Ye
        self._nue = nue
        self._rhoe = rhoe
        self._bext = bext
        self._aext = aext
        self._v0 = v0
        self._d_mask = d_mask
        self._name = name
        self._vm = ps.register_volume_mesh(f"{self._name}", self._V, self._T)
        self._dirty = True

    def _throw_if_invalid_state(self):
        if self._V is None or self._T is None:
            raise ValueError("Vertex positions and tetrahedral elements must be set.")
        if self._V.shape[1] != 3:
            raise ValueError("Vertex positions must have 3 columns.")
        if self._T.shape[1] != 4:
            raise ValueError("Tetrahedral elements must have 4 columns.")
        if self._Ye.shape[0] != self._T.shape[0]:
            raise ValueError("Young's moduli size must match number of tetrahedra.")
        if self._nue.shape[0] != self._T.shape[0]:
            raise ValueError("Poisson's ratios size must match number of tetrahedra.")
        if self._rhoe.shape[0] != self._T.shape[0]:
            raise ValueError("Mass densities size must match number of tetrahedra.")
        if self._bext.shape[0] != self._T.shape[0]:
            raise ValueError(
                "External body forces size must match number of tetrahedra."
            )
        if self._aext.shape != (3,):
            raise ValueError("External acceleration must be a 3D vector.")
        if self._v0.shape != self._V.shape:
            raise ValueError(
                "Initial velocities array must have same shape as vertex positions."
            )
        if self._d_mask.shape[0] != self._V.shape[0]:
            raise ValueError("Dirichlet mask size must match number of vertices.")

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

    Y_ranges: tuple[float, float]
    _nu_ranges: tuple[float, float]
    _rho_ranges: tuple[float, float]

    _cached_transform: np.ndarray

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
        self._cached_transform = np.eye(4)

    def draw(self):
        _, aext = imgui.InputFloat3("External Acceleration", self._aext)
        self._aext = np.array(aext)

    def set_visible(self, visible: bool):
        if self._vm is not None:
            self._vm.set_enabled(visible)
        if self._pc is not None:
            self._pc.set_enabled(visible)

    def undirty(self, n_dirichlet_groups: int):
        if self._vm is None:
            return
        self._vm.add_scalar_quantity(
            "Young's Modulus",
            np.log10(self._Ye + 1),
            defined_on="cells",
            vminmax=self.Y_ranges,
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
                f"{self._name} - Dirichlet", self.VT[d_nodes, :]
            )
            d_groups = self._d_mask[d_nodes]
            self._pc.add_scalar_quantity(
                "Group",
                d_groups,
                cmap="turbo",
                vminmax=(1, n_dirichlet_groups),
                enabled=True,
            )
        # It's possible that the object used to have a dnodes but they were all removed using a selection
        # We want to remove the point cloud
        elif self._pc is not None and ps.has_point_cloud(self._pc.get_name()):
            ps.remove_point_cloud(self._pc.get_name())
            self._pc = None

        self._cached_transform = np.array(self._vm.get_transform())
        self._dirty = False

    def on_mesh_loaded(
        self,
        name: str,
        V: np.ndarray[float],
        T: np.ndarray[float],
        Ye: np.ndarray[float] = None,
        nue: np.ndarray[float] = None,
        rhoe: np.ndarray[float] = None,
        bext: np.ndarray[float] = None,
        aext: np.ndarray[float] = None,
        v0: np.ndarray[float] = None,
        d_mask: np.ndarray[bool] = None,
    ):
        default_Y = 1e6
        default_nu = 0.45
        default_rho = 1e3
        self._V = V
        self._T = T
        self._Ye = np.full(T.shape[0], default_Y) if Ye is None else Ye
        self._nue = np.full(T.shape[0], default_nu) if nue is None else nue
        self._rhoe = np.full(T.shape[0], default_rho) if rhoe is None else rhoe
        self._bext = np.zeros((T.shape[0], 3)) if bext is None else bext
        self._aext = np.array([0.0, 0.0, -9.81]) if aext is None else aext
        self._v0 = np.zeros((V.shape[0], 3)) if v0 is None else v0
        self._d_mask = np.full(V.shape[0], 0, dtype=int) if d_mask is None else d_mask
        self._name = name
        self._throw_if_invalid_state()
        self._vm = ps.register_volume_mesh(f"{self._name}", self._V, self._T)
        self._cached_transform = np.eye(4)
        self._vm.set_transform(self._cached_transform)
        if Ye is None:
            default_Y = np.log10(default_Y)
            self.Y_ranges = (default_Y, default_Y)
        else:
            logYe = np.log10(Ye)
            self.Y_ranges = (np.min(logYe), np.max(logYe))

        if nue is None:
            self._nu_ranges = (default_nu, default_nu)
        else:
            self._nu_ranges = (np.min(lognue), np.max(lognue))

        if rhoe is None:
            default_rho = np.log10(default_rho)
            self._rho_ranges = (default_rho, default_rho)
        else:
            logrhoe = np.log10(rhoe)
            self._rho_ranges = (np.min(logrhoe), np.max(logrhoe))
        
        self._dirty = True

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
        logY = np.log10(Y)
        m, M = self.Y_ranges
        self.Y_ranges = (min(m, logY), max(M, logY))
        self._dirty = True

    def set_poisson_ratio(self, nu: float, einds: np.ndarray[int] = None):
        if einds is None:
            einds = np.arange(self._T.shape[0])
        self._nue[einds] = nu
        m, M = self.nu_ranges
        self.nu_ranges = (min(m, nu), max(M, nu))
        self._dirty = True

    def set_mass_density(self, rho: float, einds: np.ndarray[int] = None):
        if einds is None:
            einds = np.arange(self._T.shape[0])
        self._rhoe[einds] = rho
        logrho = np.log10(rho)
        m, M = self.rho_ranges
        self.rho_ranges = (min(m, logrho), max(M, logrho))
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

    
    def compare_young_modulus(self, compare):
        m, M = self.Y_ranges
        n, N = compare.Y_ranges
        global_min = min(n, m, M, N)
        global_max = max(n, m, M, N)
        self.Y_ranges = (global_min, global_max)
        compare.Y_ranges = self.Y_ranges

    def compare_rho(self, compare):
        m, M = self.rho_ranges
        n, N = compare.rho_ranges
        global_min = min(n, m, M, N)
        global_max = max(n, m, M, N)
        self.rho_ranges = (global_min, global_max)
        compare.rho_ranges = self.rho_ranges

    def compare_nu(self, compare):
        m, M = self.nu_ranges
        n, N = compare.nu_ranges
        global_min = min(n, m, M, N)
        global_max = max(n, m, M, N)
        self.nu_ranges = (global_min, global_max)
        compare.nu_ranges = self.nu_ranges

    def serialize(self, grp: h5.Group):
        grp = grp.create_group("tools.vbd.ui.TetrahedralElastodynamicsBody")
        grp["V"] = self.VT
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
        self.on_mesh_loaded(
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
    def name(self) -> str:
        return self._name

    @property
    def T(self) -> np.ndarray:
        return self._T

    @property
    def VT(self) -> np.ndarray:
        if self._V is None or self._vm is None:
            return None
        T = self._vm.get_transform()
        VH = np.vstack([self._V.T, np.ones((1, self._V.shape[0]))])
        VT = (T @ VH).T[:, :3]
        return VT

    @property
    def Ye(self) -> np.ndarray:
        return self._Ye

    @property
    def nue(self) -> np.ndarray:
        return self._nue

    @property
    def rhoe(self) -> np.ndarray:
        return self._rhoe

    @property
    def bext(self) -> np.ndarray:
        return self._bext

    @property
    def aext(self) -> np.ndarray:
        return self._aext

    @property
    def v0(self) -> np.ndarray:
        return self._v0

    @property
    def d_mask(self) -> np.ndarray:
        return self._d_mask

    @property
    def dirty(self) -> bool:
        if self._vm is None:
            return False
        return self._dirty or np.any(
            self._cached_transform != np.array(self._vm.get_transform())
        )

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

# type: ignore
import h5py
import numpy as np
from enum import Enum
import polyscope as ps
import polyscope.imgui as imgui

"""
primitive operations
all primitives require:
- 1 interval of time (begin + duration in seconds)
- list of target dirichlet groups

available primitives:
- rotate about x, y, z
- revolution (ie rotation about com)
    -> axis, rev per second
- translation 
    -> direction, speed
- implicit: empty (do nothing)
"""


def _rotate_closed_form(
    axis: np.ndarray,
    omega: float,
    pivot: np.ndarray,
    tau: np.ndarray,
    dtau: float,
    Y: np.ndarray,
    dY: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate Y about the unit axis through pivot by the angle omega * tau (Rodrigues' formula).

    Returns R (Y - p) + p and its time derivative R dY + omega dtau a x (R (Y - p)).
    """
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    theta = omega * tau
    c, s = np.cos(theta), np.sin(theta)

    def R(Z):
        return Z * c + (K @ Z) * s + axis[:, np.newaxis] * (axis @ Z) * (1 - c)

    RY = R(Y - pivot[:, np.newaxis])
    return RY + pivot[:, np.newaxis], R(dY) + omega * dtau * (K @ RY)


class TransformType(Enum):
    G_ROTATE = 0
    L_ROTATE = 1
    TRANSLATE = 2
    FIXED = 3
    L_TRANSLATE = 4


class PrimitiveTransform:
    name: str
    begin: float
    duration: float
    transform_type: TransformType
    id: int

    _mesh_dirichlet_nodes: dict[str, list[int]]  # mesh name -> dirichlet node indices
    _dirty: bool
    _pc: ps.PointCloud

    def __init__(
        self,
        name: str,
        begin: float,
        duration: float,
        transform_type: TransformType,
    ):
        self.name = name
        self.begin = begin
        self.duration = duration
        self.transform_type = (
            transform_type  # e.g., "rotate", "revolve", "translate", "empty"
        )
        self.id = -1  # to be set when added to library
        self._mesh_dirichlet_nodes = {}
        self._dirty = False
        self._pc = None

    @staticmethod
    def make_default():
        # Python doesn't support multiple constructors
        raise NotImplementedError("Implement a default constructor in your subclass!")

    @staticmethod
    def make_default(ttype: TransformType):
        if ttype == TransformType.G_ROTATE:
            return GlobalRotateTransform.make_default()
        elif ttype == TransformType.L_ROTATE:
            return LocalRotateTransform.make_default()
        elif ttype == TransformType.TRANSLATE:
            return TranslateTransform.make_default()
        elif ttype == TransformType.FIXED:
            return FixedTransform.make_default()
        elif ttype == TransformType.L_TRANSLATE:
            return LocalTranslateTransform.make_default()
        else:
            raise ValueError(f"Unknown TransformType {ttype}")

    def window_steps(self, dt: float) -> tuple[int, int]:
        """Half-open range [n_begin, n_end) of time steps during which the transform is active.

        Step n (from t_n = n * dt to t_{n+1}) is active when begin <= t_n < begin + duration.
        Boundaries are rounded to the step grid with a tolerance of 1e-3 steps, so that e.g. a 1.2 s
        window stored in float32 (1.2000000476837158) at dt = 0.01 covers exactly 120 steps.
        """
        tol = 1e-3
        n_begin = int(np.ceil(self.begin / dt - tol))
        end = self.begin + self.duration
        n_end = int(np.ceil(end / dt - tol)) if np.isfinite(end) else np.iinfo(np.int64).max
        return n_begin, n_end

    def window_times(self, dt: float) -> tuple[float, float]:
        """Time window [begin, end) rounded to the step grid, see `window_steps`."""
        n_begin, n_end = self.window_steps(dt)
        end = n_end * dt if n_end != np.iinfo(np.int64).max else np.inf
        return n_begin * dt, end

    def applicable(self, t: int, dt: float) -> bool:
        """Check if the transform is active during time step t, i.e. begin <= t * dt < begin + duration"""
        n_begin, n_end = self.window_steps(dt)
        return n_begin <= t < n_end

    def nodes(self, mesh_name: str) -> np.ndarray:
        """Get the dirichlet node indices for the given mesh"""
        return self._mesh_dirichlet_nodes[mesh_name]

    def affects(self, mesh_name: str) -> bool:
        """Check if the transform affects the given mesh"""
        return mesh_name in self._mesh_dirichlet_nodes

    def closed_form(
        self, tau: np.ndarray, dtau: float, Y: np.ndarray, dY: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the transform after it has been active for time tau.

        Args:
            tau (np.ndarray): `|# nodes|` time in seconds that the transform has been active since
                each node's anchor.
            dtau (float): Time derivative of tau, 1 while the transform is moving and 0 otherwise.
            Y (np.ndarray): `|# dims| x |# nodes|` input positions.
            dY (np.ndarray): `|# dims| x |# nodes|` time derivative of the input positions.

        Returns:
            tuple[np.ndarray, np.ndarray]: The transformed positions and their time derivative.
        """
        raise NotImplementedError("closed_form must be implemented in subclasses")

    def adjust(self):
        """Adjust parameters if necessary (e.g., normalize axes)"""
        pass

    def on_transform_added(self, mesh_names: list[str]):
        for mesh_name in mesh_names:
            self.on_mesh_added(mesh_name)

    def on_mesh_added(self, mesh_name: str):
        self._mesh_dirichlet_nodes[mesh_name] = np.array([], dtype=np.int32)

    def on_mesh_removed(self, mesh_name: str):
        self._mesh_dirichlet_nodes.pop(mesh_name, None)
        self._dirty = True

    def on_dirichlet_nodes_added(self, mesh_name: str, node_suffix: list[int]):
        nodes_prefix = self._mesh_dirichlet_nodes[mesh_name]
        self._mesh_dirichlet_nodes[mesh_name] = np.unique(
            np.concatenate((nodes_prefix, node_suffix))
        )
        self._dirty = True

    def on_dirichlet_nodes_removed(self, mesh_name: str, node_suffix: list[int]):
        nodes_prefix = self._mesh_dirichlet_nodes[mesh_name]
        self._mesh_dirichlet_nodes[mesh_name] = np.setdiff1d(nodes_prefix, node_suffix)
        self._dirty = True

    def serialize(self, h5group: h5py.Group):
        """
        Serialize the transform to an HDF5 group.
        """
        h5group.attrs["name"] = self.name
        h5group.attrs["begin"] = self.begin
        h5group.attrs["duration"] = self.duration
        h5group.attrs["transform_type"] = self.transform_type.value
        h5group.attrs["id"] = self.id
        for mesh_name, node_indices in self._mesh_dirichlet_nodes.items():
            mesh_grp = h5group.create_group(f"dirichlet/mesh/{mesh_name}")
            mesh_grp["node_indices"] = node_indices

    def deserialize(self, h5group: h5py.Group):
        """
        Deserialize the transform from an HDF5 group.
        """
        self.name = h5group.attrs["name"]
        self.begin = h5group.attrs["begin"]
        self.duration = h5group.attrs["duration"]
        self.transform_type = TransformType(h5group.attrs["transform_type"])
        self.id = h5group.attrs["id"]
        dirichlet_grp = h5group.get("dirichlet/mesh", None)
        if dirichlet_grp is not None:
            for mesh_name, mesh_grp in dirichlet_grp.items():
                node_indices = mesh_grp["node_indices"][:]
                self._mesh_dirichlet_nodes[mesh_name] = node_indices
        self._dirty = True

    def undirty(self, mesh_names: list[str], mesh_verts: list[np.ndarray]):
        if len(mesh_names) == 0:
            VD = np.array([])
        else:
            VD = np.vstack(
                [
                    V[self._mesh_dirichlet_nodes[name], :]
                    for name, V in zip(mesh_names, mesh_verts)
                ]
            )
            self._pc = ps.register_point_cloud(
                f"Transform {self.id} - Dirichlet Nodes", VD
            )
        self._dirty = False
        return VD

    def set_visible(self, visible: bool):
        if self._pc is not None:
            self._pc.set_enabled(visible)

    def on_removed(self):
        if self._pc is not None:
            ps.remove_point_cloud(self._pc.get_name())

    @property
    def dirty(self) -> bool:
        return self._dirty

    def __str__(self):
        return f"Name: {self.name}, Type: {self.transform_type.name},\n\t ID: {self.id}, \n\t\t Time: [{self.begin}, {self.begin + self.duration}]"


class GlobalRotateTransform(PrimitiveTransform):
    """Rotate all vertices around a global axis"""

    axis: np.ndarray
    degrees_per_second: float

    def __init__(
        self,
        name: str,
        begin: float,
        duration: float,
        axis: np.ndarray,
        degrees_per_second: float,
    ):
        super().__init__(name, begin, duration, TransformType.G_ROTATE)
        self.axis = axis / np.linalg.norm(axis)  # normalize axis
        self.degrees_per_second = degrees_per_second

    @staticmethod
    def make_default():
        return GlobalRotateTransform("New Global Rotate", 0, 1, np.array([1, 0, 0]), 10)

    def closed_form(self, tau, dtau, Y, dY):
        return _rotate_closed_form(
            self.axis, np.deg2rad(self.degrees_per_second), np.zeros(3), tau, dtau, Y, dY
        )

    def serialize(self, h5group: h5py.Group):
        super().serialize(h5group)
        h5group.attrs["axis"] = self.axis
        h5group.attrs["degrees_per_second"] = self.degrees_per_second

    def deserialize(self, h5group: h5py.Group):
        super().deserialize(h5group)
        self.axis = h5group.attrs["axis"]
        self.degrees_per_second = h5group.attrs["degrees_per_second"]

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}\n\t\t Axis: {self.axis},\n\t\t Degrees per second: {self.degrees_per_second}"

    def adjust(self):
        self.axis = self.axis / np.linalg.norm(self.axis)


class LocalRotateTransform(PrimitiveTransform):
    """Rotate all vertices around a local axis and origin"""

    axis: np.ndarray
    origin: np.ndarray
    degrees_per_second: float

    def __init__(
        self,
        name: str,
        begin: float,
        duration: float,
        axis: np.ndarray,
        origin: np.ndarray,
        degrees_per_second: float,
    ):
        super().__init__(name, begin, duration, TransformType.L_ROTATE)
        self.axis = axis / np.linalg.norm(axis)  # normalize axis
        self.origin = origin
        self.degrees_per_second = degrees_per_second

    @staticmethod
    def make_default():
        return LocalRotateTransform(
            "New Local Rotate", 0, 1, np.array([1, 0, 0]), np.array([0, 0, 0]), 10
        )

    def undirty(self, mesh_names: list[str], mesh_verts: list[np.ndarray]):
        VD = super().undirty(mesh_names, mesh_verts)
        # We'll use the average position of VD in order to define the local position
        if VD.shape[0] == 0:
            self.origin = np.array([0, 0, 0])
        else:
            self.origin = np.mean(VD, axis=0)
        return VD

    def closed_form(self, tau, dtau, Y, dY):
        return _rotate_closed_form(
            self.axis, np.deg2rad(self.degrees_per_second), self.origin, tau, dtau, Y, dY
        )

    def serialize(self, h5group: h5py.Group):
        super().serialize(h5group)
        h5group.attrs["axis"] = self.axis
        h5group.attrs["origin"] = self.origin
        h5group.attrs["degrees_per_second"] = self.degrees_per_second

    def deserialize(self, h5group: h5py.Group):
        super().deserialize(h5group)
        self.axis = h5group.attrs["axis"]
        self.origin = h5group.attrs["origin"]
        self.degrees_per_second = h5group.attrs["degrees_per_second"]

    def adjust(self):
        self.axis = self.axis / np.linalg.norm(self.axis)

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}\n\t\t Axis: {self.axis},\n\t\t Origin: {self.origin},\n\t\t Degrees per second: {self.degrees_per_second}"


class TranslateTransform(PrimitiveTransform):
    """Translate all vertices along a direction"""

    direction: np.ndarray
    speed: float

    def __init__(
        self,
        name: str,
        begin: float,
        duration: float,
        direction: np.ndarray,
        speed: float,
    ):
        super().__init__(name, begin, duration, TransformType.TRANSLATE)
        self.direction = direction / np.linalg.norm(direction)  # normalize direction
        self.speed = speed  # units per second

    @staticmethod
    def make_default():
        return TranslateTransform("New Translation", 0, 1, np.array([1, 0, 0]), 10)

    def closed_form(self, tau, dtau, Y, dY):
        d = self.direction[:, np.newaxis]
        return Y + self.speed * tau * d, dY + self.speed * dtau * d

    def serialize(self, h5group: h5py.Group):
        super().serialize(h5group)
        h5group.attrs["direction"] = self.direction
        h5group.attrs["speed"] = self.speed

    def deserialize(self, h5group: h5py.Group):
        super().deserialize(h5group)
        self.direction = h5group.attrs["direction"]
        self.speed = h5group.attrs["speed"]

    def adjust(self):
        self.direction = self.direction / np.linalg.norm(self.direction)

    def __str__(self):
        base_str = super().__str__()
        return (
            f"{base_str}\n\t\t Direction: {self.direction},\n\t\t Speed: {self.speed}"
        )


class LocalTranslateTransform(PrimitiveTransform):
    """Translate each vertex radially away from origin, in the xy-plane, at `speed` units per second"""

    origin: np.ndarray
    speed: float

    def __init__(
        self,
        name: str,
        begin: float,
        duration: float,
        origin: np.ndarray,
        speed: float,
    ):
        super().__init__(name, begin, duration, TransformType.L_TRANSLATE)
        self.origin = origin
        self.speed = speed  # units per second

    @staticmethod
    def make_default():
        return LocalTranslateTransform(
            "New Local Translation", 0, 1, np.array([0, 0, 0]), 10
        )

    def closed_form(self, tau, dtau, Y, dY):
        # Each node moves along its radial direction rhat in the xy-plane, which stays constant
        # while it moves. Nodes on the axis through origin have no radial direction and stay put.
        r = Y - self.origin[:, np.newaxis]
        r[2] = 0
        rho = np.linalg.norm(r, axis=0)
        rhat = np.divide(r, rho, out=np.zeros_like(r, dtype=float), where=rho > 0)
        # d rhat / dt = (I - rhat rhat^T) P_xy dY / rho
        PdY = dY.copy()
        PdY[2] = 0
        drhat = np.divide(
            PdY - rhat * np.sum(rhat * PdY, axis=0),
            rho,
            out=np.zeros_like(r, dtype=float),
            where=rho > 0,
        )
        return (
            Y + self.speed * tau * rhat,
            dY + self.speed * tau * drhat + self.speed * dtau * rhat,
        )

    def serialize(self, h5group: h5py.Group):
        super().serialize(h5group)
        h5group.attrs["origin"] = self.origin
        h5group.attrs["speed"] = self.speed

    def deserialize(self, h5group: h5py.Group):
        super().deserialize(h5group)
        self.origin = h5group.attrs["origin"]
        self.speed = h5group.attrs["speed"]

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}\n\t\t Origin: {self.origin},\n\t\t Speed: {self.speed}"


class FixedTransform(PrimitiveTransform):
    """Keep vertices fixed"""

    def __init__(self, name, begin, duration):
        super().__init__(name, begin, duration, TransformType.FIXED)

    @staticmethod
    def make_default():
        return FixedTransform("New Fixed", 0, 1)

    def closed_form(self, tau, dtau, Y, dY):
        return Y, dY

    def serialize(self, h5group):
        return super().serialize(h5group)

    def deserialize(self, h5group):
        return super().deserialize(h5group)


class TransformLibrary:
    transforms: list[PrimitiveTransform]
    _recycled_indices: list[int]
    _current_transform_index: int
    _mesh_names: list[str]
    _anchors: dict[
        str, tuple[np.ndarray, np.ndarray]
    ]  # mesh name -> (3 x n anchor positions, n anchor time steps or -1 outside constrained runs)

    def __init__(self):
        self.transforms = []
        self._recycled_indices = []
        self._current_transform_index = 0
        self._mesh_names = []
        self._anchors = {}

    def build_transform_map(self):
        self._transform_map = {}
        for ttype in TransformType:
            self._transform_map[ttype] = []

    def on_mesh_added(self, mesh_name: str):
        if mesh_name not in self._mesh_names:
            self._mesh_names.append(mesh_name)
        for transform in self.transforms:
            transform.on_mesh_added(mesh_name)

    def on_mesh_removed(self, mesh_name: str):
        if mesh_name in self._mesh_names:
            self._mesh_names.remove(mesh_name)
        for transform in self.transforms:
            transform.on_mesh_removed(mesh_name)

    def sync_mesh_names(self, mesh_names: list[str]):
        self._mesh_names = list(mesh_names)
        for transform in self.transforms:
            preserved_nodes = {}
            for name in self._mesh_names:
                if name in transform._mesh_dirichlet_nodes:
                    preserved_nodes[name] = transform._mesh_dirichlet_nodes[name]
                else:
                    preserved_nodes[name] = np.array([], dtype=np.int32)
            transform._mesh_dirichlet_nodes = preserved_nodes

    def on_dirichlet_nodes_added(
        self, transform_id: int, mesh_name: str, inds: list[int]
    ):
        transform = next(
            (
                transform
                for transform in self.transforms
                if transform.id == transform_id
            ),
            None,
        )
        if transform is None:
            ps.warning(
                f"Dirichlet group {transform_id} does not have a corresponding transform. No-op."
            )
            return
        transform.on_dirichlet_nodes_added(mesh_name, inds)

    def on_dirichlet_nodes_removed(self, mesh_name: str, inds: list[int]):
        for transform in self.transforms:
            transform.on_dirichlet_nodes_removed(mesh_name, inds)

    def add_transform(self, transform: PrimitiveTransform):
        transform.adjust()
        transform.id = self._get_new_id() + 1
        self.transforms.append(transform)
        transform.on_transform_added(self._mesh_names)

    def all_transformed_nodes(
        self, t: int, dt: float
    ) -> list[tuple[str, np.ndarray[int]]]:
        all_nodes = [[] for _ in range(len(self._mesh_names))]
        for i, name in enumerate(self._mesh_names):
            for transform in self.transforms:
                if transform.applicable(t, dt) and transform.affects(name):
                    all_nodes[i].append(transform.nodes(name))
            if len(all_nodes[i]) > 0:
                all_nodes[i] = np.unique(np.concatenate(all_nodes[i]))
        return zip(self._mesh_names, all_nodes)

    def all_transformed_nodes_with_durations(
        self,
    ) -> list[tuple[str, np.ndarray[int], np.ndarray[float]]]:
        all_nodes = [[] for _ in range(len(self._mesh_names))]
        durations = [[] for _ in range(len(self._mesh_names))]
        for i, name in enumerate(self._mesh_names):
            for transform in self.transforms:
                if transform.affects(name):
                    nodes = transform.nodes(name)
                    all_nodes[i].append(nodes)
                    durations[i].append(np.full_like(nodes, transform.duration, dtype=np.float32))
            if len(all_nodes[i]) > 0:
                all_nodes_i_cpy = np.concatenate(all_nodes[i])
                durations[i] = np.concatenate(durations[i])
                all_nodes[i], indices = np.unique(all_nodes_i_cpy, return_index=True)
                durations[i] = np.maximum.reduceat(
                    durations[i][np.argsort(all_nodes_i_cpy)], indices
                )
        return zip(self._mesh_names, all_nodes, durations)

    def evaluate(
        self, name: str, V: np.ndarray, t: int, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the prescribed motion of mesh `name` for time step t (from t * dt to (t+1) * dt).

        A node is constrained while any transform affecting it is active. When a node starts a
        constrained run at step t_a, its position x_a in V is recorded as its anchor. Its prescribed
        position at the end of the step is then the closed form

            x = T_m(tau_m) o ... o T_1(tau_1) (x_a),
            tau_k = clamp(t1, b_k, e_k) - clamp(t_a * dt, b_k, e_k),

        with the transforms in library order, t1 = (t+1) * dt, and [b_k, e_k) the transform's time
        window on the step grid. Motions therefore don't accumulate drift, and a motion that ends
        while an overlapping transform still holds the node leaves it where the motion stopped.

        Args:
            name (str): Mesh name.
            V (np.ndarray): `|# dims| x |# mesh nodes|` positions at the start of the time step.
            t (int): Time step index.
            dt (float): Time step size.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The positions at the end of the step (V with
            the constrained nodes replaced), the `|# dims| x |# mesh nodes|` prescribed velocities
            (zero for free nodes) and the `|# mesh nodes|` boolean mask of constrained nodes.
        """
        n = V.shape[1]
        XA, TA = self._anchors.get(name, (None, None))
        if XA is None or XA.shape[1] != n:
            XA = np.zeros((3, n))
            TA = np.full(n, -1, dtype=np.int64)
            self._anchors[name] = (XA, TA)
        transforms = [tr for tr in self.transforms if tr.affects(name)]
        active = np.zeros(n, dtype=bool)
        for tr in transforms:
            if tr.applicable(t, dt):
                active[tr.nodes(name)] = True
        # Close the runs of free nodes, and anchor the nodes that start a constrained run
        TA[~active] = -1
        opened = active & (TA < 0)
        XA[:, opened] = V[:, opened]
        TA[opened] = t
        Vp = np.array(V, dtype=float)
        dV = np.zeros_like(Vp)
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return Vp, dV, active
        pos = np.full(n, -1, dtype=np.int64)
        pos[idx] = np.arange(idx.size)
        Y = XA[:, idx].copy()
        dY = np.zeros_like(Y)
        ta = TA[idx] * dt
        t1 = (t + 1) * dt
        for tr in transforms:
            sub = pos[tr.nodes(name)]
            sub = sub[sub >= 0]
            if sub.size == 0:
                continue
            b, e = tr.window_times(dt)
            tau = np.clip(t1, b, e) - np.clip(ta[sub], b, e)
            # Left derivative: a node released when a motion ends keeps the motion's velocity
            dtau = 1.0 if b < t1 <= e else 0.0
            Y[:, sub], dY[:, sub] = tr.closed_form(tau, dtau, Y[:, sub], dY[:, sub])
        Vp[:, idx] = Y
        dV[:, idx] = dY
        return Vp, dV, active

    def reset_anchors(self):
        """Forget all anchors, so that constrained nodes are re-anchored at their current positions.

        Re-anchoring in the middle of a constrained run is exact unless the node follows several
        simultaneous motions that don't commute (e.g. rotations about different axes).
        """
        self._anchors = {}

    def draw(self):
        transform_type_names = [tt.name for tt in TransformType]
        _, self._current_transform_index = imgui.Combo(
            "Transform Type",
            self._current_transform_index,
            transform_type_names,
        )
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]
        if imgui.Button("Add", default_button_size):
            ttype = TransformType(self._current_transform_index)
            new_transform = PrimitiveTransform.make_default(ttype)
            self.add_transform(new_transform)

        for idx, transform in enumerate(self.transforms):
            if imgui.TreeNode(f"{transform.id} - {transform.transform_type.name}"):
                self._draw(transform, idx)
                if imgui.Button("Delete"):
                    transform = self.transforms.pop(idx)
                    transform.on_removed()
                    self._recycled_indices.append(idx)
                imgui.TreePop()

    def serialize(self, grp: h5py.Group):
        """
        Serialize the entire library to an HDF5 group.
        """
        grp["mesh_names"] = self._mesh_names
        for transform in self.transforms:
            tgrp = grp.create_group(f"transform/{transform.id}")
            transform.serialize(tgrp)
        grp["recycled_indices"] = np.array(self._recycled_indices, dtype=np.int32)

    def deserialize(self, grp: h5py.Group):
        """
        Deserialize the library from an HDF5 group.
        """
        self.empty()
        if "mesh_names" in grp:
            self._mesh_names = grp["mesh_names"][:].astype(str).tolist()
        grp_transforms = grp.get("transform", None)
        if grp_transforms is None:
            return
        for tname, tgroup in grp_transforms.items():
            transform_type = TransformType(tgroup.attrs["transform_type"])
            transform = PrimitiveTransform.make_default(transform_type)
            transform.deserialize(tgroup)
            self.transforms.append(transform)
        if "recycled_indices" in grp:
            self._recycled_indices = grp["recycled_indices"][:].tolist()

    def _draw(self, transform: PrimitiveTransform, idx: int):
        imgui.PushID(idx)
        _, transform.name = imgui.InputText("Name", transform.name)
        _, transform.begin = imgui.InputFloat("Begin Time", transform.begin)
        _, transform.duration = imgui.InputFloat("Duration", transform.duration)

        if transform.transform_type == TransformType.G_ROTATE:
            _, axis = imgui.InputFloat3("Axis", transform.axis)
            transform.axis = np.array(axis)
            if imgui.Button("Normalize"):
                transform.adjust()
            _, transform.degrees_per_second = imgui.InputFloat(
                "Degrees per Second", transform.degrees_per_second
            )

        elif transform.transform_type == TransformType.L_ROTATE:
            _, axis = imgui.InputFloat3("Axis", transform.axis)
            transform.axis = np.array(axis)
            if imgui.Button("Normalize"):
                transform.adjust()
            _, origin = imgui.InputFloat3("Origin", transform.origin)
            transform.origin = np.array(origin)
            _, transform.degrees_per_second = imgui.InputFloat(
                "Degrees per Second", transform.degrees_per_second
            )

        elif transform.transform_type == TransformType.TRANSLATE:
            _, direction = imgui.InputFloat3("Direction", transform.direction)
            transform.direction = np.array(direction)
            if imgui.Button("Normalize"):
                transform.adjust()
            _, transform.speed = imgui.InputFloat("Speed", transform.speed)

        elif transform.transform_type == TransformType.L_TRANSLATE:
            _, origin = imgui.InputFloat3("Origin", transform.origin)
            transform.origin = np.array(origin)
            _, transform.speed = imgui.InputFloat("Speed", transform.speed)
        imgui.PopID()

    def _get_new_id(self) -> int:
        if self._recycled_indices:
            return self._recycled_indices.pop()
        else:
            return len(self.transforms)

    def empty(self):
        while len(self.transforms) > 0:
            transform = self.transforms.pop()
            transform.on_removed()
        self.reset_anchors()


import unittest


class TestTransformLibrary(unittest.TestCase):
    dt = 0.01

    def _library(self, transforms_and_nodes):
        lib = TransformLibrary()
        lib.on_mesh_added("m")
        for transform, nodes in transforms_and_nodes:
            lib.add_transform(transform)
            transform.on_dirichlet_nodes_added("m", nodes)
        return lib

    def _run(self, lib, V, n_steps):
        traj = [V]
        for t in range(n_steps):
            V, dV, active = lib.evaluate("m", V, t, self.dt)
            traj.append(V)
        return traj

    def test_half_open_windows(self):
        transform = FixedTransform("f", 0.0, 1.2000000476837158)  # float32-stored 1.2 s
        self.assertEqual(transform.window_steps(self.dt), (0, 120))
        self.assertTrue(transform.applicable(119, self.dt))
        self.assertFalse(transform.applicable(120, self.dt))

    def test_rotation_is_exact(self):
        rng = np.random.default_rng(0)
        V = rng.normal(size=(3, 10))
        origin = np.array([0.5, -0.2, 0.1])
        lib = self._library(
            [
                (
                    LocalRotateTransform(
                        "r", 0.0, 1.0, np.array([0.0, 0, 1]), origin, 90.0
                    ),
                    np.arange(10),
                )
            ]
        )
        traj = self._run(lib, V, 150)
        # After the 1 s window, the nodes have rotated by exactly 90 degrees and stay there
        R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        expected = R @ (V - origin[:, np.newaxis]) + origin[:, np.newaxis]
        np.testing.assert_allclose(traj[100], expected, atol=1e-12)
        # ... and are released afterwards
        _, _, active = lib.evaluate("m", traj[100], 100, self.dt)
        self.assertFalse(np.any(active))

    def test_motion_then_hold(self):
        V = np.array([[1.0, 0, 0], [0, 2, 0], [0, 0, 1]]).T
        nodes = np.arange(3)
        lib = self._library(
            [
                (
                    LocalTranslateTransform("l", 0.0, 1.0, np.zeros(3), 0.5),
                    nodes,
                ),
                (FixedTransform("f", 0.0, 1.2), nodes),
            ]
        )
        traj = self._run(lib, V, 130)
        expected = V + 0.5 * np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 0]]).T
        for t in (100, 110, 120):
            np.testing.assert_allclose(traj[t], expected, atol=1e-12)

    def test_prescribed_velocity(self):
        rng = np.random.default_rng(1)
        V = rng.normal(size=(3, 8))
        nodes = np.arange(8)
        lib = self._library(
            [
                (
                    GlobalRotateTransform("r", 0.0, 1.0, np.array([1.0, 0, 0]), 60.0),
                    nodes,
                ),
                (
                    TranslateTransform("t", 0.0, 1.0, np.array([0.0, 1, 0]), 0.3),
                    nodes,
                ),
                (
                    LocalTranslateTransform("l", 0.0, 1.0, np.zeros(3), 0.2),
                    nodes,
                ),
            ]
        )
        traj = self._run(lib, V, 52)
        _, dV, active = lib.evaluate("m", traj[50], 50, self.dt)
        self.assertTrue(np.all(active))
        # dV is the velocity at t = 51 dt, compare with a central difference of the closed form
        central = (traj[52] - traj[50]) / (2 * self.dt)
        np.testing.assert_allclose(dV, central, atol=1e-3)


if __name__ == "__main__":
    unittest.main()

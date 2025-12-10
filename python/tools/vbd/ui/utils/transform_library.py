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


class TransformType(Enum):
    G_ROTATE = 0
    L_ROTATE = 1
    TRANSLATE = 2
    FIXED = 3


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
        else:
            raise ValueError(f"Unknown TransformType {ttype}")

    def applicable(self, t: int, dt: float) -> bool:
        """Check if the transform is applicable at time t"""
        return self.begin <= t * dt <= self.begin + self.duration

    def expired(self, t: float) -> bool:
        """Check if the transform has expired at time t"""
        return t > self.duration + self.begin

    def apply(self, t: float, dt: float, V: np.ndarray) -> np.ndarray:
        """
        Apply the transform to the vertices V at time t * dt.
        V: (n, 3) array of vertex positions
        Returns transformed vertices (n, 3)
        """
        # Implement specific transformations in subclasses
        return self.specific_apply(t, dt, V)

    def nodes(self, mesh_name: str) -> np.ndarray:
        """Get the dirichlet node indices for the given mesh"""
        return self._mesh_dirichlet_nodes[mesh_name]

    def affects(self, mesh_name: str) -> bool:
        """Check if the transform affects the given mesh"""
        return mesh_name in self._mesh_dirichlet_nodes

    def specific_apply(self, t: float, dt: float, V: np.ndarray) -> np.ndarray:
        raise NotImplementedError("specific_apply must be implemented in subclasses")

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
            self._pc = ps.register_point_cloud(f"Transform {self.id} - Dirichlet Nodes", VD)
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

    def specific_apply(self, t, dt, V):
        """Apply the global rotation transform to the given vertices.

        Args:
            t (int): The current time step.
            dt (float): The time step size.
            V (np.ndarray): `|# dims| x |# nodes|` the vertex positions.

        Returns:
            np.ndarray: The transformed vertex positions.
        """
        # Compute rotation angle
        angle_degrees = self.degrees_per_second * (dt)
        angle_radians = np.deg2rad(angle_degrees)
        # Compute rotation matrix using Rodrigues' rotation formula
        K = np.array(
            [
                [0, -self.axis[2], self.axis[1]],
                [self.axis[2], 0, -self.axis[0]],
                [-self.axis[1], self.axis[0], 0],
            ]
        )
        R = (
            np.eye(3)
            + np.sin(angle_radians) * K
            + (1 - np.cos(angle_radians)) * (K @ K)
        )
        # Apply rotation
        V_rotated = R @ V
        return V_rotated

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
        #We'll use the average position of VD in order to define the local position
        if VD.shape[0] == 0: 
            self.origin = np.array([0, 0, 0])
        else:
            self.origin = np.mean(VD, axis=0)
        return VD

    def specific_apply(self, t, dt, V):
        """Apply the local rotation transform to the given vertices.

        Args:
            t (int): The current time step.
            dt (float): The time step size.
            V (np.ndarray): `|# dims| x |# nodes|` the vertex positions.

        Returns:
            np.ndarray: The transformed vertex positions.
        """
        # Compute rotation angle
        angle_degrees = self.degrees_per_second * (dt)
        angle_radians = np.deg2rad(angle_degrees)
        # Compute rotation matrix using Rodrigues' rotation formula
        K = np.array(
            [
                [0, -self.axis[2], self.axis[1]],
                [self.axis[2], 0, -self.axis[0]],
                [-self.axis[1], self.axis[0], 0],
            ]
        )
        R = (
            np.eye(3)
            + np.sin(angle_radians) * K
            + (1 - np.cos(angle_radians)) * (K @ K)
        )
        # Translate vertices to origin
        V_translated = V - self.origin[:, np.newaxis]
        # Apply rotation
        V_rotated = R @ V_translated
        # Translate back
        V_final = V_rotated + self.origin[:, np.newaxis]
        return V_final

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

    def specific_apply(self, t, dt, V):
        """Apply the translation to the given vertices.

        Args:
            t (int): The current time step.
            dt (float): The time step size.
            V (np.ndarray): `|# dims| x |# nodes|` the vertex positions.

        Returns:
            np.ndarray: The transformed vertex positions.
        """
        # Compute translation distance
        distance = self.speed * dt
        translation_vector = self.direction * distance
        # Apply translation
        V_translated = V + translation_vector[:, np.newaxis]
        return V_translated

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


class FixedTransform(PrimitiveTransform):
    """Keep vertices fixed"""

    def __init__(self, name, begin, duration):
        super().__init__(name, begin, duration, TransformType.FIXED)

    @staticmethod
    def make_default():
        return FixedTransform("New Fixed", 0, 1)

    def specific_apply(self, t, dt, V):
        return V

    def serialize(self, h5group):
        return super().serialize(h5group)

    def deserialize(self, h5group):
        return super().deserialize(h5group)


class TransformLibrary:
    transforms: list[PrimitiveTransform]
    _recycled_indices: list[int]
    _current_transform_index: int
    _mesh_names: list[str]

    def __init__(self):
        self.transforms = []
        self._recycled_indices = []
        self._current_transform_index = 0
        self._mesh_names = []

    def build_transform_map(self):
        self._transform_map = {}
        for ttype in TransformType:
            self._transform_map[ttype] = []

    def on_mesh_added(self, mesh_name: str):
        self._mesh_names.append(mesh_name)
        for transform in self.transforms:
            transform.on_mesh_added(mesh_name)

    def on_mesh_removed(self, mesh_name: str):
        self._mesh_names.remove(mesh_name)
        for transform in self.transforms:
            transform.on_mesh_removed(mesh_name)

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

    def apply(self, name: str, V: np.ndarray, t: int, dt: float) -> np.ndarray:
        for transform in self.transforms:
            if transform.applicable(t, dt) and transform.affects(name):
                dnodes = transform.nodes(name)
                V[:, dnodes] = transform.apply(t, dt, V[:, dnodes])
        return V

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

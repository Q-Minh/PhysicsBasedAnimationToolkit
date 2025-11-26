# type: ignore
import h5py
import numpy as np
from enum import Enum
from copy import deepcopy
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
    COMPOSITE = 4


class PrimitiveTransform:
    def __init__(
        self, name: str, begin: float, duration: float, transform_type: TransformType
    ):
        self.name = name
        self.begin = begin
        self.duration = duration
        self.transform_type = (
            transform_type  # e.g., "rotate", "revolve", "translate", "empty"
        )
        self.id = -1  # to be set when added to library

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
        elif ttype == TransformType.COMPOSITE:
            return CompositeTransform.make_default()
        else:
            raise ValueError(f"Unknown TransformType {ttype}")

    def applicable(self, t: float) -> bool:
        """Check if the transform is applicable at time t"""
        return self.begin <= t <= self.duration + self.begin

    def expired(self, t: float) -> bool:
        """Check if the transform has expired at time t"""
        return t > self.duration + self.begin

    def apply(self, t: float, dt: float, V: np.ndarray) -> np.ndarray:
        """
        Apply the transform to the vertices V at time t * dt.
        V: (n, 3) array of vertex positions
        Returns transformed vertices (n, 3)
        """
        if not self.applicable(t * dt):
            return V  # no transformation outside the interval
        # Implement specific transformations in subclasses
        return self.specific_apply(t, dt, V)

    def specific_apply(self, t: float, dt: float, V: np.ndarray) -> np.ndarray:
        raise NotImplementedError("specific_apply must be implemented in subclasses")

    def adjust(self):
        """Adjust parameters if necessary (e.g., normalize axes)"""
        pass

    def serialize(self, h5group: h5py.Group):
        """
        Serialize the transform to an HDF5 group.
        """
        h5group.attrs["name"] = self.name
        h5group.attrs["begin"] = self.begin
        h5group.attrs["duration"] = self.duration
        h5group.attrs["transform_type"] = self.transform_type.value
        h5group.attrs["id"] = self.id

    def __str__(self):
        return f"Name: {self.name}, Type: {self.transform_type.name},\n\t ID: {self.id}, \n\t\t Time: [{self.begin}, {self.begin + self.duration}]"


class GlobalRotateTransform(PrimitiveTransform):
    """Rotate all vertices around a global axis"""

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
        V_rotated = (V.T @ R.T).T
        return V_rotated

    def serialize(self, h5group: h5py.Group):
        super().serialize(h5group)
        h5group.attrs["axis"] = self.axis
        h5group.attrs["degrees_per_second"] = self.degrees_per_second

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}\n\t\t Axis: {self.axis},\n\t\t Degrees per second: {self.degrees_per_second}"

    def adjust(self):
        self.axis = self.axis / np.linalg.norm(self.axis)


class LocalRotateTransform(PrimitiveTransform):
    """Rotate all vertices around a local axis and origin"""

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

    def specific_apply(self, t, dt, V):
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
        V_translated = V - self.origin
        # Apply rotation
        V_rotated = (V_translated.T @ R.T).T
        # Translate back
        V_final = V_rotated + self.origin
        return V_final

    def serialize(self, h5group: h5py.Group):
        super().serialize(h5group)
        h5group.attrs["axis"] = self.axis
        h5group.attrs["origin"] = self.origin
        h5group.attrs["degrees_per_second"] = self.degrees_per_second

    def adjust(self):
        self.axis = self.axis / np.linalg.norm(self.axis)

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}\n\t\t Axis: {self.axis},\n\t\t Origin: {self.origin},\n\t\t Degrees per second: {self.degrees_per_second}"


class TranslateTransform(PrimitiveTransform):
    """Translate all vertices along a direction"""

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
        # Compute translation distance
        distance = self.speed * (dt)
        translation_vector = self.direction * distance
        # Apply translation
        V_translated = (V.T + translation_vector.T).T
        return V_translated

    def serialize(self, h5group: h5py.Group):
        super().serialize(h5group)
        h5group.attrs["direction"] = self.direction
        h5group.attrs["speed"] = self.speed

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


class CompositeTransform(PrimitiveTransform):
    """Assemble multiple primitive transforms sequentially"""

    def __init__(self, name: str, begin: float):
        super().__init__(
            name, begin, 0, TransformType.COMPOSITE
        )  # Composite is not a primitive type
        self.transforms = []  # list of PrimitiveTransform instances

    @staticmethod
    def make_default():
        return CompositeTransform("New Composite", 0)

    def add_transform(self, transform: PrimitiveTransform):
        stored_transform = deepcopy(transform)
        stored_transform.begin = self.begin + self.duration
        self.duration += stored_transform.duration
        self.transforms.append(stored_transform)

    def specific_apply(self, t: float, dt: float, V: np.ndarray) -> np.ndarray:
        V_transformed = V
        for transform in self.transforms:
            V_transformed = transform.apply(t, dt, V_transformed)
        return V_transformed


class TransformLibrary:
    transforms: list[PrimitiveTransform]
    _recycled_indices: list[int]
    _current_transform_index: int

    def __init__(self):
        self.transforms = []
        self._recycled_indices = []
        self._current_transform_index = 0

    def build_transform_map(self):
        self._transform_map = {}
        for ttype in TransformType:
            self._transform_map[ttype] = []

    def add_transform(self, transform: PrimitiveTransform):
        transform.adjust()
        transform.id = self._get_new_id() + 1
        self.transforms.append(transform)

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
                    self.transforms.pop(idx)
                    self._recycled_indices.append(idx)
                imgui.TreePop()

    def serialize(self, grp: h5py.Group):
        """
        Serialize the entire library to an HDF5 group.
        """
        for transform in self.transforms:
            tgrp = grp.create_group(f"{transform.id}")
            transform.serialize(tgrp)

    def deserialize(self, grp: h5py.Group):
        """
        Deserialize the library from an HDF5 group.
        """
        for tname, tgroup in grp.items():
            name = tgroup.attrs["name"]
            begin = tgroup.attrs["begin"]
            duration = tgroup.attrs["duration"]
            transform_type = TransformType(tgroup.attrs["transform_type"])
            if transform_type == TransformType.G_ROTATE:
                axis = tgroup.attrs["axis"]
                degrees_per_second = tgroup.attrs["degrees_per_second"]
                transform = GlobalRotateTransform(
                    name, begin, duration, axis, degrees_per_second
                )
            elif transform_type == TransformType.L_ROTATE:
                axis = tgroup.attrs["axis"]
                origin = tgroup.attrs["origin"]
                degrees_per_second = tgroup.attrs["degrees_per_second"]
                transform = LocalRotateTransform(
                    name, begin, duration, axis, origin, degrees_per_second
                )
            elif transform_type == TransformType.TRANSLATE:
                direction = tgroup.attrs["direction"]
                speed = tgroup.attrs["speed"]
                transform = TranslateTransform(name, begin, duration, direction, speed)
            elif transform_type == TransformType.FIXED:
                transform = FixedTransform(name, begin, duration)
            else:
                raise ValueError(
                    f"Unknown transform type {transform_type} for transform {name}"
                )
            transform.id = tgroup.attrs["id"]
            self.add_transform(transform)

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

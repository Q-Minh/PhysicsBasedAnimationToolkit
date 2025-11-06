import h5py
import numpy as np
from enum import Enum
"""
primitive operations
all primitives require:
- 1 interval of time (begin-end in seconds)
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
    
class PrimitiveTransform:
    def __init__(self, name: str, begin: float, end: float, transform_type: TransformType):
        self.name = name
        self.begin = begin
        self.end = end
        self.transform_type = transform_type  # e.g., "rotate", "revolve", "translate", "empty"
        self.id = -1  # to be set when added to library

    def apply(self, t: float, V: np.ndarray) -> np.ndarray:
        """
        Apply the transform to the vertices V at time t.
        V: (n, 3) array of vertex positions
        Returns transformed vertices (n, 3)
        """
        if t < self.begin or t > self.end:
            return V  # no transformation outside the interval
        # Implement specific transformations in subclasses
        return self.specific_apply(t, V)
    
    def specific_apply(self, t: float, V: np.ndarray) -> np.ndarray:
        raise NotImplementedError("specific_apply must be implemented in subclasses")
    
    def adjust(self):
        """ Adjust parameters if necessary (e.g., normalize axes)"""
        raise NotImplementedError("adjust must be implemented in subclasses")
    
    def serialize(self, h5group: h5py.Group):
        """
        Serialize the transform to an HDF5 group.
        """
        h5group.attrs['name'] = self.name
        h5group.attrs['begin'] = self.begin
        h5group.attrs['end'] = self.end
        h5group.attrs['transform_type'] = self.transform_type.value
        h5group.attrs['id'] = self.id
    
    def __str__(self):
        return f"Name: {self.name}, Type: {self.transform_type.name},\n\t ID: {self.id}, \n\t\t Time: [{self.begin}, {self.end}]"
    
class GlobalRotateTransform(PrimitiveTransform):
    def __init__(self, name: str, begin: float, end: float, axis: np.ndarray, degrees_per_second: float):
        super().__init__(name, begin, end, TransformType.G_ROTATE)
        self.axis = axis / np.linalg.norm(axis)  # normalize axis
        self.degrees_per_second = degrees_per_second

    def __init__(self):
        super().__init__("New Global Rotate", 0, 1, TransformType.G_ROTATE)
        self.axis = np.array([1,0,0])
        self.degrees_per_second = 10

    def specific_apply(self, t, V):
        # Compute rotation angle
        angle_degrees = self.degrees_per_second * (t - self.begin)
        angle_radians = np.deg2rad(angle_degrees)
        # Compute rotation matrix using Rodrigues' rotation formula
        K = np.array([[0, -self.axis[2], self.axis[1]],
                      [self.axis[2], 0, -self.axis[0]],
                      [-self.axis[1], self.axis[0], 0]])
        R = np.eye(3) + np.sin(angle_radians) * K + (1 - np.cos(angle_radians)) * (K @ K)
        # Apply rotation
        V_rotated = V @ R.T
        return V_rotated
    
    def serialize(self, h5group: h5py.Group):
        super().serialize(h5group)
        h5group.attrs['axis'] = self.axis
        h5group.attrs['degrees_per_second'] = self.degrees_per_second

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}\n\t\t Axis: {self.axis},\n\t\t Degrees per second: {self.degrees_per_second}"
    
    def adjust(self):
        self.axis = self.axis / np.linalg.norm(self.axis)

class LocalRotateTransform(PrimitiveTransform):
    def __init__(self, name: str, begin: float, end: float, axis: np.ndarray, origin: np.ndarray, degrees_per_second: float):
        super().__init__(name, begin, end, TransformType.L_ROTATE)
        self.axis = axis / np.linalg.norm(axis)  # normalize axis
        self.origin = origin
        self.degrees_per_second = degrees_per_second

    def __init__(self):
        super().__init__("New Local Rotate", 0, 1, TransformType.L_ROTATE)
        self.axis = np.array([1,0,0])
        self.origin = np.array([0,0,0])
        self.degrees_per_second = 10

    def adjust(self):
        self.axis = self.axis / np.linalg.norm(self.axis)

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}\n\t\t Axis: {self.axis},\n\t\t Origin: {self.origin},\n\t\t Degrees per second: {self.degrees_per_second}"

class TranslateTransform(PrimitiveTransform):
    def __init__(self, name: str, begin: float, end: float, direction: np.ndarray, speed: float):
        super().__init__(name, begin, end, TransformType.TRANSLATE)
        self.direction = direction / np.linalg.norm(direction)  # normalize direction
        self.speed = speed  # units per second

    def __init__(self):
        super().__init__("New Translation", 0, 1, TransformType.TRANSLATE)
        self.direction = np.array([1,0,0])
        self.speed = 10  # units per second

    def adjust(self):
        self.direction = self.direction / np.linalg.norm(self.direction)

    def __str__(self):
        base_str = super().__str__()
        return f"{base_str}\n\t\t Direction: {self.direction},\n\t\t Speed: {self.speed}"


class TransformLibrary:
    def __init__(self):
        self.transforms = []
        # Same as transforms but with transform type as keys
        self.tranfrorm_map = {TransformType.G_ROTATE: [],
                              TransformType.L_ROTATE: [],
                              TransformType.TRANSLATE: []} 

    def add_transform(self, transform: PrimitiveTransform):
        transform.adjust()
        if transform.id == -1:
           transform.id = len(self.transforms) + 1
        self.transforms.append(transform)
        self.tranfrorm_map[transform.transform_type].append(transform)

    def serialize(self, path: str):
        """
        Serialize the entire library to an HDF5 group.
        """
        with h5py.File(path, 'w') as h5file:
            for transform in self.transforms:
                tgroup = h5file.create_group(f'transform_{transform.id}')
                transform.serialize(tgroup)

    def deserialize(self, path: str):
        """
        Deserialize the library from an HDF5 group.
        """
        with h5py.File(path, 'r') as h5file:
            self.transforms = []
            self.tranfrorm_map = {TransformType.G_ROTATE: [],
                                  TransformType.L_ROTATE: [],
                                  TransformType.TRANSLATE: []} 
            for tname, tgroup in h5file.items():
                name = tgroup.attrs['name']
                begin = tgroup.attrs['begin']
                end = tgroup.attrs['end']
                transform_type = TransformType(tgroup.attrs['transform_type'])
                if transform_type == TransformType.G_ROTATE:
                    axis = tgroup.attrs['axis']
                    degrees_per_second = tgroup.attrs['degrees_per_second']
                    transform = GlobalRotateTransform(name, begin, end, axis, degrees_per_second)
                elif transform_type == TransformType.L_ROTATE:
                    axis = tgroup.attrs['axis']
                    origin = tgroup.attrs['origin']
                    degrees_per_second = tgroup.attrs['degrees_per_second']
                    transform = LocalRotateTransform(name, begin, end, axis, origin, degrees_per_second)
                elif transform_type == TransformType.TRANSLATE:
                    direction = tgroup.attrs['direction']
                    speed = tgroup.attrs['speed']
                    transform = TranslateTransform(name, begin, end, direction, speed)
                else:
                    print(f"Unknown transform type {transform_type} for transform {name}")
                    continue
                transform.id = tgroup.attrs['id']
                self.add_transform(transform)
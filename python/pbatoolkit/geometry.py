from ._pbat import geometry as _geometry
import numpy as np
from enum import Enum

def aabb(P: np.ndarray):
    dims = P.shape[0]
    if dims != 2 and dims != 3:
        raise ValueError(f"Expected points P with dimensions (i.e. rows) 2 or 3, but got {dims}")
    class_ = getattr(_geometry, f"AxisAlignedBoundingBox{dims}")
    return class_(P)

class Cell(Enum):
    Triangle = 0
    Quadrilateral = 1
    Tetrahedron = 2
    Hexahedron = 3


def bvh(V: np.ndarray, C: np.ndarray, cell: Cell, max_points_in_leaf=10):
    if cell == Cell.Quadrilateral or cell == Cell.Hexahedron:
        raise ValueError(
            f"{cell} meshes not supported yet for BVH construction")
    dims = V.shape[0]
    if cell == Cell.Triangle:
        if dims == 2:
            return _geometry.TriangleAabbHierarchy2D(V, C, max_points_in_leaf)
        elif dims == 3:
            return _geometry.TriangleAabbHierarchy3D(V, C, max_points_in_leaf)
        else:
            raise ValueError(
                f"Expected 2 or 3 dimensional positions, but got {dims}")
    if cell == Cell.Tetrahedron:
        if dims == 3:
            return _geometry.TetrahedralAabbHierarchy(V, C, max_points_in_leaf)
        else:
            raise ValueError(
                f"Expected 3 dimensional positions, but got {dims}")

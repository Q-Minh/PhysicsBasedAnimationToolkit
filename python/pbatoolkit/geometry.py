from ._pbat import geometry as _geometry
import numpy as np
from enum import Enum

def aabb(P: np.ndarray):
    """Computes the axis aligned boundary box of the input points

    Args:
        P (np.ndarray): |#dims|x|#points| array of point positions

    Raises:
        ValueError: Only 2D and 3D positions are currently supported

    Returns:
        The axis aligned bounding box of P
    """
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
    """Computes the axis-aligned bounding box hierarchy of the mesh (V,C)

    Args:
        V (np.ndarray): |#dims|x|#vertices| mesh vertex positions. Requires column storage and 64-bit floats.
        C (np.ndarray): |#cell vertices|x|#cells| mesh cell vertex indices into V. Requires column storage and 64-bit integers.
        cell (Cell): The type of cell composing the mesh (V,C)
        max_points_in_leaf (int, optional): Maximum number of cells at BVH leaves. Defaults to 10.

    Raises:
        ValueError: Only Triangle and Tetrahedron cells are currently supported.
        ValueError: Only embedding dimensions 2 or 3 are currently supported.
        ValueError: Only embedding dimensions 2 or 3 are currently supported.

    Returns:
        The BVH over cells in mesh (V,C)
    """
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

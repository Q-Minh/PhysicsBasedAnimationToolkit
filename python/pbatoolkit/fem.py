import ._pbat
import numpy as np
from enum import Enum


class Element(Enum):
    Line = 0
    Triangle = 1
    Quadrilateral = 2
    Tetrahedron = 3
    Hexahedron = 4


def _mesh_type_name(element: str, order: int, dims: int):
    class_name = f"Mesh_{element.lower()}_Order_{order}_Dims_{dims}"
    return class_name


def mesh(V: np.ndarray, C: np.ndarray, element: Element, order: int = 1):
    dims = V.shape[0]
    class_name = _mesh_type_name(element.name, order, dims)
    class_ = getattr(_pbat, class_name)
    return class_(V, C)


class HyperElasticEnergy(Enum):
    StVk = 0
    StableNeohookean = 1


def hyper_elastic_potential(
        mesh,
        detJe: np.ndarray,
        GNe: np.ndarray,
        Y: np.ndarray,
        nu: np.ndarray,
        psi: HyperElasticEnergy = HyperElasticEnergy.StableNeohookean,
        quadrature_order: int = 1,
        dims: int = 3):
    mesh_name = _mesh_type_name(
        mesh.element, mesh.order, mesh.dims)
    class_name = f"HyperElasticPotential_{psi.name}_QuadratureOrder_{quadrature_order}_Dims_{dims}_{mesh_name}"
    class_ = getattr(_pbat, class_name)
    return class_(mesh, detJe, GNe, Y, nu)

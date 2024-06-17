from ._pbat import fem as _fem
import numpy as np
from enum import Enum


class Element(Enum):
    Line = 0
    Triangle = 1
    Quadrilateral = 2
    Tetrahedron = 3
    Hexahedron = 4


def _mesh_type_name(mesh, element: str = None, order: int = None, dims: int = None):
    if mesh is not None:
        return f"Mesh_{mesh.element.lower()}_Order_{mesh.order}_Dims_{mesh.dims}"
    class_name = f"Mesh_{element.lower()}_Order_{order}_Dims_{dims}"
    return class_name


def mesh(V: np.ndarray, C: np.ndarray, element: Element, order: int = 1):
    """Construct an FEM mesh from an input mesh

    Args:
        V (np.ndarray): |#dims|x|#vertices| mesh vertex positions
        C (np.ndarray): |#cell nodes|x|#cells| mesh cell indices into V
        element (Element): Type of element
        order (int, optional): Element shape function order. Defaults to 1.

    Returns:
        The FEM mesh of the given order
    """
    dims = V.shape[0]
    class_name = _mesh_type_name(None, element.name, order, dims)
    class_ = getattr(_fem, class_name)
    return class_(V, C)


def galerkin_gradient_matrix(mesh, detJe: np.ndarray, GNe: np.ndarray, quadrature_order: int = 1):
    """Computes the Galerkin gradient operator acting on the FEM mesh's function space.

    Args:
        mesh: The FEM mesh
        detJe (np.ndarray): The jacobian determinants evaluated at points of the specified 
        quadrature.
        GNe (np.ndarray): The shape function gradients evaluated at points of the specified
        quadrature.
        quadrature_order (int, optional): Polynomial quadrature to use. Defaults to 1.

    Returns:
        The Galerkin gradient operator
    """
    mesh_name = _mesh_type_name(mesh)
    class_name = f"GalerkinGradientMatrix_QuadratureOrder_{quadrature_order}_{mesh_name}"
    class_ = getattr(_fem, class_name)
    return class_(mesh, detJe, GNe)


def jacobian_determinants(mesh, quadrature_order: int = 1) -> np.ndarray:
    """Computes determinants of affine element jacobians, 
    i.e. the jacobians that map from the reference element to material space.

    Args:
        mesh: The FEM mesh
        quadrature_order (int, optional): Specifies the polynomial quadrature 
        to use, such that the jacobian determinants are computed at each quadrature 
        point. Defaults to 1.

    Returns:
        np.ndarray: |#element quadrature points|x|#elements| array of jacobian determinants
    """
    mesh_name = _mesh_type_name(mesh)
    function_name = f"jacobian_determinants_{mesh_name}"
    function_ = getattr(_fem, function_name)
    return function_(mesh, quadrature_order)


def reference_positions(mesh, E: np.ndarray, X: np.ndarray, max_iterations: int = 5, epsilon: float = 1e-10):
    """Computes reference positions of domain positions X in corresponding elements E, using Gauss-Newton.

    Args:
        mesh: The FEM mesh
        E (np.ndarray): 1D index array of element indices
        X (np.ndarray): |#dims|x|E.shape[0]| matrix of domain positions in corresponding elements in E
        max_iterations (int, optional): Max number of Gauss-Newton iterations. Defaults to 5.
        epsilon (float, optional): Residual early out. Defaults to 1e-10.

    Returns:
        np.ndarray: The reference positions in elements E corresponding to domain positions X in the mesh.
    """
    mesh_name = _mesh_type_name(mesh)
    function_name = f"reference_positions_{mesh_name}"
    function_ = getattr(_fem, function_name)
    return function_(mesh, E, X, max_iterations, epsilon)


def integrated_shape_functions(mesh, detJe: np.ndarray, quadrature_order: int = 1) -> np.ndarray:
    """Integrates all element shape functions via polynomial quadrature rule

    Args:
        mesh: The FEM mesh
        detJe (np.ndarray): The jacobian determinants evaluated at points of the specified 
        quadrature.
        quadrature_order (int, optional): Polynomial quadrature to use. Defaults to 1.

    Returns:
        np.ndarray: |#element nodes| x |#elements| array of integrated shape functions, i.e. 
        IN[i,e] = \\int_{\\Omega^e} N_i^e(\\X) \\partial \\Omega^e, for i^{th} shape function 
        of element e.
    """
    mesh_name = _mesh_type_name(mesh)
    function_name = f"integrated_shape_functions_{mesh_name}"
    function_ = getattr(_fem, function_name)
    return function_(mesh, detJe, quadrature_order)


def shape_function_gradients(mesh, quadrature_order: int = 1) -> np.ndarray:
    """Computes shape function gradients at all quadrature points. 
    Note that the mesh elements need to be linear transformations on the 
    reference elements for this method to work properly, even if elements 
    themselves support higher order shape functions.

    Args:
        mesh: The FEM mesh
        quadrature_order (int, optional): Polynomial quadrature rule to use. Defaults to 1.

    Returns:
        np.ndarray: |#element nodes|x|#dims * #element quad. pts. * #elements| matrix of 
        shape functions, i.e. if offset=e*dims*nquad, then the block 
        GNe[i,offset+g*dims:offset+(g+1)*dims] gives the i^{th} shape function gradient of 
        element e.
    """
    mesh_name = _mesh_type_name(mesh)
    function_name = f"shape_function_gradients_{mesh_name}"
    function_ = getattr(_fem, function_name)
    return function_(mesh, quadrature_order)


def shape_functions_at(mesh, Xi: np.ndarray):
    """Computes shape functions at reference positions Xi (i.e. positions in element space) for the given mesh.

    Args:
        mesh: The FEM mesh
        Xi (np.ndarray): Positions in element reference space
    """
    mesh_name = _mesh_type_name(mesh)
    function_name = f"shape_functions_at_{mesh_name}"
    function_ = getattr(_fem, function_name)
    return function_(mesh, Xi)


def laplacian_matrix(mesh, detJe: np.ndarray, GNe: np.ndarray, dims: int = 1, quadrature_order: int = 1):
    """Computes the input mesh's (negative semi-definite) symmetric part of the Laplacian matrix. 

    Args:
        mesh: The FEM mesh
        detJe (np.ndarray): The jacobian determinants evaluated at points of the specified 
        quadrature.
        GNe (np.ndarray): The shape function gradients evaluated at points of the specified
        quadrature.
        dims (int, optional): Image dimensionality of FEM function space. Defaults to 1.
        quadrature_order (int, optional): Polynomial quadrature rule to use. Defaults to 1.

    Returns:
        The negative semi-definite symmetric part of the FEM mesh's Laplacian matrix
    """
    mesh_name = _mesh_type_name(mesh)
    class_name = f"SymmetricLaplacianMatrix_QuadratureOrder_{quadrature_order}_{mesh_name}"
    class_ = getattr(_fem, class_name)
    L = class_(mesh, detJe, GNe)
    L.dims = dims
    return L


def mass_matrix(mesh, detJe: np.ndarray, rho: float = 1., dims: int = 3, quadrature_order: int = 2):
    """Computes the input mesh's mass matrix

    Args:
        mesh: The FEM mesh
        detJe (np.ndarray): Element jacobian determinants at quadrature points
        rho (float, optional): Uniform mass density (float) or per-element mass density (np.ndarray). Defaults to 1.
        dims (int, optional): dims (int, optional): Image dimensionality of FEM function space. Defaults to 3.
        quadrature_order (int, optional): Polynomial quadrature order to use for mass matrix computation. Defaults to 2.

    Returns:
        The mass matrix operator
    """
    mesh_name = _mesh_type_name(mesh)
    class_name = f"MassMatrix_QuadratureOrder_{quadrature_order}_{mesh_name}"
    class_ = getattr(_fem, class_name)
    M = class_(mesh, detJe, rho)
    M.dims = dims
    return M


def load_vector(mesh, detJe: np.ndarray, fe: np.ndarray, quadrature_order: int = 1):
    mesh_name = _mesh_type_name(mesh)
    dims = fe.shape[0]
    class_name = f"LoadVector_QuadratureOrder_{quadrature_order}_{mesh_name}"
    class_ = getattr(_fem, class_name)
    return class_(mesh, detJe, fe, dims)


class HyperElasticEnergy(Enum):
    StVk = 0
    StableNeoHookean = 1


def hyper_elastic_potential(
        mesh,
        detJe: np.ndarray,
        GNe: np.ndarray,
        Y: np.ndarray,
        nu: np.ndarray,
        psi: HyperElasticEnergy = HyperElasticEnergy.StableNeoHookean,
        quadrature_order: int = 1):
    """Constructs the input mesh's hyper elastic potential, which can be used to evaluate 
    the potential, its gradient given some state vector (i.e. the DOFs).

    Args:
        mesh: The FEM mesh
        detJe (np.ndarray): Element jacobian determinants at quadrature points
        GNe (np.ndarray): The shape function gradients evaluated at points of the specified
        quadrature.
        Y (np.ndarray): Array of per-element Young's modulus
        nu (np.ndarray): Array of per-element Poisson's ratio
        psi (HyperElasticEnergy, optional): The type of hyper elastic energy. Defaults to HyperElasticEnergy.StableNeohookean.
        quadrature_order (int, optional): Polynomial quadrature order to use. Defaults to 1.
        dims (int, optional): The problem's dimensionality. Defaults to 3.

    Returns:
        A hyper elastic potential instance
    """
    mesh_name = _mesh_type_name(mesh)
    class_name = f"HyperElasticPotential_{psi.name}_QuadratureOrder_{quadrature_order}_Dims_{mesh.dims}_{mesh_name}"
    class_ = getattr(_fem, class_name)
    return class_(mesh, detJe, GNe, Y, nu)

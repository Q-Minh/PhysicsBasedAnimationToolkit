from ._pbat import fem as _fem
import sys
import inspect
import numpy as np
import scipy as sp
import math
import contextlib
import io

__module = sys.modules[__name__]
for _name, _attr in inspect.getmembers(_fem):
    if not _name.startswith("__"):
        setattr(__module, _name, _attr)

_strio = io.StringIO()
with contextlib.redirect_stdout(_strio):
    help(_fem)
_strio.seek(0)
setattr(__module, "__doc__", f"{getattr(__module, "__doc__")}\n\n{_strio.read()}")

def divergence(mesh, quadrature_order: int = 1, GNe: np.ndarray = None):
    """Construct an FEM divergence operator such that composing div(grad(u)) 
    computes the Laplacian of u, i.e. D @ G = L, where L is the FEM Laplacian 
    operator.

    Args:
        mesh (pbat.fem.Mesh): 
        quadrature_order (int, optional): Polynomial order to use for operator construction. Defaults to 1.
        GNe (np.ndarray, optional): Shape function gradients at quadrature points. Defaults to None.

    Returns:
        (scipy.sparse.csc_matrix, np.ndarray): 
    """
    G, GNe = gradient(mesh, quadrature_order=quadrature_order,
                      GNe=GNe, as_matrix=True)
    qgL = _fem.inner_product_weights(
        mesh, quadrature_order=quadrature_order).flatten(order="F")
    QL = sp.sparse.diags_array([qgL], offsets=[0])
    QL = sp.sparse.kron(sp.sparse.eye_array(mesh.dims), QL)
    D = -G.T @ QL
    return D, GNe


def gradient(mesh, quadrature_order: int = 1, GNe: np.ndarray = None, as_matrix: bool = True):
    """Construct an FEM gradient operator

    Args:
        mesh (pbat.fem.Mesh): 
        quadrature_order (int, optional): Polynomial order to use for operator construction. Defaults to 1.
        GNe (np.ndarray, optional): Shape function gradients at quadrature points. Defaults to None.
        as_matrix (bool, optional): Return the operator as a sparse matrix. Defaults to True.

    Returns:
        (scipy.sparse.csc_matrix | pbat.fem.Gradient, np.ndarray): 
    """
    if GNe is None:
        GNe = _fem.shape_function_gradients(
            mesh, quadrature_order=quadrature_order)
    G = _fem.Gradient(mesh, GNe, quadrature_order=quadrature_order)
    G = G.to_matrix() if as_matrix else G
    return G, GNe


def hyper_elastic_potential(
        mesh,
        Y: float | np.ndarray = 1e6,
        nu: float | np.ndarray = 0.45,
        energy=_fem.HyperElasticEnergy.StableNeoHookean,
        precompute_sparsity: bool = True,
        quadrature_order: int = 1,
        detJe: np.ndarray = None,
        GNe: np.ndarray = None):
    """Construct an FEM hyper elastic potential

    Args:
        mesh (pbat.fem.Mesh): 
        Y (float | np.ndarray, optional): Uniform (float) or heterogeneous (|#quad.pts.|x|#elements| array) Young's modulus. Defaults to 1e6.
        nu (float | np.ndarray, optional): Uniform (float) or heterogeneous (|#quad.pts.|x|#elements| array) Poisson ratio. Defaults to 0.45.
        energy (pbat.fem.HyperElasticEnergy, optional): Constitutive model. Defaults to pbat.fem.HyperElasticEnergy.StableNeoHookean.
        precompute_sparsity (bool, optional): Precompute an acceleration data structure for fast hessian construction. Defaults to True.
        quadrature_order (int, optional): Polynomial order to use for potential (and its derivatives) evaluation. Defaults to 1.
        detJe (np.ndarray, optional): Jacobian determinants at quadrature points. Defaults to None.
        GNe (np.ndarray, optional): Shape function gradients at quadrature points. Defaults to None.

    Returns:
        (pbat.fem.HyperElasticPotential, np.ndarray, np.ndarray): 
    """
    if detJe is None:
        detJe = _fem.jacobian_determinants(
            mesh, quadrature_order=quadrature_order)
    if GNe is None:
        GNe = _fem.shape_function_gradients(
            mesh, quadrature_order=quadrature_order)
    hep = _fem.HyperElasticPotential(
        mesh, detJe, GNe, Y, nu, energy=energy, quadrature_order=quadrature_order)
    if precompute_sparsity:
        hep.precompute_hessian_sparsity()
    return hep, detJe, GNe


def laplacian(
        mesh,
        dims: int = 1,
        quadrature_order: int = 1,
        detJe: np.ndarray = None,
        GNe: np.ndarray = None,
        as_matrix: bool = True):
    """Construct an FEM Laplacian operator

    Args:
        mesh (pbat.fem.Mesh): 
        dims (int, optional): Solution space dimensions. Defaults to 1.
        quadrature_order (int, optional): Polynomial order to use for operator construction. Defaults to 1.
        detJe (np.ndarray, optional): Jacobian determinants at quadrature points. Defaults to None.
        GNe (np.ndarray, optional): Shape function gradients at quadrature points. Defaults to None.
        as_matrix (bool, optional): Return the operator as a sparse matrix. Defaults to True.

    Returns:
        (pbat.fem.Laplacian | scipy.sparse.csc_matrix, np.ndarray, np.ndarray): 
    """
    if detJe is None:
        detJe = _fem.jacobian_determinants(
            mesh, quadrature_order=quadrature_order)
    if GNe is None:
        GNe = _fem.shape_function_gradients(
            mesh, quadrature_order=quadrature_order)
    L = _fem.Laplacian(mesh, detJe, GNe, dims=dims,
                       quadrature_order=quadrature_order)
    L = L.to_matrix() if as_matrix else L
    return L, detJe, GNe


def mass_matrix(
        mesh,
        rho: float | np.ndarray = 1.0,
        dims: int = 3,
        quadrature_order: int = 2,
        lump: bool = False,
        detJe: np.ndarray = None,
        as_matrix: bool = True):
    """Construct an FEM mass matrix operator

    Args:
        mesh (pbat.fem.Mesh): 
        rho (float | np.ndarray, optional): Uniform (float) or heterogeneous (|#quad.pts.|x|#elements| array) mass density. Defaults to 1.0.
        dims (int, optional): Solution space dimensions. Defaults to 3.
        quadrature_order (int, optional): Polynomial order to use for operator construction. Defaults to 1.
        lump (bool, optional): Lump the mass matrix, if as_matrix==True. Defaults to False.
        detJe (np.ndarray, optional): Jacobian determinants at quadrature points. Defaults to None.
        as_matrix (bool, optional): Return the operator as a sparse matrix. Defaults to True.

    Returns:
        (pbat.fem.MassMatrix | scipy.sparse.csc_matrix, np.ndarray): 
    """
    if detJe is None:
        detJe = _fem.jacobian_determinants(
            mesh, quadrature_order=quadrature_order)
    M = _fem.MassMatrix(mesh, detJe, rho=rho, dims=dims,
                        quadrature_order=quadrature_order)
    if as_matrix:
        M = M.to_matrix()
        if lump:
            lumpedm = M.sum(axis=0)
            M = sp.sparse.spdiags(lumpedm, np.array(
                [0]), m=M.shape[0], n=M.shape[0])
    return M, detJe


def load_vector(
        mesh,
        fe: np.ndarray,
        quadrature_order: int = 1,
        detJe: np.ndarray = None,
        flatten: bool = True):
    """Construct an FEM load vector

    Args:
        mesh (pbat.fem.Mesh): 
        fe (np.ndarray): Uniform (|#dims| or |#dims|x1) or heterogeneous (|#dims|x|#mesh quadrature points|) load.
        quadrature_order (int, optional): Polynomial order to use for load vector construction. Defaults to 1.
        detJe (np.ndarray, optional): Jacobian determinants at quadrature points. Defaults to None.

    Returns:
        (np.ndarray, np.ndarray): 
    """
    if detJe is None:
        detJe = _fem.jacobian_determinants(
            mesh, quadrature_order=quadrature_order)
    nelems = mesh.E.shape[1]
    wg = np.tile(mesh.quadrature_weights(quadrature_order), nelems)
    qgf = detJe.flatten(order="F") * wg
    Qf = sp.sparse.diags_array([qgf], offsets=[0])
    Nf = _fem.shape_function_matrix(mesh, quadrature_order=quadrature_order)
    if len(fe.shape) == 1:
        fe = np.tile(fe[:, np.newaxis], Qf.shape[0])
    if fe.shape[1] == 1:
        fe = np.tile(fe, Qf.shape[0])
    f = fe @ Qf @ Nf
    if flatten:
        f = f.reshape(math.prod(f.shape), order="F")
    return f, detJe

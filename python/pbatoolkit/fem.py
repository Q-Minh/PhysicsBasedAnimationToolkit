from ._pbat import fem as _fem
import sys
import inspect
import numpy as np
import scipy as sp
import math
import contextlib
import io
from enum import Enum

__module = sys.modules[__name__]
for _name, _attr in inspect.getmembers(_fem):
    if not _name.startswith("__"):
        setattr(__module, _name, _attr)

_strio = io.StringIO()
with contextlib.redirect_stdout(_strio):
    help(_fem)
_strio.seek(0)
setattr(
    __module,
    "__doc__",
    f"{getattr(__module, '__doc__')}\n\n{_strio.read()}"
)


def divergence(mesh, quadrature_order: int = 1, eg: np.ndarray = None, wg: np.ndarray = None, GNeg: np.ndarray = None):
    """Construct an FEM divergence operator such that composing div(grad(u)) 
    computes the Laplacian of u, i.e. D @ G = L, where L is the FEM Laplacian 
    operator.

    Args:
        mesh (pbat.fem.Mesh): 
        quadrature_order (int, optional): Polynomial order to use for operator construction. Defaults to 1.
        eg (np.ndarray, optional): |#quad.pts.| array of elements corresponding to quadrature points. Defaults to None.
        wg (np.ndarray, optional): |#quad.pts.| array of quadrature weights. Defaults to None.
        GNeg (np.ndarray, optional): |#elem. nodes|x|#dims * #quad.pts.| array of shape function gradients at quadrature points. Defaults to None.

    Returns:
        (scipy.sparse.csc_matrix, np.ndarray, np.ndarray): 
    """
    should_compute_quadrature = eg is None or wg is None or GNeg is None
    G, eg, GNeg = gradient(mesh, quadrature_order=quadrature_order, as_matrix=True) if should_compute_quadrature else gradient(
        mesh, eg=eg, GNeg=GNeg, as_matrix=True)
    wg = _fem.inner_product_weights(
        mesh, quadrature_order=quadrature_order).flatten(order="F") if should_compute_quadrature else wg
    QL = sp.sparse.diags_array(np.asarray(wg).squeeze())
    QL = sp.sparse.kron(sp.sparse.eye_array(mesh.dims), QL)
    D = -G.T @ QL
    return D, eg, wg, GNeg


def gradient(mesh, quadrature_order: int = 1, eg: np.ndarray = None, GNeg: np.ndarray = None, as_matrix: bool = True):
    """Construct an FEM gradient operator

    Args:
        mesh (pbat.fem.Mesh): 
        quadrature_order (int, optional): Polynomial order to use for operator construction. Defaults to 1.
        eg (np.ndarray, optional): Elements corresponding to quadrature points. Defaults to None.
        GNeg (np.ndarray, optional): Shape function gradients at quadrature points. Defaults to None.
        as_matrix (bool, optional): Return the operator as a sparse matrix. Defaults to True.

    Returns:
        (scipy.sparse.csc_matrix | pbat.fem.Gradient, np.ndarray, np.ndarray): 
    """

    if GNeg is None:
        GNeg = _fem.shape_function_gradients(
            mesh, quadrature_order=quadrature_order)
        n_quadpts_per_element = GNeg.shape[1] / (mesh.dims * mesh.E.shape[1])
        eg = np.linspace(0, mesh.E.shape[1]-1,
                         num=mesh.E.shape[1], dtype=np.int64)
        eg = np.repeat(eg, n_quadpts_per_element)

    G = _fem.Gradient(mesh, eg, GNeg)
    G = G.to_matrix() if as_matrix else G
    return G, eg, GNeg


def hyper_elastic_potential(
        mesh,
        Y: float | np.ndarray = 1e6,
        nu: float | np.ndarray = 0.45,
        energy=_fem.HyperElasticEnergy.StableNeoHookean,
        precompute_sparsity: bool = True,
        quadrature_order: int = 1,
        eg: np.ndarray = None,
        wg: np.ndarray = None,
        GNeg: np.ndarray = None):
    """Construct an FEM hyper elastic potential

    Args:
        mesh (pbat.fem.Mesh): 
        Y (float | np.ndarray, optional): Uniform (float) or heterogeneous (|#quad.pts.|x|#elements| array) Young's modulus. Defaults to 1e6.
        nu (float | np.ndarray, optional): Uniform (float) or heterogeneous (|#quad.pts.|x|#elements| array) Poisson ratio. Defaults to 0.45.
        energy (pbat.fem.HyperElasticEnergy, optional): Constitutive model. Defaults to pbat.fem.HyperElasticEnergy.StableNeoHookean.
        precompute_sparsity (bool, optional): Precompute an acceleration data structure for fast hessian construction. Defaults to True.
        quadrature_order (int, optional): Polynomial order to use for potential (and its derivatives) evaluation. Defaults to 1.
        eg (np.ndarray, optional): Elements corresponding to each quadrature weight in wg.
        wg (np.ndarray, optional): Quadrature weights at quadrature points. Defaults to None.
        GNeg (np.ndarray, optional): Shape function gradients at quadrature points. Defaults to None.

    Returns:
        (pbat.fem.HyperElasticPotential, np.ndarray, np.ndarray, np.ndarray): 
    """
    if eg is None or wg is None or GNeg is None:
        wg = _fem.inner_product_weights(
            mesh, quadrature_order=quadrature_order)
        eg = np.linspace(0, mesh.E.shape[1]-1,
                         num=mesh.E.shape[1], dtype=np.int64)
        eg = np.repeat(eg, wg.shape[0])
        wg = wg.flatten(order="F")
        GNeg = _fem.shape_function_gradients(
            mesh, quadrature_order=quadrature_order)
    hep = _fem.HyperElasticPotential(
        mesh, eg, wg, GNeg, Y, nu, energy=energy)
    if precompute_sparsity:
        hep.precompute_hessian_sparsity()
    return hep, eg, wg, GNeg


def laplacian(
        mesh,
        dims: int = 1,
        quadrature_order: int = 1,
        eg: np.ndarray = None,
        wg: np.ndarray = None,
        GNeg: np.ndarray = None,
        as_matrix: bool = True):
    """Construct an FEM Laplacian operator

    Args:
        mesh (pbat.fem.Mesh): 
        dims (int, optional): Solution space dimensions. Defaults to 1.
        quadrature_order (int, optional): Polynomial order to use for operator construction. Defaults to 1.
        eg (np.ndarray, optional): |#quad.pts.| array of elements associated with quadrature points. Defaults to None.
        wg (np.ndarray, optional): |#quad.pts.| array of quadrature weights. Defaults to None.
        GNeg (np.ndarray, optional): Shape function gradients at quadrature points. Defaults to None.
        as_matrix (bool, optional): Return the operator as a sparse matrix. Defaults to True.

    Returns:
        (pbat.fem.Laplacian | scipy.sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray): 
    """
    if eg is None or wg is None or GNeg is None:
        wg = _fem.inner_product_weights(
            mesh, quadrature_order=quadrature_order)
        eg = np.linspace(0, mesh.E.shape[1]-1,
                         num=mesh.E.shape[1], dtype=np.int64)
        eg = np.repeat(eg, wg.shape[0])
        wg = wg.flatten(order="F")
        GNeg = _fem.shape_function_gradients(
            mesh, quadrature_order=quadrature_order)
    L = _fem.Laplacian(mesh, eg, wg, GNeg, dims=dims)
    L = L.to_matrix() if as_matrix else L
    return L, eg, wg, GNeg


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
            lumpedm = np.asarray(M.sum(axis=0)).squeeze()
            M = sp.sparse.diags_array(lumpedm)
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
    Qf = sp.sparse.diags_array(qgf)
    Nf = _fem.shape_function_matrix(mesh, quadrature_order=quadrature_order)
    if len(fe.shape) == 1:
        fe = np.tile(fe[:, np.newaxis], Qf.shape[0])
    if fe.shape[1] == 1:
        fe = np.tile(fe, Qf.shape[0])
    f = fe @ Qf @ Nf
    if flatten:
        f = f.reshape(math.prod(f.shape), order="F")
    return f, detJe


class QuadraturePointSelection(Enum):
    FromOutputQuadrature = 0
    FromInputRandomSampling = 1
    FromInputSmartSampling = 2


class QuadratureFittingStrategy(Enum):
    FitOutputQuadrature = 0
    FitInputQuadrature = 1


class QuadratureSingularityStrategy(Enum):
    Ignore = 0
    Constant = 1


def output_quadrature_points(omesh, obvh, iXg, selection=QuadraturePointSelection.FromOutputQuadrature, oorder=1):
    """Selects quadrature points on output mesh, given input mesh quadrature points iXg.

    Args:
        omesh: Output mesh
        obvh: BVH over output mesh
        iXg (np.ndarray): 3x|#input quad.pts.| array of input mesh quadrature points
        selection (QuadraturePointSelection, optional): Output quadrature point selection scheme. 
        Defaults to QuadraturePointSelection.FromOutputQuadrature.
        oorder (int, optional): Quadrature order on output mesh. Defaults to 1.

    Raises:
        ValueError: The selection scheme QuadraturePointSelection.FromInputSmartSampling is not yet supported

    Returns:
        (np.ndarray, np.ndarray): (oXg, oeg) where oXg is a 3x|#output quad.pts.| array of output quadrature points, 
        and oeg is the |#output quad.pts.| array of corresponding output elements
    """
    oXg = omesh.quadrature_points(oorder)
    n_output_elements = omesh.E.shape[1]
    if selection == QuadraturePointSelection.FromOutputQuadrature:
        return oXg
    if selection == QuadraturePointSelection.FromInputRandomSampling:
        inds = np.arange(iXg.shape[1])
        np.random.shuffle(inds)
        n_random_samples = oXg.shape[1]
        inds = inds[:n_random_samples]
        oXg = iXg[:, inds]
        oen = obvh.primitives_containing_points(oXg)
        oes = np.setdiff1d(list(range(n_output_elements)), oen)
        oXg = np.hstack((oXg, omesh.quadrature_points(1)[:, oes]))
        oeg = np.hstack((oen, oes))
        eorder = np.argsort(oeg)
        oXg = oXg[:, eorder]
        return oXg
    if selection == QuadraturePointSelection.FromInputSmartSampling:
        raise ValueError(
            f"selection={QuadraturePointSelection.FromInputSmartSampling} not yet supported")


def fit_output_quad_to_input_quad(
    imesh,
    omesh,
    ibvh,
    obvh,
    iorder=1,
    oorder=1,
    selection=QuadraturePointSelection.FromInputRandomSampling,
    fitting_strategy=QuadratureFittingStrategy.FitInputQuadrature,
    singular_strategy=QuadratureSingularityStrategy.Ignore,
    volerr=1e-3,
    numerical_zero=1e-12
):
    # Compute fine quadrature, which acts as the ground truth integral
    iwg = _fem.inner_product_weights(imesh, iorder).flatten(order="F")
    iXg = imesh.quadrature_points(iorder)
    # Select quadrature points
    oXg = output_quadrature_points(
        omesh, obvh, iXg, selection=selection, oorder=oorder)
    # Delete quadrature points that are outside the input mesh
    outside = ibvh.primitives_containing_points(oXg) < 0
    inside = ~outside
    oXg = oXg[:, inside]
    oeg = obvh.primitives_containing_points(oXg)

    # Find quadrature weights for the output quadrature points
    from .math import transfer_quadrature
    if fitting_strategy == QuadratureFittingStrategy.FitInputQuadrature:
        n_fine_elements = imesh.E.shape[1]
        n_fine_quad_pts = iXg.shape[1]
        ieg = np.repeat(np.arange(n_fine_elements, dtype=np.int64),
                        int(n_fine_quad_pts / n_fine_elements))
        ieg = ibvh.primitives_containing_points(iXg)
        iXi = _fem.reference_positions(imesh, ieg, iXg)
        oXi = _fem.reference_positions(
            imesh, ibvh.primitives_containing_points(oXg), oXg)
        ieg = obvh.primitives_containing_points(iXg)
        owg = np.zeros(oXg.shape[1])
        owg, err = transfer_quadrature(
            oeg, oXi, ieg, iXi, iwg, order=oorder, with_error=True, max_iters=100, precision=1e-12)
    elif fitting_strategy == QuadratureFittingStrategy.FitOutputQuadrature:
        ieg = obvh.primitives_containing_points(iXg)
        iXi = _fem.reference_positions(omesh, ieg, iXg)
        oXi = _fem.reference_positions(omesh, oeg, oXg)
        owg, err = transfer_quadrature(
            oeg, oXi, ieg, iXi, iwg, order=oorder, with_error=True, max_iters=100, precision=1e-12)

    # Delete insignificant quadrature points, i.e. those with vanishing weights
    nontrivial = owg > numerical_zero
    oXg = oXg[:, nontrivial]
    oeg = oeg[nontrivial]
    owg = owg[nontrivial]
    osg = np.full(owg.shape[0], False)

    if singular_strategy != QuadratureSingularityStrategy.Ignore:
        # Find singular elements
        n_output_elements = omesh.E.shape[1]
        ss = np.setdiff1d(list(range(n_output_elements)), oeg)
        # Inject a single quadrature point in each singular element
        soXg = omesh.quadrature_points(1)[:, ss]
        sowg = _fem.inner_product_weights(omesh, 1)[:, ss].flatten(order="F")
        n_non_singular = oXg.shape[1]
        n_singular = soXg.shape[1]
        osg = np.hstack((np.full(n_non_singular, False),
                         np.full(n_singular, True)))
        oXg = np.hstack((oXg, soXg))
        owg = np.hstack((owg, np.zeros(n_singular, dtype=owg.dtype)))
        oeg = np.hstack((oeg, ss))
        # Patch singular quadrature points
        if singular_strategy == QuadratureSingularityStrategy.Constant:
            volume = iwg.sum()
            # alpha * sowg.sum() = volerr*volume
            alpha = volerr*volume / sowg.sum()
            owg[n_non_singular:] = alpha*sowg

        ordering = np.argsort(oeg)
        oXg = oXg[:, ordering]
        owg = owg[ordering]
        oeg = oeg[ordering]
        osg = osg[ordering]

    return oXg, owg, oeg, osg, iXg, iwg, err

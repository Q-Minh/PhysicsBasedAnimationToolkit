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
setattr(__module, "__doc__", f"{getattr(__module, '__doc__')}\n\n{_strio.read()}")


def lame_coefficients(Y: float | np.ndarray, nu: float | np.ndarray):
    """Computes Lame coefficients from Young's modulus and Poisson's ratio.
    Args:
        Y (float | np.ndarray): Young's modulus.
        nu (float | np.ndarray): Poisson's ratio.
    Returns:
        Tuple[float, float]: (mu, lambda) where mu, lambda are the first and second Lame coefficients.
    """
    mue = Y / (2 * (1 + nu))
    lambdae = (Y * nu) / ((1 + nu) * (1 - 2 * nu))
    return mue, lambdae


def _per_element_vector_to_per_quad_pt_matrix(
    v: float | np.ndarray, egM: np.ndarray, dtype: np.dtype
):
    if isinstance(v, float):
        vg = np.full(egM.shape, v, dtype=dtype)
    elif isinstance(v, np.ndarray):
        if v.shape[0] != egM.shape[1]:
            raise ValueError(
                f"v must be a scalar or an array of length {egM.shape[1]}, got {v.shape[0]}."
            )
        else:
            vg = v[np.newaxis, :].repeat(egM.shape[0], axis=0)
    return vg


def rest_pose_hyper_elastic_modes(
    E: np.ndarray,
    X: np.ndarray,
    element: _fem.Element,
    order: int = 1,
    Y: float | np.ndarray = 1e6,
    nu: float | np.ndarray = 0.45,
    rho: float | np.ndarray = 1e-3,
    energy=_fem.HyperElasticEnergy.StableNeoHookean,
    modes: int = 30,
    sigma: float = -1e-5,
    zero: float = 0.0,
):
    """Computes natural (linear) displacement modes of mesh.

    Args:
        E (np.ndarray): Element matrix.
        X (np.ndarray): Node coordinates.
        element (_pbat.fem.Element): Element type.
        order (int, optional): Element order. Defaults to 1.
        Y (float | np.ndarray, optional): Young's modulus. Defaults to 1e6.
        nu (float | np.ndarray, optional): Poisson's ratio. Defaults to 0.45.
        rho (float | np.ndarray, optional): Mass density. Defaults to 1e-3.
        energy (_pbat.fem.HyperElasticEnergy, optional): Constitutive model. Defaults to _fem.HyperElasticEnergy.StableNeoHookean.
        modes (int, optional): Maximum number of modes to compute. Defaults to 30.
        sigma (float, optional): Shift (see scipy.sparse.eigsh). Defaults to -1e-5.
        zero (float, optional): Numerical zero used to cull modes. Defaults to 0.

    Returns:
        (np.ndarray, np.ndarray): (w,U) s.t. w is a |#modes| vector of amplitudes and
        U is a nx|#modes| array of displacement modes in columns.
    """
    # Compute elasticity at rest pose
    x = np.ravel(X, order="F")
    # Compute the mass matrix
    qorderM = 2 * order
    wgM = _fem.mesh_quadrature_weights(
        E, X, element, order=order, quadrature_order=qorderM
    )
    egM = _fem.mesh_quadrature_elements(E, wgM)
    rhog = _per_element_vector_to_per_quad_pt_matrix(rho, egM, wgM.dtype)
    Neg = _fem.shape_functions(
        n_elements=E.shape[1],
        element=element,
        order=order,
        quadrature_order=qorderM,
        dtype=wgM.dtype,
    )
    M = _fem.mass_matrix(
        E,
        X.shape[1],
        eg=np.ravel(egM),
        wg=np.ravel(wgM),
        rhog=np.ravel(rhog),
        Neg=Neg,
        dims=X.shape[0],
        element=element,
        order=order,
        spatial_dims=X.shape[0],
    )
    # Compute the hyper-elastic potential's hessian
    qorderU = order
    wgU = _fem.mesh_quadrature_weights(
        E, X, element, order=order, quadrature_order=qorderU
    )
    egU = _fem.mesh_quadrature_elements(E, wgU)
    GNegU = _fem.shape_function_gradients(
        E, X, element=element, order=order, dims=X.shape[0], quadrature_order=qorderU
    )
    mu, llambda = lame_coefficients(Y, nu)
    mug = _per_element_vector_to_per_quad_pt_matrix(mu, egU, wgU.dtype)
    lambdag = _per_element_vector_to_per_quad_pt_matrix(llambda, egU, wgU.dtype)
    HU = _fem.hyper_elastic_potential(
        E=E,
        n_nodes=X.shape[1],
        eg=np.ravel(egU),
        wg=np.ravel(wgU),
        GNeg=GNegU,
        mug=np.ravel(mug),
        lambdag=np.ravel(lambdag),
        x=x,
        energy=energy,
        flags=int(_fem.ElementElasticityComputationFlags.Hessian),
        element=element,
        order=order,
        dims=X.shape[0],
    )
    modes = min(modes, x.shape[0])
    l, V = sp.sparse.linalg.eigsh(HU, k=modes, M=M, sigma=sigma, which="LM")
    V = V / sp.linalg.norm(V, axis=0, keepdims=True)
    l[l <= zero] = 0
    w = np.sqrt(l)
    return w, V

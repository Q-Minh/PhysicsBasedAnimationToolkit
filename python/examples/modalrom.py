"""Quadratic-manifold reduced model of an elastic body, with fixed-energy sampling and shape reachability.

Builds a reduced displacement model u(q) = U q + 1/2 Q:qq from mass-normalized vibration modes U and their
modal derivatives Q (projected M-orthogonally off span(U)), for a CPU editor scene file, a stored FemElastoDynamics
(filepath:h5path, e.g. a Dump's fem group) or a tetrahedral mesh, under a Stable Neo-Hookean energy. Then either opens a Polyscope GUI or samples headless.

GUI panels:
    Modes:              animate one linear mode or its quadratic-manifold counterpart ("Mapping").
    Fixed-energy
    sampling:           draw shapes at a fixed potential energy (samplers "polar", "crw", "ray") or states at a
                        fixed total energy K + U ("microcanonical"), optionally with gravity in the potential, and
                        page through them. Microcanonical samples show their velocities as vectors.
    Shape reachability: author a full-order static target (gravity, a force on a box region, or a prescribed
                        displacement of a box region), solve it, and fit linear and quadratic reduced models to it.

Headless mode (--headless -o <file.h5>) writes one batch of samples in the sample-file contract format (see
rom_mcmc.write_samples). Benchmark datasets are generated with make_benchmark.py instead.

Usage:
    # GUI on a CPU editor scene file
    python modalrom.py -i C:\\git\\PhysicsBasedAnimationToolkit\\data\\scenarios\\trampoline.h5

    # GUI on a stored FemElastoDynamics (filepath:h5path), e.g. an editor Dump
    python modalrom.py -i C:\\git\\PhysicsBasedAnimationToolkit\\data\\scenarios\\MediumLongBendingBeam.Dump.h5:fem

    # Headless, 100 shapes at deformation energy E* = 10 J (polar sampler)
    python modalrom.py -i <scene>.h5 --headless -o batch.h5 --energy 10 --samples 100

    # Headless, 400 states (q, qdot) at total energy E = K + U = 10 J, pooled from 4 chains
    python modalrom.py -i <scene>.h5 --headless -o batch.h5 --sampler microcanonical --energy 10 --samples 400

    # GUI on a tetrahedral mesh with its own material
    python modalrom.py -i mesh.msh -m 1000 -Y 1e6 -n 0.45
"""

import typing
from pbatoolkit import pbat, pypbat
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
import polyscope as ps
import polyscope.imgui as imgui
import time
import meshio
import argparse
import os
import types
import rom_mcmc
import rom_reach


def mass(
    E: np.ndarray,
    X: np.ndarray,
    element: pbat.fem.Element,
    rhoe: np.ndarray,
    order: int = 1,
):
    """Compute mass matrix

    Args:
        E (np.ndarray): |# elem. nodes| x |# elements| element connectivity array.
        X (np.ndarray): 3 x |# nodes| node positions array.
        element (pbat.fem.Element): Element type.
        rhoe (np.ndarray): |# elements| x 1 element densities.

    Returns:
        np.ndarray: The mass matrix.
    """
    qorderM = 2 * order
    wgM = pbat.fem.mesh_quadrature_weights(
        E, X, element, order=order, quadrature_order=qorderM
    )
    egM = pbat.fem.mesh_quadrature_elements(E, wgM)
    Neg = pbat.fem.shape_functions(
        n_elements=E.shape[1],
        element=element,
        order=order,
        quadrature_order=qorderM,
        dtype=wgM.dtype,
    )
    rhog = rhoe[np.newaxis, :].repeat(egM.shape[0], axis=0)
    M = pbat.fem.mass_matrix(
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
    return M


def potential(
    E: np.ndarray,
    X: np.ndarray,
    element: pbat.fem.Element,
    Ye: np.ndarray,
    nue: np.ndarray,
):
    """Compute quadrature for elastic potential

    Args:
        E (np.ndarray): |# elem. nodes| x |# elements| element connectivity array.
        X (np.ndarray): 3 x |# nodes| node positions array.
        element (pbat.fem.Element): Element type.
        Ye (np.ndarray): |# elements| x 1 element Young's moduli.
        nue (np.ndarray): |# elements| x 1 element Poisson's ratios.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            |# quad.pts.| x 1 array of element indices at quadrature points,
            |# quad.pts.| x 1 array of quadrature weights,
            |# elem. nodes| x 3*|# quad.pts.| array of element shape function gradients,
            |# quad.pts.| x 1 array of element 1st Lame coefficient,
            |# quad.pts.| x 1 array of element 2nd Lame coefficient.
    """
    # Compute the hyper-elastic potential's hessian
    order = 1
    qorderU = order
    wgU = pbat.fem.mesh_quadrature_weights(
        E, X, element, order=order, quadrature_order=qorderU
    )
    egU = pbat.fem.mesh_quadrature_elements(E, wgU)
    GNegU = pbat.fem.shape_function_gradients(
        E, X, element=element, order=order, dims=X.shape[0], quadrature_order=qorderU
    )
    mug, lambdag = pypbat.fem.lame_coefficients(Ye, nue)
    return egU, wgU, GNegU, mug, lambdag


def stiffness(
    E, X, egU, wgU, GNegU, mug, lambdag, x, energy, element, order=1
) -> scipy.sparse.csr_matrix:
    """Compute the stiffness matrix for a hyper-elastic material.

    Args:
        E (np.ndarray[int]): |# elem. nodes| x |# elements| array of element nodes
        X (np.ndarray[float]): |# nodes| x 3 array of node positions
        egU (np.ndarray[float]): |# quad.pts.| x 1 array of element indices at quadrature points
        wgU (np.ndarray[float]): |# quad.pts.| x 1 array of quadrature weights
        GNegU (np.ndarray[float]): |# elem. nodes| x 3*|# quad.pts.| array of element shape function gradients
        mug (np.ndarray[float]): |# quad.pts.| x 1 array of element 1st Lame coefficient
        lambdag (np.ndarray[float]): |# quad.pts.| x 1 array of element 2nd Lame coefficient
        x (np.ndarray[float]): 3 x |# nodes| or 3*|# nodes| array of node positions
        energy (pbat.fem.HyperElasticEnergy): hyper-elastic energy model
        element (pbat.fem.Element): element type
        order (int, optional): polynomial order. Defaults to 1.

    Returns:
        scipy.sparse.csr_matrix: The stiffness matrix K
    """
    _, _, K = pbat.fem.hyper_elastic_potential(
        E=E,
        n_nodes=X.shape[1],
        eg=np.ravel(egU),
        wg=np.ravel(wgU),
        GNeg=GNegU,
        mug=np.ravel(mug),
        lambdag=np.ravel(lambdag),
        x=np.ravel(x),
        energy=energy,
        flags=pbat.fem.ElementElasticityComputationFlags.Hessian,
        spd_correction=pbat.fem.HyperElasticSpdCorrection.NoCorrection,
        element=element,
        order=order,
        dims=X.shape[0],
    )
    return K


def vibration_modes(
    M: scipy.sparse.dia_array,
    K: scipy.sparse.csr_matrix,
    modes: int = 30,
    sigma: float = -1e-5,
    zero: float = 0.0,
):
    """Computes natural (linear) displacement modes.

    Solves the generalized eigenproblem K v = lambda M v for the `modes` eigenvalues nearest sigma by shift-invert
    Lanczos (scipy.sparse.linalg.eigsh), i.e. the lowest frequencies omega = sqrt(lambda), and mass-normalizes them.

    Args:
        M (scipy.sparse.csr_matrix): Mass matrix.
        K (scipy.sparse.csr_matrix): Stiffness matrix (at rest).
        modes (int, optional): Number of modes to compute. Defaults to 30.
        sigma (float, optional): Shift of the shift-invert solve. Slightly negative, so that K - sigma M is
            nonsingular even when K has rigid (zero-energy) modes. Defaults to -1e-5.
        zero (float, optional): Eigenvalues <= zero (rigid modes, and round-off below zero) are set to 0 before
            taking square roots. Defaults to 0.

    Returns:
        (np.ndarray, np.ndarray): (w, V) s.t. w is a |# modes| vector of frequencies and V is a n x |# modes| array of mass-normalized (V^T M V = I) mode shapes in columns.
    """
    n = M.shape[0]
    modes = min(modes, n)
    l, V = sp.sparse.linalg.eigsh(K, k=modes, M=M, sigma=sigma, which="LM")
    V = V / np.sqrt(np.einsum("ni,n,ni->i", V, M.diagonal(), V))
    l[l <= zero] = 0
    w = np.sqrt(l)
    return w, V


def modal_derivatives(
    x: np.ndarray,
    Ured: np.ndarray,
    w: np.ndarray,
    Mred: scipy.sparse.dia_array,
    Keqred: scipy.sparse.csr_matrix,
    h: float,
    freedofs: np.ndarray,
    compute_stiffness: typing.Callable[[np.ndarray], scipy.sparse.csr_matrix],
):
    """Computes the modal derivatives Theta_ij = d U_i / d eta_j of the vibration modes (the quadratic manifold).

    For each pair (i, j), solves the bordered system of vibration-mode derivatives (Idelsohn & Cardona 1985)

        [ K - w_i^2 M    -M U_i ] [ Theta_ij ]   [ -(dK/d eta_j) U_i ]
        [ -U_i^T M          0   ] [   mu     ] = [         0         ],

    whose second row fixes the gauge U_i^T M Theta_ij = 0. The stiffness derivative along mode j is a central
    difference, dK/d eta_j ~ |U_j| (K(x + h U_j / |U_j|) - K(x - h U_j / |U_j|)) / (2 h), assembled once per j.
    The system for mode i is factorized once (LDLT) and reused for all j. build_rom then projects Theta M-orthogonally
    off span(U) (which removes the blow-up of near-degenerate mode pairs) and symmetrizes it into Q.

    Args:
        x (np.ndarray): 3*|# nodes| array of node positions in F order (the rest shape)
        Ured (np.ndarray): |# free dofs| x |# modes| array of vibration modes
        w (np.ndarray): |# modes| array of vibration mode frequencies
        Mred (scipy.sparse.csr_matrix): |# free dofs| x |# free dofs| mass matrix
        Keqred (scipy.sparse.csr_matrix): |# free dofs| x |# free dofs| stiffness matrix at rest
        h (float): finite-difference step, a length (build_rom uses eps times the bounding box diagonal). Too small
            and the float32 stiffness round-off dominates; too large and the difference is no longer a derivative
        freedofs (np.ndarray): array of free dofs
        compute_stiffness (typing.Callable[[np.ndarray], scipy.sparse.csr_matrix]): function to compute stiffness matrix

    Returns:
        np.ndarray: 3*|# nodes| x |# modes| x |# modes| tensor Theta, Theta[:, i, j] = Theta_ij (zero on fixed DOFs)
    """
    n, m = x.shape[0], Ured.shape[1]
    Q = np.empty((n, m, m), dtype=Ured.dtype)
    b = np.zeros((freedofs.shape[0] + 1, 1), dtype=Ured.dtype)
    thetaij = np.zeros(n, dtype=Ured.dtype)
    xleft = np.copy(x)
    xright = np.copy(x)
    # dK/d\eta_j depends only on j, so assemble it once per mode and keep its action on all modes
    dKU = np.empty((m, freedofs.shape[0], m), dtype=Ured.dtype)
    for j in range(m):
        # Difference along the unit direction, so that h is a length whatever the modes' normalization
        norm = np.linalg.norm(Ured[:, j])
        xleft[freedofs] = x[freedofs] - (h / norm) * Ured[:, j]
        xright[freedofs] = x[freedofs] + (h / norm) * Ured[:, j]
        Kleft = compute_stiffness(xleft)
        Kright = compute_stiffness(xright)
        dKdetaj = norm * (Kright - Kleft) / (2 * h)
        dKdetajred = dKdetaj.tocsr()[freedofs, :].tocsc()[:, freedofs].tocsr()
        dKU[j] = dKdetajred @ Ured
    for i in range(m):
        # Factorize modal derivative matrix
        A12 = -Mred @ Ured[:, i]
        A12 = A12.reshape(-1, 1)
        # To use Eigen LDLT
        A = scipy.sparse.bmat(
            [[Keqred - w[i] ** 2 * Mred, A12], [A12.T, None]],
            format="csr",
        )
        Ainv = pypbat.math.linalg.ldlt(A)
        Ainv.compute(A)
        # To use SuperLU,
        # A = scipy.sparse.block_array(
        #     [[Keq - w[i] ** 2 * M, A12], [A12.T, None]],
        #     format="csc",
        # )
        # Ainv = scipy.sparse.linalg.factorized(A)
        for j in range(m):
            b[:-1, 0] = -dKU[j][:, i]
            # If Eigen LDLT
            thetaij[freedofs] = Ainv.solve(b).squeeze()[:-1]
            # If SuperLU
            # thetaij = Ainv(b).squeeze()[:-1]
            Q[:, i, j] = thetaij
    return Q


def m_orthogonal_complement(V: np.ndarray, B: np.ndarray, m: np.ndarray):
    """Projects the columns of V M-orthogonally off span(B).

    Args:
        V (np.ndarray): n x k array of vectors to project.
        B (np.ndarray): n x m basis, not necessarily M-orthonormal.
        m (np.ndarray): n diagonal of the (lumped) mass matrix M.

    Returns:
        np.ndarray: V - B (B^T M B)^{-1} B^T M V
    """
    MB = m[:, np.newaxis] * B
    return V - B @ np.linalg.solve(B.T @ MB, MB.T @ V)


def elastic_energy(E, X, egU, wgU, GNegU, mug, lambdag, energy, element, order=1):
    """Returns potential(x, gradient=False, hessian=False) evaluating the total hyper-elastic energy at positions x.

    The energy is pbat's (float32), so it carries round-off noise of about 1e-5 relative (see rom_mcmc).

    Args:
        E (np.ndarray): |# elem. nodes| x |# elements| element connectivity array.
        X (np.ndarray): 3 x |# nodes| rest positions.
        egU (np.ndarray): |# quad.pts.| element indices at quadrature points.
        wgU (np.ndarray): |# quad.pts.| quadrature weights.
        GNegU (np.ndarray): |# elem. nodes| x 3*|# quad.pts.| shape function gradients.
        mug (np.ndarray): |# quad.pts.| first Lame coefficients.
        lambdag (np.ndarray): |# quad.pts.| second Lame coefficients.
        energy (pbat.fem.HyperElasticEnergy): Hyper-elastic energy model.
        element (pbat.fem.Element): Element type.
        order (int, optional): Polynomial order. Defaults to 1.

    Returns:
        typing.Callable: potential(x, gradient, hessian), where
            x (np.ndarray): 3*|# nodes| array of node positions in F order,
            gradient (bool): if True, also return the 3*|# nodes| energy gradient,
            hessian (bool): if True (with gradient), also return the SPD-corrected sparse energy Hessian,
        returning the energy, then its gradient and Hessian if requested.
    """
    flags = pbat.fem.ElementElasticityComputationFlags.Potential
    flags_grad = flags | pbat.fem.ElementElasticityComputationFlags.Gradient
    flags_hess = flags_grad | pbat.fem.ElementElasticityComputationFlags.Hessian
    # Convert once to the layouts the bindings take without copying (column-major matrices,
    # contiguous vectors), since the sampler calls potential thousands of times
    E = np.asfortranarray(E)
    GNegU = np.asfortranarray(GNegU)
    egU, wgU, mug, lambdag = (
        np.ascontiguousarray(np.ravel(a)) for a in (egU, wgU, mug, lambdag)
    )

    def potential(x: np.ndarray, gradient: bool = False, hessian: bool = False):
        U, gradU, hessU = pbat.fem.hyper_elastic_potential(
            E=E,
            n_nodes=X.shape[1],
            eg=egU,
            wg=wgU,
            GNeg=GNegU,
            mug=mug,
            lambdag=lambdag,
            x=np.asarray(x, dtype=np.float32),
            energy=energy,
            flags=flags_hess if hessian else (flags_grad if gradient else flags),
            spd_correction=pbat.fem.HyperElasticSpdCorrection.Absolute,
            element=element,
            order=order,
            dims=X.shape[0],
        )
        if hessian:
            return U, gradU.astype(np.float64), hessU
        return (U, gradU.astype(np.float64)) if gradient else U

    return potential


def elastic_energy_float64(E, X, egU, wgU, GNegU, mug, lambdag, energy, element, order=1):
    """Like elastic_energy, but evaluates the energy and its gradient in float64 with NumPy.

    pbat evaluates the energy in float32, which leaves a round-off floor of ~1e-4 J in the energy and ~3 N in the
    gradient on the test meshes. That floor limits how accurately a solve can be converged; with float64 energies and
    gradients, Newton converges to ~1e-8 N. The Hessian (only the Newton matrix, so it needn't be exact) still comes
    from pbat, with its SPD correction.

    Implements pbat's 3D Stable Neo-Hookean energy (StableNeoHookeanEnergy<3>, also the GPU solvers' snh_eval),
    psi = mu/2 (|F|^2 - 3) - mu (J - 1) + lambda/2 (J - 1)^2, for linear tetrahedra (constant F per element).

    Args:
        E, X, egU, wgU, GNegU, mug, lambdag, energy, element, order: As for elastic_energy; energy must be
            StableNeoHookean and element linear (order 1) tetrahedra.

    Returns:
        typing.Callable: potential(x, gradient, hessian), as returned by elastic_energy.
    """
    if energy != pbat.fem.HyperElasticEnergy.StableNeoHookean or order != 1:
        raise ValueError("elastic_energy_float64 implements Stable Neo-Hookean on linear tetrahedra only")
    hessian32 = elastic_energy(E, X, egU, wgU, GNegU, mug, lambdag, energy, element, order)
    eg = np.ravel(egU)
    nodes = np.asarray(E)[:, eg].T  # |# quad.pts.| x 4
    w = np.ravel(wgU).astype(np.float64)
    mu, lam = np.ravel(mug).astype(np.float64), np.ravel(lambdag).astype(np.float64)
    GN = np.asarray(GNegU).astype(np.float64).reshape(GNegU.shape[0], -1, 3).transpose(1, 0, 2)
    n_nodes = X.shape[1]

    def potential(x: np.ndarray, gradient: bool = False, hessian: bool = False):
        xe = np.asarray(x, dtype=np.float64).reshape(-1, 3)[nodes]
        F = np.einsum("gai,gaj->gij", xe, GN)
        J = np.linalg.det(F)
        psi = float(np.sum(w * (0.5 * mu * (np.einsum("gij,gij->g", F, F) - 3.0) - mu * (J - 1.0) + 0.5 * lam * (J - 1.0) ** 2)))
        if not gradient:
            return psi
        # dJ/dF = cofactor of F, whose columns are cross products of F's columns
        cof = np.stack(
            [np.cross(F[:, :, 1], F[:, :, 2]), np.cross(F[:, :, 2], F[:, :, 0]), np.cross(F[:, :, 0], F[:, :, 1])],
            axis=2,
        )
        P = mu[:, None, None] * F + (lam * (J - 1.0) - mu)[:, None, None] * cof
        g = np.zeros((n_nodes, 3))
        np.add.at(g, nodes, w[:, None, None] * np.einsum("gij,gaj->gai", P, GN))
        if hessian:
            return psi, g.ravel(), hessian32(x, True, True)[2]
        return psi, g.ravel()

    return potential


def deformation_gradient_determinants(E, egU, GNegU, x):
    """Computes det(F) at the quadrature points of linear elements.

    Args:
        E (np.ndarray): |# elem. nodes| x |# elements| element connectivity array.
        egU (np.ndarray): |# quad.pts.| element indices at quadrature points.
        GNegU (np.ndarray): |# elem. nodes| x 3*|# quad.pts.| shape function gradients.
        x (np.ndarray): 3*|# nodes| array of node positions in F order.

    Returns:
        np.ndarray: |# quad.pts.| array of det(F), which is <= 0 for inverted elements.
    """
    eg = np.ravel(egU)
    xe = x.reshape(-1, 3)[np.asarray(E)[:, eg].T]
    GN = np.asarray(GNegU).reshape(GNegU.shape[0], -1, 3).transpose(1, 0, 2)
    # F = sum_a x_a grad(N_a)^T
    return np.linalg.det(np.einsum("gai,gaj->gij", xe, GN))


def signal(w: float, t: float, c: float, k: float):
    """Mode animation amplitude c sin(k w t) at time t, for a mode of frequency w slowed down by k."""
    return c * np.sin(k * w * t)


def linear_map(U: np.ndarray, q: np.ndarray):
    """Linear reduced displacement u = U q, for the n x m basis U and m coordinates q."""
    return U @ q


def quadratic_map(U: np.ndarray, Q: np.ndarray, q: np.ndarray):
    """Quadratic-manifold displacement u = U q + 1/2 Q:qq, for the n x m x m symmetric tensor Q."""
    return U @ q + 0.5 * np.einsum("nij,i,j->n", Q, q, q)


def input_kind(input: str) -> str:
    """Classifies an input path.

    Args:
        input (str): A stored FemElastoDynamics as filepath:h5path (e.g. scene.Dump.h5:fem), a CPU editor scene
            file (.h5 or .hdf5), or a tetrahedral mesh file (anything meshio reads).

    Returns:
        str: "fem", "scene" or "mesh".
    """
    file_path, sep, _ = str(input).rpartition(":")
    # A lone drive letter (C:...) is part of a Windows path, not a filepath:h5path separator
    if sep and file_path and not (len(file_path) == 1 and file_path.isalpha()):
        return "fem"
    if str(input).lower().endswith((".h5", ".hdf5")):
        return "scene"
    return "mesh"


def check_scene_constraints(constraints, contact_geometry, path: str, t_end: float = 0.0):
    """Rejects scenes the sampled states and benchmark problems can't represent.

    The problems live in the time span [0, t_end] (t_end = 0 for sampling and static problems, the largest time
    step h for dynamic ones). They hold the Dirichlet nodes fixed at rest throughout and have no collision geometry.
    So every constraint active during the span must be fixed and cover all of it, and the scene can't have static
    colliders. Constraints entirely outside the span never act on the problems and are ignored.

    Args:
        constraints: The scene's kinematic constraints (python/tools/sim/gpu/kinematics/constraints.py), each
            with kind, begin, duration and name members.
        contact_geometry: The scene's static collider geometry, or None.
        path (str): The scene file, for error messages.
        t_end (float): End of the problems' time span in s.

    Raises:
        ValueError: Listing every unsupported collider or constraint.
    """
    problems = []
    if contact_geometry is not None:
        problems.append("it has static colliders (the problems have no collision geometry)")
    for c in constraints:
        begin, end = c.begin, c.begin + c.duration
        if begin > t_end or end <= 0.0:
            continue
        if c.kind.name != "Fixed":
            problems.append(f'constraint "{c.name}" moves its nodes ({c.kind.name}) during [0, {t_end:g}] s')
        elif begin > 0.0 or end <= t_end:
            problems.append(
                f'constraint "{c.name}" starts or ends within [0, {t_end:g}] s (active over [{begin:g}, {end:g}) s)'
            )
    if problems:
        raise ValueError(
            f"{path} can't be used: " + "; ".join(problems) + ". Constraints acting on the problems must keep "
            "their nodes fixed throughout."
        )


def count_bodies(E: np.ndarray, n_nodes: int) -> int:
    """Number of connected components (separate bodies) of a tetrahedral mesh.

    Args:
        E (np.ndarray): |# elem. nodes| x |# elements| element connectivity array.
        n_nodes (int): Number of nodes.

    Returns:
        int: The number of connected components of the node graph.
    """
    E = np.asarray(E)
    # Chaining each element's nodes connects all of them
    rows, cols = E[:-1].ravel(), E[1:].ravel()
    graph = scipy.sparse.coo_array((np.ones(rows.size), (rows, cols)), shape=(n_nodes, n_nodes))
    n_components, _ = scipy.sparse.csgraph.connected_components(graph, directed=False)
    return int(n_components)


def load_scene_fem(path: str, t_end: float = 0.0):
    """Builds the FemElastoDynamics of a CPU editor scene file, the way the editor does.

    Uses python/tools/sim/state.load_scene (which imports Warp but doesn't initialize the GPU), then checks that
    the constraints acting during [0, t_end] keep their nodes fixed and that there are no static colliders
    (check_scene_constraints).

    Args:
        path (str): The scene file (e.g. data/scenarios/trampoline.h5).
        t_end (float): End of the problems' time span in s (0 for sampling and static problems).

    Returns:
        pbat.sim.dynamics.FemElastoDynamics: The scene's FEM problem, with its fixed nodes as Dirichlet nodes.

    Raises:
        ValueError: If the scene has static colliders, constraints acting during [0, t_end] that move their nodes
            or start or end within it, or more than one body.
    """
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from python.tools.sim import state

    fem, (constraints, _, _), contact_geometry = state.load_scene(path)
    check_scene_constraints(constraints, contact_geometry, path, t_end)
    n_bodies = count_bodies(fem.E, fem.X.shape[1])
    if n_bodies > 1:
        # Rigid modes are only excluded for an unconstrained single body, so a free body next to a fixed one would
        # leave zero-frequency modes in the reduced model
        raise ValueError(f"{path} can't be used: it has {n_bodies} separate bodies, and only single-body scenes are supported")
    return fem


def load_fem(input: str):
    """Deserializes the FemElastoDynamics stored at filepath:h5path (e.g. scene.Dump.h5:fem)."""
    import gc

    file_path, _, group_path = str(input).rpartition(":")
    archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
    fem = pbat.sim.dynamics.FemElastoDynamics()
    fem.deserialize(archive[group_path] if group_path else "/")
    archive = None
    gc.collect()
    return fem


def model_from_fem(fem):
    """The model arrays of a FemElastoDynamics, as returned by load_model."""
    n = fem.X.shape[0] * fem.X.shape[1]
    M = scipy.sparse.diags_array(fem.M(), shape=(n, n))
    gravity = np.asarray(fem.aext(), dtype=np.float64).mean(axis=1)
    return (
        fem.E,
        fem.X,
        M,
        fem.egU,
        fem.wgU,
        fem.GNegU,
        fem.lamegU[0, :],
        fem.lamegU[1, :],
        fem.free_dofs,
        gravity,
    )


def load_model(input: str, rho: float, Y: float, nu: float, element, energy):
    """Loads a stored FemElastoDynamics, a CPU editor scene file, or a tetrahedral mesh.

    Args:
        input (str): filepath:h5path of a stored FemElastoDynamics, a scene file (.h5), or a tetrahedral mesh
            (see input_kind). A scene's constraints active at t = 0 must keep their nodes fixed, and it can't have
            static colliders (see check_scene_constraints).
        rho (float): Mass density (mesh input only).
        Y (float): Young's modulus (mesh input only).
        nu (float): Poisson's ratio (mesh input only).
        element (pbat.fem.Element): Element type.
        energy (pbat.fem.HyperElasticEnergy): Hyper-elastic energy model.

    Returns:
        Tuple: (E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, gravity), with M the lumped mass matrix
        and gravity the scene's mean external acceleration (3,), or (0, 0, -9.81) for a mesh.
    """
    kind = input_kind(input)
    if kind == "fem":
        return model_from_fem(load_fem(input))
    if kind == "scene":
        return model_from_fem(load_scene_fem(input))
    else:
        imesh = meshio.read(input)
        V, C = imesh.points, imesh.cells_dict["tetra"]
        X, E = pbat.fem.mesh(V.T, C.T, element=element)
        n = X.shape[0] * X.shape[1]
        Ye = np.full(E.shape[1], Y)
        nue = np.full(E.shape[1], nu)
        rhoe = np.full(E.shape[1], rho)
        M = mass(E, X, element, rhoe)
        M = scipy.sparse.diags_array(M.sum(axis=1), shape=(n, n))
        egU, wgU, GNegU, mug, lambdag = potential(E, X, element, Ye, nue)
        freedofs = np.arange(n)
        gravity = np.array([0.0, 0.0, -9.81])
    return E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, gravity


def build_rom(
    E,
    X,
    M,
    egU,
    wgU,
    GNegU,
    mug,
    lambdag,
    freedofs,
    energy,
    element,
    modes,
    eps,
    z_rtol=0.02,
):
    """Computes mass-normalized vibration modes, modal derivatives and the augmented linear basis.

    Modal derivatives are projected M-orthogonally off the span of the vibration modes. The augmented
    basis keeps the M-orthogonal SVD directions of the derivatives whose singular values are at least
    z_rtol times the largest.

    Args:
        E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs: The model, as returned by load_model.
        energy (pbat.fem.HyperElasticEnergy): Hyper-elastic energy model.
        element (pbat.fem.Element): Element type.
        modes (int): Number of vibration modes m.
        eps (float): Finite-difference step of the modal derivatives, as a fraction of the bounding box diagonal.
        z_rtol (float, optional): Relative singular value cutoff of the augmented basis. Defaults to 0.02.

    Returns:
        Tuple: (w, U, Q, Z, ws), with w the |# modes| mode frequencies, U the n x |# modes|
        vibration modes, Q the n x |# modes| x |# modes| symmetric quadratic manifold tensor,
        Z the n x |# derivative modes| augmented basis and ws the frequencies of [U Z].
    """
    n = X.shape[0] * X.shape[1]
    nfree = len(freedofs)
    mu = int(modes)
    Keq = stiffness(
        E, X, egU, wgU, GNegU, mug, lambdag, np.ravel(X, order="F"), energy, element
    )
    Mred = scipy.sparse.diags_array(M.diagonal()[freedofs], shape=(nfree, nfree))
    Keqred = Keq.tocsr()[freedofs, :].tocsc()[:, freedofs].tocsr()
    w, Ured = vibration_modes(Mred, Keqred, modes=modes)
    U = np.zeros((n, mu), dtype=Ured.dtype)
    U[freedofs, :] = Ured
    Xmax, Xmin = np.max(X, axis=1), np.min(X, axis=1)
    bbdiag = np.linalg.norm(Xmax - Xmin)
    h = eps * bbdiag
    Theta = modal_derivatives(
        X.flatten(order="F"),
        Ured,
        w,
        Mred,
        Keqred,
        h,
        freedofs,
        lambda x: stiffness(E, X, egU, wgU, GNegU, mug, lambdag, x, energy, element),
    )
    # Project out linear basis from modal derivatives. Without this, near-degenerate
    # mode pairs (w_i ~ w_j) make Theta blow up along their partner modes.
    Theta = m_orthogonal_complement(
        Theta.reshape(n, -1), U, M.diagonal().astype(Theta.dtype)
    ).reshape(n, mu, mu)
    # Compute rank-3 quadratic manifold tensor
    Q = (Theta + Theta.transpose(0, 2, 1)) / 2
    # Compute augmented linear basis
    Thetared = Theta.reshape(n, -1)[freedofs, :]
    Mredsqrt = scipy.sparse.diags_array(
        np.sqrt(M.diagonal()[freedofs]), shape=(nfree, nfree)
    )
    Mredinvsqrt = scipy.sparse.diags_array(
        1.0 / Mredsqrt.diagonal(), shape=(nfree, nfree)
    )
    # Compute M-orthogonal SVD
    Zredhat, s2, _ = np.linalg.svd(
        Mredsqrt @ Thetared, full_matrices=False, compute_uv=True
    )
    # Undo the M-norm on the SVD basis
    Zred = Mredinvsqrt @ Zredhat
    # Keep the derivative directions whose singular values are at least z_rtol times the largest
    mz = int(np.sum(s2 >= z_rtol * s2[0]))
    Zred = Zred[:, :mz]
    s = np.sqrt(s2)
    ws = np.hstack([w, s[:mz]])
    Z = np.zeros((n, mz), dtype=Zred.dtype)
    Z[freedofs, :] = Zred
    return w, U, Q, Z, ws


def reduced_model(
    mapping: str,
    U: np.ndarray,
    Q: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    x0: np.ndarray,
    potential: typing.Callable,
    rigid: int,
    weight: str,
    fext: np.ndarray | None = None,
):
    """Builds the reduced model to sample on, excluding the first `rigid` (zero-frequency) modes.

    Also sets the whitening scales scale_i = 1 / (w_i sqrt(U_i^T M U_i)), so that the linearized energy is
    1/2 |y|^2 in the samplers' coordinates y = q / scale.

    Args:
        mapping (str): "quadratic" for u = U q + 1/2 Q:qq, "linear" for u = U q.
        U (np.ndarray): n x |# modes| vibration modes.
        Q (np.ndarray): n x |# modes| x |# modes| quadratic manifold tensor.
        w (np.ndarray): |# modes| mode frequencies.
        m (np.ndarray): n lumped mass matrix diagonal.
        x0 (np.ndarray): n rest positions in F order.
        potential (typing.Callable): Full-space energy, see elastic_energy.
        rigid (int): Number of leading rigid modes to exclude (6 without Dirichlet constraints, 0 otherwise): the
            energy is flat along them, so energy level sets would be unbounded and whitening would divide by zero.
        weight (str): Distribution on the fixed-potential level set, see rom_mcmc.WEIGHTS.
        fext (np.ndarray | None): n constant external force, e.g. gravity M g, adding -fext^T u(q) to the potential,
            or None.

    Returns:
        Tuple[rom_mcmc.QuadraticManifold, np.ndarray]: The model and its mode frequencies.
    """
    keep = np.arange(rigid, U.shape[1])
    Ur = U[:, keep].astype(np.float64)
    Qr = Q[:, keep][:, :, keep].astype(np.float64) if mapping == "quadratic" else None
    m = m.astype(np.float64)
    # Whitening: the linearized energy 1/2 q^T U^T K U q = 1/2 sum_i w_i^2 (U_i^T M U_i) q_i^2
    mi = np.einsum("ni,n,ni->i", Ur, m, Ur)
    scale = 1.0 / (w[keep] * np.sqrt(mi))
    model = rom_mcmc.QuadraticManifold(
        Ur, Qr, m, x0.astype(np.float64), potential, scale, weight, fext
    )
    return model, w[keep]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Elastic stiffness quadratic manifold demo",
        description="Quadratic-manifold reduced model with fixed-energy sampling (GUI or headless) and shape "
        "reachability tests. See the module docstring for representative commands.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="A CPU editor scene file (.h5, e.g. data/scenarios/trampoline.h5; constraints active at t = 0 must "
        "keep their nodes fixed, and no static colliders), a filepath:h5path pair of an hdf5 file storing a FemElastoDynamics (e.g. "
        "scene.Dump.h5:fem), or a tetrahedral mesh (read with meshio). Scenes and stored FEMs bring their own "
        "material, masses and Dirichlet constraints; a mesh uses -m, -Y and -n and has none",
        type=str,
        dest="input",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mass-density",
        help="Mass density in kg/m^3, for mesh input only (default: %(default)s)",
        type=float,
        dest="rho",
        default=1000.0,
    )
    parser.add_argument(
        "-Y",
        "--young-modulus",
        help="Young's modulus in Pa, for mesh input only (default: %(default)s)",
        type=float,
        dest="Y",
        default=1e6,
    )
    parser.add_argument(
        "-n",
        "--poisson-ratio",
        help="Poisson's ratio, for mesh input only (default: %(default)s)",
        type=float,
        dest="nu",
        default=0.45,
    )
    parser.add_argument(
        "-k",
        "--num-modes",
        help="Number of vibration modes m, i.e. reduced coordinates (default: %(default)s). Without Dirichlet "
        "constraints, the 6 lowest (rigid) modes are excluded from sampling",
        type=int,
        dest="modes",
        default=20,
    )
    parser.add_argument(
        "--eps",
        help="Finite-difference step of the modal derivatives, as a fraction of the bounding box diagonal "
        "(default: %(default)s)",
        type=float,
        dest="eps",
        default=1e-3,
    )
    parser.add_argument(
        "--headless",
        help="Sample without the GUI and write one batch to --output (sample-file contract, see "
        "rom_mcmc.write_samples)",
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output HDF5 file for --headless sampling (required with --headless)",
        type=str,
        dest="output",
    )
    parser.add_argument(
        "--energy",
        help="Target energy in J: the potential energy E* for the level-set samplers (polar, crw, ray), the "
        "total energy E = K + U for the microcanonical sampler. Also the GUI's initial value (default: %(default)s)",
        type=float,
        dest="energy",
        default=1.0,
    )
    parser.add_argument(
        "--samples",
        help="Number of samples written by --headless, pooled over all chains (default: %(default)s)",
        type=int,
        dest="samples",
        default=100,
    )
    parser.add_argument(
        "--sampler",
        help="polar: Metropolis on directions with exact radial solves, crw: constrained random walk with "
        "projection, ray: independent random directions (biased; files require importance weights), "
        "microcanonical: positions and velocities at fixed total energy K + U (default: %(default)s)",
        choices=tuple(rom_mcmc.SAMPLERS),
        default="polar",
    )
    parser.add_argument(
        "--chains",
        help="Number of independent chains, pooled, whose agreement is reported as split R-hat. Defaults to 4 "
        "for the microcanonical sampler, 1 otherwise",
        type=int,
        dest="chains",
        default=None,
    )
    parser.add_argument(
        "--gravity",
        help="Gravitational acceleration gx gy gz in m/s^2, adding the potential -(M g)^T u(q) to headless "
        "sampling and enabling it in the GUI. Off by default; the GUI otherwise offers the scene's gravity",
        type=float,
        nargs=3,
        dest="gravity",
        default=None,
    )
    parser.add_argument(
        "--weight",
        help="Distribution on the energy level set for polar, crw and ray: mass = sqrt(det J^T M J) (independent "
        "of how q is scaled), uniform, or area (see rom_mcmc) (default: %(default)s)",
        choices=rom_mcmc.WEIGHTS,
        default="mass",
    )
    parser.add_argument(
        "--mapping",
        help="Reduced displacement map to sample on in --headless mode: quadratic u = U q + 1/2 Q:qq, or "
        "linear u = U q (default: %(default)s)",
        choices=("quadratic", "linear"),
        default="quadratic",
    )
    parser.add_argument(
        "--burn-in",
        help="Burn-in steps per chain, during which the proposal step size adapts (default: %(default)s)",
        type=int,
        dest="burn_in",
        default=300,
    )
    parser.add_argument(
        "--pilot",
        help="Steps per chain used to estimate the integrated autocorrelation time, which sets the thinning "
        "(default: %(default)s)",
        type=int,
        dest="pilot",
        default=500,
    )
    parser.add_argument(
        "--thin",
        help="Keep every thin-th step. Defaults to the autocorrelation time estimated from the pilot steps",
        type=int,
        dest="thin",
        default=None,
    )
    parser.add_argument(
        "--seed",
        help="Random seed (default: %(default)s)",
        type=int,
        dest="seed",
        default=0,
    )
    args = parser.parse_args()
    if args.headless and not args.output:
        parser.error("--headless requires --output")

    element = pbat.fem.Element.Tetrahedron
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, scene_gravity = load_model(
        args.input, args.rho, args.Y, args.nu, element, energy
    )
    n = X.shape[0] * X.shape[1]
    mu = int(args.modes)
    w, U, Q, Z, ws = build_rom(
        E,
        X,
        M,
        egU,
        wgU,
        GNegU,
        mug,
        lambdag,
        freedofs,
        energy,
        element,
        args.modes,
        args.eps,
    )
    mz = Z.shape[1]
    UZ = np.hstack([U, Z])
    x0 = X.flatten(order="F")
    psi = elastic_energy(E, X, egU, wgU, GNegU, mug, lambdag, energy, element)
    # Without Dirichlet constraints, the 6 lowest modes are rigid and the energy level set is unbounded along them
    rigid = 6 if len(freedofs) == n else 0
    def make_model(mapping: str, gravity=None):
        """The reduced model to sample on, with the gravity potential -(M g)^T u(q) if gravity is given."""
        fext = (
            rom_reach.gravity_load(M.diagonal().astype(np.float64), gravity)
            if gravity is not None
            else None
        )
        return reduced_model(
            mapping, U, Q, w, M.diagonal(), x0, psi, rigid, args.weight, fext
        )

    if args.headless:
        model, wr = make_model(args.mapping, args.gravity)
        chains = args.chains or (4 if args.sampler == "microcanonical" else 1)
        t0 = time.time()
        batch = rom_mcmc.run_chains(
            lambda rng: rom_mcmc.make_sampler(args.sampler, model, args.energy, rng),
            chains,
            args.samples,
            args.burn_in,
            args.pilot,
            args.thin,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        min_det_F = np.array(
            [
                deformation_gradient_determinants(
                    E, egU, GNegU, model.x0 + model.displacement(qs)
                ).min()
                for qs in batch.q
            ]
        )
        attrs = dict(
            input=args.input,
            n_linear_modes=mu,
            mode_normalization="mass",
            rigid_modes_excluded=rigid,
            modal_derivatives=args.mapping == "quadratic",
            eps=args.eps,
            sampler=args.sampler,
            target_energy=args.energy,
            gravity=np.asarray(args.gravity if args.gravity is not None else [0.0] * 3),
            seed=args.seed,
            git_commit=rom_mcmc.git_commit(os.path.dirname(os.path.abspath(__file__))),
            sampling_time=elapsed,
        )
        rom_mcmc.write_samples(args.output, X, model, wr, batch, attrs, min_det_F)
        s = batch.stats
        print(
            f"Wrote {args.samples} samples to {args.output} in {elapsed:.1f} s: "
            f"{chains} chain(s), acceptance={s['acceptance']:.2f}, thin={s['thin']}, tau={s['tau']:.1f}, "
            f"ESS={s['ess']:.0f}, split R-hat={s['rhat']:.3f}, star violations={s['star_violations']}, "
            f"energy noise={s['energy_noise']:.2g} J, inverted samples={np.sum(min_det_F <= 0)}"
        )
        if s["energy_noise_warning"]:
            print(
                f"Warning: the energy noise limits samples to |Psi - E*| <~ {s['energy_tolerance']:.2g} J, "
                "more than 1e-3 E*. Use a larger E* for exact level-set samples."
            )
        raise SystemExit(0)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_volume_mesh("model", X.T, E.T)
    # drop-down to select mapping type (linear/quadratic)
    mappings = ["Linear", "Quadratic"]
    mapping = mappings[1]
    mode = 6
    q = np.zeros(mu + mz, dtype=U.dtype)
    t0 = time.time()
    t = 0
    c = 0.03 * np.linalg.norm(X.max(axis=1) - X.min(axis=1))
    k = 0.05
    # Fixed-energy sampling state. The chain restarts whenever its settings change.
    samplers = list(rom_mcmc.SAMPLERS)
    sampler_kind = samplers[0]
    target_energy = args.energy
    samples_per_click = 10
    # Gravity potential for sampling, defaulting to the scene's gravity
    sampling_gravity_on = args.gravity is not None
    sampling_gravity = list(args.gravity if args.gravity is not None else scene_gravity)
    show_samples = False
    chain = None
    chain_key = None
    model = None
    samples = []
    sample_idx = 0
    stats = {}
    rng = np.random.default_rng(args.seed)
    # Shape reachability: full-order static targets authored by loads or handles, and reduced-model fits
    target_kinds = ["Gravity", "Region force", "Handle displacement"]
    fit_kinds = [
        "Linear U fit",
        "Linear [U Z] fit",
        "Quadratic fit",
        "Quadratic energy-regularized fit",
        "Quadratic ROM equilibrium",
        "Linear ROM equilibrium",
    ]
    m64, x64 = M.diagonal().astype(np.float64), x0.astype(np.float64)
    U64, UZ64 = U.astype(np.float64), UZ.astype(np.float64)
    # Mass-weighted handle penalty for ROM equilibria, stiffer than the stiffest mode
    kappa = 100.0 * float(np.max(w)) ** 2
    has_dirichlet = len(freedofs) < n
    reach = types.SimpleNamespace(
        target_kind=target_kinds[0],
        gravity=[0.0, 0.0, -0.4],
        region=[[0.95, 1.0], [0.0, 1.0], [0.0, 1.0]],
        force=[0.0, 0.0, -50.0],
        displacement=[0.0, 0.0, -0.3],
        steps=5,
        fit_kind=fit_kinds[2],
        alpha=0.01,
        target=None,
        fit=None,
        show=False,
        models={},
    )

    def reach_model(mapping: str):
        if mapping not in reach.models:
            reach.models[mapping], _ = reduced_model(
                mapping, U, Q, w, M.diagonal(), x0, psi, rigid, "uniform"
            )
        return reach.models[mapping]

    def region_nodes():
        lo, hi = np.array(reach.region).T
        return rom_reach.box_region(X, lo, hi)

    def solve_target():
        nodes = region_nodes()
        fext, handle_dofs, handle_u = None, None, None
        if reach.target_kind == "Gravity":
            fext = rom_reach.gravity_load(m64, reach.gravity)
        elif reach.target_kind == "Region force":
            fext = rom_reach.region_load(m64, nodes, reach.force)
        else:
            handle_dofs = np.intersect1d(rom_reach.node_dofs(nodes), freedofs)
            handle_u = np.asarray(reach.displacement)[handle_dofs % 3]
        t = time.time()
        u, info = rom_reach.static_equilibrium(
            psi, x64, freedofs, fext, handle_dofs, handle_u, reach.steps
        )
        reach.target = dict(
            u=u,
            fext=fext,
            handle_dofs=handle_dofs,
            handle_u=handle_u,
            energy=rom_reach.deformation_energy(psi, x64, u),
            info=info,
            time=time.time() - t,
        )
        ps.register_volume_mesh(
            "target", X.T + u.reshape(X.shape[1], 3), E.T, transparency=0.3
        )

    def fit_target():
        tgt, kind = reach.target, reach.fit_kind
        if kind == "Linear U fit":
            u, _ = rom_reach.linear_fit(U64, m64, tgt["u"])
        elif kind == "Linear [U Z] fit":
            u, _ = rom_reach.linear_fit(UZ64, m64, tgt["u"])
        elif kind == "Quadratic fit":
            u, _ = rom_reach.manifold_fit(reach_model("quadratic"), tgt["u"])
        elif kind == "Quadratic energy-regularized fit":
            u, _ = rom_reach.manifold_fit(
                reach_model("quadratic"), tgt["u"], reach.alpha, max(tgt["energy"], 1e-12)
            )
        else:
            u, _ = rom_reach.rom_equilibrium(
                reach_model("quadratic" if kind.startswith("Quadratic") else "linear"),
                tgt["fext"],
                tgt["handle_dofs"],
                tgt["handle_u"],
                kappa,
                reach.steps,
            )
        reach.fit = dict(
            u=u,
            kind=kind,
            metrics=rom_reach.fit_metrics(u, tgt["u"], m64),
            energy=rom_reach.deformation_energy(psi, x64, u),
        )

    def draw_reachability_panel():
        imgui.Separator()
        imgui.Text("Shape reachability: fit reduced models to full-order static targets")
        changed, idx = imgui.Combo(
            "Target kind", target_kinds.index(reach.target_kind), target_kinds
        )
        reach.target_kind = target_kinds[idx]
        if reach.target_kind == "Gravity":
            changed, reach.gravity = imgui.InputFloat3("Gravity (m/s^2)", reach.gravity)
            if ps.has_point_cloud("region"):
                ps.remove_point_cloud("region")
        else:
            for axis, name in enumerate("xyz"):
                changed, reach.region[axis] = imgui.SliderFloat2(
                    f"Region {name} (bounding box fraction)", reach.region[axis], 0.0, 1.0
                )
            nodes = region_nodes()
            imgui.Text(f"{len(nodes)} region nodes")
            ps.register_point_cloud("region", X.T[nodes], radius=0.004)
            if reach.target_kind == "Region force":
                changed, reach.force = imgui.InputFloat3("Force (N)", reach.force)
            else:
                changed, reach.displacement = imgui.InputFloat3(
                    "Displacement (m)", reach.displacement
                )
        changed, reach.steps = imgui.InputInt("Load steps", reach.steps)
        reach.steps = max(reach.steps, 1)
        if reach.target_kind != "Handle displacement" and not has_dirichlet:
            imgui.TextColored(
                (1.0, 0.6, 0.0, 1.0),
                "Loads need Dirichlet constraints; use handles on unconstrained meshes",
            )
        elif imgui.Button("Solve target"):
            solve_target()
            fit_target()
            reach.show = True
        if reach.target is None:
            return
        tgt = reach.target
        imgui.Text(
            f"Target: energy {tgt['energy']:.4g} J, {tgt['info']['iterations']} Newton iterations, {tgt['time']:.1f} s"
        )
        changed, idx = imgui.Combo("Fit model", fit_kinds.index(reach.fit_kind), fit_kinds)
        reach.fit_kind = fit_kinds[idx]
        if reach.fit_kind == "Quadratic energy-regularized fit":
            changed, reach.alpha = imgui.InputFloat(
                "Energy weight alpha", reach.alpha, format="%.4g"
            )
            reach.alpha = max(reach.alpha, 0.0)
        if imgui.Button("Fit"):
            fit_target()
            reach.show = True
        if reach.fit is not None:
            metrics = reach.fit["metrics"]
            imgui.Text(
                f"{reach.fit['kind']}: relative M-error {metrics['relative_error']:.3f}, "
                f"max vertex error {metrics['max_error']:.3g} m, energy {reach.fit['energy']:.4g} J"
            )
            changed, reach.show = imgui.Checkbox("Show fit and target", reach.show)

    def callback():
        global mode, c, k, q, mapping
        global sampler_kind, target_energy, samples_per_click, show_samples
        global sampling_gravity_on, sampling_gravity
        global chain, chain_key, model, samples, sample_idx, stats
        changed, map_idx = imgui.Combo("Mapping", mappings.index(mapping), mappings)
        mapping = mappings[map_idx]
        nrdofs = q.shape[0] if mapping == "Linear" else mu

        if imgui.Button("<##mode_prev"):
            mode -= 1
        imgui.SameLine()
        if imgui.Button(">##mode_next"):
            mode += 1
        imgui.SameLine()
        imgui.Text(f"Mode {mode}/{nrdofs-1}")
        changed, c = imgui.InputFloat("Wave amplitude (m)", c)
        changed, k = imgui.InputFloat("Wave frequency", k)

        imgui.Separator()
        imgui.Text("Fixed-energy sampling (Linear samples on U only)")
        imgui.Text("E is the potential energy, or K + U for the microcanonical sampler")
        changed, target_energy = imgui.InputFloat("Energy E (J)", target_energy)
        target_energy = max(target_energy, 1e-8)
        changed, sampler_idx = imgui.Combo(
            "Sampler", samplers.index(sampler_kind), samplers
        )
        sampler_kind = samplers[sampler_idx]
        changed, samples_per_click = imgui.InputInt(
            "Samples per click", samples_per_click
        )
        samples_per_click = max(samples_per_click, 1)
        changed, sampling_gravity_on = imgui.Checkbox(
            "Gravity potential -(M g)^T u(q)", sampling_gravity_on
        )
        if sampling_gravity_on:
            changed, sampling_gravity = imgui.InputFloat3(
                "Gravity acceleration (m/s^2)", sampling_gravity
            )
        if imgui.Button("Sample"):
            gravity = tuple(sampling_gravity) if sampling_gravity_on else None
            key = (mapping, target_energy, sampler_kind, gravity)
            if key != chain_key:
                model, _ = make_model(mapping.lower(), gravity)
                chain = rom_mcmc.make_sampler(sampler_kind, model, target_energy, rng)
                chain.burn_in(args.burn_in)
                chain.estimate_thinning(args.pilot)
                chain_key = key
                samples = []
            batch = chain.sample(samples_per_click)
            stats = batch.stats
            qdots = batch.qdot if batch.qdot is not None else [None] * len(batch.q)
            samples.extend(zip(batch.q, batch.energy, qdots))
            sample_idx = len(samples) - 1
            show_samples = True
            reach.show = False
        if samples:
            changed, show_samples = imgui.Checkbox("Show samples", show_samples)
            # The slider shares its label with the "Sample" button, so give it its own ID scope
            imgui.PushID("sample_slider")
            changed, sample_idx = imgui.SliderInt(
                "Sample", sample_idx, 0, len(samples) - 1
            )
            imgui.PopID()
            if chain_key[3] is not None:
                imgui.Text(f"With gravity {np.round(chain_key[3], 3).tolist()} m/s^2 in U")
            if samples[sample_idx][2] is not None:
                imgui.Text(
                    f"{chain_key[0]} map, {chain_key[2]} sampler, E {chain_key[1]:.6g} J: "
                    f"U {samples[sample_idx][1]:.6g} J, K {chain_key[1] - samples[sample_idx][1]:.6g} J"
                )
            else:
                imgui.Text(
                    f"{chain_key[0]} map, {chain_key[2]} sampler, E* {chain_key[1]:.6g} J: "
                    f"energy {samples[sample_idx][1]:.6g} J"
                )
            imgui.Text(
                f"Acceptance {stats['acceptance']:.2f}, thin {stats['thin']}, ESS {stats['ess']:.0f}, "
                f"star violations {stats['star_violations']}, energy noise {stats['energy_noise']:.2g} J"
            )
            if stats["energy_noise_warning"]:
                imgui.TextColored(
                    (1.0, 0.6, 0.0, 1.0),
                    f"Energy noise limits |Psi - E*| to ~{stats['energy_tolerance']:.2g} J (> 1e-3 E*): use a larger E*",
                )

        draw_reachability_panel()

        qdot = None
        showing_fit = reach.show and reach.fit is not None
        if ps.has_volume_mesh("target"):
            ps.get_volume_mesh("target").set_enabled(showing_fit)
        if showing_fit:
            u = reach.fit["u"]
            vm.add_scalar_quantity(
                "fit error (m)", reach.fit["metrics"]["vertex_error"], enabled=True
            )
        elif show_samples and samples:
            qs, _, qdot = samples[sample_idx]
            u = model.displacement(qs)
            if qdot is not None:
                v = model.jacobian(qs) @ qdot
                vm.add_vector_quantity(
                    "velocity", v.reshape(X.shape[1], 3), defined_on="vertices", enabled=True
                )
        else:
            mode = max(0, min(nrdofs - 1, mode))
            t = time.time() - t0
            # c is the mode's peak (linear) vertex displacement, whatever the modes' normalization
            peak = np.abs(UZ[:, mode]).max()
            etam = signal(ws[mode], t, c / peak, k)
            q[:] = 0
            q[mode] = etam
            if mapping == "Quadratic":
                u = quadratic_map(U, Q, q[:mu])
            else:
                u = linear_map(UZ, q)

        if qdot is None:
            vm.remove_quantity("velocity")
        if not showing_fit:
            vm.remove_quantity("fit error (m)")
        V = X.T + u.reshape(X.shape[1], 3)
        vm.update_vertex_positions(V)

    ps.set_user_callback(callback)
    ps.show()

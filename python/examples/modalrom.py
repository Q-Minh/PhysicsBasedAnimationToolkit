import typing
from pbatoolkit import pbat, pypbat
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import polyscope as ps
import polyscope.imgui as imgui
import time
import meshio
import argparse


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

    Args:
        M (scipy.sparse.csr_matrix): Mass matrix.
        K (scipy.sparse.csr_matrix): Stiffness matrix.
        modes (int, optional): Number of modes to compute. Defaults to 30.
        sigma (float, optional): Shift (see scipy.sparse.eigsh). Defaults to -1e-5.
        zero (float, optional): Numerical zero used to cull modes. Defaults to 0.

    Returns:
        (np.ndarray, np.ndarray): (w, V) s.t. w is a |# modes| vector of frequencies and V is a n x |# modes| array of mode shapes in columns.
    """
    n = M.shape[0]
    modes = min(modes, n)
    l, V = sp.sparse.linalg.eigsh(K, k=modes, M=M, sigma=sigma, which="LM")
    V = V / sp.linalg.norm(V, axis=0, keepdims=True)
    l[l <= zero] = 0
    w = np.sqrt(l)
    return w, V


def modal_derivatives(
    x: np.ndarray,
    Ured: np.ndarray,
    Mred: scipy.sparse.dia_array,
    Keqred: scipy.sparse.csr_matrix,
    h: float,
    freedofs: np.ndarray,
    compute_stiffness: typing.Callable[[np.ndarray], scipy.sparse.csr_matrix],
):
    """Computes the quadratic manifold.

    Args:
        x (np.ndarray): 3*|# nodes| array of node positions in F order
        Ured (np.ndarray): |# free dofs| x |# modes| array of vibration modes
        Mred (scipy.sparse.csr_matrix): |# free dofs| x |# free dofs| mass matrix
        Keqred (scipy.sparse.csr_matrix): |# free dofs| x |# free dofs| stiffness matrix
        h (float): step size for finite difference
        freedofs (np.ndarray): array of free dofs
        compute_stiffness (typing.Callable[[np.ndarray], scipy.sparse.csr_matrix]): function to compute stiffness matrix
        symmetrize (bool, optional): whether to symmetrize the quadratic manifold. Defaults to True.

    Returns:
        np.ndarray: The quadratic manifold rank-3 tensor
    """
    n, m = x.shape[0], Ured.shape[1]
    Q = np.empty((n, m, m), dtype=Ured.dtype)
    b = np.zeros((freedofs.shape[0] + 1, 1), dtype=Ured.dtype)
    thetaij = np.zeros(n, dtype=Ured.dtype)
    xleft = np.copy(x)
    xright = np.copy(x)
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
            # Compute dK/d\eta_j
            xleft[freedofs] = x[freedofs] - h * Ured[:, j]
            xright[freedofs] = x[freedofs] + h * Ured[:, j]
            Kleft = compute_stiffness(xleft)
            Kright = compute_stiffness(xright)
            dKdetaj = (Kright - Kleft) / h
            dKdetajred = dKdetaj.tocsr()[freedofs, :].tocsc()[:, freedofs].tocsr()
            b[:-1, 0] = -dKdetajred @ Ured[:, i]
            # If Eigen LDLT
            thetaij[freedofs] = Ainv.solve(b).squeeze()[:-1]
            # If SuperLU
            # thetaij = Ainv(b).squeeze()[:-1]
            Q[:, i, j] = thetaij
    return Q


def signal(w: float, t: float, c: float, k: float):
    return c * np.sin(k * w * t)


def linear_map(U: np.ndarray, q: np.ndarray):
    return U @ q


def quadratic_map(U: np.ndarray, Q: np.ndarray, q: np.ndarray):
    return U @ q + 0.5 * np.einsum("nij,i,j->n", Q, q, q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Elastic stiffness quadratic manifold demo",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input tetrahedral mesh or filepath:h5path pair "
        "of hdf5 file containing an FemElastoDynamics and the hdf5 group "
        "path to the FemElastoDynamics",
        type=str,
        dest="input",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mass-density",
        help="Mass density",
        type=float,
        dest="rho",
        default=1000.0,
    )
    parser.add_argument(
        "-Y",
        "--young-modulus",
        help="Young's modulus",
        type=float,
        dest="Y",
        default=1e6,
    )
    parser.add_argument(
        "-n",
        "--poisson-ratio",
        help="Poisson's ratio",
        type=float,
        dest="nu",
        default=0.45,
    )
    parser.add_argument(
        "-k",
        "--num-modes",
        help="Number of modes to compute",
        type=int,
        dest="modes",
        default=20,
    )
    parser.add_argument(
        "--eps",
        help="Epsilon for numerical differentiation",
        type=float,
        dest="eps",
        default=1e-3,
    )
    parser.add_argument(
        "--mapping",
        help="Mapping type (linear | quadratic)",
        type=str,
        dest="mapping",
        default="linear",
    )
    args = parser.parse_args()

    input_tokens = str(args.input).split(":")
    element = pbat.fem.Element.Tetrahedron
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    if len(input_tokens) == 2:
        import gc

        file_path, group_path = input_tokens[0], input_tokens[1]
        archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
        fem = pbat.sim.dynamics.FemElastoDynamics()
        fem.deserialize(archive[group_path] if group_path else "/")
        archive = None
        gc.collect()
        n = fem.X.shape[0] * fem.X.shape[1]
        E = fem.E
        X = fem.X
        M = scipy.sparse.diags_array(fem.M(), shape=(n, n))
        egU, wgU, GNegU, mug, lambdag = (
            fem.egU,
            fem.wgU,
            fem.GNegU,
            fem.lamegU[0, :],
            fem.lamegU[1, :],
        )
        freedofs = fem.free_dofs
    else:
        imesh = meshio.read(args.input)
        V, C = imesh.points, imesh.cells_dict["tetra"]
        X, E = pbat.fem.mesh(V.T, C.T, element=element)
        n = X.shape[0] * X.shape[1]
        Ye = np.full(E.shape[1], args.Y)
        nue = np.full(E.shape[1], args.nu)
        rhoe = np.full(E.shape[1], args.rho)
        M = mass(E, X, element, rhoe)
        M = scipy.sparse.diags_array(M.sum(axis=1), shape=(n, n))
        egU, wgU, GNegU, mug, lambdag = potential(E, X, element, Ye, nue)
        freedofs = np.arange(n)

    n = X.shape[0] * X.shape[1]
    nfree = len(freedofs)
    mu = int(args.modes)
    Keq = stiffness(
        E, X, egU, wgU, GNegU, mug, lambdag, np.ravel(X, order="F"), energy, element
    )
    Mred = scipy.sparse.diags_array(M.diagonal()[freedofs], shape=(nfree, nfree))
    Keqred = Keq.tocsr()[freedofs, :].tocsc()[:, freedofs].tocsr()
    w, Ured = vibration_modes(Mred, Keqred, modes=args.modes)
    U = np.zeros((n, mu), dtype=Ured.dtype)
    U[freedofs, :] = Ured
    Xmax, Xmin = np.max(X, axis=1), np.min(X, axis=1)
    bbdiag = np.linalg.norm(Xmax - Xmin)
    h = args.eps * bbdiag
    Theta = modal_derivatives(
        X.flatten(order="F"),
        Ured,
        Mred,
        Keqred,
        h,
        freedofs,
        lambda x: stiffness(E, X, egU, wgU, GNegU, mug, lambdag, x, energy, element),
    )
    # Compute rank-3 quadratic manifold tensor
    Q = (Theta + Theta.transpose(0, 2, 1)) / 2
    # Compute augmented linear basis
    Thetared = Theta.reshape(n, -1)[freedofs, :]
    # Project out linear basis from modal derivatives
    Mredsqrt = scipy.sparse.diags_array(
        np.sqrt(M.diagonal()[freedofs]), shape=(nfree, nfree)
    )
    Mredinvsqrt = scipy.sparse.diags_array(
        1.0 / Mredsqrt.diagonal(), shape=(nfree, nfree)
    )
    Thetared = np.eye(nfree) @ Thetared - Ured @ (Ured.T @ (Mred @ Thetared))
    # Compute M-orthogonal SVD
    Zredhat, s2, _ = np.linalg.svd(
        Mredsqrt @ Thetared, full_matrices=False, compute_uv=True
    )
    # Undo the M-norm on the SVD basis
    Zred = Mredinvsqrt @ Zredhat
    tau = h
    mz = np.argmax(s2 < tau) if s2[-1] < tau else s2.shape[0]
    Zred = Zred[:, :mz]
    s = np.sqrt(s2)
    ws = np.hstack([w, s[:mz]])
    Z = np.zeros((n, mz), dtype=Zred.dtype)
    Z[freedofs, :] = Zred
    UZ = np.hstack([U, Z])

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_volume_mesh("model", X.T, E.T)
    # drop-down to select mapping type (linear/quadratic)
    mappings = ["Linear", "Quadratic"]
    mapping = mappings[0]
    mode = 6
    q = np.zeros(mu + mz, dtype=U.dtype)
    t0 = time.time()
    t = 0
    c = 0.15
    k = 0.05

    def callback():
        global mode, c, k, q, mapping
        changed, map_idx = imgui.Combo("Mapping", mappings.index(mapping), mappings)
        mapping = mappings[map_idx]
        nrdofs = q.shape[0] if mapping == "Linear" else mu
        changed, mode = imgui.InputInt(f"Mode {mode}/{nrdofs-1}", mode)
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)

        mode = max(0, min(nrdofs - 1, mode))
        t = time.time() - t0
        etam = signal(ws[mode], t, c, k)
        q[:] = 0
        q[mode] = etam
        if mapping == "Quadratic":
            u = quadratic_map(U, Q, q[:mu])
        else:
            u = linear_map(UZ, q)

        V = X.T + u.reshape(X.shape[1], 3)
        vm.update_vertex_positions(V)

    ps.set_user_callback(callback)
    ps.show()

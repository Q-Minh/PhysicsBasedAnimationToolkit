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


def rest_pose_hyper_elastic_modes(
    E: np.ndarray,
    X: np.ndarray,
    element: pbat.fem.Element,
    Ye: np.ndarray,
    nue: np.ndarray,
    rhoe: np.ndarray,
    energy=pbat.fem.HyperElasticEnergy.StableNeoHookean,
    modes: int = 30,
    sigma: float = -1e-5,
    zero: float = 0.0,
):
    """Computes natural (linear) displacement modes of mesh.

    Args:
        E (np.ndarray): Element matrix.
        X (np.ndarray): Node coordinates.
        element (_pbat.fem.Element): Element type.
        Ye (np.ndarray): |# elements| x 1 Young's modulus.
        nue (np.ndarray): |# elements| x 1 Poisson's ratio.
        rhoe (np.ndarray): |# elements| x 1 Mass density.
        energy (_pbat.fem.HyperElasticEnergy, optional): Constitutive model. Defaults to _fem.HyperElasticEnergy.StableNeoHookean.
        modes (int, optional): Maximum number of modes to compute. Defaults to 30.
        sigma (float, optional): Shift (see scipy.sparse.eigsh). Defaults to -1e-5.
        zero (float, optional): Numerical zero used to cull modes. Defaults to 0.

    Returns:
        (np.ndarray, np.ndarray, sp.sparse.csr_matrix, sp.sparse.csr_matrix): (w,U,M,K) s.t. w is
        a |# modes| vector of amplitudes, U is a n x |# modes| array of displacement modes in columns,
        M is the mass matrix and K is the stiffness matrix.
    """
    # Compute elasticity at rest pose
    x = np.ravel(X, order="F")
    # Compute the mass matrix
    order = 1
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
    # Compute the hyper-elastic potential's hessian
    qorderU = order
    wgU = pbat.fem.mesh_quadrature_weights(
        E, X, element, order=order, quadrature_order=qorderU
    )
    egU = pbat.fem.mesh_quadrature_elements(E, wgU)
    GNegU = pbat.fem.shape_function_gradients(
        E, X, element=element, order=order, dims=X.shape[0], quadrature_order=qorderU
    )
    mug, lambdag = pypbat.fem.lame_coefficients(Ye, nue)
    K = stiffness(E, X, egU, wgU, GNegU, mug, lambdag, x, energy, element, order)
    modes = min(modes, x.shape[0])
    l, V = sp.sparse.linalg.eigsh(K, k=modes, M=M, sigma=sigma, which="LM")
    V = V / sp.linalg.norm(V, axis=0, keepdims=True)
    l[l <= zero] = 0
    w = np.sqrt(l)
    return w, V, M, K, egU, wgU, GNegU, mug, lambdag


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
        help="Path to input tetrahedral mesh",
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
        default=1e-4,
    )
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    element = pbat.fem.Element.Tetrahedron
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    X, E = pbat.fem.mesh(V.T, C.T, element=element)
    Ye = np.full(E.shape[1], args.Y)
    nue = np.full(E.shape[1], args.nu)
    rhoe = np.full(E.shape[1], args.rho)
    w, U, M, Keq, egU, wgU, GNegU, mug, lambdag = rest_pose_hyper_elastic_modes(
        E, X, element, Ye=Ye, nue=nue, rhoe=rhoe, energy=energy, modes=args.modes
    )
    n, m = U.shape
    # compute bounding box diagonal length
    Xmax, Xmin = np.max(X, axis=1), np.min(X, axis=1)
    bbdiag = np.linalg.norm(Xmax - Xmin)
    h = args.eps * bbdiag
    Q = np.empty((n, m, m), dtype=U.dtype)
    b = np.zeros((n + 1, 1), dtype=U.dtype)
    for i in range(m):
        # Factorize modal derivative matrix
        A12 = -M @ U[:, i]
        A12 = A12.reshape(-1, 1)
        # To use Eigen LDLT
        A = scipy.sparse.bmat(
            [[Keq - w[i] ** 2 * M, A12], [A12.T, None]],
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
            xleft = np.ravel(X, order="F") - h * U[:, j]
            xright = np.ravel(X, order="F") + h * U[:, j]
            Kleft = stiffness(
                E, X, egU, wgU, GNegU, mug, lambdag, xleft, energy, element
            )
            Kright = stiffness(
                E, X, egU, wgU, GNegU, mug, lambdag, xright, energy, element
            )
            dKdetaj = (Kright - Kleft) / h
            b[:-1, 0] = -dKdetaj @ U[:, i]
            # If Eigen LDLT
            thetaij = Ainv.solve(b).squeeze()[:-1]
            Q[:, i, j] = thetaij
            # If SuperLU
            # thetaij = Ainv(b).squeeze()[:-1]
    # symmetrize Q
    Q = (Q + Q.transpose(0, 2, 1)) / 2

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_volume_mesh("model", X.T, E.T)
    # drop-down to select mapping type (linear/quadratic)
    mappings = ["Linear", "Quadratic"]
    mapping = mappings[0]
    mode = 6
    q = np.zeros(m, dtype=U.dtype)
    t0 = time.time()
    t = 0
    c = 0.15
    k = 0.05

    def callback():
        global mode, c, k, q, mapping
        changed, map_idx = imgui.Combo("Mapping", mappings.index(mapping), mappings)
        changed, mode = imgui.InputInt("Mode", mode)
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)

        mode = max(0, min(m - 1, mode))
        mapping = mappings[map_idx]
        t = time.time() - t0
        etam = signal(w[mode], t, c, k)
        q[:] = 0
        q[mode] = etam
        if mapping == "Quadratic":
            u = quadratic_map(U, Q, q)
        else:
            u = linear_map(U, q)

        V = X.T + u.reshape(X.shape[1], 3)
        vm.update_vertex_positions(V)

    ps.set_user_callback(callback)
    ps.show()

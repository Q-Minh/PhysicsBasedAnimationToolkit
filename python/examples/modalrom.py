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
import os
import rom_mcmc


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
    w: np.ndarray,
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
        w (np.ndarray): |# modes| array of vibration mode frequencies
        Mred (scipy.sparse.csr_matrix): |# free dofs| x |# free dofs| mass matrix
        Keqred (scipy.sparse.csr_matrix): |# free dofs| x |# free dofs| stiffness matrix
        h (float): step size for finite difference
        freedofs (np.ndarray): array of free dofs
        compute_stiffness (typing.Callable[[np.ndarray], scipy.sparse.csr_matrix]): function to compute stiffness matrix

    Returns:
        np.ndarray: The quadratic manifold rank-3 tensor
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
        xleft[freedofs] = x[freedofs] - h * Ured[:, j]
        xright[freedofs] = x[freedofs] + h * Ured[:, j]
        Kleft = compute_stiffness(xleft)
        Kright = compute_stiffness(xright)
        dKdetaj = (Kright - Kleft) / (2 * h)
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
    """Returns potential(x, gradient=False) evaluating the total hyper-elastic energy at positions x.

    Args:
        x (np.ndarray): 3*|# nodes| array of node positions in F order
        gradient (bool): if True, also return the 3*|# nodes| energy gradient

    Returns:
        float | Tuple[float, np.ndarray]: The energy, and its gradient if requested
    """
    flags = pbat.fem.ElementElasticityComputationFlags.Potential
    flags_grad = flags | pbat.fem.ElementElasticityComputationFlags.Gradient

    def potential(x: np.ndarray, gradient: bool = False):
        U, gradU, _ = pbat.fem.hyper_elastic_potential(
            E=E,
            n_nodes=X.shape[1],
            eg=np.ravel(egU),
            wg=np.ravel(wgU),
            GNeg=GNegU,
            mug=np.ravel(mug),
            lambdag=np.ravel(lambdag),
            x=np.asarray(x, dtype=np.float32),
            energy=energy,
            flags=flags_grad if gradient else flags,
            element=element,
            order=order,
            dims=X.shape[0],
        )
        return (U, gradU.astype(np.float64)) if gradient else U

    return potential


def signal(w: float, t: float, c: float, k: float):
    return c * np.sin(k * w * t)


def linear_map(U: np.ndarray, q: np.ndarray):
    return U @ q


def quadratic_map(U: np.ndarray, Q: np.ndarray, q: np.ndarray):
    return U @ q + 0.5 * np.einsum("nij,i,j->n", Q, q, q)


def load_model(input: str, rho: float, Y: float, nu: float, element, energy):
    """Loads a tetrahedral mesh, or an FemElastoDynamics from an HDF5 archive.

    Args:
        input (str): Path to a tetrahedral mesh, or filepath:h5path of an FemElastoDynamics.
        rho (float): Mass density (mesh input only).
        Y (float): Young's modulus (mesh input only).
        nu (float): Poisson's ratio (mesh input only).
        element (pbat.fem.Element): Element type.
        energy (pbat.fem.HyperElasticEnergy): Hyper-elastic energy model.

    Returns:
        Tuple: (E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs), with M the lumped mass matrix.
    """
    file_path, sep, group_path = str(input).rpartition(":")
    if sep and file_path and not (len(file_path) == 1 and file_path.isalpha()):
        import gc

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
    return E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs


def build_rom(
    E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, energy, element, modes, eps
):
    """Computes vibration modes, modal derivatives and the augmented linear basis.

    Modal derivatives are projected M-orthogonally off the span of the vibration modes.

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
    tau = h
    mz = np.argmax(s2 < tau) if s2[-1] < tau else s2.shape[0]
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
):
    """Builds the reduced model to sample on, excluding the first `rigid` (zero-frequency) modes.

    Args:
        mapping (str): "quadratic" for u = U q + 1/2 Q:qq, "linear" for u = U q.
        U (np.ndarray): n x |# modes| vibration modes.
        Q (np.ndarray): n x |# modes| x |# modes| quadratic manifold tensor.
        w (np.ndarray): |# modes| mode frequencies.
        m (np.ndarray): n lumped mass matrix diagonal.
        x0 (np.ndarray): n rest positions in F order.
        potential (typing.Callable): Full-space energy, see elastic_energy.
        rigid (int): Number of leading rigid modes to exclude.
        weight (str): Sampling weight, see rom_mcmc.WEIGHTS.

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
        Ur, Qr, m, x0.astype(np.float64), potential, scale, weight
    )
    return model, w[keep]


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
        "--headless",
        help="Sample without the GUI and write the samples to --output",
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output HDF5 file for --headless sampling",
        type=str,
        dest="output",
    )
    parser.add_argument(
        "--energy",
        help="Target deformation energy E* in J",
        type=float,
        dest="energy",
        default=1.0,
    )
    parser.add_argument(
        "--samples",
        help="Number of samples",
        type=int,
        dest="samples",
        default=100,
    )
    parser.add_argument(
        "--sampler",
        help="polar: Metropolis on directions with exact radial solves, "
        "crw: constrained random walk with projection, "
        "ray: independent random directions (not distributed as the target)",
        choices=tuple(rom_mcmc.SAMPLERS),
        default="polar",
    )
    parser.add_argument(
        "--weight",
        help="Target distribution on the energy level set (see rom_mcmc)",
        choices=rom_mcmc.WEIGHTS,
        default="mass",
    )
    parser.add_argument(
        "--mapping",
        help="Reduced displacement map to sample on",
        choices=("quadratic", "linear"),
        default="quadratic",
    )
    parser.add_argument(
        "--burn-in",
        help="Number of burn-in steps, during which the step size adapts",
        type=int,
        dest="burn_in",
        default=1000,
    )
    parser.add_argument(
        "--pilot",
        help="Number of steps used to estimate the autocorrelation time for automatic thinning",
        type=int,
        dest="pilot",
        default=1000,
    )
    parser.add_argument(
        "--thin",
        help="Keep every thin-th step. Defaults to the estimated autocorrelation time.",
        type=int,
        dest="thin",
        default=None,
    )
    parser.add_argument(
        "--seed",
        help="Random seed",
        type=int,
        dest="seed",
        default=0,
    )
    args = parser.parse_args()
    if args.headless and not args.output:
        parser.error("--headless requires --output")

    element = pbat.fem.Element.Tetrahedron
    energy = pbat.fem.HyperElasticEnergy.StableNeoHookean
    E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs = load_model(
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

    def make_model(mapping: str):
        return reduced_model(
            mapping, U, Q, w, M.diagonal(), x0, psi, rigid, args.weight
        )

    if args.headless:
        model, wr = make_model(args.mapping)
        rng = np.random.default_rng(args.seed)
        sampler = rom_mcmc.make_sampler(args.sampler, model, args.energy, rng)
        t0 = time.time()
        batch = rom_mcmc.run_chain(
            sampler, args.samples, args.burn_in, args.pilot, args.thin
        )
        elapsed = time.time() - t0
        attrs = dict(
            input=args.input,
            n_linear_modes=mu,
            rigid_modes_excluded=rigid,
            modal_derivatives=args.mapping == "quadratic",
            eps=args.eps,
            sampler=args.sampler,
            target_energy=args.energy,
            seed=args.seed,
            git_commit=rom_mcmc.git_commit(os.path.dirname(os.path.abspath(__file__))),
            sampling_time=elapsed,
        )
        rom_mcmc.write_samples(args.output, X, model, wr, batch, attrs)
        s = batch.stats
        print(
            f"Wrote {args.samples} samples to {args.output} in {elapsed:.1f} s: "
            f"acceptance={s['acceptance']:.2f}, thin={s['thin']}, tau={s['tau']:.1f}, "
            f"ESS={s['ess']:.0f}, star violations={s['star_violations']}, "
            f"energy noise={s['energy_noise']:.2g} J"
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
    c = 0.15
    k = 0.05
    # Fixed-energy sampling state. The chain restarts whenever its settings change.
    samplers = list(rom_mcmc.SAMPLERS)
    sampler_kind = samplers[0]
    target_energy = args.energy
    samples_per_click = 10
    show_samples = False
    chain = None
    chain_key = None
    model = None
    samples = []
    sample_idx = 0
    stats = {}
    rng = np.random.default_rng(args.seed)

    def callback():
        global mode, c, k, q, mapping
        global sampler_kind, target_energy, samples_per_click, show_samples
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
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)

        imgui.Separator()
        imgui.Text("Fixed-energy sampling (Linear samples on U only)")
        changed, target_energy = imgui.InputFloat("Energy E* (J)", target_energy)
        target_energy = max(target_energy, 1e-8)
        changed, sampler_idx = imgui.Combo(
            "Sampler", samplers.index(sampler_kind), samplers
        )
        sampler_kind = samplers[sampler_idx]
        changed, samples_per_click = imgui.InputInt(
            "Samples per click", samples_per_click
        )
        samples_per_click = max(samples_per_click, 1)
        if imgui.Button("Sample"):
            key = (mapping, target_energy, sampler_kind)
            if key != chain_key:
                model, _ = make_model(mapping.lower())
                chain = rom_mcmc.make_sampler(sampler_kind, model, target_energy, rng)
                chain.burn_in(args.burn_in)
                chain.estimate_thinning(args.pilot)
                chain_key = key
                samples = []
            batch = chain.sample(samples_per_click)
            stats = batch.stats
            samples.extend(zip(batch.q, batch.energy))
            sample_idx = len(samples) - 1
            show_samples = True
        if samples:
            changed, show_samples = imgui.Checkbox("Show samples", show_samples)
            # The slider shares its label with the "Sample" button, so give it its own ID scope
            imgui.PushID("sample_slider")
            changed, sample_idx = imgui.SliderInt(
                "Sample", sample_idx, 0, len(samples) - 1
            )
            imgui.PopID()
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

        if show_samples and samples:
            u = model.displacement(samples[sample_idx][0])
        else:
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

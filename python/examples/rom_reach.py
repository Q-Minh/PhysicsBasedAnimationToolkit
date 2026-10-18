"""Full-order static targets and reduced-model fits, to test which shapes a reduced model can reach.

Targets are full-order static equilibria under authored loads (gravity, a force on a region of the mesh)
or under prescribed displacements of a region (handles). Reduced models are then fit to a target in
three ways:

- geometric least squares, min_q 1/2 |u(q) - u_t|_M^2 (linear bases in closed form),
- least squares regularized by the elastic energy, which trades the geometric match for plausibility,
- the reduced model's own equilibrium under the same load or handles, i.e. what it predicts.

static_equilibrium also solves one implicit Euler (BDF1) step when given an inertia term, which
make_benchmark.py uses for the reference solutions of dynamic problems. Its accuracy is limited by pbat's float32
energy: it stops at the gradient noise floor (a residual of a few N on the beam and trampoline).

modalrom.py's "Shape reachability" GUI panel drives this module. Usage from a script, with the model from
modalrom.load_model, modalrom.build_rom and modalrom.reduced_model:

    tip = rom_reach.box_region(X, [0.95, 0, 0], [1, 1, 1])
    ut, info = rom_reach.static_equilibrium(potential, x0, freedofs, rom_reach.gravity_load(m, [0, 0, -0.4]))
    u, q = rom_reach.manifold_fit(quadratic_model, ut)              # geometric best fit
    u, q = rom_reach.rom_equilibrium(quadratic_model, rom_reach.gravity_load(m, [0, 0, -0.4]))  # ROM prediction
    print(rom_reach.fit_metrics(u, ut, m)["relative_error"])
"""

import typing

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg


def box_region(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Nodes whose rest positions lie in a box given as fractions of the mesh's bounding box.

    Args:
        X (np.ndarray): 3 x |# nodes| rest positions.
        lo (np.ndarray): 3 lower box bounds, as fractions in [0, 1] of the bounding box along x, y, z.
        hi (np.ndarray): 3 upper box bounds, as fractions in [0, 1].

    Returns:
        np.ndarray: Indices of the nodes inside the box.
    """
    Xmin, Xmax = X.min(axis=1), X.max(axis=1)
    t = (X - Xmin[:, np.newaxis]) / np.maximum(Xmax - Xmin, 1e-12)[:, np.newaxis]
    lo, hi = np.asarray(lo)[:, np.newaxis], np.asarray(hi)[:, np.newaxis]
    return np.flatnonzero(np.all((t >= lo) & (t <= hi), axis=0))


def node_dofs(nodes: np.ndarray) -> np.ndarray:
    """DOFs of nodes, with DOFs interleaved per node (3 * i + d)."""
    return (3 * np.asarray(nodes)[:, np.newaxis] + np.arange(3)).ravel()


def gravity_load(m: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Full-space gravity load M g.

    Args:
        m (np.ndarray): 3*|# nodes| lumped masses per DOF, interleaved per node.
        g (np.ndarray): 3 gravitational acceleration in m/s^2.

    Returns:
        np.ndarray: 3*|# nodes| load in N.
    """
    return m * np.tile(np.asarray(g, dtype=np.float64), m.shape[0] // 3)


def region_load(m: np.ndarray, nodes: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Full-space load applying a total force to nodes, distributed in proportion to their masses.

    Args:
        m (np.ndarray): 3*|# nodes| lumped masses per DOF, interleaved per node.
        nodes (np.ndarray): Indices of the loaded nodes, e.g. from box_region.
        force (np.ndarray): 3 total force in N.

    Returns:
        np.ndarray: 3*|# nodes| load in N, zero outside the nodes.
    """
    f = np.zeros(m.shape[0])
    if len(nodes) == 0:
        return f
    mn = m[3 * np.asarray(nodes)]
    f[node_dofs(nodes)] = ((mn / mn.sum())[:, np.newaxis] * np.asarray(force)[np.newaxis, :]).ravel()
    return f


def static_equilibrium(
    potential: typing.Callable,
    x0: np.ndarray,
    free: np.ndarray,
    fext: np.ndarray | None = None,
    handle_dofs: np.ndarray | None = None,
    handle_u: np.ndarray | None = None,
    steps: int = 5,
    max_iters: int = 50,
    rtol: float = 1e-9,
    inertia: typing.Tuple[np.ndarray, np.ndarray] | None = None,
    u0: np.ndarray | None = None,
) -> typing.Tuple[np.ndarray, dict]:
    """Full-order equilibrium, min_u Psi(x0 + u) - fext^T u [+ 1/2 sum_i k_i (u_i - ut_i)^2], by load-stepped
    Newton with backtracking.

    With inertia = (k, ut), k = M / h^2 and ut = xtilde - x0, this is one implicit Euler (BDF1) step,
    min_x 1/(2 h^2) |x - xtilde|_M^2 + Psi(x). The load and the handle displacements are ramped linearly
    over `steps` increments, each solved by Newton from the previous one. Each Newton solve stops when the
    decrement drops below rtol * (1 + |Psi|), or when the line search can't decrease the objective any
    more, which happens at the energy's round-off floor (pbat evaluates it in float32).

    Algorithm (one Newton iteration on the unknown DOFs, i.e. free DOFs that aren't handles):
        1. Assemble the gradient r = grad Psi - f + k (u - ut) and the Hessian H = hess Psi + diag(k), with pbat's
           absolute-value SPD correction of the element Hessians, so H is positive definite and the Newton step a
           descent direction.
        2. Solve H du = -r (sparse direct solve) and stop if the decrement -r^T du is small enough.
        3. Backtracking line search: halve the step from 1 until the Armijo condition (c = 1e-4) holds; give up
           (round-off floor reached) below a step of 1e-8.

    Args:
        potential (typing.Callable): potential(x, gradient, hessian) returns Psi(x), or
            (Psi(x), grad Psi(x), Hessian of Psi(x) as a sparse matrix) if hessian.
        x0 (np.ndarray): n rest positions in F order.
        free (np.ndarray): DOFs that may move (e.g. all non-Dirichlet DOFs).
        fext (np.ndarray | None): n external load, or None.
        handle_dofs (np.ndarray | None): Subset of free DOFs with prescribed displacements, or None.
        handle_u (np.ndarray | None): Prescribed displacements of handle_dofs.
        steps (int): Number of load increments. More increments make large loads and handle displacements converge
            reliably (each Newton solve starts close to its solution) at the cost of more solves; 1 suffices for
            well-conditioned problems such as implicit Euler steps.
        max_iters (int): Maximum Newton iterations per increment.
        rtol (float): Relative Newton decrement tolerance. Deliberately tight: in practice the solve stops at the
            float32 round-off floor, when the line search can no longer decrease the objective.
        inertia (typing.Tuple[np.ndarray, np.ndarray] | None): (k, ut), n diagonal inertia stiffness (M / h^2 for a
            time step h) and n inertial target displacement (xtilde - x0), or None for a static solve.
        u0 (np.ndarray | None): n initial displacement (e.g. the inertial target, a good guess for small h), or
            zero. Fixed DOFs keep their u0 values.

    Returns:
        typing.Tuple[np.ndarray, dict]: The n equilibrium displacement u, and solve statistics: Newton
        iterations, the last decrement, and the largest entry of the last Newton correction, which estimates
        how far u is from the exact solution.
    """
    n = x0.shape[0]
    fext = np.zeros(n) if fext is None else fext
    handle_dofs = np.zeros(0, dtype=np.int64) if handle_dofs is None else np.asarray(handle_dofs)
    unknown = np.setdiff1d(free, handle_dofs)
    u = np.zeros(n) if u0 is None else np.array(u0, dtype=np.float64)
    k, ut = (np.zeros(n), np.zeros(n)) if inertia is None else inertia
    iterations, decrement, correction = 0, np.inf, np.inf
    for s in range(1, steps + 1):
        scale = s / steps
        if handle_dofs.size:
            u[handle_dofs] = scale * np.asarray(handle_u)
        f = scale * fext

        def objective(u):
            d = u - ut
            return float(potential(x0 + u, False, False)) - f @ u + 0.5 * d @ (k * d)

        for _ in range(max_iters):
            psi, g, H = potential(x0 + u, True, True)
            r = (g - f + k * (u - ut))[unknown]
            Huu = H.tocsr()[unknown, :].tocsc()[:, unknown].astype(np.float64)
            Huu = Huu + scipy.sparse.diags(k[unknown])
            du = scipy.sparse.linalg.spsolve(Huu, -r)
            correction = float(np.abs(du).max())
            iterations += 1
            decrement = -(r @ du)
            if 0.5 * decrement <= rtol * (1.0 + abs(psi)):
                break
            f0, alpha, step = objective(u), 1.0, np.zeros(n)
            while alpha > 1e-8:
                step[unknown] = alpha * du
                if objective(u + step) <= f0 - 1e-4 * alpha * decrement:
                    break
                alpha *= 0.5
            else:
                break
            u += step
    return u, dict(iterations=iterations, decrement=decrement, correction=correction)


def deformation_energy(potential: typing.Callable, x0: np.ndarray, u: np.ndarray) -> float:
    """Deformation energy Psi(x0 + u) - Psi(x0).

    Args:
        potential (typing.Callable): potential(x, gradient, hessian), see static_equilibrium.
        x0 (np.ndarray): n rest positions in F order.
        u (np.ndarray): n displacement.

    Returns:
        float: The deformation energy in J.
    """
    return float(potential(x0 + u, False, False)) - float(potential(x0, False, False))


def linear_fit(B: np.ndarray, m: np.ndarray, ut: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """M-orthogonal projection of ut onto span(B), the best fit min_c |B c - ut|_M.

    Args:
        B (np.ndarray): n x k basis, e.g. the modes U or the augmented basis [U Z].
        m (np.ndarray): n lumped masses (the diagonal of M).
        ut (np.ndarray): n target displacement.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: The fitted displacement B c and the coefficients c.
    """
    MB = m[:, np.newaxis] * B
    c = np.linalg.solve(B.T @ MB, MB.T @ ut)
    return B @ c, c


def manifold_fit(
    model,
    ut: np.ndarray,
    alpha: float = 0.0,
    energy_scale: float = 1.0,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Fits a reduced model to a target displacement, optionally regularized by the elastic energy.

    Minimizes 1/2 |u(q) - ut|_M^2 / |ut|_M^2 + alpha * U(q) / energy_scale, so that alpha is
    dimensionless when energy_scale is the target's deformation energy. alpha = 0 is the geometric best
    fit (Levenberg-Marquardt); alpha > 0 trades the match for lower energy (L-BFGS). Starts from the
    projection of ut onto the model's linear basis.

    Args:
        model: Reduced model (rom_mcmc.QuadraticManifold, without an external force).
        ut (np.ndarray): n target displacement.
        alpha (float): Weight of the energy regularizer. With energy_scale the target's deformation energy, the
            two terms are comparable for alpha ~ 1; in practice alpha ~ 1e-3-1e-2 trims spurious strain while
            keeping the shape close, and alpha ~ 1 already collapses the fit toward the rest shape.
        energy_scale (float): Energy that normalizes the regularizer, e.g. the target's deformation energy.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: The fitted displacement u(q) and the coordinates q.
    """
    m = model.M
    norm2 = max(ut @ (m * ut), 1e-300)
    _, q0 = linear_fit(model.U, m, ut)
    if alpha == 0.0:
        w = np.sqrt(m / norm2)
        result = scipy.optimize.least_squares(
            lambda q: w * (model.displacement(q) - ut),
            q0,
            jac=lambda q: w[:, np.newaxis] * model.jacobian(q),
            method="lm",
        )
        q = result.x
    else:

        def objective(q):
            d = model.displacement(q) - ut
            e, gq = model.energy_and_gradient(q)
            fit = 0.5 * d @ (m * d) / norm2
            return fit + alpha * e / energy_scale, model.jacobian(q).T @ (m * d) / norm2 + alpha * gq / energy_scale

        q = scipy.optimize.minimize(objective, q0, jac=True, method="L-BFGS-B").x
    return model.displacement(q), q


def rom_equilibrium(
    model,
    fext: np.ndarray | None = None,
    handle_dofs: np.ndarray | None = None,
    handle_u: np.ndarray | None = None,
    kappa: float = 0.0,
    steps: int = 5,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """The reduced model's own static equilibrium under a load and/or handles.

    Minimizes U(q) - fext^T u(q) + kappa/2 sum_h m_h (u_h(q) - handle_u_h)^2 over q, ramping the load and
    handles over `steps` increments with warm starts. Handles are penalized because a reduced model can't
    in general match many prescribed DOFs exactly; kappa (1/s^2) should exceed the model's stiffest
    squared frequency.

    Args:
        model: Reduced model (rom_mcmc.QuadraticManifold, without an external force).
        fext (np.ndarray | None): n external load.
        handle_dofs (np.ndarray | None): Handle DOFs.
        handle_u (np.ndarray | None): Prescribed displacements of handle_dofs.
        kappa (float): Mass-weighted handle penalty stiffness in 1/s^2. It acts like a spring of stiffness
            kappa m_h on each handle DOF, so values well above the model's largest squared frequency make the
            handles nearly hard constraints (the GUI uses 100 omega_max^2); larger values make L-BFGS converge slower.
        steps (int): Number of load increments, each warm-started from the previous solution, which keeps the
            solve on the load path from rest (the reduced energy may have several minima).

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: The equilibrium displacement u(q) and coordinates q.
    """
    n = model.U.shape[0]
    fext = np.zeros(n) if fext is None else fext
    handle_dofs = np.zeros(0, dtype=np.int64) if handle_dofs is None else np.asarray(handle_dofs)
    mh = model.M[handle_dofs]
    q = np.zeros(model.dims)
    for s in range(1, steps + 1):
        scale = s / steps
        f, d = scale * fext, scale * np.asarray(handle_u) if handle_dofs.size else None

        def objective(q):
            u = model.displacement(q)
            e, gq = model.energy_and_gradient(q)
            g = -f
            value = e - f @ u
            if handle_dofs.size:
                r = u[handle_dofs] - d
                value += 0.5 * kappa * r @ (mh * r)
                g = g.copy()
                g[handle_dofs] += kappa * mh * r
            return value, gq + model.jacobian(q).T @ g

        q = scipy.optimize.minimize(objective, q, jac=True, method="L-BFGS-B").x
    return model.displacement(q), q


def fit_metrics(u: np.ndarray, ut: np.ndarray, m: np.ndarray) -> dict:
    """Errors of a fitted displacement u against a target ut.

    Args:
        u (np.ndarray): n fitted displacement.
        ut (np.ndarray): n target displacement.
        m (np.ndarray): n lumped masses (the diagonal of M).

    Returns:
        dict: relative_error |u - ut|_M / |ut|_M, max_error (largest vertex error in m), and vertex_error
        (|# nodes| per-vertex error magnitudes in m).
    """
    d = u - ut
    vertex_error = np.linalg.norm(d.reshape(-1, 3), axis=1)
    return dict(
        relative_error=float(np.sqrt(d @ (m * d) / max(ut @ (m * ut), 1e-300))),
        max_error=float(vertex_error.max()),
        vertex_error=vertex_error,
    )

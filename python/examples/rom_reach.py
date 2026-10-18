"""Full-order static targets and reduced-model fits, to test which shapes a reduced model can reach.

Targets are full-order static equilibria under authored loads (gravity, a force on a region of the mesh)
or under prescribed displacements of a region (handles). Reduced models are then fit to a target in
three ways:

- geometric least squares, min_q 1/2 |u(q) - u_t|_M^2 (linear bases in closed form),
- least squares regularized by the elastic energy, which trades the geometric match for plausibility,
- the reduced model's own equilibrium under the same load or handles, i.e. what it predicts.
"""

import typing

import numpy as np
import scipy.optimize
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
    """Full-space gravity load M g, given the lumped DOF masses m and the acceleration g (3,)."""
    return m * np.tile(np.asarray(g, dtype=np.float64), m.shape[0] // 3)


def region_load(m: np.ndarray, nodes: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Full-space load applying the total force (3,) to nodes, distributed in proportion to their masses."""
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
) -> typing.Tuple[np.ndarray, dict]:
    """Full-order static equilibrium, min_u Psi(x0 + u) - fext^T u, by load-stepped Newton with backtracking.

    The load and the handle displacements are ramped linearly over `steps` increments, each solved by
    Newton from the previous one. Each Newton solve stops when the decrement drops below
    rtol * (1 + |Psi|), or when the line search can't decrease the objective any more, which happens
    at the energy's round-off floor (pbat evaluates it in float32).

    Args:
        potential (typing.Callable): potential(x, gradient, hessian) returns Psi(x), or
            (Psi(x), grad Psi(x), Hessian of Psi(x) as a sparse matrix) if hessian.
        x0 (np.ndarray): n rest positions in F order.
        free (np.ndarray): DOFs that may move (e.g. all non-Dirichlet DOFs).
        fext (np.ndarray | None): n external load, or None.
        handle_dofs (np.ndarray | None): Subset of free DOFs with prescribed displacements, or None.
        handle_u (np.ndarray | None): Prescribed displacements of handle_dofs.
        steps (int): Number of load increments.
        max_iters (int): Maximum Newton iterations per increment.
        rtol (float): Relative Newton decrement tolerance.

    Returns:
        typing.Tuple[np.ndarray, dict]: The n equilibrium displacement u, and solve statistics.
    """
    n = x0.shape[0]
    fext = np.zeros(n) if fext is None else fext
    handle_dofs = np.zeros(0, dtype=np.int64) if handle_dofs is None else np.asarray(handle_dofs)
    unknown = np.setdiff1d(free, handle_dofs)
    u = np.zeros(n)
    iterations, decrement = 0, np.inf
    for s in range(1, steps + 1):
        scale = s / steps
        if handle_dofs.size:
            u[handle_dofs] = scale * np.asarray(handle_u)
        f = scale * fext

        def objective(u):
            return float(potential(x0 + u, False, False)) - f @ u

        for _ in range(max_iters):
            psi, g, H = potential(x0 + u, True, True)
            r = (g - f)[unknown]
            Huu = H.tocsr()[unknown, :].tocsc()[:, unknown].astype(np.float64)
            du = scipy.sparse.linalg.spsolve(Huu, -r)
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
    return u, dict(iterations=iterations, decrement=decrement)


def deformation_energy(potential: typing.Callable, x0: np.ndarray, u: np.ndarray) -> float:
    """Psi(x0 + u) - Psi(x0)."""
    return float(potential(x0 + u, False, False)) - float(potential(x0, False, False))


def linear_fit(B: np.ndarray, m: np.ndarray, ut: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """M-orthogonal projection of ut onto span(B), the best fit min_c |B c - ut|_M.

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
        alpha (float): Weight of the energy regularizer.
        energy_scale (float): Energy that normalizes the regularizer, e.g. the target's.

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
        kappa (float): Mass-weighted handle penalty stiffness.
        steps (int): Number of load increments.

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
    """Relative M-norm error, maximum vertex error, and per-vertex error magnitudes of u against ut."""
    d = u - ut
    vertex_error = np.linalg.norm(d.reshape(-1, 3), axis=1)
    return dict(
        relative_error=float(np.sqrt(d @ (m * d) / max(ut @ (m * ut), 1e-300))),
        max_error=float(vertex_error.max()),
        vertex_error=vertex_error,
    )

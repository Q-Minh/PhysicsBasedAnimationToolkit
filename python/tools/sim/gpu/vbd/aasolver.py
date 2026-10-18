import warp as wp
from ..common import profiling

from python.tools.sim.gpu.vbd.solver import (
    finalize_subproblem,
    initialize_solve,
    iterate,
    prepare_subproblem,
    solve_outer_loop,
)

from ..contact.mesh.cd import ContactDetection
from ..elasticity.fem import FemElastoDynamics
from .params import AndersonLinearSolver, AndersonParams
from ..contact.dynamics import MeshDynamics as ContactDynamics


# Gram kernel tiling: each block multiplies one (m+1, _GRAM_TILE_K) tile of [F f]^T by its transpose.
_GRAM_TILE_K = 256
_GRAM_BLOCK_DIM = 64
_SOLVE_BLOCK_DIM = 32
# float32 Gram entries carry relative rounding errors of about m * eps, so pivots below that level
# (relative to the largest diagonal entry) are noise. This floors the Cholesky rank threshold.
_FLOAT32_EPS = 1.1920929e-07
# Relative residual at which the Jacobi-PCG solve of the (normalized) normal equations stops.
_PCG_RTOL = 1e-6


@wp.kernel
def _anderson_pre_iterate_kernel(
    x: wp.array[wp.vec3f],
    xkm1: wp.array[wp.vec3f],
    Xk: wp.array2d[wp.vec3f], 
    dkl: int,
):
    i = wp.tid()
    xi = x[i]
    Xk[dkl, i] = xi - xkm1[i]  # pyright: ignore[reportIndexIssue]
    xkm1[i] = xi  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _anderson_post_iterate_kernel(
    x: wp.array[wp.vec3f],
    xkm1: wp.array[wp.vec3f],
    fk: wp.array[wp.vec3f],
    fkm1: wp.array[wp.vec3f],
    Fk: wp.array2d[wp.vec3f],
    dkl: int,
):
    i = wp.tid()
    fki = x[i] - xkm1[i]  # pyright: ignore[reportIndexIssue]
    fk[i] = fki  # pyright: ignore[reportIndexIssue]
    Fk[dkl, i] = fki - fkm1[i]  # pyright: ignore[reportIndexIssue]
    fkm1[i] = fki  # pyright: ignore[reportIndexIssue]


@wp.kernel
def _anderson_residual_kernel(
    x: wp.array[wp.vec3f],
    xkm1: wp.array[wp.vec3f],
    fkm1: wp.array[wp.vec3f],
):
    i = wp.tid()
    fkm1[i] = x[i] - xkm1[i]  # pyright: ignore[reportIndexIssue]


_anderson_kernel_cache: dict = {}


def anderson_kernels(m: int):
    """Build (or fetch) the Anderson least-squares kernels specialized on the window size ``m``.

    Tile shapes must be compile-time constants, so the kernels are generated per ``m``. The active
    window size ``mk <= m`` is a runtime argument: slots ``j >= mk`` are masked out, which keeps
    one captured CUDA graph valid while the window fills up.

    Returns:
        (gram, solve_cholesky, solve_pcg): Warp kernels. ``gram`` accumulates the Gram matrix of
        ``[F f]`` into ``G`` (tiled launch over ``3N``). Both solves read ``G`` and write
        ``gamma``, using ``G[:mk, :mk] = F^T F`` and ``G[:mk, m] = F^T f``.
    """
    if m in _anderson_kernel_cache:
        return _anderson_kernel_cache[m]

    M = m
    MP = m + 1
    vecM = wp.types.vector(length=M, dtype=wp.float32)
    matM = wp.types.matrix(shape=(M, M), dtype=wp.float32)
    rank_floor = float(M) * _FLOAT32_EPS

    @wp.func
    def _scale_and_mask_entry(a: wp.float32, i: int, j: int, mk: int, s: wp.float32):
        # Active block is scaled by s = 1 / max diag. Inactive slots get a unit diagonal and no
        # coupling, so their coefficients solve to exactly 0.
        if i >= mk or j >= mk:
            if i == j:
                return wp.float32(1.0)
            return wp.float32(0.0)
        return a * s

    @wp.func
    def _scale_and_mask_rhs(b: wp.float32, i: int, mk: int, s: wp.float32):
        if i >= mk:
            return wp.float32(0.0)
        return b * s

    @wp.func
    def _keep_if(x: wp.float32, keep: int):
        if keep != 0:
            return x
        return wp.float32(0.0)

    @wp.kernel(module="unique")
    def gram(FfT: wp.array2d[wp.float32], G: wp.array2d[wp.float32]):
        """Accumulate ``G += T T^T`` for this block's ``(m+1, TILE_K)`` tile ``T`` of ``[F f]^T``."""
        i = wp.tid()
        t = wp.tile_load(FfT, shape=(MP, _GRAM_TILE_K), offset=(0, i * _GRAM_TILE_K))
        wp.tile_atomic_add(G, wp.tile_matmul(t, wp.tile_transpose(t)))

    @wp.kernel(module="unique")
    def solve_cholesky(
        G: wp.array2d[wp.float32],
        mk: int,
        rank_threshold: float,
        gamma: wp.array[wp.float32],
    ):
        """Solve ``F^T F gamma = F^T f`` by Cholesky, regularizing only when needed.

        Mimics a rank-revealing COD with relative threshold ``tau = rank_threshold``: the system is
        first scaled by its largest active diagonal entry, and factorized unregularized. If any
        pivot satisfies ``L_jj^2 <= tau_eff^2`` (equivalently, ``F`` has a relative singular value
        around ``tau_eff`` or below), or the factorization breaks down, the scaled system is
        refactorized with the ridge ``tau_eff^2 I``, i.e. a ridge relative to ``max_j (F^T F)_jj``.
        Here ``tau_eff^2 = max(tau^2, m * eps_float32)`` accounts for float32 rounding of ``G``. A
        zero history (``max diag = 0``) or a non-finite solution yields ``gamma = 0``, i.e. the
        plain fixed-point step.
        """
        dmax = wp.float32(0.0)
        for j in range(mk):
            dmax = wp.max(dmax, G[j, j])
        if not (dmax > wp.float32(0.0)) or not wp.isfinite(dmax):
            for j in range(M):
                gamma[j] = wp.float32(0.0)
            return
        s = wp.float32(1.0) / dmax
        tau2 = wp.max(wp.float32(rank_threshold) * wp.float32(rank_threshold), wp.float32(rank_floor))

        rows = wp.tile_broadcast(wp.tile_reshape(wp.tile_arange(M, dtype=int), shape=(M, 1)), shape=(M, M))
        cols = wp.tile_broadcast(wp.tile_reshape(wp.tile_arange(M, dtype=int), shape=(1, M)), shape=(M, M))
        A = wp.tile_map(
            _scale_and_mask_entry,
            wp.tile_load(G, shape=(M, M), offset=(0, 0)),
            rows,
            cols,
            wp.tile_full(shape=(M, M), value=mk, dtype=int),
            wp.tile_full(shape=(M, M), value=s, dtype=wp.float32),
        )
        b = wp.tile_map(
            _scale_and_mask_rhs,
            wp.tile_load(G, shape=(M, 1), offset=(0, M)),
            wp.tile_reshape(wp.tile_arange(M, dtype=int), shape=(M, 1)),
            wp.tile_full(shape=(M, 1), value=mk, dtype=int),
            wp.tile_full(shape=(M, 1), value=s, dtype=wp.float32),
        )

        # Unregularized attempt, then decide (uniformly across the block) whether to regularize.
        L0 = wp.tile_cholesky(A)
        needs_ridge = int(0)
        for j in range(M):
            ljj = wp.tile_extract(L0, j, j)
            if j < mk and (not wp.isfinite(ljj) or ljj * ljj <= tau2):
                needs_ridge = 1
        ridge = wp.float32(0.0)
        if needs_ridge != 0:
            ridge = tau2
        L = wp.tile_cholesky(wp.tile_diag_add(A, wp.tile_full(shape=M, value=ridge, dtype=wp.float32)))
        x = wp.tile_cholesky_solve(L, b)

        finite = int(1)
        for j in range(M):
            if not wp.isfinite(wp.tile_extract(x, j, 0)):
                finite = 0
        x = wp.tile_map(_keep_if, x, wp.tile_full(shape=(M, 1), value=finite, dtype=int))
        wp.tile_store(gamma, wp.tile_reshape(x, shape=(M,)))

    @wp.kernel(module="unique")
    def solve_pcg(G: wp.array2d[wp.float32], mk: int, gamma: wp.array[wp.float32]):
        """Solve ``F^T F gamma = F^T f`` by Jacobi-preconditioned CG, without regularization.

        ``F^T F`` is only positive semi-definite, but ``F^T f`` lies in its range, so CG started
        from 0 converges to a least-squares solution in at most ``rank(F) <= mk`` steps (exact
        arithmetic). Runs on the symmetrically scaled system ``D^-1/2 F^T F D^-1/2`` (unit
        diagonal), with the right-hand side normalized to unit length to avoid float32 underflow.
        Stops after ``mk`` iterations, when the relative residual drops below ``_PCG_RTOL``, or
        when a search direction has no curvature (it lies in the numerical null space).
        """
        if wp.tid() != 0:
            return
        dinv = vecM()
        for j in range(M):
            djj = G[j, j]
            if j < mk and djj > wp.float32(0.0):
                dinv[j] = wp.float32(1.0) / wp.sqrt(djj)
        A = matM()
        b = vecM()
        for i in range(M):
            b[i] = G[i, M] * dinv[i]
            for j in range(M):
                A[i, j] = G[i, j] * dinv[i] * dinv[j]
        bnorm = wp.length(b)
        y = vecM()
        if bnorm > wp.float32(0.0) and wp.isfinite(bnorm):
            r = b / bnorm
            p = r
            rr = wp.dot(r, r)
            rtol2 = wp.float32(_PCG_RTOL * _PCG_RTOL)
            active = int(1)
            for it in range(M):
                if active != 0 and it < mk and rr > rtol2:
                    Ap = A * p
                    pAp = wp.dot(p, Ap)
                    if pAp > wp.float32(0.0):
                        alpha = rr / pAp
                        y = y + alpha * p
                        r = r - alpha * Ap
                        rr_next = wp.dot(r, r)
                        p = r + (rr_next / rr) * p
                        rr = rr_next
                    else:
                        active = 0
            y = y * bnorm
        for j in range(M):
            gj = y[j] * dinv[j]
            if not wp.isfinite(gj):
                gj = wp.float32(0.0)
            gamma[j] = gj

    kernels = (gram, solve_cholesky, solve_pcg)
    _anderson_kernel_cache[m] = kernels
    return kernels


@wp.kernel
def _anderson_mixing_kernel(
    x: wp.array[wp.vec3f],
    xkm1: wp.array[wp.vec3f],
    fk: wp.array[wp.vec3f],
    Xk: wp.array2d[wp.vec3f], 
    Fk: wp.array2d[wp.vec3f], 
    gammak: wp.array[wp.float32],
    beta: float,
    mk: int,
):
    i = wp.tid()
    xi = xkm1[i] + beta * fk[i]  # pyright: ignore[reportIndexIssue]
    for j in range(mk):
        gammaj = gammak[j]  # pyright: ignore[reportIndexIssue]
        xi -= gammaj * (Xk[j, i] + beta * Fk[j, i])  # pyright: ignore[reportIndexIssue]
    x[i] = xi  # pyright: ignore[reportIndexIssue]


@profiling.scoped("anderson.subproblem")
def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: AndersonParams,
):
    """Run ``n_subproblem_max_iters`` Anderson iterations on the current subproblem.

    The accelerated fixed-point map ``g`` is ``vbd_iters_per_anderson_iter`` consecutive VBD
    sweeps (each preceded by a slack dual update, as in the plain VBD subproblem loop), so the
    subproblem runs ``n_subproblem_max_iters * vbd_iters_per_anderson_iter`` sweeps in total.
    """
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    n_vbd_iters = params.vbd_iters_per_anderson_iter
    x = fem.data.x
    n_nodes = x.shape[0]

    def _vbd_iterate():
        for _ in range(n_vbd_iters):
            contact.update_dual(
                fem.data.x,
                fem.xt,
                request_slack_update=True,
                request_decay_update=False,
                request_lagrange_multiplier_update=False,
            )
            iterate(fem, contact, params)

    Xk = params.Xk
    Fk = params.Fk

    # kp == 0
    wp.copy(params.xkm1, x)
    _vbd_iterate()
    wp.launch(
        kernel=_anderson_residual_kernel,
        dim=n_nodes,
        inputs=[x, params.xkm1, params.fkm1],
    )
    for kp in range(1, n_subproblem_max_iters):
        dkl = (kp - 1) % params.m
        wp.launch(
            kernel=_anderson_pre_iterate_kernel,
            dim=n_nodes,
            inputs=[x, params.xkm1, Xk, dkl],
        )
        _vbd_iterate()
        wp.launch(
            kernel=_anderson_post_iterate_kernel,
            dim=n_nodes,
            inputs=[x, params.xkm1, params.fk, params.fkm1, Fk, dkl],
        )
        mk = min(params.m, kp)
        anderson_mix(x, params, mk)


@profiling.scoped("anderson.mix")
def anderson_mix(x: wp.array, params: AndersonParams, mk: int):
    """Solve the Anderson least-squares problem and mix the accelerated iterate into ``x``.

    Forms the Gram matrix of ``[F f]`` in one pass over ``F`` (``G[:mk, :mk] = F^T F``,
    ``G[:mk, m] = F^T f``), solves ``F^T F gamma = F^T f`` with the selected method (see
    :class:`AndersonLinearSolver`), then applies ``x = xkm1 + beta * fk - (X + beta * F) gamma``.
    """
    n_nodes = x.shape[0]
    gram, solve_cholesky, solve_pcg = params.kernels
    with profiling.scope("anderson.gram"):
        params.G.zero_()
        wp.launch_tiled(
            kernel=gram,
            dim=[(3 * n_nodes + _GRAM_TILE_K - 1) // _GRAM_TILE_K],
            inputs=[params.FfT_flat, params.G],
            block_dim=_GRAM_BLOCK_DIM,
        )
    with profiling.scope("anderson.solve"):
        if params.linear_solver == AndersonLinearSolver.NormalEquations:
            wp.launch_tiled(
                kernel=solve_cholesky,
                dim=[1],
                inputs=[params.G, mk, params.cod_numerical_zero, params.gammak],
                block_dim=_SOLVE_BLOCK_DIM,
            )
        else:
            wp.launch(kernel=solve_pcg, dim=1, inputs=[params.G, mk, params.gammak])
    wp.launch(
        kernel=_anderson_mixing_kernel,
        dim=n_nodes,
        inputs=[x, params.xkm1, params.fk, params.Xk, params.Fk, params.gammak, params.beta, mk],
    )


class AndersonSolver:

    def __init__(self):
        pass

    def solve(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        cd: ContactDetection,
        params: AndersonParams,
    ) -> bool:
        initialize_solve(fem, contact, cd, params)

        def outer_iteration(k: int):
            prepare_subproblem(fem, contact, cd, params)
            solve_subproblem(fem, contact, params)
            finalize_subproblem(fem, contact, cd, params)

        converged = solve_outer_loop(fem, contact, cd, params, outer_iteration)
        fem.back_substitute_velocities()
        cd.on_time_step_ended()
        return converged

    @property
    def supports_graph_capture(self):
        return True


import unittest


class TestAndersonMix(unittest.TestCase):
    """Checks the Gram-based least-squares solves against float64 least squares."""

    n_nodes = 2000
    m = 5

    def _make_params(self, linear_solver):
        import numpy as np
        from pbatoolkit import pbat

        E = np.arange(self.n_nodes, dtype=np.int64).reshape(-1, 4).T  # disjoint tetrahedra
        GVGp, GVGe, GVGilocal = pbat.sim.algorithm.vbd.vertex_element_adjacency_graph(E, self.n_nodes)
        GVVp, GVVadj, colors = pbat.sim.algorithm.vbd.vertex_colors(E, self.n_nodes)
        vbd = pbat.sim.algorithm.vbd.Params()
        vbd.with_vertex_element_adjacency_graph(GVGp, GVGe, GVGilocal)
        vbd.with_vertex_colors(GVVp, GVVadj, colors).construct()
        anderson = pbat.sim.algorithm.vbd.AndersonParams()
        anderson.m = self.m
        anderson.beta = 1.0
        return AndersonParams(vbd, anderson, linear_solver)

    def _fill(self, params, F, f, X):
        import cupy as cp

        params.FkT[...] = cp.asarray(F.T, dtype=cp.float32)
        params.FfT[self.m] = cp.asarray(f, dtype=cp.float32)
        params.XkT[...] = cp.asarray(X.T, dtype=cp.float32)

    def _solve(self, linear_solver, F, f, X, mk):
        params = self._make_params(linear_solver)
        self._fill(params, F, f, X)
        x = wp.zeros((self.n_nodes,), dtype=wp.vec3f)
        anderson_mix(x, params, mk)
        return params, x

    def _histories(self, seed=0, scale=1e-4):
        import numpy as np

        rng = np.random.default_rng(seed)
        K = 3 * self.n_nodes
        F = scale * rng.standard_normal((K, self.m))
        f = scale * rng.standard_normal(K)
        X = scale * rng.standard_normal((K, self.m))
        return F, f, X

    def test_matches_least_squares(self):
        import numpy as np

        F, f, X = self._histories()
        g_ref = np.linalg.lstsq(F, f, rcond=None)[0]
        for solver in AndersonLinearSolver:
            params, _ = self._solve(solver, F, f, X, self.m)
            g = params.gammak.numpy().astype(np.float64)
            self.assertLess(np.abs(g - g_ref).max() / np.abs(g_ref).max(), 1e-3, solver.name)

    def test_inactive_slots_are_ignored(self):
        import numpy as np

        F, f, X = self._histories(seed=1)
        mk = 2
        g_ref = np.linalg.lstsq(F[:, :mk], f, rcond=None)[0]
        for solver in AndersonLinearSolver:
            params, _ = self._solve(solver, F, f, X, mk)
            g = params.gammak.numpy().astype(np.float64)
            self.assertTrue(np.all(g[mk:] == 0.0), solver.name)
            self.assertLess(np.abs(g[:mk] - g_ref).max() / np.abs(g_ref).max(), 1e-3, solver.name)

    def test_rank_deficient_history(self):
        import numpy as np

        F, f, X = self._histories(seed=2)
        F[:, 3] = F[:, 1]  # exact linear dependence
        F[:, 4] = 0.0  # zero column
        g_opt = np.linalg.lstsq(F, f, rcond=None)[0]
        r_opt = np.linalg.norm(F @ g_opt - f)
        for solver in AndersonLinearSolver:
            params, _ = self._solve(solver, F, f, X, self.m)
            g = params.gammak.numpy().astype(np.float64)
            self.assertTrue(np.all(np.isfinite(g)), solver.name)
            self.assertLess(np.linalg.norm(F @ g - f) / r_opt, 1.0 + 1e-3, solver.name)

    def test_zero_history_gives_plain_step(self):
        import numpy as np

        F, f, X = self._histories(seed=3)
        F[...] = 0.0
        for solver in AndersonLinearSolver:
            params, _ = self._solve(solver, F, f, X, self.m)
            self.assertTrue(np.all(params.gammak.numpy() == 0.0), solver.name)

    def test_mixing_and_graph_capture(self):
        import numpy as np

        F, f, X = self._histories(seed=4)
        for solver in AndersonLinearSolver:
            params = self._make_params(solver)
            self._fill(params, F, f, X)
            xkm1 = np.random.default_rng(5).standard_normal((self.n_nodes, 3)).astype(np.float32)
            params.xkm1.assign(xkm1)
            x = wp.zeros((self.n_nodes,), dtype=wp.vec3f)
            with wp.ScopedCapture() as capture:
                anderson_mix(x, params, self.m)
            wp.capture_launch(capture.graph)
            g = params.gammak.numpy().astype(np.float64)
            x_ref = xkm1.reshape(-1) + params.beta * f - (X + params.beta * F) @ g
            self.assertLess(np.abs(x.numpy().reshape(-1) - x_ref).max(), 1e-4, solver.name)


if __name__ == "__main__":
    wp.init()
    unittest.main()

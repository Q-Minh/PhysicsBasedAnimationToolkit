import warp as wp
import cupy as cp

from python.tools.sim.gpu.vbd.solver import (
    check_convergence,
    finalize_subproblem,
    initialize_solve,
    iterate,
    prepare_subproblem,
)

from ..common.stream import Stream
from ..contact.mesh.cd import ContactDetection
from ..elasticity.fem import FemElastoDynamics
from .params import AndersonParams
from ..contact.dynamics import MeshDynamics as ContactDynamics


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


@wp.kernel
def _anderson_solve_normal_equations_kernel(
    A: wp.array2d[wp.float32],
    b: wp.array[wp.float32],
    gammak: wp.array[wp.float32],
    mk: int,
    regularization: float,
):
    """Solve ``A gamma = b`` with an in-place LDLT factorization in one thread."""
    if wp.tid() != 0:
        return

    # The user-configured diagonal shift makes the normal system positive definite when F is rank
    # deficient.
    for i in range(mk):
        A[i, i] += regularization  # pyright: ignore[reportIndexIssue]
        for j in range(i):
            lij = A[i, j]  # pyright: ignore[reportIndexIssue]
            for k in range(j):
                lij -= A[i, k] * A[j, k] * A[k, k]  # pyright: ignore[reportIndexIssue]
            A[i, j] = lij / A[j, j]  # pyright: ignore[reportIndexIssue]

        dii = A[i, i]  # pyright: ignore[reportIndexIssue]
        for k in range(i):
            dii -= A[i, k] * A[i, k] * A[k, k]  # pyright: ignore[reportIndexIssue]
        A[i, i] = dii  # pyright: ignore[reportIndexIssue]

    # Forward solve: L y = b.
    for i in range(mk):
        yi = b[i]  # pyright: ignore[reportIndexIssue]
        for j in range(i):
            yi -= A[i, j] * gammak[j]  # pyright: ignore[reportIndexIssue]
        gammak[i] = yi  # pyright: ignore[reportIndexIssue]

    # Diagonal solve: D z = y.
    for i in range(mk):
        gammak[i] /= A[i, i]  # pyright: ignore[reportIndexIssue]

    # Backward solve: L^T gamma = z.
    for i in range(mk - 1, -1, -1):
        gammai = gammak[i]  # pyright: ignore[reportIndexIssue]
        for j in range(i + 1, mk):
            gammai -= A[j, i] * gammak[j]  # pyright: ignore[reportIndexIssue]
        gammak[i] = gammai  # pyright: ignore[reportIndexIssue]


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


def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: AndersonParams,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    x = fem.data.x
    n_nodes = x.shape[0]

    def _vbd_iterate():
        contact.update_dual(
            fem.data.x,
            fem.xt,
            request_slack_update=True,
            request_decay_update=False,
            request_lagrange_multiplier_update=False,
        )
        iterate(fem, contact, params)

    device = wp.get_cuda_device()
    Xk = wp.array(
        params.XkT,
        dtype=wp.vec3f,
        shape=(params.m, n_nodes),
        device=device,
        copy=False,
    )
    Fk = wp.array(
        params.FkT,
        dtype=wp.vec3f,
        shape=(params.m, n_nodes),
        device=device,
        copy=False,
    )

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


def anderson_mix(x: wp.array, params: AndersonParams, mk: int):
    """Solve the Anderson normal equations and mix the accelerated iterate into ``x`` in place.

    ``params.FkT[:mk]`` stores ``F^T`` and has shape ``(mk, 3N)``. Therefore this solves
    ``(F^T F + eps I) gamma = F^T f``, where ``eps`` is the UI-exposed
    ``cod_numerical_zero`` setting, with an in-kernel LDLT factorization. It then applies
    ``x = xkm1 + beta * fk - (X + beta * F) gamma``.
    """
    device = wp.get_cuda_device()
    n_nodes = x.shape[0]
    stream = Stream(wp.get_stream())
    with cp.cuda.Stream.from_external(stream):
        FkT = params.FkT[:mk]
        fk = cp.asarray(params.fk).ravel()
        cp.matmul(FkT, fk, out=params.FkTfk[:mk])
        cp.matmul(FkT, FkT.T, out=params.FkTFk[:mk, :mk])

    Xk = wp.array(
        params.XkT,
        dtype=wp.vec3f,
        shape=(params.m, n_nodes),
        device=device,
        copy=False,
    )
    Fk = wp.array(
        params.FkT,
        dtype=wp.vec3f,
        shape=(params.m, n_nodes),
        device=device,
        copy=False,
    )
    A = wp.array(
        params.FkTFk,
        dtype=wp.float32,
        shape=(params.m, params.m),
        device=device,
        copy=False,
    )
    b = wp.array(
        params.FkTfk,
        dtype=wp.float32,
        shape=(params.m,),
        device=device,
        copy=False,
    )
    wp.launch(
        kernel=_anderson_solve_normal_equations_kernel,
        dim=1,
        inputs=[A, b, params.gammak, mk, params.cod_numerical_zero],
    )
    wp.launch(
        kernel=_anderson_mixing_kernel,
        dim=n_nodes,
        inputs=[x, params.xkm1, params.fk, Xk, Fk, params.gammak, params.beta, mk],
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
        converged = False
        initialize_solve(fem, contact, cd, params)
        for k in range(params.data.n_max_iters):
            if check_convergence(fem, contact, params):
                converged = True
                break
            prepare_subproblem(fem, contact, cd, params)
            solve_subproblem(fem, contact, params)
            finalize_subproblem(fem, contact, cd, params)
        fem.back_substitute_velocities()
        cd.on_time_step_ended()
        return converged

    @property
    def supports_graph_capture(self):
        return True

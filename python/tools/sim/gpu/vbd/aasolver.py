import warp as wp
import warp.optim.linear as wol
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
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from .params import (
    AndersonParams,
    ParamsData,
)
from ..contact.dynamics import (
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
    PenaltyAdaptivity,
)


@wp.kernel
def _anderson_pre_iterate_kernel(
    x: wp.array[wp.vec3f],
    xkm1: wp.array[wp.vec3f],
    Xk: wp.array[wp.vec3f],
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
    Fk: wp.array[wp.vec3f],
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
        params.Xk,
        dtype=wp.vec3f,
        shape=(params.m, n_nodes),
        device=device,
        copy=False,
    )
    Fk = wp.array(
        params.Fk,
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
    """Solve the Anderson mixing least-squares subproblem via CG on the normal equations, and
    mix the accelerated iterate into `x` in place, mirroring `Anderson.h`'s `Iterate` (`kp > 0`
    branch, after `Fk`/`fk` have been updated):

        gammak[:mk] = argmin_gamma || Fk[:mk] @ gamma - fk ||
                    <=> Fk[:mk]^T Fk[:mk] gammak[:mk] = Fk[:mk]^T fk    (normal equations)
        x = xkm1 + beta * fk - Xk[:mk] @ gammak[:mk] - beta * (Fk[:mk] @ gammak[:mk])

    The normal equations are solved with `warp.optim.linear.cg`, using a custom
    `warp.optim.linear.LinearOperator` whose matvec routine performs the `Fk[:mk]^T Fk[:mk] @ v`
    product as a `cupy` matmul. All CuPy operations (including the mixing update) are enqueued
    on the current Warp stream (wrapped via `Stream`, which implements the CUDA stream protocol)
    so that they interleave correctly with the surrounding Warp kernel launches, instead of
    running on CuPy's own default stream.
    """
    device = wp.get_cuda_device()
    n_nodes = x.shape[0]
    stream = Stream(wp.get_stream())
    with cp.cuda.Stream.from_external(stream):
        FkT = params.FkT
        Fk = FkT.T
        # Form normal equations
        cp.matmul(FkT, params.fk, out=params.FkTfk)
        cp.matmul(FkT, Fk, out=params.FkTFk)
        # Solve normal equations (i.e. least squares)
        wol.cg(params.FkTFk, params.FkTfk, params.gammak, maxiter=params.m)
        # TODO: Implement mixing
        # xkp1 = xkm1 + beta*fk - Xk[:mk] @ gammak[:mk] - beta * (Fk[:mk] @ gammak[:mk])
        Xk = params.XkT.T
        cp.matmul(Xk[:mk], params.gammak[:mk], out=params.Xkgammak)
        cp.matmul(Fk[:mk], params.gammak[:mk], out=params.Fkgammak)
        x = cp.asarray(x).ravel(order="F")
        xkm1 = cp.asarray(params.xkm1).ravel(order="F")
        wp.launch(
            kernel=_anderson_mixing_kernel,
            dim=(n_nodes*3,),
            inputs=[x, xkm1, params.fk, params.Xkgammak, params.Fkgammak, params.beta],
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

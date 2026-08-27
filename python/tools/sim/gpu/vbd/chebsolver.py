import warp as wp

from python.tools.sim.gpu.vbd.solver import (
    check_convergence,
    finalize_subproblem,
    initialize_solve,
    iterate,
    prepare_subproblem,
)

from ..contact.mesh.cd import ContactDetection
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from .params import (
    ChebyshevParams,
    ParamsData,
)
from ..contact.dynamics import (
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
    PenaltyAdaptivity,
)


def _chebyshev_omega(k: int, rho2: float, omega: float) -> float:
    """Mirrors `pbat::sim::algorithm::vbd::kernels::ChebyshevOmega`."""
    if k == 0:
        return 1.0
    elif k == 1:
        return 2.0 / (2.0 - rho2)
    else:
        return 4.0 / (4.0 - rho2 * omega)


@wp.kernel
def _chebyshev_momentum_update_kernel(
    x: wp.array[wp.vec3f],
    xkm1: wp.array[wp.vec3f],
    xkm2: wp.array[wp.vec3f],
    omega: wp.float32,
    apply_omega: wp.int32,
):
    """Mirrors lines 149-155 of `source/pbat/sim/algorithm/vbd/Chebyshev.h`:

        if (k > 1)
            xk = omega * (xk - xkm2) + xkm2;
        xkm2 = xkm1;
        xkm1 = xk;
    """
    i = wp.tid()
    xk = x[i]
    if apply_omega != 0:
        xk = omega * (xk - xkm2[i]) + xkm2[i]
        x[i] = xk  # pyright: ignore[reportIndexIssue]
    xkm2[i] = xkm1[i]  # pyright: ignore[reportIndexIssue]
    xkm1[i] = xk  # pyright: ignore[reportIndexIssue]


def chebyshev_momentum_update(fem: FemElastoDynamics, cheb: ChebyshevParams, k: int):
    """Update `fem.data.x` in place via the Chebyshev semi-iterative momentum recurrence,
    and advance `cheb`'s `omega`, `xkm1`, `xkm2` state.

    Mirrors `pbat::sim::algorithm::vbd::Chebyshev.h`'s `Iterate` Chebyshev update.
    """
    rho = cheb.rho  # pyright: ignore[reportArgumentType]
    rho2 = rho * rho
    omega = _chebyshev_omega(k, rho2, cheb.omega)  # pyright: ignore[reportArgumentType]
    cheb.omega = omega
    n_nodes = fem.data.x.shape[0]
    wp.launch(
        kernel=_chebyshev_momentum_update_kernel,
        dim=n_nodes,
        inputs=[fem.data.x, cheb.xkm1, cheb.xkm2, omega, 1 if k > 1 else 0],
    )


def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: ChebyshevParams,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    for kp in range(n_subproblem_max_iters):
        contact.update_dual(
            fem.data.x,
            fem.xt,
            request_slack_update=True,
            request_decay_update=False,
            request_lagrange_multiplier_update=False,
        )
        iterate(fem, contact, params)
        chebyshev_momentum_update(fem, params, kp)


class ChebyshevSolver:

    def __init__(self):
        pass

    def solve(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        cd: ContactDetection,
        params: ChebyshevParams,
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

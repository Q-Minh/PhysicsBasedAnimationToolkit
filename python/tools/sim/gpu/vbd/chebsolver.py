import warp as wp
from ..common import profiling

from python.tools.sim.gpu.vbd.solver import (
    finalize_subproblem,
    initialize_solve,
    iterate,
    prepare_subproblem,
    solve_outer_loop,
    IterationSchedule,
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


@wp.kernel
def _chebyshev_omega_kernel(
    omega: wp.array[wp.float32],
    rho2: wp.float32,
    k: wp.int32,
):
    """Mirrors `pbat::sim::algorithm::vbd::kernels::ChebyshevOmega`, updating omega[0] in place.

    Kept on the device so that a recorded iteration (k > 1) can be replayed, e.g. inside
    `wp.capture_while`.
    """
    if k == 0:
        omega[0] = wp.float32(1.0)
    elif k == 1:
        omega[0] = wp.float32(2.0) / (wp.float32(2.0) - rho2)
    else:
        omega[0] = wp.float32(4.0) / (wp.float32(4.0) - rho2 * omega[0])


@wp.kernel
def _chebyshev_momentum_update_kernel(
    x: wp.array[wp.vec3f],
    xkm1: wp.array[wp.vec3f],
    xkm2: wp.array[wp.vec3f],
    omega: wp.array[wp.float32],
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
        xk = omega[0] * (xk - xkm2[i]) + xkm2[i]
        x[i] = xk  # pyright: ignore[reportIndexIssue]
    xkm2[i] = xkm1[i]  # pyright: ignore[reportIndexIssue]
    xkm1[i] = xk  # pyright: ignore[reportIndexIssue]


@profiling.scoped("chebyshev.mix")
def chebyshev_momentum_update(fem: FemElastoDynamics, cheb: ChebyshevParams, k: int):
    """Update `fem.data.x` in place via the Chebyshev semi-iterative momentum recurrence,
    and advance `cheb`'s `omega`, `xkm1`, `xkm2` state.

    Mirrors `pbat::sim::algorithm::vbd::Chebyshev.h`'s `Iterate` Chebyshev update.
    """
    rho = cheb.rho  # pyright: ignore[reportArgumentType]
    rho2 = rho * rho
    wp.launch(
        kernel=_chebyshev_omega_kernel, dim=1, inputs=[cheb.omega, rho2, min(k, 2)]
    )
    n_nodes = fem.data.x.shape[0]
    wp.launch(
        kernel=_chebyshev_momentum_update_kernel,
        dim=n_nodes,
        inputs=[fem.data.x, cheb.xkm1, cheb.xkm2, cheb.omega, 1 if k > 1 else 0],
    )


def subproblem_iteration(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: ChebyshevParams,
    kp: int,
):
    """Iteration ``kp`` of the subproblem: slack dual update, VBD sweep, Chebyshev momentum.

    Iterations ``kp >= 2`` record identical work, so they can be replayed.
    """
    contact.update_dual(
        fem.data.x,
        fem.xt,
        request_slack_update=True,
        request_decay_update=False,
        request_lagrange_multiplier_update=False,
    )
    iterate(fem, contact, params)
    chebyshev_momentum_update(fem, params, kp)


@profiling.scoped("chebyshev.subproblem")
def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: ChebyshevParams,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    for kp in range(n_subproblem_max_iters):
        subproblem_iteration(fem, contact, params, kp)


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
        initialize_solve(fem, contact, cd, params)
        converged = solve_outer_loop(
            fem, contact, cd, params, self.schedule(fem, contact, params)
        )
        fem.back_substitute_velocities()
        cd.on_time_step_ended()
        return converged

    def schedule(
        self, fem: FemElastoDynamics, contact: ContactDynamics, params: ChebyshevParams
    ) -> IterationSchedule:
        """The first two iterations differ (omega's recurrence starts); later ones are alike."""
        return IterationSchedule(
            lambda k, kp: subproblem_iteration(fem, contact, params, kp),
            prefix=2,
            scope="chebyshev.subproblem",
        )

    @property
    def supports_graph_capture(self):
        return True

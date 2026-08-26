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
    Params,
    ParamsData,
)
from ..contact.dynamics import (
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
    PenaltyAdaptivity,
)


def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
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
        # TODO: Add Chebyshev acceleration here
        # ...


class ChebyshevSolver:

    def __init__(self):
        pass

    def solve(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        cd: ContactDetection,
        params: Params,
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

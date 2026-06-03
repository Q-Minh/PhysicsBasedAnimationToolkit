"""
Newton solver for FEM elasto-dynamics with contact (augmented Lagrangian).

Outer loop mirrors ``gpu/vbd/solver.py``.  Newton-specific work lives in
``prepare_subproblem`` and ``solve_subproblem``; everything else is identical
to the VBD solver.
"""

import warp as wp
import warp.sparse
import warp.optim.linear

from pbatoolkit import pbat
from ..contact.mesh.cd import ContactDetection
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from ..contact.dynamics import (
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
    PenaltyAdaptivity,
)
from ..elasticity import snh


@wp.struct
class ParamsData:
    """GPU-side Newton solver parameters."""

    # Outer AL iteration control
    n_max_iters: wp.int32
    k: wp.int32

    # Inner Newton iteration control
    n_subproblem_max_iters: wp.int32
    gtol2: wp.float32  # squared gradient-norm convergence threshold

    # Backtracking line search (Armijo)
    ls_max_iters: wp.int32
    ls_tau: wp.float32  # step shrink factor
    ls_c: wp.float32  # Armijo slope constant
    ls_alpha: wp.float32  # initial step size

    # Vertex-element adjacency (for elastic gradient / Hessian)
    GVGp: wp.array[wp.int32]
    GVGadj: wp.array[wp.int32]

    # BSR Hessian 3x3 block triplets
    Hrows: wp.array[wp.int32]
    Hcols: wp.array[wp.int32]
    Hvals: wp.array[wp.mat33f]


@wp.kernel
def _initialize_hessian_triplets(
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
):
    tid = wp.tid()
    n_nodes = fem.X.shape[0]
    n_elems = fem.E.shape[0]
    if tid < n_nodes:
        # Nodal lumped mass diagonal
        offset = wp.int32(0)
        i = tid
        params.Hrows[offset + i] = i
        params.Hcols[offset + i] = i
    if tid < n_elems:
        # Element elastic Hessian
        offset = n_nodes
        e = tid
        nodes = fem.E[e]
        begin = offset + e * 4 * 4
        for i in range(4):
            for j in range(4):
                idx = begin + i * 4 + j
                params.Hrows[idx] = nodes[i]
                params.Hcols[idx] = nodes[j]


class Params:
    """Python wrapper that builds and owns the GPU :class:`ParamsData` struct.

    Parameters
    ----------
    n_nodes:
        Total number of FEM vertices.
    n_max_iters:
        Maximum augmented-Lagrangian outer iterations.
    n_subproblem_max_iters:
        Maximum Newton steps per AL subproblem.
    gtol2:
        Squared gradient-norm convergence threshold.
    ls_max_iters:
        Maximum backtracking iterations.
    ls_tau:
        Step shrink factor for backtracking.
    ls_c:
        Armijo sufficient-decrease constant.
    ls_alpha:
        Initial step size.
    """

    _H: warp.sparse.BsrMatrix  # Assembled sparse Hessian

    def __init__(
        self,
        params_cpu: pbat.sim.algorithm.newton.Params,
    ):
        newton: pbat.math.optimization.Newton = params_cpu.newton
        line_search: pbat.math.optimization.BackTrackingLineSearch | None = (
            newton.line_search
        )
        self._data = ParamsData()
        self._data.n_max_iters = params_cpu.n_max_iters
        self._data.k = 0
        self._data.n_subproblem_max_iters = newton.n_max_iters
        self._data.gtol2 = newton.gtol2
        self._data.ls_max_iters = line_search.n_max_iters if line_search else 0
        self._data.ls_tau = line_search.tau if line_search else 0.5
        self._data.ls_c = line_search.c if line_search else 1e-4
        self._data.ls_alpha = line_search.alpha if line_search else 1.0

    def construct(self, fem: FemElastoDynamics, contact: ContactDynamics):
        vv_capacity, ve_capacity, vf_capacity, ee_capacity = (
            contact.cvv.capacity,
            contact.cve.capacity,
            contact.cvf.capacity,
            contact.cee.capacity,
        )
        n_elems = fem.data.E.shape[0]
        n_nodes = fem.data.x.shape[0]
        n_fem_triplets = n_nodes + n_elems * (
            4**2
        )  # 1x1 nodal lumped mass + 4x4 element elastic Hessian
        n_contact_triplets = (
            vv_capacity * (2**2)  # 2x2 vertex-vertex contact Hessian
            + ve_capacity * (3**2)  # 3x3 vertex-edge contact Hessian
            + vf_capacity * (4**2)  # 4x4 vertex-triangle contact Hessian
            + ee_capacity * (4**2)  # 4x4 edge-edge contact Hessian
        )
        n_max_triplets = n_fem_triplets + n_contact_triplets
        self._data.Hrows = wp.zeros(n_max_triplets, dtype=wp.int32)
        self._data.Hcols = wp.zeros(n_max_triplets, dtype=wp.int32)
        self._data.Hvals = wp.zeros(n_max_triplets, dtype=wp.mat33f)
        # Create template hessian triplets
        wp.launch(
            kernel=_initialize_hessian_triplets,
            dim=n_elems,
            inputs=[fem.data, self._data],
        )
        self._H = warp.sparse.bsr_from_triplets(
            n_nodes,
            n_nodes,
            self._data.Hrows,
            self._data.Hcols,
            self._data.Hvals,
            prune_numerical_zeros=True,
        )

    @property
    def data(self) -> ParamsData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data


def initialize_solve(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
) -> None:
    """Detect contacts and initialise the constraint set for the time step."""
    cd.on_time_step_started()
    cd.detect_contacts(from_xt=True)
    contact.update_constraint_set()
    cd.filter_step()


def check_convergence(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> bool:
    """TODO: return True when ||gradient||^2 <= gtol2."""
    return False


def prepare_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
) -> None:
    """TODO: assemble Hessian, adapt penalty parameters."""
    pass


def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> None:
    """TODO: Newton solve where each iteration is a Jacobi-preconditioned CG linear solve + backtracking line search."""
    pass


def finalize_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
) -> None:
    """Dual update and contact re-detection -- mirrors ``vbd.solver.finalize_subproblem``."""
    contact.update_dual(
        fem.data.x,
        fem.xt,
        request_slack_update=True,
        request_decay_update=True,
        request_lagrange_multiplier_update=True,
    )
    cd.filter_step()
    cd.detect_contacts()
    contact.update_constraint_set()


def serialize_newton_cpu_params(params: pbat.sim.algorithm.newton.Params, grp) -> None:
    """Serialize a Newton CPU Params object to an h5py group."""
    grp.attrs["n_max_iters"] = params.n_max_iters
    grp.attrs["linear_solver"] = params.linear_solver.value
    newton = params.newton
    ng = grp.require_group("newton")
    ng.attrs["n_max_iters"] = newton.n_max_iters
    ng.attrs["gtol2"] = newton.gtol2
    ls = newton.line_search
    if ls is not None:
        lg = ng.require_group("line_search")
        lg.attrs["n_max_iters"] = ls.n_max_iters
        lg.attrs["tau"] = ls.tau
        lg.attrs["c"] = ls.c
        lg.attrs["alpha"] = ls.alpha


def deserialize_newton_cpu_params(
    params: pbat.sim.algorithm.newton.Params, grp
) -> None:
    """Deserialize a Newton CPU Params object from an h5py group."""
    if "n_max_iters" in grp.attrs:
        params.n_max_iters = int(grp.attrs["n_max_iters"])
    if "linear_solver" in grp.attrs:
        params.linear_solver = pbat.sim.algorithm.newton.ELinearSolver(
            int(grp.attrs["linear_solver"])
        )
    newton = params.newton
    if "newton" in grp:
        ng = grp["newton"]
        if "n_max_iters" in ng.attrs:
            newton.n_max_iters = int(ng.attrs["n_max_iters"])
        if "gtol2" in ng.attrs:
            newton.gtol2 = float(ng.attrs["gtol2"])
        ls = newton.line_search
        if ls is not None and "line_search" in ng:
            lg = ng["line_search"]
            if "n_max_iters" in lg.attrs:
                ls.n_max_iters = int(lg.attrs["n_max_iters"])
            if "tau" in lg.attrs:
                ls.tau = float(lg.attrs["tau"])
            if "c" in lg.attrs:
                ls.c = float(lg.attrs["c"])
            if "alpha" in lg.attrs:
                ls.alpha = float(lg.attrs["alpha"])


class NewtonSolver:
    """Newton solver for FEM elasto-dynamics with contact.

    Outer loop structure mirrors :class:`gpu.vbd.solver.VbdSolver`.
    """

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
        for _ in range(int(params.data.n_max_iters)):
            if check_convergence(fem, contact, params):
                converged = True
                break
            prepare_subproblem(fem, contact, cd, params)
            solve_subproblem(fem, contact, params)
            finalize_subproblem(fem, contact, cd, params)
        fem.back_substitute_velocities()
        cd.on_time_step_ended()
        return converged

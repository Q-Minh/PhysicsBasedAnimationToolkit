"""
VBD (Vertex Block Descent) solver for FEM elasto-dynamics.

Mirrors `pbat::sim::algorithm::vbd::Solve` from `source/pbat/sim/algorithm/vbd/Core.h`.
Contact constraints are left as placeholders for future implementation.
"""

from dataclasses import dataclass
import math
from typing import Callable, Tuple
import warp as wp
from ..common import profiling
import numpy as np

from ..contact.mesh.cd import ContactDetection
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from .params import (
    Params,
    ParamsData,
)
from .kernels import (
    local_elastic_derivatives,
    local_contact_derivatives,
    # local_contact_rayleigh_quotients,
    add_inertia_or_external_load_derivatives,
    integrate_positions,
    local_elastic_hessians,
)
from ..contact.dynamics import (
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
    PenaltyAdaptivity,
)


@wp.kernel
def _vertex_solve_kernel(
    pbegin: int,
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    contact: ContactDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    h2: float,
):
    """Process one vertex in the current color partition."""
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid // block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]
    i = params.Padj[
        pbegin + block_id  # pyright: ignore[reportOperatorIssue, reportIndexIssue]
    ]
    # Skip Dirichlet nodes
    if is_dirichlet_node(fem.dmask, i):  # pyright: ignore[reportArgumentType]
        return
    xi = fem.x[i]  # pyright: ignore[reportIndexIssue]
    xtildei = fem.xtilde[i]  # pyright: ignore[reportIndexIssue]
    mi = fem.m[i]  # pyright: ignore[reportIndexIssue]
    # Accumulate elastic energy derivatives
    gil, Hil = local_elastic_derivatives(
        i,
        fem.x,
        fem.E,
        fem.wg,
        fem.GNeg,
        fem.mug,
        fem.lambdag,
        params.GVGp,
        params.GVGadj,
        local_tid,  # type: ignore
        block_dims,  # type: ignore
    )
    gil *= h2  # type: ignore
    Hil *= h2  # type: ignore
    vi = contact.meshes.GXV[i]
    if vi >= 0:
        gil_c, Hil_c = local_contact_derivatives(
            i,
            vi,
            fem.xt,
            params.xb,
            contact,
            local_tid,  # type: ignore
            block_dims,  # type: ignore
        )
        gil += gil_c
        Hil += Hil_c
    gis, His = (
        wp.tile(gil, preserve_type=True),  # pyright: ignore[reportArgumentType]
        wp.tile(Hil, preserve_type=True),  # pyright: ignore[reportArgumentType]
    )
    gi, Hi = (
        wp.tile_reduce(wp.add, gis)[0],  # pyright: ignore[reportIndexIssue]
        wp.tile_reduce(wp.add, His)[0],  # pyright: ignore[reportIndexIssue]
    )
    if local_tid > 0:
        return
    # Add inertia (K = m, already in position space) or, for static problems, external load derivatives
    gi, Hi = add_inertia_or_external_load_derivatives(
        fem.is_static, mi, xtildei, fem.fext[i], xi, gi, Hi
    )  # pyright: ignore[reportArgumentType]
    # Solve local system: x -= H^{-1} g
    dxi = integrate_positions(
        gi,
        Hi,
        params.vls_solver,
        params.vls_max_iters,
        params.vls_eps,
        params.hess_zero,
    )  # pyright: ignore[reportArgumentType]
    fem.x[i] -= dxi  # pyright: ignore[reportIndexIssue]


@wp.kernel(launch_bounds=32)
def _compute_vertex_hessians(
    fem: FemElastoDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    params: ParamsData,  # pyright: ignore[reportGeneralTypeIssues]
    Hi: wp.array[wp.mat33f],
    h2: float,
):
    """Compute per-surface-vertex dynamics Hessian blocks (elastic + mass, or elastic only for static problems).
    Results are stored in Hi[v] for each surface vertex index v.
    """
    tid = wp.tid()
    block_dims = wp.block_dim()
    block_id = tid // block_dims  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dims  # pyright: ignore[reportOperatorIssue]
    i = block_id
    Hil = local_elastic_hessians(
        i,  # type: ignore
        fem.x,
        fem.E,
        fem.wg,
        fem.GNeg,
        fem.mug,
        fem.lambdag,
        params.GVGp,
        params.GVGadj,
        local_tid,  # type: ignore
        block_dims,  # type: ignore
    )
    Hil *= h2  # type: ignore
    His = wp.tile(Hil, preserve_type=wp.bool(True))
    Hib = wp.tile_sum(His)
    Hil = wp.tile_extract(Hib, 0)  # type: ignore
    if local_tid == 0:
        if fem.is_static == 0:
            mi = fem.m[i]
            for d in range(3):
                Hil[d, d] += mi
        Hi[i] = Hil  # type: ignore


def linearize_constraints(
    fem: FemElastoDynamics, contact: ContactDynamics, params: Params
):
    """TODO: Linearize contact constraints at current iterate."""
    pass


def check_convergence(
    fem: FemElastoDynamics, contact: ContactDynamics, params: Params
) -> bool:
    """Check gradient norm convergence (elastic + momentum or external load + contact), |g| <= gtol.

    Mirrors ``pbat::sim::algorithm::vbd::CheckConvergence``. Reads the result back to the host, so
    it can't be captured; solves test convergence on the device instead (see
    :func:`solve_outer_loop`). Returns False if ``params.convergence`` wasn't built.
    """
    if params.convergence is None:
        return False
    params.convergence.gradient_norm.compute(wp.get_stream())
    return params.convergence.converged(params.data.gtol)


@dataclass
class IterationSchedule:
    """How a VBD-family solver's subproblem iterations are recorded, for replay.

    ``iteration(k, kp)`` runs iteration ``kp`` of outer iteration ``k``. Iterations ``kp < prefix``
    record distinct work; later ones repeat with ``period`` (their host-side inputs depend on ``kp``
    only through ``kp % period``), so a block of a multiple of ``period`` iterations records once and
    replays. Each iteration applies ``sweeps_per_iteration`` VBD sweeps. ``scope`` names the
    subproblem's profiling scope.

    A subproblem runs ``n_iterations`` iterations (``n_subproblem_max_iters`` if None), then
    ``tail(k)`` if given, e.g. the sweeps left over when the sweep budget isn't a multiple of
    ``sweeps_per_iteration``. A convergence test that ends the subproblem early skips the tail too.
    """

    iteration: Callable[[int, int], None]
    prefix: int = 0
    period: int = 1
    sweeps_per_iteration: int = 1
    scope: str = "vbd.subproblem"
    n_iterations: int | None = None
    tail: Callable[[int], None] | None = None


def _subproblem_layout(
    schedule: IterationSchedule, n_sub: int, check_interval: int
) -> Tuple[int, int, int, int]:
    """Split a subproblem's ``n_sub`` iterations into (prefix, block, n_blocks, remainder).

    A block holds about ``check_interval`` sweeps, rounded up to a multiple of the period.
    """
    prefix = min(schedule.prefix, n_sub)
    per_block = max(1, math.ceil(check_interval / max(1, schedule.sweeps_per_iteration)))
    block = math.ceil(per_block / schedule.period) * schedule.period
    n_blocks = (n_sub - prefix) // block
    remainder = n_sub - prefix - n_blocks * block
    return prefix, block, n_blocks, remainder


def solve_outer_loop(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
    schedule: IterationSchedule,
) -> bool:
    """Run the outer (augmented-Lagrangian) loop shared by the VBD-family solvers.

    Equivalent to C++ ``vbd::Solve``: ``for k < n_max_iters: if converged: break; prepare; sweep;
    finalize``, plus a convergence test every ``check_interval`` sweeps inside each subproblem
    (:class:`ConvergenceCheck`). A test that passes inside a subproblem ends it; finalize and the
    outer test follow, as after a full subproblem.

    Recording, so that captured graphs stop early and stay small:

    - Outer iterations are unrolled on the host. Each runs inside ``wp.capture_if(active)``, so a
      converged step skips the rest.
    - Within a subproblem, the distinct first iterations are recorded once, then a block of identical
      iterations replays inside ``wp.capture_while``, then any remainder. The graph holds about one
      block per outer iteration instead of every sweep.
    - Conditional graph bodies must not allocate. ``cd.prepare_detection()`` (e.g. a BVH refit) runs
      between the conditional nodes. Detection and the feasibility filter run inside them only if
      ``cd.supports_conditional_capture``; otherwise they stay outside and also run after convergence,
      where they leave the solution as it is.

    Without ``params.convergence``, runs all ``n_max_iters`` outer iterations without tests.

    Returns:
        bool: Whether the solve converged. Always False while capturing, where the result stays on
        the device (``params.convergence.keep``).
    """
    check = params.convergence
    n_max_iters = int(params.data.n_max_iters)
    n_sub = (
        int(params.data.n_subproblem_max_iters)
        if schedule.n_iterations is None
        else int(schedule.n_iterations)
    )
    tail = schedule.tail
    if check is None:
        for k in range(n_max_iters):
            prepare_subproblem(fem, contact, cd, params)
            with profiling.scope(schedule.scope):
                for kp in range(n_sub):
                    schedule.iteration(k, kp)
                if tail is not None:
                    tail(k)
            finalize_subproblem(fem, contact, cd, params)
        return False
    gtol = float(params.data.gtol)
    prefix, block, n_blocks, remainder = _subproblem_layout(
        schedule, n_sub, int(check.params.check_interval)
    )
    detect_inside = cd.supports_conditional_capture

    def subproblem(k: int):
        prepare_subproblem(fem, contact, cd, params)
        with profiling.scope(schedule.scope):
            for kp in range(prefix):
                schedule.iteration(k, kp)
            if n_blocks == 0:
                for kp in range(prefix, n_sub):
                    schedule.iteration(k, kp)
                if tail is not None:
                    tail(k)
            else:

                def sweep_block():
                    for kp in range(prefix, prefix + block):
                        schedule.iteration(k, kp)
                    check.after_block(gtol, n_blocks)

                check.begin_inner(n_blocks)
                wp.capture_while(check.inner_keep, while_body=sweep_block)
                if remainder > 0 or tail is not None:
                    begin = prefix + n_blocks * block

                    def rest():
                        for kp in range(begin, begin + remainder):
                            schedule.iteration(k, kp)
                        if tail is not None:
                            tail(k)

                    wp.capture_if(check.subproblem_keep, on_true=rest)
        with profiling.scope("solve.finalize_subproblem"):
            update_dual(fem, contact)
            if detect_inside:
                cd.filter_step()

    def update_constraints():
        with profiling.scope("solve.finalize_subproblem"):
            if detect_inside:
                cd.detect_contacts(prepared=True)
            contact.update_constraint_set()
        check.end_outer(gtol, n_max_iters)

    check.start(gtol, n_max_iters)
    for k in range(n_max_iters):
        check.begin_outer()
        wp.capture_if(check.active, on_true=subproblem, k=k)
        with profiling.scope("solve.finalize_subproblem"):
            if detect_inside:
                cd.prepare_detection()
            else:
                cd.filter_step()
                cd.detect_contacts()
        wp.capture_if(check.active, on_true=update_constraints)
    return False if wp.get_device().is_capturing else check.converged(gtol)


@profiling.scoped("contact.penalty")
def compute_augmented_lagrangian_penalty(
    fem: FemElastoDynamics, contact: ContactDynamics, params: Params
):
    h2 = fem.h2
    n_nodes = fem.data.x.shape[0]
    block_dim = 32
    wp.launch(
        kernel=_compute_vertex_hessians,
        dim=n_nodes * block_dim,
        inputs=[fem.data, params.data, contact.Hi, h2],
        block_dim=block_dim,
    )
    contact.adapt_penalty_parameters()


@profiling.scoped("solve.prepare_subproblem")
def prepare_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
):
    """Assemble block-diagonal Hessian, update penalty parameter."""
    if contact.params.penalty_adaptivity == PenaltyAdaptivity.SUBPROBLEM:
        compute_augmented_lagrangian_penalty(fem, contact, params)


@profiling.scoped("solve.initialize")
def initialize_solve(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
):
    """Initialize the VBD solve by updating contact constraint set and restoring feasibility.

    Mirrors ``pbat::sim::algorithm::vbd::InitializeSolve``.
    Called once after :meth:`FemElastoDynamics.setup_time_integration_optimization`,
    before the first call to :func:`solve`.
    """
    # Compute contact set from current state
    cd.on_time_step_started()
    cd.detect_contacts(from_xt=True)
    contact.update_constraint_set()
    cd.filter_step()
    if contact.params.penalty_adaptivity == PenaltyAdaptivity.TIMESTEP:
        compute_augmented_lagrangian_penalty(fem, contact, params)


def update_dual(fem: FemElastoDynamics, contact: ContactDynamics):
    """Full dual update of a finished subproblem: slack, decay and Lagrange multipliers."""
    contact.update_dual(
        fem.data.x,
        fem.xt,
        request_slack_update=True,
        request_decay_update=True,
        request_lagrange_multiplier_update=True,
    )


@profiling.scoped("solve.finalize_subproblem")
def finalize_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
):
    """Finalize the current linearized subproblem.

    Mirrors ``pbat::sim::algorithm::vbd::FinalizeSubproblem``:
      1. Full dual update (slack + decay + Lagrange multipliers).
      2. Restore feasibility.
      3. Update constraint set for the next subproblem.
    """
    update_dual(fem, contact)
    cd.filter_step()
    cd.detect_contacts()
    contact.update_constraint_set()


@profiling.scoped("vbd.sweep")
def iterate(fem: FemElastoDynamics, contact: ContactDynamics, params: Params):
    """One VBD Gauss-Seidel sweep over all color partitions."""
    h2 = fem.h2
    # Copy current positions to buffer (for contact lagging)
    wp.copy(dest=params.data.xb, src=fem.data.x)
    # Process each color partition sequentially
    Pptr = params.Pptr
    n_partitions = len(Pptr) - 1
    for p in range(n_partitions):
        p_begin = int(Pptr[p])
        p_end = int(Pptr[p + 1])
        n_verts_in_partition = p_end - p_begin
        if n_verts_in_partition > 0:
            block_dim = 32
            wp.launch(
                kernel=_vertex_solve_kernel,
                dim=n_verts_in_partition * block_dim,
                inputs=[p_begin, fem.data, contact.data, params.data, h2],
                block_dim=block_dim,
            )


def subproblem_iteration(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
    kp: int,
):
    """Iteration ``kp`` of the subproblem: slack dual update, then one VBD sweep."""
    contact.update_dual(
        fem.data.x,
        fem.xt,
        request_slack_update=True,
        request_decay_update=False,
        request_lagrange_multiplier_update=False,
    )
    iterate(fem, contact, params)


@profiling.scoped("vbd.subproblem")
def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
):
    n_subproblem_max_iters = params.data.n_subproblem_max_iters
    for kp in range(n_subproblem_max_iters):
        subproblem_iteration(fem, contact, params, kp)


class VbdSolver:

    def __init__(self):
        pass

    def solve(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        cd: ContactDetection,
        params: Params,
    ) -> bool:
        initialize_solve(fem, contact, cd, params)
        converged = solve_outer_loop(
            fem, contact, cd, params, self.schedule(fem, contact, params)
        )
        fem.back_substitute_velocities()
        cd.on_time_step_ended()
        return converged

    def schedule(
        self, fem: FemElastoDynamics, contact: ContactDynamics, params: Params
    ) -> IterationSchedule:
        """Plain VBD iterations are all alike."""
        return IterationSchedule(
            lambda k, kp: subproblem_iteration(fem, contact, params, kp), scope="vbd.subproblem"
        )

    @property
    def supports_graph_capture(self):
        return True

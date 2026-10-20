"""
Device-side convergence test of the VBD-family solves.

Mirrors ``pbat::sim::algorithm::vbd::CheckConvergence``: stop once ``|g|^2 <= gtol^2``, where ``g`` is
the gradient of the time step's Lagrangian at the current contact multipliers, over free DOFs
(:class:`gpu.gradient.GradientNorm` without the augmented-Lagrangian penalty term, as C++
``ToGradient(..., bForAugmentedLagrangian = false)``). Without contact it is the objective's gradient,
so the test means the same inside a subproblem as between subproblems.

The test runs between outer iterations, as in C++, and also every ``check_interval`` sweeps inside a
subproblem. Between outer iterations, the step must also be feasible: no contact may penetrate its
margin by more than ``ftol * dmin``, i.e. ``max_j max(0, dmin - c_j(x)) <= ftol * dmin`` over the
current contacts (the maintainer's choice; C++ has no feasibility test). Inside a subproblem only the
gradient is tested: passing ends the subproblem, and both tests follow its finalization. Its outcome stays on the device in flags that conditional graph nodes
(``wp.capture_if``/``wp.capture_while``) read, so a captured step stops early too. See
``vbd.solver.solve_outer_loop`` for how the flags drive the loops.

The same kernels record each solve's outcome (:class:`SolveStatus`: outer iterations and sweeps used,
how it ended, and how far from the tolerances) in a small device ring, one row per solve. The host
reads it back with data it already synchronizes on (``SimulationState.step``), see
:meth:`ConvergenceCheck.enqueue_status_readback`.
"""

import enum
from typing import NamedTuple

import numpy as np
import warp as wp

import cupy as cp
import cuda.compute

from ...common.fields import DocField, SerializableMixin
from ..common.reduce import Reduce
from ..contact.dynamics import MeshDynamics as ContactDynamics, MeshDynamicsData
from ..contact import halfedges
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from ..gradient import GradientNorm


class ConvergenceParams(SerializableMixin):
    """GPU-only settings of the VBD-family convergence test.

    Held by ``SimulationState.convergence_params`` (kept across resets) and serialized under
    ``Solver/Convergence``, so configs set them as ``Solver.Convergence.check_interval`` and
    ``Solver.Convergence.ftol``. The gradient tolerance is relative (``rtol > 0``, default 1e-2, see below)
    or, with ``rtol = 0``, each solver's absolute ``gtol`` (``Solver.<solver>.gtol``; Newton's ``gtol2``). GPU Newton
    uses the same relative test (see ``gpu.newton.solver.check_convergence``).

    Relative test (``rtol > 0``): the gradient g of the time step's Lagrangian is turned into a
    position-like residual by the inverse lumped mass, M^-1 g (the correction an inertia-only Newton step
    would make), and compared with the step's free-flight displacement d = x~ - x^t (h v^t + h^2 a_ext
    for BDF1): ``RMS_M(M^-1 g) <= rtol * RMS_M(d)``, with mass-weighted RMS over free nodes, i.e.
    ``sum |g_i|^2 / m_i <= rtol^2 * sum m_i |d_i|^2``. It doesn't depend on the mesh's size or mass. The
    scale is floored, ``max(RMS_M(d), displacement_floor)``, so it never vanishes (e.g. for a body at
    rest without loads). Static problems have no inertial displacement and use the absolute ``gtol``.
    """

    check_interval = DocField(
        20, "VBD sweeps between convergence tests inside a subproblem (each test costs about a quarter sweep)"
    )
    ftol = DocField(
        0.5,
        "Feasibility tolerance between outer iterations: largest allowed penetration into the contact margin, as a fraction of dmin",
    )
    rtol = DocField(
        1e-2,
        "Relative gradient tolerance (0 = off: the solver's absolute gtol applies). Stops when the mass-normalized "
        "residual RMS(M^-1 g) is at most rtol times the step's free-flight displacement RMS(x~ - x^t) "
        "(h v^t + h^2 a_ext for BDF1), both mass-weighted over free nodes",
    )
    displacement_floor = DocField(
        1e-6,
        "Smallest displacement scale (m) of the relative gradient test: the scale is max(RMS(x~ - x^t), "
        "displacement_floor), so it never vanishes for bodies at rest",
    )


class SolveEnd(enum.IntEnum):
    """How a VBD-family solve ended (:attr:`SolveStatus.reason`)."""

    RUNNING = 0  # not finished: an unwritten row (no convergence check)
    CONVERGED = 1  # gradient test and feasibility met, within the outer-iteration budget
    GRADIENT_UNMET = 2  # outer iterations exhausted with the gradient test unmet (feasible)
    FEASIBILITY_UNMET = 3  # outer iterations exhausted, gradient test met but infeasible
    BOTH_UNMET = 4  # outer iterations exhausted with neither met


SOLVE_RUNNING = wp.constant(int(SolveEnd.RUNNING))
SOLVE_CONVERGED = wp.constant(int(SolveEnd.CONVERGED))
SOLVE_GRADIENT_UNMET = wp.constant(int(SolveEnd.GRADIENT_UNMET))
SOLVE_FEASIBILITY_UNMET = wp.constant(int(SolveEnd.FEASIBILITY_UNMET))
SOLVE_BOTH_UNMET = wp.constant(int(SolveEnd.BOTH_UNMET))


@wp.struct
class SolveStatus:
    """One VBD-family solve (time step or substep), as the convergence kernels record it."""

    solve: wp.int32  # index of the solve since the check was built (ConvergenceCheck.solves)
    outer: wp.int32  # outer iterations used
    sweeps: wp.int32  # VBD sweeps used (each Anderson step counts its p sweeps)
    reason: wp.int32  # SolveEnd
    gradient_ratio: wp.float32  # gradient measure / tolerance at the end (<= 1 passes)
    feasibility_ratio: wp.float32  # penetration / (ftol dmin) at the end (<= 1 passes)


class SweepLayout(NamedTuple):
    """How many sweeps a subproblem used, from its recorded layout (``solve_outer_loop``)."""

    full: int  # sweeps of a subproblem that ran its whole budget
    prefix: int  # sweeps before the first tested block
    block: int  # sweeps per tested block
    n_blocks: int  # tested blocks (0: no test inside the subproblem, so it always runs in full)


def select_new_status_rows(rows: np.ndarray, next_solve: int) -> tuple[np.ndarray, int, int]:
    """Pick the finished solves at or after ``next_solve`` from a copy of the status ring.

    Args:
        rows (np.ndarray): Structured array of :class:`SolveStatus` rows (the ring, in any order).
        next_solve (int): First solve index not read yet.

    Returns:
        tuple[np.ndarray, int, int]: The new rows by solve index, the next unread solve index, and how
        many solves were lost to the ring wrapping between reads.
    """
    new = rows[(rows["reason"] != int(SolveEnd.RUNNING)) & (rows["solve"] >= next_solve)]
    new = np.sort(new, order="solve")
    if new.shape[0] == 0:
        return new, next_solve, 0
    first, last = int(new["solve"][0]), int(new["solve"][-1])
    lost = (first - next_solve) + (last - first + 1 - new.shape[0])
    return new, last + 1, lost


@wp.func
def _gradient_ratio(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    gm2: wp.array[wp.float32],
    dm2: wp.array[wp.float32],
    rtol2: wp.float32,
) -> wp.float32:
    """The gradient test's measure over its tolerance, sqrt(sum |g|^2/m / (rtol^2 scale)) or
    |g| / gtol (see :func:`_is_converged`): at most 1 when it passes."""
    num = gnorm2[0]
    den = gtol2
    if rtol2 > wp.float32(0):
        num = gm2[0]
        den = rtol2 * dm2[0]
    ratio = wp.float32(0)
    if den > wp.float32(0):
        ratio = wp.sqrt(num / den)
    elif num > wp.float32(0):
        ratio = wp.float32(wp.inf)
    return ratio


@wp.func
def _feasibility_ratio(
    penetration: wp.array[wp.float32], ftol: wp.float32, dmin: wp.float32
) -> wp.float32:
    """Largest penetration over its tolerance ftol * dmin: at most 1 when feasible."""
    tol = ftol * dmin
    p = penetration[0]
    ratio = wp.float32(0)
    if tol > wp.float32(0):
        ratio = p / tol
    elif p > wp.float32(0):
        ratio = wp.float32(wp.inf)
    return ratio


@wp.func
def _solve_end(converged: bool, feasible: bool, iterations_left: bool) -> wp.int32:
    reason = SOLVE_BOTH_UNMET
    if converged and feasible:
        reason = SOLVE_CONVERGED
    elif iterations_left:
        reason = SOLVE_RUNNING
    elif feasible:
        reason = SOLVE_GRADIENT_UNMET
    elif converged:
        reason = SOLVE_FEASIBILITY_UNMET
    return reason


@wp.func
def _is_converged(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    gm2: wp.array[wp.float32],
    dm2: wp.array[wp.float32],
    rtol2: wp.float32,
) -> bool:
    """Relative test ``sum |g|^2/m <= rtol^2 max(sum m |d|^2, M floor^2)`` if enabled (``rtol2 > 0``;
    ``dm2`` holds the floored scale), else ``|g|^2 <= gtol^2``."""
    if rtol2 > wp.float32(0):
        return gm2[0] <= rtol2 * dm2[0]
    return gnorm2[0] <= gtol2


@wp.kernel
def _mass_normalized_residual_terms(
    g: wp.array[wp.vec3f], fem: FemElastoDynamicsData, terms: wp.array[wp.float32]
):
    i = wp.tid()
    terms[i] = wp.float32(0)
    if not is_dirichlet_node(fem.dmask, i):
        terms[i] = wp.dot(g[i], g[i]) / fem.m[i]


@wp.kernel
def _free_flight_displacement_terms(
    fem: FemElastoDynamicsData, terms: wp.array[wp.float32], masses: wp.array[wp.float32]
):
    i = wp.tid()
    terms[i] = wp.float32(0)
    masses[i] = wp.float32(0)
    if not is_dirichlet_node(fem.dmask, i):
        d = fem.xtilde[i] - fem.xt[i]
        terms[i] = fem.m[i] * wp.dot(d, d)
        masses[i] = fem.m[i]


@wp.kernel
def _floor_displacement_scale(
    dm2: wp.array[wp.float32], free_mass: wp.array[wp.float32], floor2: wp.float32
):
    dm2[0] = wp.max(dm2[0], free_mass[0] * floor2)


@wp.func
def _is_feasible(penetration: wp.array[wp.float32], ftol: wp.float32, dmin: wp.float32) -> bool:
    return penetration[0] <= ftol * dmin


# --- Feasibility: largest penetration into the contact margin, max_j max(0, dmin - c_j(x)) ---
# The gaps are the contact energy's linearized ones, (xcp1 - xcp2) . n, as in gpu/gradient.py.
# Slots whose partner index is the empty sentinel (v >= number of partners) are skipped: vertex SDF
# detection can count such slots (seen on CubeFallingOnCube), and their positions would be garbage.


@wp.func
def _record_penetration(gap: wp.float32, dmin: wp.float32, out: wp.array[wp.float32]):
    p = dmin - gap
    if p > wp.float32(0):
        wp.atomic_max(out, 0, p)


@wp.kernel
def _vv_penetration(
    x: wp.array[wp.vec3f], contact: MeshDynamicsData, n_u: wp.int32, out: wp.array[wp.float32]
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vv.prefix[n_u]:
        return
    vj = contact.contacts.vv.v[c]
    if wp.int32(vj) >= contact.meshes.V.shape[0]:
        return
    i = contact.meshes.V[contact.contacts.vv.u[c]]
    j = contact.meshes.V[vj]
    n = contact.contacts.vv_bases.n[c]
    _record_penetration(wp.dot(x[i] - x[j], n), contact.dmin, out)


@wp.kernel
def _ve_penetration(
    x: wp.array[wp.vec3f], contact: MeshDynamicsData, n_u: wp.int32, out: wp.array[wp.float32]
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ve.prefix[n_u]:
        return
    i = contact.meshes.V[contact.contacts.ve.u[c]]
    he = contact.contacts.ve.v[c]
    if wp.int32(he) >= 3 * contact.meshes.F.shape[0]:
        return
    b1 = contact.contacts.ve_bary[c]
    b0 = wp.float32(1) - b1
    ea = halfedges.incoming_vertex(contact.meshes.F, he)
    eb = halfedges.outgoing_vertex(contact.meshes.F, he)
    n = contact.contacts.ve_bases.n[c]
    _record_penetration(wp.dot(x[i] - (b0 * x[ea] + b1 * x[eb]), n), contact.dmin, out)


@wp.kernel
def _vf_penetration(
    x: wp.array[wp.vec3f], contact: MeshDynamicsData, n_u: wp.int32, out: wp.array[wp.float32]
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vf.prefix[n_u]:
        return
    i = contact.meshes.V[contact.contacts.vf.u[c]]
    f = contact.contacts.vf.v[c]
    if wp.int32(f) >= contact.meshes.F.shape[0]:
        return
    finds = contact.meshes.F[f]
    uv = contact.contacts.vf_bary[c]
    b0 = wp.float32(1) - uv[0] - uv[1]
    xc = b0 * x[finds[0]] + uv[0] * x[finds[1]] + uv[1] * x[finds[2]]
    n = contact.contacts.vf_bases.n[c]
    _record_penetration(wp.dot(x[i] - xc, n), contact.dmin, out)


@wp.kernel
def _ee_penetration(
    x: wp.array[wp.vec3f], contact: MeshDynamicsData, n_u: wp.int32, out: wp.array[wp.float32]
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ee.prefix[n_u]:
        return
    he_u = contact.contacts.ee.u[c]
    he_v = contact.contacts.ee.v[c]
    if wp.int32(he_v) >= 3 * contact.meshes.F.shape[0]:
        return
    st = contact.contacts.ee_bary[c]
    ia = halfedges.incoming_vertex(contact.meshes.F, he_u)
    ib = halfedges.outgoing_vertex(contact.meshes.F, he_u)
    ic = halfedges.incoming_vertex(contact.meshes.F, he_v)
    id = halfedges.outgoing_vertex(contact.meshes.F, he_v)
    xcp1 = (wp.float32(1) - st[0]) * x[ia] + st[0] * x[ib]
    xcp2 = (wp.float32(1) - st[1]) * x[ic] + st[1] * x[id]
    n = contact.contacts.ee_bases.n[c]
    _record_penetration(wp.dot(xcp1 - xcp2, n), contact.dmin, out)


@wp.kernel
def _start_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    gm2: wp.array[wp.float32],
    dm2: wp.array[wp.float32],
    rtol2: wp.float32,
    penetration: wp.array[wp.float32],
    ftol: wp.float32,
    contact: MeshDynamicsData,
    n_max_iters: wp.int32,
    k: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    status: wp.array[SolveStatus],
    solves: wp.array[wp.int32],
):
    k[0] = 0
    converged = _is_converged(gnorm2, gtol2, gm2, dm2, rtol2)
    feasible = _is_feasible(penetration, ftol, contact.dmin)
    done = converged and feasible
    keep[0] = wp.where(n_max_iters > 0 and not done, 1, 0)
    # A new solve: advance the counter and start its status row
    n = solves[0]
    solves[0] = n + 1
    row = SolveStatus()
    row.solve = n
    row.outer = 0
    row.sweeps = 0
    row.reason = _solve_end(converged, feasible, n_max_iters > 0)
    row.gradient_ratio = _gradient_ratio(gnorm2, gtol2, gm2, dm2, rtol2)
    row.feasibility_ratio = _feasibility_ratio(penetration, ftol, contact.dmin)
    status[n % status.shape[0]] = row


@wp.kernel
def _begin_outer_kernel(keep: wp.array[wp.int32], active: wp.array[wp.int32]):
    active[0] = keep[0]


@wp.kernel
def _begin_inner_kernel(
    n_blocks: wp.int32,
    blocks: wp.array[wp.int32],
    inner_keep: wp.array[wp.int32],
    subproblem_keep: wp.array[wp.int32],
):
    blocks[0] = 0
    inner_keep[0] = wp.where(n_blocks > 0, 1, 0)
    subproblem_keep[0] = 1


@wp.kernel
def _after_block_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    gm2: wp.array[wp.float32],
    dm2: wp.array[wp.float32],
    rtol2: wp.float32,
    n_blocks: wp.int32,
    blocks: wp.array[wp.int32],
    inner_keep: wp.array[wp.int32],
    subproblem_keep: wp.array[wp.int32],
):
    b = blocks[0] + 1
    blocks[0] = b
    converged = _is_converged(gnorm2, gtol2, gm2, dm2, rtol2)
    subproblem_keep[0] = wp.where(converged, 0, 1)
    inner_keep[0] = wp.where(not converged and b < n_blocks, 1, 0)


@wp.kernel
def _end_outer_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    gm2: wp.array[wp.float32],
    dm2: wp.array[wp.float32],
    rtol2: wp.float32,
    penetration: wp.array[wp.float32],
    ftol: wp.float32,
    contact: MeshDynamicsData,
    n_max_iters: wp.int32,
    k: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    status: wp.array[SolveStatus],
    solves: wp.array[wp.int32],
    sweeps_full: wp.int32,
    sweeps_prefix: wp.int32,
    sweeps_block: wp.int32,
    n_blocks: wp.int32,
    blocks: wp.array[wp.int32],
    subproblem_keep: wp.array[wp.int32],
):
    kk = k[0] + 1
    k[0] = kk
    converged = _is_converged(gnorm2, gtol2, gm2, dm2, rtol2)
    feasible = _is_feasible(penetration, ftol, contact.dmin)
    done = converged and feasible
    keep[0] = wp.where(kk < n_max_iters and not done, 1, 0)
    # Update the solve's status row with the subproblem that just ended (see SweepLayout)
    idx = (solves[0] - 1) % status.shape[0]
    row = status[idx]
    row.outer = kk
    if n_blocks == 0 or subproblem_keep[0] != 0:
        row.sweeps = row.sweeps + sweeps_full
    else:
        row.sweeps = row.sweeps + sweeps_prefix + blocks[0] * sweeps_block
    row.reason = _solve_end(converged, feasible, kk < n_max_iters)
    row.gradient_ratio = _gradient_ratio(gnorm2, gtol2, gm2, dm2, rtol2)
    row.feasibility_ratio = _feasibility_ratio(penetration, ftol, contact.dmin)
    status[idx] = row


class ConvergenceCheck:
    """Outer-iteration counter and loop flags of a solve, on the device.

    Flags (each a (1,) int32 array, read by conditional graph nodes):

    - ``keep``: the step goes on (not both optimal and feasible, and outer iterations left).
    - ``active``: the current outer iteration runs (``keep`` when it started).
    - ``inner_keep``: the current subproblem's sweep blocks go on.
    - ``subproblem_keep``: the current subproblem's remaining sweeps run (not converged inside it).

    ``k`` counts the outer iterations done, ``penetration`` holds the last feasibility measure.
    :meth:`converged` reads the last gradient test on the host.

    ``solves`` counts the solves started (:meth:`start`), and ``status`` is a ring of
    ``status_capacity`` :class:`SolveStatus` rows, row ``solve % status_capacity`` per solve. The
    kernels that run anyway write them, so recording adds no graph nodes.
    """

    def __init__(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        params: ConvergenceParams | None = None,
        status_capacity: int = 1,
    ):
        """
        Args:
            fem (FemElastoDynamics): The problem whose gradient is tested.
            contact (ContactDynamics): Its contact model (multipliers for the gradient, contact sets
                for the feasibility test).
            params (ConvergenceParams | None): Check interval and feasibility tolerance; defaults if None.
            status_capacity (int): Rows of the status ring: at least the solves between two
                :meth:`read_status` calls (``SimulationState`` reads every step: its substeps).
        """
        self.params = params or ConvergenceParams()
        capacity = max(1, int(status_capacity))
        self.status = wp.zeros(capacity, dtype=SolveStatus)
        self.solves = wp.zeros(1, dtype=wp.int32)
        self._status_host = wp.zeros(capacity, dtype=SolveStatus, device="cpu", pinned=True)
        self._next_solve = 0
        self._fem = fem
        self._contact = contact
        self.gradient_norm = GradientNorm(fem, contact, augmented_lagrangian=False)
        # Relative test (params.rtol > 0): sum |g_i|^2 / m_i and sum m_i |d_i|^2 over free nodes
        n_nodes = fem.data.x.shape[0]
        self._terms = wp.zeros(n_nodes, dtype=wp.float32)
        self._masses = wp.zeros(n_nodes, dtype=wp.float32)
        self.residual_m2 = wp.zeros(1, dtype=wp.float32)
        self.displacement_m2 = wp.zeros(1, dtype=wp.float32)  # floored scale, see ConvergenceParams
        self.free_mass = wp.zeros(1, dtype=wp.float32)
        self._reduce_residual = Reduce(
            d_in=cp.asarray(self._terms),
            d_out=cp.asarray(self.residual_m2),
            num_items=n_nodes,
            op=cuda.compute.OpKind.PLUS,
        )
        self._reduce_displacement = Reduce(
            d_in=cp.asarray(self._terms),
            d_out=cp.asarray(self.displacement_m2),
            num_items=n_nodes,
            op=cuda.compute.OpKind.PLUS,
        )
        self._reduce_free_mass = Reduce(
            d_in=cp.asarray(self._masses),
            d_out=cp.asarray(self.free_mass),
            num_items=n_nodes,
            op=cuda.compute.OpKind.PLUS,
        )
        self.penetration = wp.zeros(1, dtype=wp.float32)
        self.k = wp.zeros(1, dtype=wp.int32)
        self.keep = wp.zeros(1, dtype=wp.int32)
        self.active = wp.zeros(1, dtype=wp.int32)
        self.blocks = wp.zeros(1, dtype=wp.int32)
        self.inner_keep = wp.zeros(1, dtype=wp.int32)
        self.subproblem_keep = wp.zeros(1, dtype=wp.int32)

    @property
    def relative(self) -> bool:
        """Whether the relative test applies: enabled, and not a static problem."""
        return float(self.params.rtol) > 0.0 and not self._fem.static

    def _test(self) -> None:
        stream = wp.get_stream()
        self.gradient_norm.compute(stream)
        if self.relative:
            self._measure_residual(self.gradient_norm.g)

    def _measure_residual(self, g: wp.array) -> None:
        wp.launch(
            _mass_normalized_residual_terms,
            dim=self._terms.shape[0],
            inputs=[g, self._fem.data, self._terms],
        )
        self._reduce_residual(wp.get_stream())

    def measure_scale(self) -> None:
        """Measure the relative test's scale for the time step: the step's sum m_i |x~_i - x^t_i|^2
        over free nodes, floored at (free mass) * displacement_floor^2. No-op unless :attr:`relative`.
        Capturable."""
        if self.relative:
            stream = wp.get_stream()
            wp.launch(
                _free_flight_displacement_terms,
                dim=self._terms.shape[0],
                inputs=[self._fem.data, self._terms, self._masses],
            )
            self._reduce_displacement(stream)
            self._reduce_free_mass(stream)
            wp.launch(
                _floor_displacement_scale,
                dim=1,
                inputs=[self.displacement_m2, self.free_mass, float(self.params.displacement_floor) ** 2],
            )

    def _tolerance_inputs(self, gtol: float) -> list:
        return [
            self.gradient_norm.gnorm2,
            float(gtol) ** 2,
            self.residual_m2,
            self.displacement_m2,
            float(self.params.rtol) ** 2 if self.relative else 0.0,
        ]

    def _measure_penetration(self) -> None:
        self.penetration.zero_()
        contact = self._contact
        x = self._fem.data.x
        for kernel, cs in zip(
            [_vv_penetration, _ve_penetration, _vf_penetration, _ee_penetration],
            [contact.cvv, contact.cve, contact.cvf, contact.cee],
        ):
            wp.launch(kernel, dim=cs.capacity, inputs=[x, contact.data, cs.n_u, self.penetration])

    def _outer_test_inputs(self, gtol: float, n_max_iters: int) -> list:
        return self._tolerance_inputs(gtol) + [
            self.penetration,
            float(self.params.ftol),
            self._contact.data,
            int(n_max_iters),
            self.k,
            self.keep,
        ]

    def start(self, gtol: float, n_max_iters: int) -> None:
        """Test the initial positions: ``k = 0``, ``keep`` = not (optimal and feasible).

        Args:
            gtol (float): Gradient-norm tolerance (absolute; used when the relative test is off).
            n_max_iters (int): Maximum outer iterations; ``keep`` is 0 if it's 0.
        """
        self.measure_scale()
        self._test()
        self._measure_penetration()
        wp.launch(
            _start_kernel,
            dim=1,
            inputs=self._outer_test_inputs(gtol, n_max_iters) + [self.status, self.solves],
        )

    def begin_outer(self) -> None:
        """Latch ``keep`` into ``active`` for the outer iteration that starts."""
        wp.launch(_begin_outer_kernel, dim=1, inputs=[self.keep, self.active])

    def begin_inner(self, n_blocks: int) -> None:
        """Start a subproblem's loop over sweep blocks: ``inner_keep`` = 1 if ``n_blocks > 0``.

        Args:
            n_blocks (int): Number of blocks in the subproblem.
        """
        wp.launch(
            _begin_inner_kernel,
            dim=1,
            inputs=[int(n_blocks), self.blocks, self.inner_keep, self.subproblem_keep],
        )

    def after_block(self, gtol: float, n_blocks: int) -> None:
        """Test after a sweep block: a converged subproblem stops, and so do its remaining sweeps.

        Only the gradient is tested here; feasibility is tested between outer iterations.

        Args:
            gtol (float): Gradient-norm tolerance (absolute).
            n_blocks (int): Number of blocks in the subproblem.
        """
        self._test()
        wp.launch(
            _after_block_kernel,
            dim=1,
            inputs=self._tolerance_inputs(gtol)
            + [
                int(n_blocks),
                self.blocks,
                self.inner_keep,
                self.subproblem_keep,
            ],
        )

    def end_outer(
        self, gtol: float, n_max_iters: int, layout: SweepLayout | None = None
    ) -> None:
        """Test after an outer iteration (C++'s check before the next one, plus feasibility):
        ``k += 1``, update ``keep`` and the solve's status row.

        Args:
            gtol (float): Gradient-norm tolerance (absolute).
            n_max_iters (int): Maximum outer iterations.
            layout (SweepLayout | None): The subproblem's sweep layout, to count the sweeps it used;
                None counts none.
        """
        layout = layout or SweepLayout(0, 0, 0, 0)
        self._test()
        self._measure_penetration()
        wp.launch(
            _end_outer_kernel,
            dim=1,
            inputs=self._outer_test_inputs(gtol, n_max_iters)
            + [
                self.status,
                self.solves,
                int(layout.full),
                int(layout.prefix),
                int(layout.block),
                int(layout.n_blocks),
                self.blocks,
                self.subproblem_keep,
            ],
        )

    def enqueue_status_readback(self) -> None:
        """Queue an asynchronous copy of the status ring into pinned host memory on the current
        stream. Doesn't synchronize: :meth:`read_status` is valid after the next synchronization of
        that stream (``SimulationState.step`` queues it right before its own)."""
        wp.copy(self._status_host, self.status)

    def read_status(self) -> tuple[np.ndarray, int]:
        """The solves finished since the last call, from the last :meth:`enqueue_status_readback`
        (after a synchronization). Host-only, no GPU calls.

        Returns:
            tuple[np.ndarray, int]: :class:`SolveStatus` rows by solve index, and how many solves were
            lost because the ring wrapped between reads.
        """
        rows, self._next_solve, lost = select_new_status_rows(
            np.array(self._status_host.numpy()), self._next_solve
        )
        return rows, lost

    def converged(self, gtol: float) -> bool:
        """Host-side gradient test from the last test (relative or absolute, see
        :class:`ConvergenceParams`). Synchronizes, so not capturable.

        Args:
            gtol (float): Gradient-norm tolerance (absolute).

        Returns:
            bool: Whether the last gradient test passed (feasibility isn't included).
        """
        if self.relative:
            return self._relative_passes()
        return float(self.gradient_norm.gnorm2.numpy()[0]) <= float(gtol) * float(gtol)

    def _relative_passes(self) -> bool:
        dm2 = float(self.displacement_m2.numpy()[0])
        return float(self.residual_m2.numpy()[0]) <= float(self.params.rtol) ** 2 * dm2

    def test_gradient(self) -> bool:
        """Host-side gradient test of the current positions (relative or absolute with ``gtol``
        unused: see :meth:`converged`). Synchronizes, so not capturable.

        Returns:
            bool: Whether the relative test passes (False if it's off).
        """
        if not self.relative:
            return False
        self._test()
        return self._relative_passes()

    def test_outer(self) -> bool:
        """Host-side outer test of the current positions, as the VBD family's between outer
        iterations: the relative gradient test (:meth:`test_gradient`) and feasibility, the largest
        penetration into the contact margin at most ``ftol * dmin``. Synchronizes, so not capturable.

        Returns:
            bool: Whether both pass (False if the relative test is off).
        """
        if not self.test_gradient():
            return False
        self._measure_penetration()
        ftol, dmin = float(self.params.ftol), float(self._contact.data.dmin)
        return float(self.penetration.numpy()[0]) <= ftol * dmin

    def relative_test(self, g: wp.array) -> bool:
        """Host-side relative test of a given gradient, ``sum |g_i|^2 / m_i <= rtol^2 * scale``, against
        the scale from the last :meth:`measure_scale`. Synchronizes, so not capturable.

        Newton uses it for its augmented-Lagrangian subproblem gradient.

        Args:
            g (wp.array): (N,) vec3f gradient.

        Returns:
            bool: Whether ``g`` passes (False if the relative test is off).
        """
        if not self.relative:
            return False
        self._measure_residual(g)
        return self._relative_passes()

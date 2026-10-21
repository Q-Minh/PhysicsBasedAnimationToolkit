"""
Device-side convergence test of the VBD-family solves.

Mirrors ``pbat::sim::algorithm::vbd::CheckConvergence``: stop once ``|g|^2 <= gtol^2``, where ``g`` is
the gradient of the time step's Lagrangian at the current contact multipliers, over free DOFs
(:class:`gpu.gradient.GradientNorm` without the augmented-Lagrangian penalty term, as C++
``ToGradient(..., bForAugmentedLagrangian = false)``). Without contact it is the objective's gradient,
so the test means the same inside a subproblem as between subproblems.

The test runs between outer iterations, as in C++, and also every ``check_interval`` sweeps inside a
subproblem. Between outer iterations, the step must also be feasible: no contact may penetrate its
margin by more than ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``, i.e.
``max_j max(0, dmin - c_j(x)) <= ftol * max(|dmin|, FEASIBILITY_FLOOR)`` over the
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
    ``Solver.Convergence.ftol``. The stationarity (gradient) test of dynamic problems uses the relative and
    absolute stationarity tolerances ``rstol`` and ``astol`` (see below); with both 0, and for static
    problems, each solver's absolute ``gtol`` applies instead (``Solver.<solver>.gtol``; Newton's ``gtol2``).
    GPU Newton uses the same test (see ``gpu.newton.solver.check_convergence``).

    Stationarity test: the gradient g of the time step's Lagrangian is turned into a position-like residual
    by the inverse lumped mass, M^-1 g (the correction an inertia-only Newton step would make), and passes
    when ``RMS_M(M^-1 g) <= max(rstol * RMS_M(d), astol * L)``, with d = x~ - x^t the step's free-flight
    displacement (h v^t + h^2 a_ext for BDF1), mass-weighted RMS over free nodes, and L the mean
    bounding-box diagonal of the scene's bodies at rest (:func:`mean_body_diagonal`), i.e.
    ``sum |g_i|^2 / m_i <= max(rstol^2 sum m_i |d_i|^2, (astol L)^2 sum m_i)``. Neither term depends on the
    mesh's resolution or mass, and both rstol and astol are dimensionless. The absolute term is a float32-noise
    fallback for steps with almost no motion (RMS_M(d) ~ 0), where the relative term would ask for a zero
    residual: astol = 1e-6 is about 8 float32 ulps of L. It must stay well below the step's motion, or the test
    passes on a step that hasn't moved, so it isn't a per-scene tuning knob. Stiff scenes near rest may then
    run their full budget, since their float32 residual floor, about (h omega)^2 ulps for a node's
    stiffness-to-mass frequency omega, can exceed rstol * RMS_M(d). Static problems have no inertial
    displacement and use the absolute ``gtol``.
    """

    check_interval = DocField(
        20, "VBD sweeps between convergence tests inside a subproblem (each test costs about a quarter sweep)"
    )
    ftol = DocField(
        0.5,
        "Feasibility tolerance between outer iterations: largest allowed penetration into the contact margin, as a "
        "fraction of |dmin| (at least 1e-6 m, so a zero or negative margin still has a positive tolerance)",
    )
    rstol = DocField(
        1e-2,
        "Relative stationarity tolerance: the gradient test passes when the mass-normalized residual RMS(M^-1 g) is "
        "at most max(rstol * RMS(x~ - x^t), astol * L), x~ - x^t being the step's free-flight displacement "
        "(h v^t + h^2 a_ext for BDF1), both mass-weighted over free nodes. With rstol = astol = 0, the solver's "
        "absolute gtol applies instead",
    )
    astol = DocField(
        1e-6,
        "Absolute stationarity tolerance, relative to the scene's size: the gradient test passes when the "
        "mass-normalized residual RMS(M^-1 g) is at most max(rstol * RMS(x~ - x^t), astol * L), L being the mean "
        "bounding-box diagonal of the scene's bodies at rest. astol * L is a float32-noise fallback for steps with "
        "almost no motion (RMS(x~ - x^t) ~ 0); the default 1e-6 is about 8 float32 ulps of L. It must stay well "
        "below the step's motion, or the test passes on a step that hasn't moved, so it isn't a per-scene tuning "
        "knob (a former astol = 5e-4 passed the square trampoline's ~1.5 mm residual against ~1 mm of gravity "
        "sag per step)",
    )

    def deserialize(self, grp) -> None:
        """Read the parameters, taking ``rstol`` from the former ``rtol`` in files that predate it (configs
        go through ``config.RENAMED`` instead)."""
        super().deserialize(grp)
        if "rstol" not in grp.attrs and "rtol" in grp.attrs:
            self.rstol = float(grp.attrs["rtol"])


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
    feasibility_ratio: wp.float32  # penetration / (ftol max(|dmin|, floor)) at the end (<= 1 passes)


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
    stol2: wp.array[wp.float32],
    relative: wp.int32,
) -> wp.float32:
    """The gradient test's measure over its tolerance, sqrt(sum |g|^2/m / stol2) or |g| / gtol (see
    :func:`_is_converged`): at most 1 when it passes."""
    num = gnorm2[0]
    den = gtol2
    if relative != 0:
        num = gm2[0]
        den = stol2[0]
    ratio = wp.float32(0)
    if den > wp.float32(0):
        ratio = wp.sqrt(num / den)
    elif num > wp.float32(0):
        ratio = wp.float32(wp.inf)
    return ratio


FEASIBILITY_FLOOR = 1e-6
"""Smallest length scale (m) of the feasibility tolerance, ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``:
a zero or negative contact margin (the paper configs use dmin = -1e-4) still gets a positive tolerance."""


def feasibility_tolerance(ftol: float, dmin: float) -> float:
    """Host-side ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``, as :func:`_feasibility_tolerance`."""
    return float(ftol) * max(abs(float(dmin)), FEASIBILITY_FLOOR)


def mean_body_diagonal(X: np.ndarray, E: np.ndarray) -> float:
    """Mean bounding-box diagonal (m) of a mesh's bodies, the connected components of its elements, at
    the given positions. Nodes in no element (e.g. static colliders' contact nodes) belong to no body.

    Args:
        X (np.ndarray): (N, 3) positions.
        E (np.ndarray): (|# elements|, 4) element nodes.

    Returns:
        float: The mean diagonal over bodies (0 without elements).
    """
    import scipy.sparse
    import scipy.sparse.csgraph

    E = np.asarray(E).reshape(-1, 4)
    if E.shape[0] == 0:
        return 0.0
    n = X.shape[0]
    rows = np.repeat(E[:, 0], 3)
    cols = E[:, 1:].reshape(-1)
    graph = scipy.sparse.coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
    _, labels = scipy.sparse.csgraph.connected_components(graph, directed=False)
    nodes = np.unique(E)
    diagonals = []
    for body in np.unique(labels[nodes]):
        P = X[nodes[labels[nodes] == body]]
        diagonals.append(float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))))
    return float(np.mean(diagonals))


def ratio_to_tolerance(value: float, tolerance: float) -> float:
    """Host-side ``value / tolerance`` as the status kernels compute their ratios (:func:`_gradient_ratio`,
    :func:`_feasibility_ratio`): 0 for a zero value, inf for a positive value over a zero tolerance."""
    if tolerance > 0.0:
        return value / tolerance
    return float("inf") if value > 0.0 else 0.0


@wp.func
def _feasibility_tolerance(ftol: wp.float32, dmin: wp.float32) -> wp.float32:
    """Largest allowed penetration into the contact margin, ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``.
    Equals ``ftol * dmin`` for a margin of at least the floor."""
    return ftol * wp.max(wp.abs(dmin), wp.float32(FEASIBILITY_FLOOR))


@wp.func
def _feasibility_ratio(
    penetration: wp.array[wp.float32], ftol: wp.float32, dmin: wp.float32
) -> wp.float32:
    """Largest penetration over its tolerance (:func:`_feasibility_tolerance`): at most 1 when feasible."""
    tol = _feasibility_tolerance(ftol, dmin)
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
    stol2: wp.array[wp.float32],
    relative: wp.int32,
) -> bool:
    """Stationarity test ``sum |g|^2/m <= stol2 = max(rstol^2 sum m |d|^2, (astol L)^2 M)`` if enabled
    (``relative``), else ``|g|^2 <= gtol^2``."""
    if relative != 0:
        return gm2[0] <= stol2[0]
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


@wp.func
def _stationarity_tol2(
    dm2: wp.float32, free_mass: wp.float32, rstol2: wp.float32, atol2: wp.float32
) -> wp.float32:
    """Squared stationarity tolerance on sum |g|^2/m: max(rstol^2 sum m |d|^2, atol^2 sum m), atol = astol L
    being the absolute tolerance in m."""
    return wp.max(rstol2 * dm2, atol2 * free_mass)


@wp.kernel
def _stationarity_tolerance(
    dm2: wp.array[wp.float32],
    free_mass: wp.array[wp.float32],
    rstol2: wp.float32,
    atol2: wp.float32,
    stol2: wp.array[wp.float32],
):
    stol2[0] = _stationarity_tol2(dm2[0], free_mass[0], rstol2, atol2)


@wp.func
def _is_feasible(penetration: wp.array[wp.float32], ftol: wp.float32, dmin: wp.float32) -> bool:
    return penetration[0] <= _feasibility_tolerance(ftol, dmin)


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
    stol2: wp.array[wp.float32],
    relative: wp.int32,
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
    converged = _is_converged(gnorm2, gtol2, gm2, stol2, relative)
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
    row.gradient_ratio = _gradient_ratio(gnorm2, gtol2, gm2, stol2, relative)
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
    stol2: wp.array[wp.float32],
    relative: wp.int32,
    n_blocks: wp.int32,
    blocks: wp.array[wp.int32],
    inner_keep: wp.array[wp.int32],
    subproblem_keep: wp.array[wp.int32],
):
    b = blocks[0] + 1
    blocks[0] = b
    converged = _is_converged(gnorm2, gtol2, gm2, stol2, relative)
    subproblem_keep[0] = wp.where(converged, 0, 1)
    inner_keep[0] = wp.where(not converged and b < n_blocks, 1, 0)


@wp.kernel
def _end_outer_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    gm2: wp.array[wp.float32],
    stol2: wp.array[wp.float32],
    relative: wp.int32,
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
    converged = _is_converged(gnorm2, gtol2, gm2, stol2, relative)
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
    row.gradient_ratio = _gradient_ratio(gnorm2, gtol2, gm2, stol2, relative)
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
        length_scale: float | None = None,
    ):
        """
        Args:
            fem (FemElastoDynamics): The problem whose gradient is tested.
            contact (ContactDynamics): Its contact model (multipliers for the gradient, contact sets
                for the feasibility test).
            params (ConvergenceParams | None): Check interval and feasibility tolerance; defaults if None.
            status_capacity (int): Rows of the status ring: at least the solves between two
                :meth:`read_status` calls (``SimulationState`` reads every step: its substeps).
            length_scale (float | None): The scene's size L (m) that scales ``astol``: the mean
                bounding-box diagonal of its bodies (:func:`mean_body_diagonal`), computed from the rest
                positions if None.
        """
        self.params = params or ConvergenceParams()
        if length_scale is None:
            length_scale = mean_body_diagonal(fem.data.X.numpy(), fem.data.E.numpy())
        self.length_scale = float(length_scale)
        capacity = max(1, int(status_capacity))
        self.status = wp.zeros(capacity, dtype=SolveStatus)
        self.solves = wp.zeros(1, dtype=wp.int32)
        self._status_host = wp.zeros(capacity, dtype=SolveStatus, device="cpu", pinned=True)
        self._next_solve = 0
        self._fem = fem
        self._contact = contact
        self.gradient_norm = GradientNorm(fem, contact, augmented_lagrangian=False)
        # Stationarity test (rstol or astol > 0): sum |g_i|^2 / m_i against
        # stol2 = max(rstol^2 sum m_i |d_i|^2, (astol L)^2 sum m_i), sums over free nodes
        n_nodes = fem.data.x.shape[0]
        self._terms = wp.zeros(n_nodes, dtype=wp.float32)
        self._masses = wp.zeros(n_nodes, dtype=wp.float32)
        self.residual_m2 = wp.zeros(1, dtype=wp.float32)
        self.displacement_m2 = wp.zeros(1, dtype=wp.float32)  # sum m_i |d_i|^2, the relative term's scale
        self.stationarity_tol2 = wp.zeros(1, dtype=wp.float32)  # stol2, see ConvergenceParams
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
        """Whether the stationarity test with ``rstol``/``astol`` applies (rather than the solvers'
        absolute ``gtol``): either is positive, and the problem isn't static."""
        enabled = float(self.params.rstol) > 0.0 or float(self.params.astol) > 0.0
        return enabled and not self._fem.static

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
        """Measure the stationarity tolerance for the time step, ``stol2 = max(rstol^2 sum m_i |x~_i -
        x^t_i|^2, (astol L)^2 sum m_i)`` over free nodes. No-op unless :attr:`relative`. Capturable."""
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
                _stationarity_tolerance,
                dim=1,
                inputs=[
                    self.displacement_m2,
                    self.free_mass,
                    float(self.params.rstol) ** 2,
                    (float(self.params.astol) * self.length_scale) ** 2,
                    self.stationarity_tol2,
                ],
            )

    def _tolerance_inputs(self, gtol: float) -> list:
        return [
            self.gradient_norm.gnorm2,
            float(gtol) ** 2,
            self.residual_m2,
            self.stationarity_tol2,
            int(self.relative),
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
        return float(self.residual_m2.numpy()[0]) <= float(self.stationarity_tol2.numpy()[0])

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
        penetration into the contact margin at most ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``.
        Both are always measured, so :meth:`gradient_ratio` and :meth:`feasibility_ratio` describe
        this test. Synchronizes, so not capturable.

        Returns:
            bool: Whether both pass (False if the relative test is off).
        """
        gradient_passes = self.test_gradient()
        self._measure_penetration()
        tol = feasibility_tolerance(self.params.ftol, self._contact.data.dmin)
        return gradient_passes and float(self.penetration.numpy()[0]) <= tol

    def gradient_ratio(self) -> float:
        """Host-side stationarity measure of the last test over its tolerance,
        ``sqrt(sum |g_i|^2 / m_i / stol2)``, as the status ring's ``gradient_ratio``: at most 1 when it
        passes. Synchronizes.

        Returns:
            float: The ratio (from the last :meth:`test_gradient`, :meth:`test_outer` or
            :meth:`relative_test`).
        """
        den = float(self.stationarity_tol2.numpy()[0])
        return float(np.sqrt(ratio_to_tolerance(float(self.residual_m2.numpy()[0]), den)))

    def feasibility_ratio(self, measure: bool = False) -> float:
        """Host-side largest penetration into the contact margin over its tolerance
        ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``, as the status ring's ``feasibility_ratio``: at most 1
        when feasible. Synchronizes.

        Args:
            measure (bool): Measure the penetration at the current positions first; otherwise use
                the last :meth:`test_outer`'s.

        Returns:
            float: The ratio.
        """
        if measure:
            self._measure_penetration()
        tol = feasibility_tolerance(self.params.ftol, self._contact.data.dmin)
        return ratio_to_tolerance(float(self.penetration.numpy()[0]), tol)

    def relative_test(self, g: wp.array) -> bool:
        """Host-side stationarity test of a given gradient, ``sum |g_i|^2 / m_i <= stol2``, against the
        tolerance from the last :meth:`measure_scale`. Synchronizes, so not capturable.

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


import unittest


class TestFeasibilityTolerance(unittest.TestCase):
    def test_tolerance(self):
        wp.init()

        @wp.kernel
        def k(
            ftol: wp.array[wp.float32],
            dmin: wp.array[wp.float32],
            penetration: wp.array[wp.float32],
            tol: wp.array[wp.float32],
            ratio: wp.array[wp.float32],
            feasible: wp.array[wp.int32],
        ):
            i = wp.tid()
            p = wp.array(ptr=penetration.ptr + wp.uint64(i * 4), shape=(1,), dtype=wp.float32)
            tol[i] = _feasibility_tolerance(ftol[i], dmin[i])
            ratio[i] = _feasibility_ratio(p, ftol[i], dmin[i])
            feasible[i] = wp.where(_is_feasible(p, ftol[i], dmin[i]), 1, 0)

        cases = [  # (ftol, dmin, penetration)
            (0.5, 2e-3, 0.0),  # default margin: unchanged
            (0.5, 2e-3, 1.5e-3),
            (0.5, -1e-4, 0.0),  # negative margin (paper configs): |dmin|
            (0.5, -1e-4, 4e-5),
            (0.5, -1e-4, 6e-5),
            (0.5, 0.0, 0.0),  # zero margin: the floor
            (0.5, 0.0, 1e-6),
        ]
        f, d, pen = (np.array(c, dtype=np.float32) for c in zip(*cases))
        n = len(cases)
        tol, ratio, feasible = (wp.zeros(n, dtype=t) for t in (wp.float32, wp.float32, wp.int32))
        wp.launch(k, dim=n, inputs=[wp.array(f), wp.array(d), wp.array(pen), tol, ratio, feasible])
        tol, ratio, feasible = tol.numpy(), ratio.numpy(), feasible.numpy()
        self.assertTrue(np.isfinite(ratio).all())
        # dmin > 0: bit-identical to ftol * dmin
        self.assertEqual(tol[0], np.float32(0.5) * np.float32(2e-3))
        self.assertEqual(feasible[:2].tolist(), [1, 0])
        # dmin < 0: |dmin|, so zero penetration passes
        self.assertAlmostEqual(float(tol[2]), 5e-5, delta=1e-6 * 5e-5)
        self.assertEqual(feasible[2:5].tolist(), [1, 1, 0])
        self.assertAlmostEqual(float(ratio[3]), 0.8, places=5)
        # dmin = 0: the floor
        self.assertAlmostEqual(float(tol[5]), 0.5 * FEASIBILITY_FLOOR, delta=1e-6 * 0.5 * FEASIBILITY_FLOOR)
        self.assertEqual(feasible[5:].tolist(), [1, 0])
        self.assertAlmostEqual(float(ratio[6]), 2.0, places=5)
        # Host side agrees
        for (fi, di, _), t in zip(cases, tol):
            self.assertAlmostEqual(feasibility_tolerance(fi, di), float(t), delta=1e-7 * float(t))


class TestStationarityTolerance(unittest.TestCase):
    def test_relative_and_absolute_terms(self):
        wp.init()

        @wp.kernel
        def k(
            dm2: wp.array[wp.float32],
            free_mass: wp.array[wp.float32],
            gm2: wp.array[wp.float32],
            rstol2: wp.float32,
            atol2: wp.float32,
            stol2: wp.array[wp.float32],
            converged: wp.array[wp.int32],
            ratio: wp.array[wp.float32],
        ):
            i = wp.tid()
            g = wp.array(ptr=gm2.ptr + wp.uint64(i * 4), shape=(1,), dtype=wp.float32)
            s = wp.array(ptr=stol2.ptr + wp.uint64(i * 4), shape=(1,), dtype=wp.float32)
            s[0] = _stationarity_tol2(dm2[i], free_mass[i], rstol2, atol2)
            # The absolute-test inputs (gnorm2, gtol2) are unused by the stationarity branch
            converged[i] = wp.where(_is_converged(g, wp.float32(0), g, s, 1), 1, 0)
            ratio[i] = _gradient_ratio(g, wp.float32(0), g, s, 1)

        # Free mass 2 kg; RMS residual r and RMS free-flight displacement D give gm2 = 2 r^2, dm2 = 2 D^2
        rstol, atol = 1e-2, 1e-4  # atol = astol * L, in m
        cases = [  # (D, r)
            (1e-1, 0.9e-3),  # relative term (1e-3) dominates: passes
            (1e-1, 1.1e-3),
            (1e-3, 0.9e-4),  # near rest: the absolute term (1e-4) dominates over rstol D = 1e-5
            (1e-3, 1.1e-4),
            (0.0, 0.5e-4),  # no displacement at all: astol alone
        ]
        D, r = (np.array(c, dtype=np.float64) for c in zip(*cases))
        n = len(cases)
        dm2, gm2 = wp.array((2 * D**2).astype(np.float32)), wp.array((2 * r**2).astype(np.float32))
        mass = wp.array(np.full(n, 2.0, dtype=np.float32))
        stol2, converged, ratio = wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.int32), wp.zeros(n, dtype=wp.float32)
        wp.launch(k, dim=n, inputs=[dm2, mass, gm2, rstol**2, atol**2, stol2, converged, ratio])
        self.assertEqual(converged.numpy().tolist(), [1, 0, 1, 0, 1])
        expected = r / np.maximum(rstol * D, atol)
        np.testing.assert_allclose(ratio.numpy(), expected, rtol=1e-5)

    def test_mean_body_diagonal(self):
        # Two bodies: a unit tet (diagonal sqrt 3) and a tet of edge 2 sharing no node with it (2 sqrt 3),
        # plus a node in no element (a collider's contact node), which belongs to no body
        unit = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        X = np.vstack([unit, 10 + 2 * unit, [[100.0, 100.0, 100.0]]])
        E = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertAlmostEqual(mean_body_diagonal(X, E), 1.5 * np.sqrt(3))
        self.assertEqual(mean_body_diagonal(X, np.zeros((0, 4), dtype=int)), 0.0)


if __name__ == "__main__":
    unittest.main()

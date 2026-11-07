"""
Device-side convergence test of the VBD-family solves.

Mirrors ``pbat::sim::algorithm::vbd::CheckConvergence``: stop once ``|g|^2 <= gtol^2``, where ``g`` is
the gradient of the time step's Lagrangian at the current contact multipliers, over free DOFs
(:class:`gpu.gradient.GradientNorm` without the augmented-Lagrangian penalty term, as C++
``ToGradient(..., bForAugmentedLagrangian = false)``). Without contact it is the objective's gradient,
so the test means the same inside a subproblem as between subproblems. The contact terms are weighted by the
decay weights gamma the positions were solved with: between outer iterations the full dual update's new weights
are still pending (``ConstraintSet.gamma_next``, committed after the test), so the test sees the new multipliers
with the subproblem's gamma, which with ``multiplier_step`` 1 is the subproblem's augmented gradient at the end of
its sweeps. C++ tests with the new weights.

The test runs between outer iterations, as in C++, and also every ``check_interval`` sweeps inside a
subproblem. Between outer iterations, the gradient is measured on the contact set the subproblem solved,
after the full dual update and the feasibility filter but before re-detection (``measure_outer_gradient``), since
the augmented-Lagrangian method solves each subproblem for a fixed constraint set (maintainer, 2026-11-06; C++ tests
after re-detection). The step must also be feasible: no contact may penetrate its
margin by more than ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``, i.e.
``max_j max(0, dmin - c_j(x)) <= ftol * max(|dmin|, FEASIBILITY_FLOOR)`` over the
re-detected contacts (the maintainer's choice; C++ has no feasibility test), measured by ``end_outer``, which
decides the stop once both are known. Inside a subproblem only the gradient is tested, and with the stationarity
test (``rstol`` or ``astol`` > 0) it is the subproblem's own residual, its augmented-Lagrangian gradient, which with
``multiplier_step`` 1 the next outer test measures again after the dual update (:meth:`ConvergenceCheck.after_block`):
passing ends the subproblem only, never the solve, and both outer tests follow its finalization. With dynamic problems
the test's tolerance is relative to each node's forces and can pass at the float32 floor (:class:`ConvergenceParams`).
Its outcome stays on the device in flags that conditional graph nodes
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

    Stationarity test, in acceleration units (the gradient g = h^2 f_res over h^2 and the lumped mass): node i's
    residual acceleration r_i = f_res,i / m_i may exceed a relative allowance rstol s_i / m_i, and what's left over,
    mass-weighted RMS over free nodes, must be within the absolute tolerance:

        RMS_M(e) <= astol |a|,   e_i = max(0, |r_i|^2 - (rstol s_i / m_i)^2)^(1/2),
        s_i = |m_i (x_i - x~_i)| / h^2 + |f_el,i| + |f_c,i| + |f_ext,i|,

    s_i being the magnitudes of the terms whose sum is node i's force residual: the inertial term (the net force the
    motion needs; it holds -f_ext for BDF1, so f_ext also counts on its own and a free fall, x = x~, keeps a scale),
    the elastic force, the contact force (normal and friction) and the external load, each net at the node
    (gpu.gradient.Gradient, ``force_scale``). h is the solve's time step (dt / substeps) and |a| the scene's external
    acceleration (:func:`external_acceleration`: the gravity's norm, or :data:`STANDARD_GRAVITY` without a load). In
    the gradient's units: ``sum max(0, |g_i|^2 - (rstol h^2 s_i)^2) / m_i <= stol2 = (astol |a| h^2)^2 sum m_i``.
    The relative part is a componentwise relative residual, |f_res,i| <= rstol sum_k |f_k,i| (the componentwise
    backward error of Oettli and Prager, Numer. Math. 1964, used as a stopping test by Arioli, Duff and Ruiz, SIAM J.
    Matrix Anal. Appl. 1992; Higham, Accuracy and Stability of Numerical Algorithms, ch. 7), with the forces netted
    per node rather than per element, so that internal stresses that balance don't loosen it. Every term is a force,
    so for a fixed physical motion the test doesn't depend on h. The allowances depend on the iterate, so each test
    measures them with the gradient. A normwise form, RMS_M(r) <= rstol RMS_M(s / m), was tried first and rejected:
    contact-loaded light surface nodes dominate its scale (ArmadilloSpaghetti.DragonImpact at 7 s: RMS_M(s / m) =
    1.2e5 m/s^2 from elastic and contact forces of ~1e7 m/s^2 that cancel at the top 1% of nodes, against a median
    node's 105 m/s^2), so it accepted the substeps' initial guesses. astol |a| is the tolerated residual
    acceleration where the forces are small; with 4 rstol^2 + astol^2 < 1, a step that hasn't moved (r = a, s = 2 m a
    for a free node at rest) can't pass. Changed meaning (2026-11-06): the test used to be RMS_M(r) <=
    max(rstol RMS_M(x~ - x^t) / h^2, astol |a|), whose relative term rstol |v^t / h + a| grew as 1/h for a moving
    body (Trampoline.SoftSoft, first 0.3 s, median 1e-2 RMS_M(x~ - x^t) / h^2: 0.97, 1.91 and 9.5 m/s^2 at h = 10, 5
    and 1 ms, the last above astol |a| = 4.9; 1e-2 RMS_M(s / m): 0.31, 0.30, 0.27). rstol keeps its value 1e-2: at
    the scenes' production h the former relative term was below the absolute one on the soft scenes, and the new
    allowances change their early exits by under 3% (Trampoline.SoftSoft, CapsulesIntoBowl); at 5e-2 the floor test
    below stopped DragonImpact earlier than its accuracy warranted. Static problems use the absolute ``gtol``.

    Float32 floor: a position error of one float32 spacing makes the force f_i = |H_i|_F eps32 |x_i| / h^2 (H_i node
    i's Hessian block in the gradient's units, inertia included, from the last sweep; eps32 = 2^-23), the
    acceleration phi_i = f_i / m_i. No float32 iterate reduces the residual below it (stiff bodies near rest, large
    coordinates: the residual of the exactly rounded solution is about 0.15 RMS_M(phi), 2026-11-06). The test also
    passes when both

        RMS_M(e') <= astol |a|,   e'_i = max(0, |r_i|^2 - max(rstol s_i / m_i, kl phi_i)^2)^(1/2),
        RMS_M(r) <= kg RMS_M(phi),

    i.e. the residual beyond each node's relative allowance or floor is within the absolute tolerance, and the
    residual is near the floor overall, kl = ``floor_local_kappa`` 1.5 and kg = ``floor_kappa`` 0.2. The first alone
    stops smooth (bending) modes whose residual is coherent but below each node's floor; the second alone stops soft
    bodies early when stiff ones set the floor. The floor ratio RMS_M(phi) / (astol |a|) is recorded in the solve
    status. GPU Newton has no floor (TODO in gpu/newton/solver.py).

    Inside a subproblem the test measures the subproblem's own residual, its augmented-Lagrangian gradient (see
    :meth:`ConvergenceCheck.after_block`), and passing ends the subproblem only; between subproblems, the Lagrangian
    gradient at the new multipliers, and passing it (and feasibility) ends the solve.

    ``early_exit`` (default on) lets passing the tests end a solve. Off, solves run their full budgets (fixed
    compute) while the tests still run and the status still records their ratios to the tolerances, unlike
    zero tolerances, whose ratios are infinite.
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
        "Relative stationarity tolerance, relative to the forces in each node's balance: node i's residual "
        "acceleration may exceed rstol s_i / m_i, s_i = |m_i (x_i - x~_i)| / h^2 + |f_el,i| + |f_c,i| + |f_ext,i| the "
        "magnitudes of its inertial, elastic, contact and external forces, and the mass-weighted RMS of what's left "
        "over must be at most astol * |a|. Every term is a force, so it doesn't depend on the time step. With rstol = "
        "astol = 0, the solver's absolute gtol applies instead. Changed meaning (2026-11-06): the test used to pass at "
        "RMS(M^-1 f_res) <= max(rstol * RMS(x~ - x^t) / h^2, astol * |a|), rstol |v / h + a|, default 1e-2",
    )
    astol = DocField(
        0.5,
        "Absolute stationarity tolerance, as a fraction of the scene's external acceleration |a| (its gravity's "
        "norm, or 9.81 m/s^2 without a load): the mass-weighted RMS of the residual acceleration beyond each node's "
        "relative allowance (see rstol) must be at most astol * |a|. Time-step and scene-size invariant; with "
        "4 rstol^2 + astol^2 < 1 a step that hasn't moved can't pass. "
        "Changed meaning (2026-10-20): astol used to be a fraction of the scene's size L (astol * L), so an old "
        "file's astol = 1e-4 is now extremely strict",
    )
    floor_kappa = DocField(
        0.2,
        "Float32 floor, global factor kg (0: no floor): the gradient test also passes when the residual RMS(M^-1 "
        "f_res) is at most kg times the float32 floor RMS(M^-1 f) and the residual beyond each node's larger "
        "allowance, max(rstol s_i, floor_local_kappa f_i), is within astol * |a| (RMS). f_i = |H_i|_F eps32 |x_i| / "
        "h^2 is the force a position error of one float32 spacing makes through node i's Hessian block (the last "
        "sweep's; VBD family only). The floor ratio is recorded in the solve status either way",
    )
    floor_local_kappa = DocField(
        1.5,
        "Float32 floor, per-node factor kl (0: no floor): the floor test allows each node kl times its float32 floor "
        "force f_i (see floor_kappa)",
    )
    early_exit = DocField(
        True,
        "End a solve (or a subproblem) once it passes its tests. Off, every solve runs its full budget of outer "
        "iterations and sweeps, at fixed compute, and still tests and records its gradient and feasibility ratios "
        "against the usual tolerances",
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
    floor_ratio: wp.float32  # float32 floor / tolerance, sqrt(sum f_i^2/m_i / stol2), of the same test (0: none)


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
    sums: wp.array[wp.float32],
    stol2: wp.array[wp.float32],
    relative: wp.int32,
) -> wp.float32:
    """The gradient test's measure over its tolerance (see :func:`_is_converged`): the residual beyond the relative
    allowances over the absolute tolerance, sqrt(sum max(0, |g_i|^2 - (rstol s_i)^2) / m_i / stol2), or |g| / gtol. At
    most 1 passes; above 1, the stationarity test can still pass at the float32 floor."""
    num = gnorm2[0]
    den = gtol2
    if relative != 0:
        num = sums[1]  # SUM_RELATIVE
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


STANDARD_GRAVITY = 9.81
"""External acceleration (m/s^2) the absolute stationarity tolerance uses for a scene without a load."""


def external_acceleration(fext: np.ndarray, m: np.ndarray, dmask: np.ndarray) -> float:
    """The scene's external acceleration |a| (m/s^2), the scale of the absolute stationarity tolerance
    ``astol * |a| * h^2``: the mass-weighted RMS of f_i / m_i over free nodes, sqrt(sum |f_i|^2 / m_i /
    sum m_i), which is the gravity's norm for a uniform gravity; :data:`STANDARD_GRAVITY` without a load.

    Args:
        fext (np.ndarray): (N, 3) external forces.
        m (np.ndarray): (N,) lumped masses.
        dmask (np.ndarray): (N,) Dirichlet mask (nonzero: fixed node, excluded).

    Returns:
        float: |a|.
    """
    m = np.asarray(m, dtype=np.float64).reshape(-1)
    free = (np.asarray(dmask).reshape(-1) == 0) & (m > 0)
    f = np.asarray(fext, dtype=np.float64).reshape(-1, 3)[free]
    mass = m[free].sum()
    a = np.sqrt(np.sum(f * f, axis=1) @ (1.0 / m[free]) / mass) if mass > 0 else 0.0
    return float(a) if a > 0 else STANDARD_GRAVITY


def absolute_stationarity_tolerance(astol: float, acceleration: float, h: float) -> float:
    """The absolute term of the stationarity test, ``astol * |a| * h^2`` (m), for the external
    acceleration ``acceleration`` (m/s^2) and the time step ``h`` (s)."""
    # The float32 floor (2026-11-06): the backward-error bound of the force sums, eps32 * sum_i (|m_i (x_i - x~_i)| +
    # |h^2 f_el,i| + |h^2 f_c,i|), proposed here before, is 1000-10000x below the attainable residual. The floor is
    # the quantization of x, f_i = |H_i|_F eps32 |x_i| through the vertex Hessian (the residual of the exactly rounded
    # solution is ~0.15 RMS_M(M^-1 f) on BendingBeam.100MPa and DragonImpact, also translated by 32 m), so the test
    # passes at it (ConvergenceParams.floor_kappa, floor_local_kappa; _is_converged). The force sum became the
    # relative term's scale instead (ConvergenceParams). Evidence: investigate/float32-residual-floor (8eadc920).
    # TODO: Solve for the increment dx = x - x^t instead of x, F = F(x^t) + grad dx: |dx| ~ 1 mm instead of |x| ~ 1 m
    # would lower the position quantization, and with it the floor, ~1000x. Why: the floor, not the tolerance, ends
    # stiff scenes' solves (BendingBeam.100MPa, DragonImpact at 7 s: floor 18-100x the tolerance). Solver-wide change
    # (elastic, contact and inertia kernels; 9 floats per element to cache D_s(x^t)); not tried.
    # TODO: A stall exit for the in-subproblem checks: stop when, at one check, the residual stopped falling
    # (r_k >= (1 - X) r_prev) AND no free vertex moved since the previous check by more than
    # max(rxtol * P, kappa eps32 |x_i|), P = max_i |x_i - x_i^start| the subproblem's progress. Why: on DragonImpact
    # the residual plateaus after ~40 of 100 sweeps per substep. Calibrated (2026-10-24): its armadillo and dragon
    # nodes still converge then (creep, cos of consecutive windows +0.85...+1), so it saves only 20-40% at rxtol
    # 2e-2...5e-2; a movement-only test stalls stiff bodies, which move (1 + (h omega)^2) less per residual (the
    # 1e8 beam); with the residual condition, the beam's single-check residual noise after ~2000 sweeps still
    # stalls it at 1.5x its 4000-sweep residual. Not pursued (maintainer, 2026-10-24): no scene needs it, and it
    # risks the timings and other scenes. Notes: python/tools/sim/docs/solver-accuracy-notes.md, entry 7.
    return float(astol) * float(acceleration) * float(h) ** 2


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


# The stationarity test's sums over free nodes (ConvergenceCheck.sums), each in the gradient's units squared per mass
SUM_RESIDUAL = wp.constant(0)  # sum |g_i|^2 / m_i
SUM_RELATIVE = wp.constant(1)  # sum max(0, |g_i|^2 - (rstol s_i)^2) / m_i: beyond the relative allowances
SUM_FLOOR = wp.constant(2)  # sum f_i^2 / m_i, f_i = |H_i|_F EPS32 |x_i| the float32 floor (0 without Hessian norms)
SUM_EXCESS = wp.constant(3)  # sum max(0, |g_i|^2 - max(rstol s_i, kl f_i)^2) / m_i: beyond the floors too
N_SUMS = 4

EPS32 = 1.1920928955078125e-07
"""float32 machine epsilon 2^-23: the spacing of float32 numbers at x is between 0.5 and 1 times EPS32 |x|."""


@wp.func
def _is_converged(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    sums: wp.array[wp.float32],
    stol2: wp.array[wp.float32],
    relative: wp.int32,
    floor_kappa2: wp.float32,
) -> bool:
    """Stationarity test if enabled (``relative``), else ``|g|^2 <= gtol^2``. The stationarity test passes when the
    residual beyond each vertex's relative allowance rstol s_i is within the absolute tolerance, sum max(0, |g_i|^2
    - (rstol s_i)^2) / m_i <= stol2, or at the float32 floor: the residual beyond each vertex's larger allowance
    max(rstol s_i, kl f_i) is within it and the residual is at most kg times the floor, sum |g_i|^2 / m_i <= kg^2
    sum f_i^2 / m_i (see :class:`ConvergenceParams`)."""
    if relative != 0:
        tol2 = stol2[0]
        at_floor = sums[SUM_EXCESS] <= tol2 and sums[SUM_RESIDUAL] <= floor_kappa2 * sums[SUM_FLOOR]
        return sums[SUM_RELATIVE] <= tol2 or at_floor
    return gnorm2[0] <= gtol2


@wp.func
def _floor_ratio(sums: wp.array[wp.float32], stol2: wp.array[wp.float32]) -> wp.float32:
    """The float32 floor over the absolute tolerance, sqrt(sum f_i^2 / m_i / stol2) (0 without a floor)."""
    num = sums[SUM_FLOOR]
    ratio = wp.float32(0)
    if stol2[0] > wp.float32(0):
        ratio = wp.sqrt(num / stol2[0])
    elif num > wp.float32(0):
        ratio = wp.float32(wp.inf)
    return ratio


@wp.kernel
def _stationarity_terms(
    g: wp.array[wp.vec3f],
    scale: wp.array[wp.float32],
    hnorm: wp.array[wp.float32],
    use_floor: wp.int32,
    rstol: wp.float32,
    local_kappa: wp.float32,
    fem: FemElastoDynamicsData,
    terms: wp.array2d[wp.float32],
):
    """Free node i's terms of the stationarity sums (``SUM_*``). Its relative allowance is rstol s_i, s_i the sum of
    the magnitudes of the terms of its force balance (``scale``, gpu.gradient.Gradient's force scale), and its float32
    floor f_i = |H_i|_F EPS32 |x_i|: the force (in the gradient's units) that a position error of about one float32
    spacing at x_i makes through the vertex's own Hessian block |H_i|_F (``hnorm``, the last sweep's, inertia
    included)."""
    i = wp.tid()
    for k in range(N_SUMS):
        terms[k, i] = wp.float32(0)
    if is_dirichlet_node(fem.dmask, i):
        return
    mi = fem.m[i]
    g2 = wp.dot(g[i], g[i])
    a = rstol * scale[i]
    f = wp.float32(0)
    if use_floor != 0:
        f = hnorm[i] * wp.float32(EPS32) * wp.length(fem.x[i])
    af = wp.max(a, local_kappa * f)
    terms[SUM_RESIDUAL, i] = g2 / mi
    terms[SUM_RELATIVE, i] = wp.max(g2 - a * a, wp.float32(0)) / mi
    terms[SUM_FLOOR, i] = f * f / mi
    terms[SUM_EXCESS, i] = wp.max(g2 - af * af, wp.float32(0)) / mi


@wp.kernel
def _free_mass_terms(fem: FemElastoDynamicsData, masses: wp.array[wp.float32]):
    i = wp.tid()
    masses[i] = wp.float32(0)
    if not is_dirichlet_node(fem.dmask, i):
        masses[i] = fem.m[i]


@wp.kernel
def _stationarity_tolerance(free_mass: wp.array[wp.float32], atol2: wp.float32, stol2: wp.array[wp.float32]):
    """Squared absolute tolerance on the sums, stol2 = atol^2 sum m_i, atol = astol |a| h^2 (m)."""
    stol2[0] = atol2 * free_mass[0]


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
    sums: wp.array[wp.float32],
    stol2: wp.array[wp.float32],
    relative: wp.int32,
    early_exit: wp.int32,
    floor_kappa2: wp.float32,
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
    converged = _is_converged(gnorm2, gtol2, sums, stol2, relative, floor_kappa2)
    feasible = _is_feasible(penetration, ftol, contact.dmin)
    done = converged and feasible and early_exit != 0
    keep[0] = wp.where(n_max_iters > 0 and not done, 1, 0)
    # A new solve: advance the counter and start its status row
    n = solves[0]
    solves[0] = n + 1
    row = SolveStatus()
    row.solve = n
    row.outer = 0
    row.sweeps = 0
    row.reason = _solve_end(converged, feasible, n_max_iters > 0)
    row.gradient_ratio = _gradient_ratio(gnorm2, gtol2, sums, stol2, relative)
    row.feasibility_ratio = _feasibility_ratio(penetration, ftol, contact.dmin)
    row.floor_ratio = wp.where(relative != 0, _floor_ratio(sums, stol2), wp.float32(0))
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
    sums: wp.array[wp.float32],
    stol2: wp.array[wp.float32],
    relative: wp.int32,
    early_exit: wp.int32,
    floor_kappa2: wp.float32,
    n_blocks: wp.int32,
    blocks: wp.array[wp.int32],
    inner_keep: wp.array[wp.int32],
    subproblem_keep: wp.array[wp.int32],
):
    b = blocks[0] + 1
    blocks[0] = b
    stop = _is_converged(gnorm2, gtol2, sums, stol2, relative, floor_kappa2) and early_exit != 0
    subproblem_keep[0] = wp.where(stop, 0, 1)
    inner_keep[0] = wp.where(not stop and b < n_blocks, 1, 0)


@wp.kernel
def _end_outer_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    sums: wp.array[wp.float32],
    stol2: wp.array[wp.float32],
    relative: wp.int32,
    early_exit: wp.int32,
    floor_kappa2: wp.float32,
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
    converged = _is_converged(gnorm2, gtol2, sums, stol2, relative, floor_kappa2)
    feasible = _is_feasible(penetration, ftol, contact.dmin)
    done = converged and feasible and early_exit != 0
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
    row.gradient_ratio = _gradient_ratio(gnorm2, gtol2, sums, stol2, relative)
    row.feasibility_ratio = _feasibility_ratio(penetration, ftol, contact.dmin)
    row.floor_ratio = wp.where(relative != 0, _floor_ratio(sums, stol2), wp.float32(0))
    status[idx] = row


class ConvergenceCheck:
    """Outer-iteration counter and loop flags of a solve, on the device.

    Flags (each a (1,) int32 array, read by conditional graph nodes):

    - ``keep``: the step goes on (not both optimal and feasible, or ``early_exit`` off, and outer iterations
      left).
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
        acceleration: float | None = None,
    ):
        """
        Args:
            fem (FemElastoDynamics): The problem whose gradient is tested.
            contact (ContactDynamics): Its contact model (multipliers for the gradient, contact sets
                for the feasibility test).
            params (ConvergenceParams | None): Check interval and feasibility tolerance; defaults if None.
            status_capacity (int): Rows of the status ring: at least the solves between two
                :meth:`read_status` calls (``SimulationState`` reads every step: its substeps).
            acceleration (float | None): The scene's external acceleration |a| (m/s^2) that scales
                ``astol`` (:func:`external_acceleration`), computed from ``fem``'s loads if None.
        """
        self.params = params or ConvergenceParams()
        if acceleration is None:
            acceleration = external_acceleration(
                fem.data.fext.numpy(), fem.data.m.numpy(), fem.data.dmask.numpy()
            )
        self.acceleration = float(acceleration)
        capacity = max(1, int(status_capacity))
        self.status = wp.zeros(capacity, dtype=SolveStatus)
        self.solves = wp.zeros(1, dtype=wp.int32)
        self._status_host = wp.zeros(capacity, dtype=SolveStatus, device="cpu", pinned=True)
        self._next_solve = 0
        self._fem = fem
        self._contact = contact
        # The outer test's Lagrangian gradient, and the subproblem's augmented one for the tests inside it, each with
        # its per-node force scale (the stationarity test's relative term)
        self.gradient_norm = GradientNorm(fem, contact, augmented_lagrangian=False, force_scale=True)
        self.augmented_norm = GradientNorm(fem, contact, augmented_lagrangian=True, force_scale=True)
        # Stationarity test (rstol or astol > 0), sums over free nodes (SUM_*) against stol2 = (astol |a| h^2)^2 sum m_i
        n_nodes = fem.data.x.shape[0]
        self._terms = wp.zeros((N_SUMS, n_nodes), dtype=wp.float32)
        self._masses = wp.zeros(n_nodes, dtype=wp.float32)
        self.sums = wp.zeros(N_SUMS, dtype=wp.float32)
        self.residual_m2 = self.sums[0:1]  # sum |g_i|^2 / m_i (a view of sums)
        self.stationarity_tol2 = wp.zeros(1, dtype=wp.float32)  # stol2, see ConvergenceParams
        self.free_mass = wp.zeros(1, dtype=wp.float32)
        # The solver's (N,) vertex Hessian norms |H_i|_F for the float32 floor (set_hessian_norms); None: no floor
        self._hessian_norms = None
        self._no_hessian_norms = wp.zeros(1, dtype=wp.float32)
        terms, sums = cp.asarray(self._terms), cp.asarray(self.sums)
        self._reduce_sums = [
            Reduce(d_in=terms[k], d_out=sums[k : k + 1], num_items=n_nodes, op=cuda.compute.OpKind.PLUS)
            for k in range(N_SUMS)
        ]
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

    @property
    def absolute_tolerance(self) -> float:
        """The stationarity test's absolute term ``astol * |a| * h^2`` (m), h being the time step of
        ``fem``'s integrator (dt / substeps)."""
        return absolute_stationarity_tolerance(self.params.astol, self.acceleration, self._fem.bdf.h)

    def set_hessian_norms(self, hnorm: wp.array | None) -> None:
        """Use the solver's (N,) vertex Hessian norms |H_i|_F (``Params.data.Hnk``, written by every sweep) for the
        float32 floor; None: no floor. Host-side: recorded kernels read the array, so it must stay the same while a
        captured graph replays."""
        self._hessian_norms = hnorm

    def _test(self) -> None:
        """Measure the Lagrangian gradient at the current multipliers (the outer test's), and with the stationarity
        test its sums and tolerance."""
        stream = wp.get_stream()
        self.gradient_norm.compute(stream)
        if self.relative:
            self._measure_residual(self.gradient_norm)

    def _test_subproblem(self) -> None:
        """Measure the test inside a subproblem. With the stationarity test, that is the subproblem's own residual,
        the augmented-Lagrangian gradient the sweeps minimize: multipliers max(0, lambda_k - sigma_n c(x)) (one-sided
        as the vertex solves with ``vbd_slack_at_evaluation``, otherwise with the stored slack), friction capped at
        mu_f times that normal force, re-evaluated at this test (the Coulomb limits are unfrozen first; the
        augmented gradient freezes them, GPU Newton's rule), at the subproblem's decay weights. With
        ``multiplier_step`` 1 it is the next outer test's quantity before the dual update's Coulomb projection and
        the feasibility filter. With the absolute ``gtol``, the Lagrangian gradient, as between subproblems (C++)."""
        if not self.relative:
            self._test()
            return
        a = self.augmented_norm
        a.one_sided = bool(int(self._contact.data.vbd_slack_at_evaluation))
        self._contact.reset_friction_limits()
        a.compute_gradient(wp.get_stream())
        self._measure_residual(a)

    def _measure_residual(self, gradient: GradientNorm) -> None:
        """The stationarity sums of ``gradient`` (``g`` and its force ``scale``)."""
        stream = wp.get_stream()
        hnorm = self._hessian_norms
        wp.launch(
            _stationarity_terms,
            dim=self._terms.shape[1],
            inputs=[
                gradient.g,
                gradient.scale,
                hnorm if hnorm is not None else self._no_hessian_norms,
                int(hnorm is not None),
                float(self.params.rstol),
                float(self.params.floor_local_kappa),
                self._fem.data,
                self._terms,
            ],
        )
        for reduce in self._reduce_sums:
            reduce(stream)

    def measure_scale(self) -> None:
        """Measure the time step's absolute tolerance on the stationarity sums, ``stol2 = (astol |a| h^2)^2 sum m_i``
        over free nodes (the relative allowances come from each test's forces). No-op unless :attr:`relative`.
        Capturable."""
        if self.relative:
            wp.launch(_free_mass_terms, dim=self._masses.shape[0], inputs=[self._fem.data, self._masses])
            self._reduce_free_mass(wp.get_stream())
            wp.launch(
                _stationarity_tolerance,
                dim=1,
                inputs=[self.free_mass, self.absolute_tolerance**2, self.stationarity_tol2],
            )

    def _tolerance_inputs(self, gtol: float) -> list:
        return [
            self.gradient_norm.gnorm2,
            float(gtol) ** 2,
            self.sums,
            self.stationarity_tol2,
            int(self.relative),
            int(bool(self.params.early_exit)),
            float(self.params.floor_kappa) ** 2,
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
        """Test the initial positions: ``k = 0``, ``keep`` = not (optimal and feasible), or always with
        ``early_exit`` off.

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
        """Test after a sweep block: a converged subproblem stops, and so do its remaining sweeps (unless
        ``early_exit`` is off).

        Only the subproblem's residual is tested here (:meth:`_test_subproblem`): passing ends the subproblem, never
        the solve, which is decided between outer iterations only (:meth:`end_outer`), after the full dual update,
        the filter, the solved set's gradient, the re-detection and feasibility.

        Args:
            gtol (float): Gradient-norm tolerance (absolute).
            n_blocks (int): Number of blocks in the subproblem.
        """
        self._test_subproblem()
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

    def measure_outer_gradient(self) -> None:
        """The gradient part of the test after an outer iteration, on the contact set its subproblem solved:
        call it after the full dual update and the feasibility filter, before re-detection and the
        constraint-set update, and before the pending decay weights are committed
        (``ContactDynamics.commit_decay``), so it measures the new multipliers with the decay weights the
        subproblem solved with. It stores scalars only (``gradient_norm.gnorm2``, ``residual_m2``), which
        :meth:`end_outer` reads after the re-detection. Capturable."""
        self._test()

    def end_outer(
        self, gtol: float, n_max_iters: int, layout: SweepLayout | None = None
    ) -> None:
        """Test after an outer iteration (C++'s check before the next one, plus feasibility):
        ``k += 1``, update ``keep`` and the solve's status row. Its gradient is the one
        :meth:`measure_outer_gradient` measured before re-detection; feasibility is measured here, on the
        current (re-detected) contacts. Call it before the dual update's pending decay weights are committed
        (``ContactDynamics.commit_decay``).

        Args:
            gtol (float): Gradient-norm tolerance (absolute).
            n_max_iters (int): Maximum outer iterations.
            layout (SweepLayout | None): The subproblem's sweep layout, to count the sweeps it used;
                None counts none.
        """
        layout = layout or SweepLayout(0, 0, 0, 0)
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
        sums, tol2 = self.sums.numpy(), float(self.stationarity_tol2.numpy()[0])
        kg2 = float(self.params.floor_kappa) ** 2
        at_floor = float(sums[3]) <= tol2 and float(sums[0]) <= kg2 * float(sums[2])
        return float(sums[1]) <= tol2 or at_floor

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
        return self.test_feasibility() and gradient_passes

    def test_feasibility(self) -> bool:
        """Host-side feasibility test of the current positions on the current contacts: the largest
        penetration into the contact margin at most ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``. Sets what
        :meth:`feasibility_ratio` reads. Synchronizes, so not capturable.

        Returns:
            bool: Whether it passes.
        """
        self._measure_penetration()
        tol = feasibility_tolerance(self.params.ftol, self._contact.data.dmin)
        return float(self.penetration.numpy()[0]) <= tol

    def gradient_ratio(self) -> float:
        """Host-side stationarity measure of the last test over its tolerance, the residual beyond the relative
        allowances ``sqrt(sum max(0, |g_i|^2 - (rstol s_i)^2) / m_i / stol2)``, as the status ring's
        ``gradient_ratio``: at most 1 passes. Synchronizes.

        Returns:
            float: The ratio (from the last :meth:`test_gradient`, :meth:`test_outer` or
            :meth:`relative_test`).
        """
        den = float(self.stationarity_tol2.numpy()[0])
        return float(np.sqrt(ratio_to_tolerance(float(self.sums.numpy()[1]), den)))

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

    def relative_test(self, gradient: GradientNorm) -> bool:
        """Host-side stationarity test of a given gradient against its tolerance, which comes from the gradient's
        own force scale (see :class:`ConvergenceParams`). Synchronizes, so not capturable.

        Newton uses it for its augmented-Lagrangian subproblem gradient.

        Args:
            gradient (GradientNorm): A gradient computed with ``force_scale``.

        Returns:
            bool: Whether it passes (False if the relative test is off).
        """
        if not self.relative:
            return False
        self._measure_residual(gradient)
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
    def test_decision(self):
        wp.init()

        @wp.kernel
        def k(
            sums: wp.array2d[wp.float32],
            stol2: wp.array[wp.float32],
            floor_kappa2: wp.float32,
            converged: wp.array[wp.int32],
            ratio: wp.array[wp.float32],
            floor: wp.array[wp.float32],
        ):
            i = wp.tid()
            si = wp.array(ptr=sums.ptr + wp.uint64(i * N_SUMS * 4), shape=(N_SUMS,), dtype=wp.float32)
            # The absolute-test inputs (gnorm2, gtol2) are unused by the stationarity branch
            converged[i] = wp.where(_is_converged(si, wp.float32(0), si, stol2, 1, floor_kappa2), 1, 0)
            ratio[i] = _gradient_ratio(si, wp.float32(0), si, stol2, 1)
            floor[i] = _floor_ratio(si, stol2)

        # Free mass 2 kg, so stol2 = 2 atol^2; RMS values x (residual r, residual beyond the relative allowances R,
        # float32 floor F, residual beyond the larger of the relative and floor allowances X) give each sum as 2 x^2
        atol, kg = 1e-4, 0.2  # atol = astol |a| h^2, in m
        cases = [  # (r, R, F, X)
            (5e-3, 0.9e-4, 0.0, 0.9e-4),  # within the relative allowances but for 0.9 atol: passes
            (5e-3, 1.1e-4, 0.0, 1.1e-4),
            (0.9e-4, 0.9e-4, 0.0, 0.9e-4),  # no allowance (no force): the absolute tolerance alone
            (3e-4, 1.1e-4, 2e-3, 0.9e-4),  # at the float32 floor: r <= kg F = 4e-4 and beyond the floors within: passes
            (3e-4, 1.1e-4, 2e-3, 1.1e-4),  # beyond the floors more than the tolerance
            (5e-4, 1.1e-4, 2e-3, 0.9e-4),  # r above kg F
        ]
        r, R, F, X = (np.array(c, dtype=np.float64) for c in zip(*cases))
        n = len(cases)
        sums = np.zeros((n, N_SUMS), dtype=np.float32)
        sums[:, 0], sums[:, 1], sums[:, 2], sums[:, 3] = 2 * r**2, 2 * R**2, 2 * F**2, 2 * X**2
        stol2 = wp.array(np.array([2 * atol**2], dtype=np.float32))
        converged, ratio, floor = wp.zeros(n, dtype=wp.int32), wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.float32)
        wp.launch(k, dim=n, inputs=[wp.array(sums), stol2, kg**2, converged, ratio, floor])
        self.assertEqual(converged.numpy().tolist(), [1, 0, 1, 1, 0, 0])
        np.testing.assert_allclose(ratio.numpy(), R / atol, rtol=1e-5)
        np.testing.assert_allclose(floor.numpy(), F / atol, rtol=1e-5)
        # Without a floor factor, the floor case fails like any ratio above 1
        wp.launch(k, dim=n, inputs=[wp.array(sums), stol2, 0.0, converged, ratio, floor])
        self.assertEqual(converged.numpy().tolist(), [1, 0, 1, 0, 0, 0])

    def test_external_acceleration(self):
        # Uniform gravity on unequal masses: its norm; fixed and massless nodes don't count
        m = np.array([1.0, 3.0, 2.0, 0.0])
        fext = m[:, None] * np.array([0.0, 0.0, -9.81])
        fext[2] = [100.0, 0.0, 0.0]  # a fixed node's load
        self.assertAlmostEqual(external_acceleration(fext, m, np.array([0, 0, 1, 0])), 9.81, places=12)
        # No load: standard gravity
        self.assertEqual(external_acceleration(np.zeros((4, 3)), m, np.zeros(4)), STANDARD_GRAVITY)

    def test_absolute_tolerance_scales_with_h2(self):
        astol = ConvergenceParams().astol
        # The default matches the former astol * L = 1e-4 * 5.66 m of the trampolines at h = 1e-2
        self.assertAlmostEqual(absolute_stationarity_tolerance(astol, 9.81, 1e-2), 0.5 * 9.81e-4)
        tol = [absolute_stationarity_tolerance(astol, 9.81, h) for h in (1e-2, 2e-3)]
        self.assertAlmostEqual(tol[0] / tol[1], 25.0)
        # Below one step's gravity sag h^2 |a| at any h, so a step that hasn't moved can't pass
        for h in (1e-2, 2e-3, 1e-4):
            self.assertLess(absolute_stationarity_tolerance(astol, 9.81, h), 9.81 * h * h)


def _two_resting_cubes(solver_name: str = "VBD"):
    """A test scene: a free cube resting on a fixed one, settled into contact with one outer iteration per
    step, early exit off and rstol 0. Returns the ``SimulationState``."""
    from pbatoolkit import pbat, pypbat
    from ...state import SimulationState, SolverType, load_params

    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    C = np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [6, 7, 5, 3], [0, 5, 3, 6]])
    X = np.vstack([V, V + [0.21, 0.13, 1.002]]).astype(np.float64)
    fem_cpu = pbat.sim.dynamics.FemElastoDynamics(X.T, np.vstack([C, C + 8]).T)
    fem_cpu.set_mass_matrix(1e3)
    mu, lam = pypbat.fem.lame_coefficients(1e6, 0.45)
    fem_cpu.set_elastic_energy(mu, lam)  # pyright: ignore[reportArgumentType]
    fem_cpu.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
    fem_cpu.set_time_integration_scheme(dt=1e-2, s=1)
    fem_cpu.constrain(np.repeat([1, 0], 8))
    state = SimulationState(fem_cpu, load_params(fem_cpu))
    state.solver = SolverType[solver_name]
    state.convergence_params.early_exit = False
    # No relative allowances, so that the recorded gradient ratios are the residual itself over the absolute tolerance
    state.convergence_params.rstol = 0.0
    state.reset()
    state.params[state.solver].data.n_max_iters = 1
    for _ in range(5):  # settle into contact
        state.step()
    return state


def _recorded_gradient_ratio(check: "ConvergenceCheck") -> float:
    """The last solve's recorded gradient ratio (synchronizes)."""
    check.enqueue_status_readback()
    wp.synchronize()
    rows, _ = check.read_status()
    return float(rows[-1]["gradient_ratio"])


class TestOuterGradientBeforeRedetection(unittest.TestCase):
    """The outer test's gradient is measured on the contact set the subproblem solved: after the full dual update
    and the feasibility filter, before re-detection and the constraint-set update, which re-evaluate the contacts
    at the new positions (features, bases, newly detected pairs with lambda = 0). Two cubes resting in contact, one
    outer iteration per step, the measured step run eagerly; an independent check measures the gradient ratio
    right before the re-detection and after the constraint-set update (measured: 2e-4 before, 5e-3 after)."""

    def test_status_is_solved_set_gradient(self):
        for solver_name in ("VBD", "AAAVBD"):
            with self.subTest(solver=solver_name):
                state = _two_resting_cubes(solver_name)
                contact, check, cd = state.contact, state.convergence, state.detector
                probe = ConvergenceCheck(state.fem, contact, check.params, acceleration=check.acceleration)
                measured = {}
                detect_contacts, commit_decay = cd.detect_contacts, contact.commit_decay

                def ratio() -> float:
                    return probe.gradient_ratio()

                def hooked_detect_contacts(*args, **kwargs):
                    if kwargs.get("from_xt"):  # the time step's first detection
                        result = detect_contacts(*args, **kwargs)
                        probe.measure_scale()
                        return result
                    probe._test()  # the subproblem's contact set, after the dual update and the filter
                    measured["solved"] = ratio()
                    return detect_contacts(*args, **kwargs)

                def hooked_commit_decay():
                    probe._test()  # the re-detected contact set, with the same pending decay weights
                    measured["redetected"] = ratio()
                    commit_decay()

                cd.detect_contacts, contact.commit_decay = hooked_detect_contacts, hooked_commit_decay
                state._advance()  # eager, so the hooks run
                status = _recorded_gradient_ratio(check)
                # Re-detection changes the measure a lot, and the status records the solved set's
                self.assertGreater(measured["redetected"], 5.0 * measured["solved"])
                np.testing.assert_allclose(status, measured["solved"], rtol=1e-4)


class TestOuterTestDecayWeights(unittest.TestCase):
    """The outer test after a full dual update measures the positions with the decay weights gamma_k their
    subproblem solved with and the new multipliers lambda_{k+1}: with ``multiplier_step`` 1 that is the
    subproblem's augmented-Lagrangian gradient at the end of its sweeps (slack refreshed there), also when the
    update resets softened contacts to gamma = 1, which the next subproblem then uses. The recorded test follows
    the feasibility filter (which leaves resting cubes in place) and precedes the re-detection. Two cubes resting in
    contact, VBD with one outer iteration per step, the measured step run eagerly."""

    def test_reset_contacts(self):
        state = _two_resting_cubes("VBD")
        self.assertEqual(float(state.contact.params.multiplier_step), 1.0)
        contact, check = state.contact, state.convergence
        sets = (contact.cvv, contact.cve, contact.cvf, contact.cee)
        for cs in sets:
            cs.gamma.current.fill_(0.5)

        augmented = GradientNorm(state.fem, contact, augmented_lagrangian=True, force_scale=True)
        measured = {}
        update_dual, commit_decay = contact.update_dual, contact.commit_decay

        def ratio() -> float:
            return check.gradient_ratio()

        def hooked_update_dual(x, xt, **kwargs):
            full = kwargs.get("request_decay_update", True)
            if full:
                # The subproblem's augmented gradient at the end of its sweeps, with the slack refreshed there and
                # the Coulomb limits unset (the dual update projects onto the new multiplier's cone)
                update_dual(x, xt, request_decay_update=False, request_lagrange_multiplier_update=False)
                contact.reset_friction_limits()
                augmented.compute(wp.get_stream())
                check._measure_residual(augmented)
                measured["augmented"] = ratio()
                contact.reset_friction_limits()
            update_dual(x, xt, **kwargs)
            if full:  # The outer test's quantity, before the filter and the re-detection
                check._test()
                measured["post_update"] = ratio()

        def hooked_commit_decay():
            measured["resets"] = sum(
                int(np.sum((cs.gamma.current.numpy() == 0.5) & (cs.gamma_next.current.numpy() == 1.0))) for cs in sets
            )
            commit_decay()
            # The former outer test's quantity: the new multipliers with the new weights
            check._test()
            measured["committed"] = ratio()

        contact.update_dual, contact.commit_decay = hooked_update_dual, hooked_commit_decay
        state._advance()  # eager, so the hooks run
        status = _recorded_gradient_ratio(check)
        self.assertGreater(measured["resets"], 0)
        np.testing.assert_allclose(measured["post_update"], measured["augmented"], rtol=1e-5)
        # The positions never felt the reset contacts' full force, which the former test charged them with
        # (measured: committed 1.9, augmented 2.3e-4)
        self.assertGreater(measured["committed"], 100.0 * measured["augmented"])
        np.testing.assert_allclose(status, measured["augmented"], rtol=1e-3)
        # The next subproblem solves with the reset weights
        self.assertTrue(all(np.all(cs.gamma.current.numpy() == cs.gamma_next.current.numpy()) for cs in sets))


class TestForceRelativeTolerance(unittest.TestCase):
    """The relative term is relative to the forces in the balance, so in acceleration units it doesn't depend on the
    time step: a deformed free cube moving at constant velocity v (x = x^t + h v, so x - x~ = -h^2 a for BDF1), no
    contact, at h and h/2. Its force scale per mass, RMS_M(M^-1 s) / h^2, is the same at both; the former term,
    RMS_M(x~ - x^t) / h^2 = RMS |v / h + a|, doubles with v dominating."""

    @staticmethod
    def _tolerances(dt: float) -> tuple[float, float]:
        from pbatoolkit import pbat, pypbat
        from ...state import SimulationState, load_params

        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        C = np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [6, 7, 5, 3], [0, 5, 3, 6]])
        X = np.vstack([V, V + [0.0, 0.0, 3.0]]).astype(np.float64)  # the free cube well above the fixed one
        fem_cpu = pbat.sim.dynamics.FemElastoDynamics(X.T, np.vstack([C, C + 8]).T)
        fem_cpu.set_mass_matrix(1e3)
        mu, lam = pypbat.fem.lame_coefficients(1e6, 0.45)
        fem_cpu.set_elastic_energy(mu, lam)  # pyright: ignore[reportArgumentType]
        fem_cpu.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
        fem_cpu.set_time_integration_scheme(dt=dt, s=1)
        fem_cpu.constrain(np.repeat([1, 0], 8))
        state = SimulationState(fem_cpu, load_params(fem_cpu))
        state.dt = dt
        state.reset()
        fem, check = state.fem, state.convergence
        assert abs(float(fem.bdf.h) - dt) < 1e-9
        rng = np.random.default_rng(3)
        xt = X + np.vstack([np.zeros((8, 3)), 0.02 * rng.standard_normal((8, 3))])  # deformed free cube
        v = np.zeros_like(X)
        v[8:] = [2.0, -1.0, 0.5]
        a = np.array([0.0, 0.0, -9.81])
        h = float(fem.bdf.h)
        fem.data.xt.assign(xt.astype(np.float32))
        fem.data.x.assign((xt + h * v).astype(np.float32))
        xtilde = xt + h * v
        xtilde[8:] += h * h * a
        fem.data.xtilde.assign(xtilde.astype(np.float32))
        check.measure_scale()
        check._test()
        m = fem.data.m.numpy()[8:].astype(np.float64)
        allowance = check.gradient_norm.scale.numpy()[8:] / m / h**2  # s_i / (m_i h^2), m/s^2, free nodes
        former = float(np.sqrt(np.mean(np.sum((xtilde[8:] - xt[8:]) ** 2, axis=1)))) / h**2
        return allowance, former

    def test_h_independent(self):
        wp.init()
        s1, f1 = self._tolerances(1e-2)
        s2, f2 = self._tolerances(5e-3)
        np.testing.assert_allclose(s2, s1, rtol=5e-3)  # float32 rounding of x - x~ (h^2 |a| = 0.25 mm at h/2)
        self.assertTrue(np.all(s1 > 2 * 9.81 * (1 - 1e-4)))  # inertial and external terms (2 |a|) plus elastic forces
        self.assertGreater(f2 / f1, 1.8)


def _block_check_state(n_max_iters: int, n_sub: int, early_exit: bool):
    """Two resting cubes (VBD, see :func:`_two_resting_cubes`) set to ``n_max_iters`` x ``n_sub`` sweeps with one test
    block of ``check_interval`` = 20 sweeps per 20 (VBD's schedule has no prefix)."""
    state = _two_resting_cubes("VBD")
    params = state.params[state.solver].data
    params.n_max_iters = n_max_iters
    params.n_subproblem_max_iters = n_sub
    state.convergence_params.check_interval = 20
    state.convergence_params.early_exit = early_exit
    return state


class TestInnerTestEndsSubproblemOnly(unittest.TestCase):
    """A pass inside a subproblem ends that subproblem only: the solve is decided between subproblems, after the full
    dual update, the filter, the solved set's gradient, the re-detection and feasibility (``end_outer``). Resting
    cubes, VBD 3 x 40 with a test every 20 sweeps and a huge rstol, so every test's gradient part passes; feasibility
    is forced to fail. Each subproblem stops after its first block, yet the solve runs all 3 outer iterations, each
    with its full dual update and outer test, and ends infeasible."""

    def test_inner_pass_doesnt_end_solve(self):
        state = _block_check_state(3, 40, early_exit=True)
        state.convergence_params.rstol = 1e6
        contact, check = state.contact, state.convergence
        calls = {"dual": 0, "end_outer": 0, "after_block": 0}
        update_dual, end_outer, after_block = contact.update_dual, check.end_outer, check.after_block

        def hooked_update_dual(*args, **kwargs):
            calls["dual"] += int(kwargs.get("request_decay_update", True) and kwargs.get(
                "request_lagrange_multiplier_update", True))
            return update_dual(*args, **kwargs)

        def hooked_end_outer(*args, **kwargs):
            calls["end_outer"] += 1
            return end_outer(*args, **kwargs)

        def hooked_after_block(*args, **kwargs):
            calls["after_block"] += 1
            return after_block(*args, **kwargs)

        def infeasible():
            check.penetration.fill_(1.0)  # 1 m into the margin

        contact.update_dual, check.end_outer, check.after_block = hooked_update_dual, hooked_end_outer, hooked_after_block
        check._measure_penetration = infeasible
        state._advance()  # eager, so the hooks run on the host
        check.enqueue_status_readback()
        wp.synchronize()
        row = check.read_status()[0][-1]
        self.assertEqual(int(row["outer"]), 3)
        self.assertEqual(int(row["sweeps"]), 3 * 20)  # each subproblem stopped after its first block
        self.assertEqual(int(row["reason"]), int(SolveEnd.FEASIBILITY_UNMET))
        self.assertEqual(calls, {"dual": 3, "end_outer": 3, "after_block": 3})


class TestInnerRatioMatchesOuter(unittest.TestCase):
    """With ``multiplier_step`` 1, the test inside a subproblem (its augmented gradient, multipliers max(0, lambda_k -
    sigma c) and friction capped from them) measures what the next outer test measures after the full dual update
    (Lagrangian gradient at lambda_k+1, Coulomb-projected friction), up to that projection and the feasibility filter.
    Resting cubes, VBD 1 x 20 with its one test block at the end of the subproblem, early exit off."""

    def test_inner_equals_outer(self):
        state = _block_check_state(1, 20, early_exit=False)
        self.assertEqual(float(state.contact.params.multiplier_step), 1.0)
        check = state.convergence
        inner = {}
        after_block = check.after_block

        def hooked_after_block(*args, **kwargs):
            after_block(*args, **kwargs)
            inner["ratio"] = check.gradient_ratio()
            inner["floor"] = float(np.sqrt(check.sums.numpy()[2] / check.stationarity_tol2.numpy()[0]))

        check.after_block = hooked_after_block
        state._advance()
        check.enqueue_status_readback()
        wp.synchronize()
        row = check.read_status()[0][-1]
        outer = float(row["gradient_ratio"])
        self.assertGreater(inner["floor"], 0.0)  # VBD stores its vertex Hessian norms for the floor
        # Measured: equal to the last bit, with mu_f 0.2 and 0 (2026-11-06)
        np.testing.assert_allclose(outer, inner["ratio"], rtol=1e-3)


if __name__ == "__main__":
    unittest.main()

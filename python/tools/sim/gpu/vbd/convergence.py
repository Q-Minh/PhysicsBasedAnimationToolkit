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
the test is the mass-weighted RMS (or, with ``node_norm`` MAX, the largest) of each node's residual over its own
allowance, relative to its forces, absolute, or at the float32 floor (:class:`ConvergenceParams`).
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
from ..gradient import GradientNorm, _elastic_gradient, _inertial_gradient


class NodeNorm(enum.Enum):
    """The norm over free nodes that the stationarity test applies to its node quantities
    (:attr:`ConvergenceParams.node_norm`)."""

    RMS = 0  # mass-weighted RMS, RMS_M(v) = (sum_i m_i v_i^2 / sum_i m_i)^(1/2)
    MAX = 1  # largest over free nodes, max_i |v_i|


NODE_NORM_RMS = wp.constant(int(NodeNorm.RMS.value))
NODE_NORM_MAX = wp.constant(int(NodeNorm.MAX.value))


class ConvergenceParams(SerializableMixin):
    """GPU-only settings of the VBD-family convergence test.

    Held by ``SimulationState.convergence_params`` (kept across resets) and serialized under
    ``Solver/Convergence``, so configs set them as ``Solver.Convergence.check_interval`` and
    ``Solver.Convergence.ftol``. The stationarity (gradient) test of dynamic problems uses the relative and
    absolute stationarity tolerances ``rstol`` and ``astol`` (see below); with both 0, and for static
    problems, each solver's absolute ``gtol`` applies instead (``Solver.<solver>.gtol``; Newton's ``gtol2``).
    GPU Newton uses the same test (see ``gpu.newton.solver.check_convergence``).

    Stationarity test (dynamic problems, rstol or astol > 0):

        RMS_M(q) <= 1   and   (RMS_M(q0) <= 1   or   RMS_M(r) <= kg RMS_M(phi))

        q_i      = |r_i| / max(rho_i, alpha, kl phi_i)
        q0_i     = |r_i| / max(rho_i, alpha)
        r_i      = f_res,i / m_i = g_i / (m_i h^2)                                 residual acceleration
        rho_i    = rstol s_i / m_i                                                  relative allowance
        s_i      = max(|m_i (x_i - x~_i)| / h^2, |f_el,i|, |f_c,i|, |f_ext,i|)       largest force term
        alpha    = max(astol |a|, ACCELERATION_FLOOR)                               absolute allowance
        phi_i    = |H_i|_F eps32 |x_i| / (m_i h^2)                                  float32 floor
        RMS_M(v) = (sum_i m_i v_i^2 / sum_i m_i)^(1/2)                              over free nodes

    with astol = 0.1, kl = ``floor_local_kappa`` 1.5 and kg = ``floor_kappa`` 0.2. s_i is the largest of the terms
    whose sum is node i's force residual, each net at the node (gpu.gradient.Gradient, ``force_scale``): the inertial term (it holds
    -f_ext for BDF1, so f_ext also counts on its own and a free fall, x = x~, keeps a scale), the elastic force, the
    contact force (normal and friction as one term) and the external load. h is the solve's time step (dt /
    substeps), |a| the scene's external acceleration (:func:`external_acceleration`: the gravity's norm, or
    :data:`STANDARD_GRAVITY` without a load), H_i node i's Hessian block in the gradient's units (inertia included,
    from the last sweep) and eps32 = 2^-23.

    q_i is node i's residual over its allowance: relative to its forces (a componentwise relative residual,
    |f_res,i| <= rstol max_k |f_k,i|; cf. Oettli and Prager, Numer. Math. 1964; Arioli, Duff and Ruiz, SIAM J. Matrix
    Anal. Appl. 1992), absolute where the forces are small, or at the float32 floor, the acceleration a position error
    of one float32 spacing makes, below which no float32 iterate gets (the exactly rounded solution's residual is about
    0.15 RMS_M(phi)). The floor clause stops a solve at the floor only when the residual is near the floor overall:
    the per-node floor alone stops stiff bodies' smooth modes early (BendingBeam.100MPa), and the global clause alone
    stops soft bodies when stiff ones set the floor. Every term is a force, so for a fixed physical motion the test
    doesn't depend on h; with rstol < 1 and astol < 1 a step that hasn't moved (r = a, s = m a for a free node at
    rest) can't pass. The status records RMS_M(q) (``gradient_ratio``), max_i q_i, RMS_M(q0) and RMS_M(r) / (kg
    RMS_M(phi)) (:class:`SolveStatus`). In the gradient's units the node ratio is |g_i| / max(rstol h^2 s_i,
    alpha h^2 m_i, kl |H_i|_F eps32 |x_i|). GPU Newton has no floor (TODO in gpu/newton/solver.py), so it tests
    RMS_M(q0) <= 1. Static problems use the absolute ``gtol``.

    The RMS decides, not the largest ratio: max_i q_i <= 1 is held by single nodes at their float32 floor (light
    surface nodes, a node next to a clamped rim, an interior beam node). Measured (2026-11-06, AAAVBD, against the
    former test): Trampoline.SoftSoft passed 0 of 421 steps (+49% sweeps), CapsulesIntoBowl +132% sweeps,
    ArmadilloSpaghetti.DragonImpact 0 of 30 substeps (+31%). The RMS form against the former test:
    Trampoline.SoftSoft and BendingBeam.100MPa bit-identical, DragonImpact and CapsulesIntoBowl -1% sweeps with the
    same exits; DragonImpact and the beam pass through the floor clause (python/tools/sim/CLAUDE.md).

    ``node_norm`` MAX (not the default) replaces every RMS_M of the test by the largest value over free nodes:

        max_i q_i <= 1   and   (max_i q0_i <= 1   or   max_i |r_i| <= kg max_i phi_i)

    The floor clause keeps its global form, the residual's norm against the floor's in the same norm (a per-node
    max_i |r_i| / phi_i would be the per-node floor again, at kg < kl). The status then records max_i q_i,
    max_i q0_i and max_i |r_i| / (kg max_i phi_i) as ``gradient_ratio``, ``no_floor_ratio`` and
    ``floor_clause_ratio`` (``max_ratio`` is max_i q_i under either norm), so
    ``status_ratio.stationarity_test_ratio`` decides as the test does. The RMS lets a heavy body hide a light stiff
    one (Celebration: the base plate's mass hides the letters, whose residuals are 6-9x their float32 floor when
    every solver passes the RMS test); the max catches it, at the cost of the single-node failures above.

    ``smoothing_passes`` k > 0 (not the default; k = 0 adds no work) adds a smoothed test, which must pass too: the
    same test with r_i replaced by r_bar_i, k passes of mass-weighted averaging over each free node's closed 1-ring
    of the tet mesh (free nodes only: fixed nodes neither contribute nor receive), and phi_i by a floor under the
    same filter, phi_bar_i:

        r_bar_i^(t+1) = sum_j m_j r_bar_j^(t) / sum_j m_j         j in {i} and i's mesh neighbours, free
        phi_bar_i     = c |S_k(p)|_i,   p_i = (g(x + xi) - g(x))_i / (m_i h^2)

    S_k the same k passes, xi a random step of one float32 spacing (+-) per coordinate, g the elastic and inertial
    gradient in float32 (contact left out: its Lagrangian force doesn't depend on x), c = :data:`SMOOTHED_FLOOR_SCALE`
    = 1.4. p is measured once per solve (time step or substep), at its initial positions (:meth:`ConvergenceCheck.
    measure_scale`), and only where the floor applies (the VBD family). rho_i, alpha, kl, kg and the test's form are
    unchanged. A solve passes only if the unsmoothed test and the smoothed one both pass, and the status records the
    ratios of the one whose decisive ratio is larger (``_keep_stricter_sums``), so ``status_ratio.
    stationarity_test_ratio`` stays exact. The smoothed test alone isn't safe: the filter also averages away real
    high-frequency residual, and on Trampoline.SoftSoft (AAAVBD 5 x 20, 0-1 s, k = 16) it passed every step after
    5000 sweeps instead of 9500, 1.0-5.0 mm from an early-exit-off reference instead of 0.10-0.29 mm. Why smooth:
    float32 rounding noise is high-frequency
    and dominates the per-node residual of stiff bodies, so no per-node norm of r ranks the solvers on Celebration
    (the letters sag under Chebyshev, Anderson and VBD), while it averages away under the filter and the solvers'
    smooth error doesn't: RMS r_bar_16 at step 100 is 0.56 g for AAAVBD, 5.5 (Chebyshev), 4.8 (Anderson) and 15.9
    (VBD). Raw phi can't be the floor after filtering: the real rounding noise falls 94x by k = 16, phi-sized white
    noise 16x, so phi's 58 g letter floor would pass Chebyshev. The probe tracks the measured filtered floor F_k
    (exact minimizer's rounding plus Warp's evaluation error) at 2.8-3.2x for every k from 0 to 32, within 7% across
    states. Every choice errs strict (the test must never pass an unconverged solve): c makes the floor clause's
    threshold kg phi_bar at most F_k (0.77-0.91 F_k measured, at most F_k by construction, see
    :data:`SMOOTHED_FLOOR_SCALE`), so the floor clause can't accept more than the real noise, where the unsmoothed
    test's kg phi is 1.16 F_0. Predicted offline (Celebration step 100, k = 16), decisive ratio: AAAVBD 1.72,
    Chebyshev 23.8, Anderson 18.5, VBD 52.2; AAAVBD 5 x 400 1.43, 5 x 6000 1.30 (k = 4: 2.73, 21.7, 18.6, 54.0;
    1.49, 1.44), so a fully converged AAAVBD state still fails narrowly (accepted: more work, never less). Measured
    (2026-11-07, 1-step resumes at step 100, 5 x 80 sweeps, k = 16): AAAVBD 1.71, Chebyshev 23.8, Anderson 18.5, an
    AAAVBD 5 x 2000 1.36 (the unsmoothed test passes the three 5 x 80 states at 0.78-0.85); Trampoline.SoftSoft 0-1 s at 10 ms
    bit-identical to k = 0 (the unsmoothed test is the stricter at every pass). Data:
    data/profiling/investigations/residual-smoothing. Each test costs k filter passes (82 us each on Celebration's
    196k nodes) and one more ratio kernel and reductions, 0.22 sweep at k = 4 and 0.77 at k = 16, and each solve two
    elastic gradients and k passes: +1.3% and +4.1% per Celebration step (21 tests). GPU Newton smooths r too,
    without a floor (as without smoothing).

    astol 0.1 (about 1 m/s^2; 0.5 until 2026-11-06), from a sweep against early-exit-off long-budget references
    (AAAVBD, Trampoline.SoftSoft 0-2.8 s and CapsulesIntoBowl 0-0.8 s; RMS_M distance in mm, the trampoline at 1 s
    and 2.8 s, the capsules at 0.2 s and 0.8 s):

        astol   h = 10 ms: trampoline sweeps, steps passing, distance     capsules sweeps, distance
        0.05    27860   172/280   0.22 / 2.5                              6540   0.05 / 1.9
        0.1     25460   251/280   0.29 / 2.5                              5780   0.11 / 2.1
        0.2     22720   261/280   0.66 / 2.7                              4840   0.18 / 3.5
        0.5     18860   280/280   1.15 / 1.5                              3900   0.92 / 4.9

    At h = 1 ms every substep of both scenes passes at its first test (20 sweeps) for every astol in this range, with
    the same distances (trampoline 0.01 mm before the sphere's impact at 2.4 s). The feasibility
    ratio's max is about the same at every astol (the trampoline's 1.05 matches its reference's 1.04).

    Changed form (2026-11-06): the test was RMS_M(e) <= alpha, e_i = max(0, |r_i|^2 - rho_i^2)^(1/2) the residual
    beyond the relative allowance, or RMS_M(e') <= alpha and RMS_M(r) <= kg RMS_M(phi), e'_i beyond
    max(rho_i, kl phi_i). Changed meaning (2026-11-06): the relative term used to be rstol RMS_M(x~ - x^t) / h^2,
    rstol |v^t / h + a|, which grew as 1/h for a moving body; rstol went from 1e-2 to 2e-2.

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
        2e-2,
        "Relative stationarity tolerance: node i's relative allowance is rstol s_i / m_i, s_i = max(|m_i (x_i - x~_i)| "
        "/ h^2, |f_el,i|, |f_c,i|, |f_ext,i|) the largest of its inertial, elastic, contact and external forces. The "
        "gradient test passes at RMS_M(q) <= 1 and (RMS_M(q0) <= 1 or RMS_M(r) <= floor_kappa RMS_M(phi)), q_i = |r_i| "
        "/ max(rstol s_i / m_i, astol |a|, floor_local_kappa phi_i), q0_i the same without the float32 floor phi_i, "
        "r_i the residual acceleration. With rstol = astol = 0, the solver's absolute gtol applies instead. Changed "
        "meaning (2026-11-06): the relative term used to be rstol * RMS(x~ - x^t) / h^2, default 1e-2",
    )
    astol = DocField(
        0.1,
        "Absolute stationarity tolerance, as a fraction of the scene's external acceleration |a| (its gravity's "
        "norm, or 9.81 m/s^2 without a load): node i's absolute allowance is astol * |a| (at least 1e-6 m/s^2), see "
        "rstol. With rstol < 1 and astol < 1 a step that hasn't moved can't pass. Default 0.1 since 2026-11-06 (was "
        "0.5). Changed meaning (2026-10-20): astol used to be a fraction of the scene's size L (astol * L), so an old "
        "file's astol = 1e-4 is now extremely strict",
    )
    floor_kappa = DocField(
        0.2,
        "Float32 floor clause kg: a test that passes only with the float32 floor in the node allowances (RMS_M(q) "
        "<= 1 < RMS_M(q0)) also needs RMS_M(r) <= kg RMS_M(phi), the residual near the floor overall (0: the floor "
        "never decides)",
    )
    floor_local_kappa = DocField(
        1.5,
        "Float32 floor factor kl (0: no floor): node i's allowance is at least kl phi_i, phi_i = |H_i|_F eps32 |x_i| "
        "/ (m_i h^2) the acceleration a position error of one float32 spacing makes through node i's Hessian block "
        "(the last sweep's; VBD family only)",
    )
    # TODO: A per-body norm, the largest of each body's (connected component's) mass-weighted RMS, would keep a heavy
    # body from hiding a light stiff one without the max's single-node sensitivity. Measured offline (2026-11-07,
    # Celebration step 100, production budget): the letters' own RMS_M(q) is 1.46 / 1.68 / 1.36 (AAAVBD / Chebyshev /
    # Anderson) and the confetti's 2.24 / 1.38 / 1.55 where the whole scene's is 0.41 / 0.35 / 0.33, so it would fail
    # all three but not rank AAAVBD best either; MAX gives 31.4 / 29.4 / 35.3, set by single light confetti nodes.
    # Not implemented: needs a per-body segmented reduction; data/profiling/investigations/max-node-norm.
    node_norm = DocField(
        NodeNorm.RMS,
        "Norm over free nodes in the stationarity test: RMS, the mass-weighted RMS (RMS_M(q) <= 1 and (RMS_M(q0) <= 1 "
        "or RMS_M(r) <= floor_kappa RMS_M(phi))), or MAX, the largest node value in every term (max q_i <= 1 and "
        "(max q0_i <= 1 or max |r_i| <= floor_kappa max phi_i)). MAX is stricter: a light stiff body can't hide "
        "behind a heavy one's mass, but single nodes at their float32 floor can hold a step. The status ratios "
        "follow it (GPU Newton too)",
    )
    smoothing_passes = DocField(
        0,
        "Residual smoothing passes k (0: off, no extra work): the stationarity test must also pass with r_bar_i, k "
        "passes of mass-weighted averaging of the residual acceleration over each free node's closed 1-ring of the "
        "tet mesh, in place of r_i, and a float32 floor under the same filter (1.4 |S_k(g(x + xi) - g(x))| / (m h^2), "
        "xi a random +-1 float32 spacing, measured once per solve) in place of phi_i; the status shows the stricter "
        "test. Float32 noise averages away, the solvers' smooth errors don't, so it fails solvers whose stiff bodies "
        "hide a smooth error under the per-node floor. Each test costs k filter passes (VBD family and GPU Newton; "
        "Newton has no floor)",
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
    gradient_ratio: wp.float32  # gradient measure / tolerance at the end, ||q|| (<= 1 passes; ||.|| node_norm)
    feasibility_ratio: wp.float32  # penetration / (ftol max(|dmin|, floor)) at the end (<= 1 passes)
    max_ratio: wp.float32  # the same test's largest node ratio, max_i q_i (0: absolute test)
    no_floor_ratio: wp.float32  # ||q0||, the node ratios without the float32 floor (0: absolute test)
    floor_clause_ratio: wp.float32  # ||r|| / (kg ||phi||) (0: absolute test)


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
    free_mass: wp.array[wp.float32],
    relative: wp.int32,
) -> wp.float32:
    """The gradient test's measure over its tolerance (see :func:`_is_converged`): ||q|| (``free_mass`` the norm
    weight, see ``SUM_*``), or |g| / gtol. At most 1 passes the stationarity test's first condition."""
    ratio = wp.float32(0)
    if relative != 0:
        ratio = _mass_rms(sums, SUM_RATIO, free_mass)
    elif gtol2 > wp.float32(0):
        ratio = wp.sqrt(gnorm2[0] / gtol2)
    elif gnorm2[0] > wp.float32(0):
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

ACCELERATION_FLOOR = 1e-6
"""Smallest absolute stationarity tolerance (m/s^2), ``max(astol * |a|, ACCELERATION_FLOOR)``: with astol = 0 (the
relative test alone), a node without forces still has a positive allowance, so its ratio stays finite."""


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
    """The absolute term of the stationarity test, ``max(astol * |a|, ACCELERATION_FLOOR) * h^2`` (m), for the
    external acceleration ``acceleration`` (m/s^2) and the time step ``h`` (s)."""
    # The float32 floor (2026-11-06): the backward-error bound of the force sums, eps32 * sum_i (|m_i (x_i - x~_i)| +
    # |h^2 f_el,i| + |h^2 f_c,i|), proposed here before, is 1000-10000x below the attainable residual. The floor is
    # the quantization of x, f_i = |H_i|_F eps32 |x_i| through the vertex Hessian (the residual of the exactly rounded
    # solution is ~0.15 RMS_M(M^-1 f) on BendingBeam.100MPa and DragonImpact, also translated by 32 m), so each node's
    # allowance is at least kl times it, with the floor clause (ConvergenceParams.floor_local_kappa, floor_kappa;
    # _is_converged). The force sum
    # became the relative term's scale instead (ConvergenceParams). Evidence: investigate/float32-residual-floor
    # (8eadc920).
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
    return max(float(astol) * float(acceleration), ACCELERATION_FLOOR) * float(h) ** 2


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


# The stationarity test's reductions over free nodes (ConvergenceCheck.sums), from node i's ratios q_i and q0_i (see
# _node_ratio) and, in the gradient's units, its residual g_i and float32 floor f_i. With NodeNorm.RMS they are sums,
# divided by the norm weight sum m_i; with NodeNorm.MAX maxima of the same quantities per unit mass (q_i^2,
# |g_i|^2 / m_i^2, ...), whose norm weight is 1, so that the same comparisons and ratios apply to either norm.
SUM_RATIO_MAX = wp.constant(0)  # max_i q_i (either norm)
SUM_RATIO = wp.constant(1)  # sum m_i q_i^2 (MAX: max_i q_i^2)
SUM_RATIO_NO_FLOOR = wp.constant(2)  # sum m_i q0_i^2 (MAX: max_i q0_i^2)
SUM_RESIDUAL = wp.constant(3)  # sum |g_i|^2 / m_i (MAX: max_i |g_i|^2 / m_i^2)
SUM_FLOOR = wp.constant(4)  # sum f_i^2 / m_i, 0 without Hessian norms (MAX: max_i f_i^2 / m_i^2)
N_SUMS = 5

EPS32 = 1.1920928955078125e-07
"""float32 machine epsilon 2^-23: the spacing of float32 numbers at x is between 0.5 and 1 times EPS32 |x|."""


@wp.func
def _is_converged(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    sums: wp.array[wp.float32],
    free_mass: wp.array[wp.float32],
    relative: wp.int32,
    floor_kappa2: wp.float32,
) -> bool:
    """Stationarity test if enabled (``relative``), else ``|g|^2 <= gtol^2``. The stationarity test (see
    :class:`ConvergenceParams`) is ||q|| <= 1 and (||q0|| <= 1 or ||r|| <= kg ||phi||), i.e. with the RMS
    sum m_i q_i^2 <= sum m_i and (sum m_i q0_i^2 <= sum m_i or sum |g_i|^2 / m_i <= kg^2 sum f_i^2 / m_i), and with
    the max the same comparisons of the maxima against a norm weight ``free_mass`` of 1 (see ``SUM_*``)."""
    if relative != 0:
        mass = free_mass[0]
        at_floor = sums[SUM_RESIDUAL] <= floor_kappa2 * sums[SUM_FLOOR]
        return sums[SUM_RATIO] <= mass and (sums[SUM_RATIO_NO_FLOOR] <= mass or at_floor)
    return gnorm2[0] <= gtol2


@wp.func
def _mass_rms(sums: wp.array[wp.float32], k: wp.int32, free_mass: wp.array[wp.float32]) -> wp.float32:
    """Norm of a node ratio, sqrt(sums[k] / w) for the norm weight w = ``free_mass``: the mass-weighted RMS for
    w = sum m_i, the largest node value for NodeNorm.MAX's w = 1 (0 for w = 0)."""
    ratio = wp.float32(0)
    if free_mass[0] > wp.float32(0):
        ratio = wp.sqrt(sums[k] / free_mass[0])
    return ratio


@wp.func
def _floor_clause_ratio(sums: wp.array[wp.float32], floor_kappa2: wp.float32) -> wp.float32:
    """||r|| / (kg ||phi||), with the RMS sqrt(sum |g_i|^2 / m_i / (kg^2 sum f_i^2 / m_i)) and with the max
    sqrt(max |g_i|^2 / m_i^2 / (kg^2 max f_i^2 / m_i^2)): at most 1 passes the floor clause (inf without a floor)."""
    num = sums[SUM_RESIDUAL]
    den = floor_kappa2 * sums[SUM_FLOOR]
    ratio = wp.float32(0)
    if den > wp.float32(0):
        ratio = wp.sqrt(num / den)
    elif num > wp.float32(0):
        ratio = wp.float32(wp.inf)
    return ratio


@wp.func
def _node_ratio(
    g: wp.vec3f, scale: wp.float32, m: wp.float32, f: wp.float32, rstol: wp.float32, atol: wp.float32, kl: wp.float32
) -> wp.float32:
    """Node i's residual over its allowance, in the gradient's units, q_i = |g_i| / max(rstol s_i, atol m_i, kl f_i):
    s_i the largest magnitude of the terms of its force balance (``scale``), atol = max(astol |a|, ACCELERATION_FLOOR)
    h^2 the absolute tolerance (m), and f_i = |H_i|_F EPS32 |x_i| its float32 floor (0 gives q0_i)."""
    return wp.length(g) / wp.max(wp.max(rstol * scale, atol * m), kl * f)


@wp.kernel
def _stationarity_ratios(
    g: wp.array[wp.vec3f],
    scale: wp.array[wp.float32],
    hnorm: wp.array[wp.float32],
    use_floor: wp.int32,
    rstol: wp.float32,
    atol: wp.float32,
    local_kappa: wp.float32,
    node_norm: wp.int32,
    fem: FemElastoDynamicsData,
    terms: wp.array2d[wp.float32],
):
    """Free node i's terms of the stationarity reductions (``SUM_*``), from its ratios q_i and q0_i
    (:func:`_node_ratio`). Its relative allowance is rstol s_i, s_i the largest magnitude of the terms of its force
    balance (``scale``, gpu.gradient.Gradient's force scale), and its float32 floor f_i = |H_i|_F EPS32 |x_i|: the
    force (in the gradient's units) that a position error of about one float32 spacing at x_i makes through the
    vertex's own Hessian block |H_i|_F (``hnorm``, the last sweep's, inertia included). Fixed and massless nodes don't
    count. With NodeNorm.MAX the terms are per unit mass, for max reductions (see ``SUM_*``)."""
    i = wp.tid()
    for k in range(N_SUMS):
        terms[k, i] = wp.float32(0)
    mi = fem.m[i]
    if is_dirichlet_node(fem.dmask, i) or mi <= wp.float32(0):
        return
    f = wp.float32(0)
    if use_floor != 0:
        f = hnorm[i] * wp.float32(EPS32) * wp.length(fem.x[i])
    q = _node_ratio(g[i], scale[i], mi, f, rstol, atol, local_kappa)
    q0 = _node_ratio(g[i], scale[i], mi, wp.float32(0), rstol, atol, local_kappa)
    terms[SUM_RATIO_MAX, i] = q
    if node_norm == NODE_NORM_MAX:
        terms[SUM_RATIO, i] = q * q
        terms[SUM_RATIO_NO_FLOOR, i] = q0 * q0
        terms[SUM_RESIDUAL, i] = wp.dot(g[i], g[i]) / (mi * mi)
        terms[SUM_FLOOR, i] = (f * f) / (mi * mi)
    else:
        terms[SUM_RATIO, i] = mi * q * q
        terms[SUM_RATIO_NO_FLOOR, i] = mi * q0 * q0
        terms[SUM_RESIDUAL, i] = wp.dot(g[i], g[i]) / mi
        terms[SUM_FLOOR, i] = f * f / mi


@wp.kernel
def _free_mass_terms(fem: FemElastoDynamicsData, masses: wp.array[wp.float32]):
    i = wp.tid()
    masses[i] = wp.float32(0)
    if not is_dirichlet_node(fem.dmask, i):
        masses[i] = fem.m[i]


# --- Residual smoothing (ConvergenceParams.smoothing_passes) ---

SMOOTHED_FLOOR_SCALE = 1.4
"""c of the smoothed test's float32 floor phi_bar_i = c |S_k(p)|_i / (m_i h^2), p = g(x + xi) - g(x) for a random
+-1 float32 spacing xi per coordinate (:class:`ConvergenceParams`). Chosen on the strict side (maintainer: never pass
an unconverged solve; failing a nearly converged one is acceptable), so that the floor clause's threshold kg phi_bar
can't exceed the real floor F_k, the filtered residual of the exactly rounded solution (its rounding, uniform within
+-1/2 spacing, plus the float32 evaluation error): p's rounding part is sqrt(12) = 3.46 times F_k's (the variance of
+-1 against U(-1/2, 1/2)) and its evaluation part sqrt(2) times, so p <= 3.46 F_k and kg c p <= F_k for c <=
1 / (0.2 sqrt(12)) = 1.44 at kg = 0.2. Measured on Celebration (step 100): p / F_k = 2.75-3.25 for k = 0-32, within 7%
across states, so kg c p = 0.77-0.91 F_k. phi_bar is then below phi's scale at k = 0 (RMS_M(phi) / RMS_M(p) = 1.78)."""

PROBE_SEED = 20261107
"""Seed of the floor probe's random signs (fixed, so the probe and the test are reproducible)."""


@wp.kernel
def _stationarity_ratios_smoothed(
    g: wp.array[wp.vec3f],
    scale: wp.array[wp.float32],
    floor: wp.array[wp.float32],
    use_floor: wp.int32,
    rstol: wp.float32,
    atol: wp.float32,
    local_kappa: wp.float32,
    node_norm: wp.int32,
    fem: FemElastoDynamicsData,
    terms: wp.array2d[wp.float32],
):
    """:func:`_stationarity_ratios` with the smoothed residual ``g`` (r_bar_i m_i h^2) and the smoothed float32 floor
    ``floor`` (phi_bar_i m_i h^2) in the gradient's units; the same terms otherwise."""
    i = wp.tid()
    for k in range(N_SUMS):
        terms[k, i] = wp.float32(0)
    mi = fem.m[i]
    if is_dirichlet_node(fem.dmask, i) or mi <= wp.float32(0):
        return
    f = wp.float32(0)
    if use_floor != 0:
        f = floor[i]
    q = _node_ratio(g[i], scale[i], mi, f, rstol, atol, local_kappa)
    q0 = _node_ratio(g[i], scale[i], mi, wp.float32(0), rstol, atol, local_kappa)
    terms[SUM_RATIO_MAX, i] = q
    if node_norm == NODE_NORM_MAX:
        terms[SUM_RATIO, i] = q * q
        terms[SUM_RATIO_NO_FLOOR, i] = q0 * q0
        terms[SUM_RESIDUAL, i] = wp.dot(g[i], g[i]) / (mi * mi)
        terms[SUM_FLOOR, i] = (f * f) / (mi * mi)
    else:
        terms[SUM_RATIO, i] = mi * q * q
        terms[SUM_RATIO_NO_FLOOR, i] = mi * q0 * q0
        terms[SUM_RESIDUAL, i] = wp.dot(g[i], g[i]) / mi
        terms[SUM_FLOOR, i] = f * f / mi


@wp.func
def _decisive_ratio(sums: wp.array[wp.float32], weight: wp.array[wp.float32], floor_kappa2: wp.float32) -> wp.float32:
    """The stationarity test's decisive ratio from its reductions, max(||q||, min(||q0||, ||r|| / (kg ||phi||))): at
    most 1 iff it passes (as ``status_ratio.stationarity_test_ratio``)."""
    return wp.max(
        _mass_rms(sums, SUM_RATIO, weight),
        wp.min(_mass_rms(sums, SUM_RATIO_NO_FLOOR, weight), _floor_clause_ratio(sums, floor_kappa2)),
    )


@wp.kernel
def _keep_stricter_sums(
    unsmoothed: wp.array[wp.float32],
    weight: wp.array[wp.float32],
    floor_kappa2: wp.float32,
    sums: wp.array[wp.float32],
):
    """With smoothing, a test passes only if the unsmoothed and the smoothed tests both pass: ``sums`` (the smoothed
    test's reductions) becomes the unsmoothed test's where its decisive ratio is larger, so every reader of ``sums``
    (decisions, status rows, the host) sees the stricter of the two."""
    # The decisions first (as _is_converged), so rounding of the ratios at 1 can't pick a passing test over a failing one
    passes = _is_converged(sums, wp.float32(0), sums, weight, 1, floor_kappa2)
    unsmoothed_passes = _is_converged(unsmoothed, wp.float32(0), unsmoothed, weight, 1, floor_kappa2)
    stricter = passes and not unsmoothed_passes
    if passes == unsmoothed_passes:
        stricter = _decisive_ratio(unsmoothed, weight, floor_kappa2) > _decisive_ratio(sums, weight, floor_kappa2)
    if stricter:
        for k in range(N_SUMS):
            sums[k] = unsmoothed[k]


@wp.kernel
def _smoothing_pass(
    src: wp.array[wp.vec3f],
    masses: wp.array[wp.float32],
    GVVp: wp.array[wp.int32],
    GVVadj: wp.array[wp.int32],
    dst: wp.array[wp.vec3f],
):
    """One pass of the residual filter, in the gradient's units (u_i = m_i v_i h^2 for an acceleration field v):
    dst_i = m_i sum_j src_j / sum_j m_j over j in {i} and i's mesh neighbours (``GVVp``/``GVVadj``, which may list i
    itself), free nodes only (``masses`` the free masses, 0 on fixed nodes), i.e. m_i h^2 times the mass-weighted
    average of v. Fixed and massless nodes get 0."""
    # TODO: A cheaper filter. Why: a pass takes 82 us on Celebration (196k nodes, 13.1-node rings; memory-bound at
    # ~0.6 TB/s), so a k = 16 test costs 0.77 sweep (k = 4: 0.22; 2026-11-07, CUPTI). Ideas: store each ring's mass
    # sum once per solve (drops the neighbours' mass loads), fuse several passes per block through shared memory, or
    # a wider stencil with fewer passes. Not tried.
    i = wp.tid()
    mi = masses[i]
    out = wp.vec3f(0.0)
    if mi > wp.float32(0):
        s = src[i]
        w = mi
        for k in range(GVVp[i], GVVp[i + 1]):
            j = GVVadj[k]
            if j != i:
                mj = masses[j]
                if mj > wp.float32(0):
                    s += src[j]
                    w += mj
        out = (mi / w) * s
    dst[i] = out


@wp.func
def _step_one_spacing(v: wp.float32, up: bool) -> wp.float32:
    """v moved by about one float32 spacing up or down (exactly one, or two next to a power of 2); 0 stays 0."""
    a = wp.abs(v)
    out = v
    if a >= wp.float32(1.17549435e-38):  # normal numbers
        spacing = wp.pow(wp.float32(2), wp.floor(wp.log2(a)) - wp.float32(23))
        out = wp.where(up, v + spacing, v - spacing)
    return out


@wp.kernel
def _perturb_positions(x: wp.array[wp.vec3f], seed: wp.int32, xp: wp.array[wp.vec3f]):
    """xp = x moved by one float32 spacing per coordinate, with random signs (the floor probe's xi)."""
    i = wp.tid()
    state = wp.rand_init(seed, i)
    xi = x[i]
    out = xi
    for d in range(3):
        out[d] = _step_one_spacing(xi[d], wp.randf(state) < wp.float32(0.5))
    xp[i] = out


@wp.kernel
def _subtract(a: wp.array[wp.vec3f], b: wp.array[wp.vec3f]):
    """a -= b."""
    i = wp.tid()
    a[i] = a[i] - b[i]


@wp.kernel
def _floor_from_probe(p: wp.array[wp.vec3f], c: wp.float32, floor: wp.array[wp.float32]):
    """The smoothed float32 floor in the gradient's units, c |S_k(p)|_i."""
    i = wp.tid()
    floor[i] = c * wp.length(p[i])


def _vertex_adjacency(E: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-vertex adjacency of tets ``E`` (Ne, 4), each vertex listing itself and its mesh neighbours, as
    ``pbat.sim.algorithm.vbd.vertex_colors``'s GVVp/GVVadj (CSR, int32)."""
    E = np.asarray(E, dtype=np.int64).reshape(-1, 4)
    i = np.repeat(E, 4, axis=1).ravel()
    j = np.tile(E, (1, 4)).ravel()
    keys = np.unique(i * n_nodes + j)
    rows, cols = keys // n_nodes, keys % n_nodes
    GVVp = np.zeros(n_nodes + 1, dtype=np.int32)
    GVVp[1:] = np.cumsum(np.bincount(rows, minlength=n_nodes))
    return GVVp, cols.astype(np.int32)


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


@wp.func
def _node_ratios(
    sums: wp.array[wp.float32],
    free_mass: wp.array[wp.float32],
    relative: wp.int32,
    floor_kappa2: wp.float32,
) -> wp.vec3f:
    """The status row's stationarity ratios besides ``gradient_ratio``: (max_i q_i, ||q0||, ||r|| / (kg ||phi||)),
    0 for the absolute test."""
    ratios = wp.vec3f(0.0)
    if relative != 0:
        ratios = wp.vec3f(
            sums[SUM_RATIO_MAX], _mass_rms(sums, SUM_RATIO_NO_FLOOR, free_mass), _floor_clause_ratio(sums, floor_kappa2)
        )
    return ratios


@wp.func
def _with_node_ratios(row: SolveStatus, ratios: wp.vec3f) -> SolveStatus:
    out = row
    out.max_ratio = ratios[0]
    out.no_floor_ratio = ratios[1]
    out.floor_clause_ratio = ratios[2]
    return out


@wp.kernel
def _start_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    sums: wp.array[wp.float32],
    free_mass: wp.array[wp.float32],
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
    converged = _is_converged(gnorm2, gtol2, sums, free_mass, relative, floor_kappa2)
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
    row.gradient_ratio = _gradient_ratio(gnorm2, gtol2, sums, free_mass, relative)
    row.feasibility_ratio = _feasibility_ratio(penetration, ftol, contact.dmin)
    row = _with_node_ratios(row, _node_ratios(sums, free_mass, relative, floor_kappa2))
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
    free_mass: wp.array[wp.float32],
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
    stop = _is_converged(gnorm2, gtol2, sums, free_mass, relative, floor_kappa2) and early_exit != 0
    subproblem_keep[0] = wp.where(stop, 0, 1)
    inner_keep[0] = wp.where(not stop and b < n_blocks, 1, 0)


@wp.kernel
def _end_outer_kernel(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    sums: wp.array[wp.float32],
    free_mass: wp.array[wp.float32],
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
    converged = _is_converged(gnorm2, gtol2, sums, free_mass, relative, floor_kappa2)
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
    row.gradient_ratio = _gradient_ratio(gnorm2, gtol2, sums, free_mass, relative)
    row.feasibility_ratio = _feasibility_ratio(penetration, ftol, contact.dmin)
    row = _with_node_ratios(row, _node_ratios(sums, free_mass, relative, floor_kappa2))
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
        vertex_adjacency: tuple[wp.array, wp.array] | None = None,
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
            vertex_adjacency (tuple[wp.array, wp.array] | None): ``fem``'s vertex-vertex adjacency (GVVp, GVVadj)
                on the device, the residual smoothing's 1-rings (``smoothing_passes``), e.g. the VBD params'; built
                from ``fem``'s elements when smoothing first runs if None.
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
        # Stationarity test (rstol or astol > 0): reductions of the node ratios over free nodes (SUM_*), and their
        # free mass sum m_i (the RMSs' weight)
        n_nodes = fem.data.x.shape[0]
        self._terms = wp.zeros((N_SUMS, n_nodes), dtype=wp.float32)
        self._masses = wp.zeros(n_nodes, dtype=wp.float32)
        self.sums = wp.zeros(N_SUMS, dtype=wp.float32)
        self.free_mass = wp.zeros(1, dtype=wp.float32)
        # The solver's (N,) vertex Hessian norms |H_i|_F for the float32 floor (set_hessian_norms); None: no floor
        self._hessian_norms = None
        self._no_hessian_norms = wp.zeros(1, dtype=wp.float32)
        terms, sums = cp.asarray(self._terms), cp.asarray(self.sums)
        # Per node norm (NodeNorm): sums for the RMS, maxima for the max; max_i q_i under either
        maximum, plus = cuda.compute.OpKind.MAXIMUM, cuda.compute.OpKind.PLUS
        self._reduce_sums = {
            norm: [
                Reduce(d_in=terms[k], d_out=sums[k : k + 1], num_items=n_nodes, op=op)
                for k, op in [(SUM_RATIO_MAX, maximum)] + [(k, rest) for k in range(SUM_RATIO, N_SUMS)]
            ]
            for norm, rest in ((NodeNorm.RMS, plus), (NodeNorm.MAX, maximum))
        }
        # NodeNorm.MAX's norm weight (see SUM_*): its reductions are maxima per unit mass
        self._unit_weight = wp.ones(1, dtype=wp.float32)
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
        # Residual smoothing (smoothing_passes > 0): the 1-rings, and buffers allocated when it's first used
        # (prepare_smoothing): two ping-pong fields, the probe's positions and the smoothed floor
        self._vertex_adjacency = vertex_adjacency
        self._smooth = None
        self._probe_x = None
        self._smoothed_floor = None

    @property
    def smoothing_passes(self) -> int:
        """The residual smoothing's passes k (``ConvergenceParams.smoothing_passes``; 0: off)."""
        return max(0, int(self.params.smoothing_passes))

    def prepare_smoothing(self) -> None:
        """Allocate the residual smoothing's buffers if ``smoothing_passes > 0`` and they don't exist yet. Host-side:
        call it before a graph capture (``SimulationState.step`` does), since captured work must not allocate; the
        tests also call it when they first run."""
        if self.smoothing_passes == 0 or self._smooth is not None:
            return
        n_nodes = self._fem.data.x.shape[0]
        if self._vertex_adjacency is None:
            GVVp, GVVadj = _vertex_adjacency(self._fem.data.E.numpy().reshape(-1, 4), n_nodes)
            self._vertex_adjacency = (wp.array(GVVp, dtype=wp.int32), wp.array(GVVadj, dtype=wp.int32))
        self._smooth = (wp.zeros(n_nodes, dtype=wp.vec3f), wp.zeros(n_nodes, dtype=wp.vec3f))
        self._probe_x = wp.zeros(n_nodes, dtype=wp.vec3f)
        self._smoothed_floor = wp.zeros(n_nodes, dtype=wp.float32)
        # The unsmoothed test's reductions, kept beside the smoothed test's (sums): the stricter one decides
        self._unsmoothed_sums = wp.zeros(N_SUMS, dtype=wp.float32)
        terms, sums = cp.asarray(self._terms), cp.asarray(self._unsmoothed_sums)
        maximum, plus = cuda.compute.OpKind.MAXIMUM, cuda.compute.OpKind.PLUS
        self._reduce_unsmoothed_sums = {
            norm: [
                Reduce(d_in=terms[k], d_out=sums[k : k + 1], num_items=n_nodes, op=op)
                for k, op in [(SUM_RATIO_MAX, maximum)] + [(k, rest) for k in range(SUM_RATIO, N_SUMS)]
            ]
            for norm, rest in ((NodeNorm.RMS, plus), (NodeNorm.MAX, maximum))
        }

    def _smooth_field(self, src: wp.array, k: int) -> wp.array:
        """k filter passes of ``src`` (gradient units, see :func:`_smoothing_pass`) through the ping-pong buffers,
        leaving ``src`` as it is; returns the buffer holding the result."""
        GVVp, GVVadj = self._vertex_adjacency
        for t in range(k):
            dst = self._smooth[t % 2]
            wp.launch(_smoothing_pass, dim=dst.shape[0], inputs=[src, self._masses, GVVp, GVVadj, dst])
            src = dst
        return src

    def _measure_smoothed_floor(self) -> None:
        """The smoothed test's float32 floor c |S_k(p)|_i in the gradient's units (:class:`ConvergenceParams`), p =
        g(x + xi) - g(x) of the elastic and inertial gradient at the current positions, xi a random step of one
        float32 spacing per coordinate. Reads the free masses (:meth:`measure_scale`). Capturable."""
        # TODO: Include the contact penalty's response in the probe (the subproblems' augmented contact stiffness).
        # Why: an AAAVBD Celebration step solved to 5 x 2000 sweeps still fails the smoothed test at 1.36 (offline:
        # 5 x 6000 at 1.30), its filtered residual 1.4x the elastic-and-inertial floor; the probe leaves contact out
        # (the strict side), which may be the missing noise. Not measured.
        fem = self._fem
        h2 = wp.float32(fem.h2)
        g0, g1 = self._smooth
        n_nodes, n_elems = fem.data.x.shape[0], fem.data.E.shape[0]
        wp.launch(_perturb_positions, dim=n_nodes, inputs=[fem.data.x, PROBE_SEED, self._probe_x])
        perturbed = FemElastoDynamicsData()
        for name in FemElastoDynamicsData.vars:
            setattr(perturbed, name, getattr(fem.data, name))
        perturbed.x = self._probe_x
        for data, g in ((fem.data, g0), (perturbed, g1)):
            g.zero_()
            wp.launch(_elastic_gradient, dim=n_elems, inputs=[data, h2, g])
            wp.launch(_inertial_gradient, dim=n_nodes, inputs=[data, g])
        wp.launch(_subtract, dim=n_nodes, inputs=[g1, g0])
        # The passes write g0, g1, g0, ..., so the first reads p from g1 and none writes its own input
        p = self._smooth_field(g1, self.smoothing_passes)
        wp.launch(_floor_from_probe, dim=n_nodes, inputs=[p, float(SMOOTHED_FLOOR_SCALE), self._smoothed_floor])

    @property
    def relative(self) -> bool:
        """Whether the stationarity test with ``rstol``/``astol`` applies (rather than the solvers'
        absolute ``gtol``): either is positive, and the problem isn't static."""
        enabled = float(self.params.rstol) > 0.0 or float(self.params.astol) > 0.0
        return enabled and not self._fem.static

    @property
    def absolute_tolerance(self) -> float:
        """The stationarity test's absolute term ``max(astol * |a|, ACCELERATION_FLOOR) * h^2`` (m), h being the time
        step of ``fem``'s integrator (dt / substeps)."""
        return absolute_stationarity_tolerance(self.params.astol, self.acceleration, self._fem.bdf.h)

    @property
    def node_norm(self) -> NodeNorm:
        """The norm over free nodes of the stationarity test (``ConvergenceParams.node_norm``)."""
        return NodeNorm(self.params.node_norm)

    @property
    def norm_weight(self) -> wp.array:
        """The (1,) weight the reductions ``sums`` are divided by: the free mass sum m_i for the RMS, 1 for the
        max (see ``SUM_*``)."""
        return self._unit_weight if self.node_norm == NodeNorm.MAX else self.free_mass

    def set_hessian_norms(self, hnorm: wp.array | None) -> None:
        """Use the solver's (N,) vertex Hessian norms |H_i|_F (``Params.data.Hnk``, written by every sweep) for the
        float32 floor; None: no floor. Host-side: recorded kernels read the array, so it must stay the same while a
        captured graph replays."""
        self._hessian_norms = hnorm

    def _test(self) -> None:
        """Measure the Lagrangian gradient at the current multipliers (the outer test's), and with the stationarity
        test its node ratios' reductions."""
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
        """The stationarity reductions of ``gradient``'s node ratios (``g`` and its force ``scale``). With
        ``smoothing_passes`` > 0, also those of its smoothed residual (``g`` itself is left as it is), and ``sums``
        keeps the stricter test's (both must pass)."""
        stream = wp.get_stream()
        hnorm = self._hessian_norms
        k = self.smoothing_passes
        if k > 0:
            self.prepare_smoothing()
            # The unsmoothed test, as with k = 0, into its own reductions
            wp.launch(
                _stationarity_ratios,
                dim=self._terms.shape[1],
                inputs=[
                    gradient.g,
                    gradient.scale,
                    hnorm if hnorm is not None else self._no_hessian_norms,
                    int(hnorm is not None),
                    float(self.params.rstol),
                    self.absolute_tolerance,
                    float(self.params.floor_local_kappa),
                    int(self.node_norm.value),
                    self._fem.data,
                    self._terms,
                ],
            )
            for reduce in self._reduce_unsmoothed_sums[self.node_norm]:
                reduce(stream)
            # The smoothed test
            wp.launch(
                _stationarity_ratios_smoothed,
                dim=self._terms.shape[1],
                inputs=[
                    self._smooth_field(gradient.g, k),
                    gradient.scale,
                    self._smoothed_floor,
                    int(hnorm is not None),
                    float(self.params.rstol),
                    self.absolute_tolerance,
                    float(self.params.floor_local_kappa),
                    int(self.node_norm.value),
                    self._fem.data,
                    self._terms,
                ],
            )
            for reduce in self._reduce_sums[self.node_norm]:
                reduce(stream)
            # Both must pass: keep the reductions of the test with the larger decisive ratio
            wp.launch(
                _keep_stricter_sums,
                dim=1,
                inputs=[self._unsmoothed_sums, self.norm_weight, float(self.params.floor_kappa) ** 2, self.sums],
            )
            return
        wp.launch(
            _stationarity_ratios,
            dim=self._terms.shape[1],
            inputs=[
                gradient.g,
                gradient.scale,
                hnorm if hnorm is not None else self._no_hessian_norms,
                int(hnorm is not None),
                float(self.params.rstol),
                self.absolute_tolerance,
                float(self.params.floor_local_kappa),
                int(self.node_norm.value),
                self._fem.data,
                self._terms,
            ],
        )
        for reduce in self._reduce_sums[self.node_norm]:
            reduce(stream)

    def measure_scale(self) -> None:
        """Measure the time step's free mass sum m_i over free nodes, the weight of the node ratios' RMSs (the
        allowances come from each test's forces and the absolute tolerance), and with ``smoothing_passes`` > 0 and
        the float32 floor on (:meth:`set_hessian_norms`) the smoothed floor at the current positions. No-op unless
        :attr:`relative`. Capturable."""
        if self.relative:
            wp.launch(_free_mass_terms, dim=self._masses.shape[0], inputs=[self._fem.data, self._masses])
            self._reduce_free_mass(wp.get_stream())
            if self.smoothing_passes > 0 and self._hessian_norms is not None:
                self.prepare_smoothing()
                self._measure_smoothed_floor()

    def _tolerance_inputs(self, gtol: float) -> list:
        return [
            self.gradient_norm.gnorm2,
            float(gtol) ** 2,
            self.sums,
            self.norm_weight,
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
        subproblem solved with. It stores scalars only (``gradient_norm.gnorm2``, ``sums``), which
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
        sums, mass = self.sums.numpy().astype(np.float64), float(self.norm_weight.numpy()[0])
        at_floor = sums[SUM_RESIDUAL] <= float(self.params.floor_kappa) ** 2 * sums[SUM_FLOOR]
        return bool(sums[SUM_RATIO] <= mass and (sums[SUM_RATIO_NO_FLOOR] <= mass or at_floor))

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
        """Host-side stationarity measure of the last test over its tolerance, ||q|| (:class:`ConvergenceParams`,
        ``node_norm``), as the status ring's ``gradient_ratio``. Synchronizes.

        Returns:
            float: The ratio (from the last :meth:`test_gradient`, :meth:`test_outer` or
            :meth:`relative_test`).
        """
        mass = float(self.norm_weight.numpy()[0])
        return float(np.sqrt(self.sums.numpy()[SUM_RATIO] / mass)) if mass > 0.0 else 0.0

    def max_ratio(self) -> float:
        """Host-side largest node ratio of the last test, max_i q_i, as the status ring's ``max_ratio``.
        Synchronizes."""
        return float(self.sums.numpy()[SUM_RATIO_MAX])

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
    def test_node_ratio(self):
        wp.init()

        @wp.kernel
        def k(
            cases: wp.array2d[wp.float32],
            rstol: wp.float32,
            atol: wp.float32,
            kl: wp.float32,
            q: wp.array[wp.float32],
        ):
            i = wp.tid()
            g = wp.vec3f(cases[i, 0], wp.float32(0), wp.float32(0))
            q[i] = _node_ratio(g, cases[i, 1], cases[i, 2], cases[i, 3], rstol, atol, kl)

        rstol, atol, kl = 2e-2, 1e-4, 1.5  # atol = astol |a| h^2, in m
        cases = [  # (|g_i|, s_i, m_i, f_i), in the gradient's units
            (0.9 * rstol * 1.0, 1.0, 1.0, 0.0),  # the relative allowance rstol s_i decides
            (1.1 * atol * 2.0, 1e-6, 2.0, 0.0),  # small forces: the absolute allowance atol m_i
            (0.5 * kl * 1e-3, 1e-3, 1.0, 1e-3),  # at the float32 floor: kl f_i
            (1e-9, 0.0, 1e-3, 0.0),  # no force, no floor: atol m_i keeps the ratio finite
        ]
        q = wp.zeros(len(cases), dtype=wp.float32)
        wp.launch(k, dim=len(cases), inputs=[wp.array(np.array(cases, dtype=np.float32)), rstol, atol, kl, q])
        np.testing.assert_allclose(q.numpy(), [0.9, 1.1, 0.5, 1e-9 / (atol * 1e-3)], rtol=1e-5)

    def test_decision(self):
        wp.init()

        @wp.kernel
        def k(
            sums: wp.array2d[wp.float32],
            free_mass: wp.array[wp.float32],
            floor_kappa2: wp.float32,
            converged: wp.array[wp.int32],
            ratio: wp.array[wp.float32],
            row_out: wp.array[SolveStatus],
        ):
            i = wp.tid()
            si = wp.array(ptr=sums.ptr + wp.uint64(i * N_SUMS * 4), shape=(N_SUMS,), dtype=wp.float32)
            # The absolute-test inputs (gnorm2, gtol2) are unused by the stationarity branch
            converged[i] = wp.where(_is_converged(si, wp.float32(0), si, free_mass, 1, floor_kappa2), 1, 0)
            ratio[i] = _gradient_ratio(si, wp.float32(0), si, free_mass, 1)
            row_out[i] = _with_node_ratios(SolveStatus(), _node_ratios(si, free_mass, 1, floor_kappa2))

        # Free mass 2 kg, so each sum m_i v_i^2 is 2 RMS_M(v)^2; RMS_M(r) and RMS_M(phi) in the gradient's units per
        # mass, so sum |g_i|^2 / m_i = 2 r^2 and sum f_i^2 / m_i = 2 F^2
        kg = 0.2
        cases = [  # (max_i q_i, RMS_M(q), RMS_M(q0), r, F)
            (3.0, 0.9, 0.95, 1.0, 1.0),  # within the allowances without the floor: passes, whatever the largest ratio
            (0.9, 0.5, 2.0, 0.15, 1.0),  # needs the floor, residual near it overall (r <= kg F): passes
            (0.9, 0.5, 2.0, 0.25, 1.0),  # needs the floor, residual above kg F: fails
            (1.5, 1.1, 1.1, 0.1, 1.0),  # beyond the allowances with the floor: fails
        ]
        qmax, q, q0, r, F = (np.array(c, dtype=np.float64) for c in zip(*cases))
        n = len(cases)
        sums = np.zeros((n, N_SUMS), dtype=np.float32)
        sums[:, SUM_RATIO_MAX], sums[:, SUM_RATIO], sums[:, SUM_RATIO_NO_FLOOR] = qmax, 2 * q**2, 2 * q0**2
        sums[:, SUM_RESIDUAL], sums[:, SUM_FLOOR] = 2 * r**2, 2 * F**2
        converged, ratio = wp.zeros(n, dtype=wp.int32), wp.zeros(n, dtype=wp.float32)
        rows = wp.zeros(n, dtype=SolveStatus)
        free_mass = wp.array(np.array([2.0], dtype=np.float32))
        wp.launch(k, dim=n, inputs=[wp.array(sums), free_mass, kg**2, converged, ratio, rows])
        self.assertEqual(converged.numpy().tolist(), [1, 1, 0, 0])
        np.testing.assert_allclose(ratio.numpy(), q, rtol=1e-6)
        out = rows.numpy()
        np.testing.assert_allclose(out["max_ratio"], qmax, rtol=1e-6)
        np.testing.assert_allclose(out["no_floor_ratio"], q0, rtol=1e-6)
        np.testing.assert_allclose(out["floor_clause_ratio"], r / (kg * F), rtol=1e-6)
        # kg = 0: the floor never decides
        wp.launch(k, dim=n, inputs=[wp.array(sums), free_mass, 0.0, converged, ratio, rows])
        self.assertEqual(converged.numpy().tolist(), [1, 0, 0, 0])
        # NodeNorm.MAX: the reductions are maxima per unit mass (max q_i^2, max q0_i^2, max |g_i|^2 / m_i^2,
        # max f_i^2 / m_i^2) and the norm weight is 1, so the same cases decide alike and record the maxima
        cases = [  # (max_i q_i, max_i q0_i, max_i |r_i|, max_i phi_i)
            (0.9, 0.95, 1.0, 1.0),  # within the allowances without the floor: passes
            (0.9, 2.0, 0.15, 1.0),  # needs the floor, residual near the largest floor (r <= kg F): passes
            (0.9, 2.0, 0.25, 1.0),  # needs the floor, residual above kg F: fails
            (1.1, 1.1, 0.1, 1.0),  # a node beyond its allowance with the floor: fails
        ]
        q, q0, r, F = (np.array(c, dtype=np.float64) for c in zip(*cases))
        sums = np.zeros((n, N_SUMS), dtype=np.float32)
        sums[:, SUM_RATIO_MAX], sums[:, SUM_RATIO], sums[:, SUM_RATIO_NO_FLOOR] = q, q**2, q0**2
        sums[:, SUM_RESIDUAL], sums[:, SUM_FLOOR] = r**2, F**2
        unit = wp.array(np.array([1.0], dtype=np.float32))
        wp.launch(k, dim=n, inputs=[wp.array(sums), unit, kg**2, converged, ratio, rows])
        self.assertEqual(converged.numpy().tolist(), [1, 1, 0, 0])
        out = rows.numpy()
        np.testing.assert_allclose(ratio.numpy(), q, rtol=1e-6)
        np.testing.assert_allclose(out["max_ratio"], q, rtol=1e-6)
        np.testing.assert_allclose(out["no_floor_ratio"], q0, rtol=1e-6)
        np.testing.assert_allclose(out["floor_clause_ratio"], r / (kg * F), rtol=1e-6)

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
        self.assertAlmostEqual(absolute_stationarity_tolerance(astol, 9.81, 1e-2), 0.1 * 9.81e-4)
        tol = [absolute_stationarity_tolerance(astol, 9.81, h) for h in (1e-2, 2e-3)]
        self.assertAlmostEqual(tol[0] / tol[1], 25.0)
        # Below one step's gravity sag h^2 |a| at any h, so a step that hasn't moved can't pass
        for h in (1e-2, 2e-3, 1e-4):
            self.assertLess(absolute_stationarity_tolerance(astol, 9.81, h), 9.81 * h * h)
        # astol = 0 (the relative test alone): still positive, so a node without forces has a finite ratio
        self.assertEqual(absolute_stationarity_tolerance(0.0, 9.81, 1e-2), ACCELERATION_FLOOR * 1e-4)


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
    # No relative allowances, so that the recorded gradient ratios are the residuals over the absolute tolerance (or the
    # float32 floor)
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


class TestNodeNorm(unittest.TestCase):
    """``node_norm`` MAX measures the same node quantities as the RMS and takes the largest over free nodes in every
    term: max q_i, max q0_i and max |r_i| / (kg max phi_i), recorded as the status's ratios; the device decision, the
    host's (GPU Newton's) and ``status_ratio.stationarity_test_ratio`` agree. Two resting cubes (VBD), measured
    at the same positions with both norms."""

    def test_max_norm(self):
        from ...status_ratio import stationarity_test_ratio

        state = _two_resting_cubes("VBD")
        check = state.convergence
        params = check.params
        params.rstol, params.astol = 2e-2, 0.1
        kg = float(params.floor_kappa)
        self.assertEqual(params.node_norm, NodeNorm.RMS)  # the default
        check.measure_scale()
        check._test()
        terms = check._terms.numpy().astype(np.float64)
        m = state.fem.data.m.numpy().astype(np.float64)
        free = (state.fem.data.dmask.numpy() == 0) & (m > 0)
        q2, q02 = terms[SUM_RATIO][free] / m[free], terms[SUM_RATIO_NO_FLOOR][free] / m[free]
        r2, f2 = terms[SUM_RESIDUAL][free] / m[free], terms[SUM_FLOOR][free] / m[free]
        rms_ratio, rms_passes = check.gradient_ratio(), check._relative_passes()
        self.assertGreater(float(np.max(f2)), 0.0)  # VBD's floor is on
        params.node_norm = NodeNorm.MAX
        check._test()
        expected = [np.sqrt(np.max(q2)), np.sqrt(np.max(q02)), np.sqrt(np.max(r2) / (kg**2 * np.max(f2)))]
        np.testing.assert_allclose(check.gradient_ratio(), expected[0], rtol=1e-5)
        self.assertEqual(check.gradient_ratio(), check.max_ratio())
        self.assertGreater(check.gradient_ratio(), rms_ratio)  # the max is at least the RMS
        # The device's decision and status row, from the same reductions
        check.penetration.zero_()
        check.k.zero_()
        wp.launch(_end_outer_kernel, dim=1, inputs=check._outer_test_inputs(1.0, 1) + [
            check.status, check.solves, 0, 0, 0, 0, check.blocks, check.subproblem_keep])
        row = check.status.numpy()[(int(check.solves.numpy()[0]) - 1) % check.status.shape[0]]
        np.testing.assert_allclose(
            [row["gradient_ratio"], row["no_floor_ratio"], row["floor_clause_ratio"]], expected, rtol=1e-5)
        passes = check._relative_passes()
        self.assertEqual(int(row["reason"]) == int(SolveEnd.CONVERGED), passes)
        self.assertEqual(stationarity_test_ratio(row) <= 1.0, passes)
        # Back to the RMS: the same measure as before
        params.node_norm = NodeNorm.RMS
        check._test()
        self.assertEqual(check.gradient_ratio(), rms_ratio)
        self.assertEqual(check._relative_passes(), rms_passes)


def _smooth_reference(u: np.ndarray, m: np.ndarray, GVVp: np.ndarray, GVVadj: np.ndarray, k: int) -> np.ndarray:
    """float64 reference of k :func:`_smoothing_pass` passes of ``u`` (gradient units), ``m`` the free masses."""
    u = np.asarray(u, dtype=np.float64)
    for _ in range(k):
        out = np.zeros_like(u)
        for i in np.flatnonzero(m > 0):
            ring = [j for j in GVVadj[GVVp[i] : GVVp[i + 1]] if j != i and m[j] > 0]
            out[i] = m[i] * (u[i] + u[ring].sum(axis=0)) / (m[i] + m[ring].sum())
        u = out
    return u


class TestResidualSmoothing(unittest.TestCase):
    """``smoothing_passes`` k: the stationarity test of the smoothed residual r_bar_k against the smoothed float32
    floor (:class:`ConvergenceParams`). Two resting cubes, the lower one fixed."""

    def _state(self, solver_name: str = "VBD"):
        state = _two_resting_cubes(solver_name)
        check = state.convergence
        check.params.rstol, check.params.astol = 2e-2, 0.1
        return state, check

    def test_k0_adds_no_work(self):
        # k = 0 (the default): the test launches exactly what it launched before smoothing existed and measures the
        # same values, bit for bit, as the unsmoothed kernel launched directly; no smoothing buffer is allocated
        state, check = self._state()
        self.assertEqual(ConvergenceParams().smoothing_passes, 0)
        smoothing = {_stationarity_ratios_smoothed, _smoothing_pass, _perturb_positions, _subtract, _floor_from_probe,
                     _keep_stricter_sums}
        launched = []
        launch = wp.launch

        def recording_launch(kernel, *args, **kwargs):
            launched.append(kernel)
            return launch(kernel, *args, **kwargs)

        wp.launch = recording_launch
        try:
            check.measure_scale()
            check._test()
            check._test_subproblem()
            check._test()
        finally:
            wp.launch = launch
        self.assertFalse(smoothing & set(launched))
        self.assertEqual(launched.count(_stationarity_ratios), 3)
        self.assertEqual(launched.count(_elastic_gradient), 3)  # the tests' gradients, no probe
        self.assertIsNone(check._smooth)
        terms, sums = check._terms.numpy().copy(), check.sums.numpy().copy()
        # The unsmoothed kernel and reductions, launched directly on the same gradient
        gradient, hnorm = check.gradient_norm, check._hessian_norms
        wp.launch(_stationarity_ratios, dim=terms.shape[1], inputs=[
            gradient.g, gradient.scale, hnorm, 1, float(check.params.rstol), check.absolute_tolerance,
            float(check.params.floor_local_kappa), int(check.node_norm.value), state.fem.data, check._terms])
        for reduce in check._reduce_sums[check.node_norm]:
            reduce(wp.get_stream())
        np.testing.assert_array_equal(check._terms.numpy(), terms)
        np.testing.assert_array_equal(check.sums.numpy(), sums)
        # A whole captured step: no smoothing buffer either
        state.step()
        self.assertIsNone(check._smooth)

    def test_adjacency(self):
        # The fallback 1-rings equal the VBD params' (pbat's vertex_colors), each vertex listing itself
        state, check = self._state()
        vbd = state.params[state.solver].data
        GVVp, GVVadj = _vertex_adjacency(state.fem.data.E.numpy().reshape(-1, 4), state.fem.data.x.shape[0])
        np.testing.assert_array_equal(GVVp, vbd.GVVp.numpy())
        np.testing.assert_array_equal(GVVadj, vbd.GVVadj.numpy())

    def test_filter(self):
        # k passes of mass-weighted averaging over closed 1-rings of free nodes, in the gradient's units: as a float64
        # reference; a uniform acceleration stays uniform; random-sign node noise averages away; fixed nodes get 0
        state, check = self._state()
        check.params.smoothing_passes = 3
        check.measure_scale()
        check.prepare_smoothing()
        m = check._masses.numpy().astype(np.float64)
        GVVp, GVVadj = (a.numpy() for a in check._vertex_adjacency)
        free = m > 0
        rng = np.random.default_rng(5)
        u = (m[:, None] * rng.standard_normal((m.shape[0], 3))).astype(np.float32)
        out = check._smooth_field(wp.array(u, dtype=wp.vec3f), 3).numpy()
        np.testing.assert_allclose(out, _smooth_reference(u, m, GVVp, GVVadj, 3), rtol=1e-5, atol=1e-6 * np.abs(u).max())
        self.assertTrue(np.all(out[~free] == 0.0))
        a = np.array([0.3, -2.0, 9.81])
        uniform = check._smooth_field(wp.array((m[:, None] * a).astype(np.float32), dtype=wp.vec3f), 3).numpy()
        np.testing.assert_allclose(uniform[free] / m[free, None], np.tile(a, (int(free.sum()), 1)), rtol=1e-6)

        def rms(v):
            return np.sqrt(np.sum(np.sum((v[free] / m[free, None]) ** 2, axis=1) * m[free]) / m[free].sum())

        self.assertLess(rms(out), 0.6 * rms(u))

    def test_floor_probe(self):
        # The probe moves every nonzero coordinate by exactly one float32 spacing (zeros stay), and the smoothed floor
        # is positive on free nodes, 0 on fixed ones, and below the unsmoothed floor phi
        state, check = self._state()
        check.params.smoothing_passes = 4
        check.measure_scale()
        x, xp = state.fem.data.x.numpy(), check._probe_x.numpy()
        nonzero = x != 0
        np.testing.assert_array_equal(np.abs(xp - x)[nonzero], np.spacing(np.abs(x[nonzero])))
        np.testing.assert_array_equal(xp[~nonzero], x[~nonzero])
        floor = check._smoothed_floor.numpy()
        m = check._masses.numpy()
        phi = check._hessian_norms.numpy() * EPS32 * np.linalg.norm(x, axis=1)
        self.assertTrue(np.all(floor[m > 0] > 0.0) and np.all(floor[m == 0] == 0.0))
        self.assertTrue(np.all(floor[m > 0] < phi[m > 0]))
        # Without the floor (GPU Newton), no probe
        check._smoothed_floor.zero_()
        check.set_hessian_norms(None)
        check.measure_scale()
        self.assertTrue(np.all(check._smoothed_floor.numpy() == 0.0))

    def test_smoothed_test(self):
        # k = 4: the test passes only if the unsmoothed test and the smoothed one (the smoothed residual against the
        # smoothed floor) both pass, and records the ratios of the one with the larger decisive ratio (float64
        # references from the device's gradient, force scale and floors), under both norms; the device's decision,
        # the host's and status_ratio.stationarity_test_ratio agree; the test leaves the gradient as it was
        from ...status_ratio import stationarity_test_ratio

        state, check = self._state()
        params = check.params
        params.smoothing_passes = 4
        kl, kg = float(params.floor_local_kappa), float(params.floor_kappa)
        check.measure_scale()
        for norm in NodeNorm:
            with self.subTest(norm=norm.name):
                params.node_norm = norm
                check._test()
                g = check.gradient_norm.g.numpy()
                check._measure_residual(check.gradient_norm)
                np.testing.assert_array_equal(check.gradient_norm.g.numpy(), g)  # g isn't smoothed in place
                m = check._masses.numpy().astype(np.float64)
                free = m > 0
                GVVp, GVVadj = (a.numpy() for a in check._vertex_adjacency)
                s, mf = check.gradient_norm.scale.numpy()[free], m[free]
                allowance = np.maximum(float(params.rstol) * s, check.absolute_tolerance * mf)

                def ratios(r, f):  # (||q||, ||q0||, floor clause), max q, decisive; r, f in the gradient's units
                    q, q0 = r / np.maximum(allowance, kl * f), r / allowance
                    if norm == NodeNorm.RMS:
                        out = [np.sqrt(np.sum(mf * q**2) / mf.sum()), np.sqrt(np.sum(mf * q0**2) / mf.sum()),
                               np.sqrt(np.sum(r**2 / mf) / (kg**2 * np.sum(f**2 / mf)))]
                    else:
                        out = [q.max(), q0.max(), np.sqrt(np.max(r**2 / mf**2) / (kg**2 * np.max(f**2 / mf**2)))]
                    return out, q.max(), max(out[0], min(out[1], out[2]))

                x = state.fem.data.x.numpy()
                phi = (check._hessian_norms.numpy() * EPS32 * np.linalg.norm(x, axis=1))[free]
                unsmoothed = ratios(np.linalg.norm(g, axis=1)[free], phi)
                smoothed = ratios(np.linalg.norm(_smooth_reference(g, m, GVVp, GVVadj, 4), axis=1)[free],
                                  check._smoothed_floor.numpy()[free])
                expected = max(unsmoothed, smoothed, key=lambda e: e[2])
                check.penetration.zero_()
                check.k.zero_()
                wp.launch(_end_outer_kernel, dim=1, inputs=check._outer_test_inputs(1.0, 1) + [
                    check.status, check.solves, 0, 0, 0, 0, check.blocks, check.subproblem_keep])
                row = check.status.numpy()[(int(check.solves.numpy()[0]) - 1) % check.status.shape[0]]
                np.testing.assert_allclose(
                    [row["gradient_ratio"], row["no_floor_ratio"], row["floor_clause_ratio"]], expected[0], rtol=1e-4)
                np.testing.assert_allclose(row["max_ratio"], expected[1], rtol=1e-4)
                np.testing.assert_allclose(stationarity_test_ratio(row), max(unsmoothed[2], smoothed[2]), rtol=1e-4)
                passes = check._relative_passes()
                self.assertEqual(int(row["reason"]) == int(SolveEnd.CONVERGED), passes)
                self.assertEqual(stationarity_test_ratio(row) <= 1.0, passes)
        params.node_norm = NodeNorm.RMS

    def test_keep_stricter(self):
        # The combined decision passes iff both tests pass, and keeps the stricter test's reductions
        wp.init()
        kg = 0.2

        def sums(q, q0, r, F):  # one free node of unit mass: the RMS sums are the squares
            out = np.zeros(N_SUMS, dtype=np.float32)
            out[SUM_RATIO_MAX], out[SUM_RATIO], out[SUM_RATIO_NO_FLOOR] = q, q**2, q0**2
            out[SUM_RESIDUAL], out[SUM_FLOOR] = r**2, F**2
            return out

        passing, failing, worse = sums(0.5, 0.9, 1.0, 1.0), sums(0.5, 2.0, 0.5, 1.0), sums(0.5, 3.0, 0.9, 1.0)
        weight = wp.array(np.ones(1, dtype=np.float32))
        for unsmoothed, smoothed, kept in [(passing, failing, failing), (failing, passing, failing),
                                           (failing, worse, worse), (worse, failing, worse), (passing, passing, passing)]:
            out = wp.array(smoothed.copy())
            wp.launch(_keep_stricter_sums, dim=1, inputs=[wp.array(unsmoothed.copy()), weight, kg**2, out])
            np.testing.assert_array_equal(out.numpy(), kept)

    def test_steps(self):
        # Whole steps with k = 4: VBD-family solvers under graph capture, and GPU Newton (no floor, so its floor clause
        # never decides), record finite ratios
        for solver_name in ("VBD", "AAAVBD", "Newton"):
            with self.subTest(solver=solver_name):
                state, check = self._state(solver_name)
                check.params.smoothing_passes = 4
                state.capture = None
                state.step()
                state.step()
                if solver_name == "Newton":
                    ratios = state.params[state.solver].last_outer_ratios
                    self.assertTrue(np.all(np.isfinite(ratios)))
                else:
                    self.assertIsNotNone(check._smooth)
                    row = check.status.numpy()[(int(check.solves.numpy()[0]) - 1) % check.status.shape[0]]
                    self.assertTrue(np.isfinite(float(row["gradient_ratio"])))
                    self.assertGreater(float(row["floor_clause_ratio"]), 0.0)


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
        self.assertTrue(np.all(s1 > 9.81 * (1 - 1e-3)))  # the largest of |a| (inertial, external) and the elastic force
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
            inner["ratio"], inner["max"] = check.gradient_ratio(), check.max_ratio()

        check.after_block = hooked_after_block
        state._advance()
        check.enqueue_status_readback()
        wp.synchronize()
        row = check.read_status()[0][-1]
        self.assertIsNotNone(check._hessian_norms)  # VBD stores its vertex Hessian norms for the floor
        # Measured: equal to the last bit, with mu_f 0.2 and 0 (2026-11-06)
        np.testing.assert_allclose(float(row["gradient_ratio"]), inner["ratio"], rtol=1e-3)
        np.testing.assert_allclose(float(row["max_ratio"]), inner["max"], rtol=1e-3)


if __name__ == "__main__":
    unittest.main()

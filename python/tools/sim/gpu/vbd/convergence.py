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
"""

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
    or, with ``rtol = 0``, each solver's absolute ``gtol`` (``Solver.<solver>.gtol``).

    Relative test (``rtol > 0``): the gradient g of the time step's Lagrangian is turned into a
    position-like residual by the inverse lumped mass, M^-1 g (the correction an inertia-only Newton step
    would make), and compared with the step's free-flight displacement d = x~ - x^t (h v^t + h^2 a_ext
    for BDF1): ``RMS_M(M^-1 g) <= rtol * RMS_M(d)``, with mass-weighted RMS over free nodes, i.e.
    ``sum |g_i|^2 / m_i <= rtol^2 * sum m_i |d_i|^2``. It doesn't depend on the mesh's size or mass. When
    d is 0 (a body at rest without loads) or for static problems, the absolute ``gtol`` test applies.
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


@wp.func
def _is_converged(
    gnorm2: wp.array[wp.float32],
    gtol2: wp.float32,
    gm2: wp.array[wp.float32],
    dm2: wp.array[wp.float32],
    rtol2: wp.float32,
) -> bool:
    """Relative test ``sum |g|^2/m <= rtol^2 sum m |d|^2`` if enabled and d != 0, else ``|g|^2 <= gtol^2``."""
    if rtol2 > wp.float32(0) and dm2[0] > wp.float32(0):
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
def _free_flight_displacement_terms(fem: FemElastoDynamicsData, terms: wp.array[wp.float32]):
    i = wp.tid()
    terms[i] = wp.float32(0)
    if fem.is_static == 0 and not is_dirichlet_node(fem.dmask, i):
        d = fem.xtilde[i] - fem.xt[i]
        terms[i] = fem.m[i] * wp.dot(d, d)


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
):
    k[0] = 0
    done = _is_converged(gnorm2, gtol2, gm2, dm2, rtol2) and _is_feasible(penetration, ftol, contact.dmin)
    keep[0] = wp.where(n_max_iters > 0 and not done, 1, 0)


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
):
    kk = k[0] + 1
    k[0] = kk
    done = _is_converged(gnorm2, gtol2, gm2, dm2, rtol2) and _is_feasible(penetration, ftol, contact.dmin)
    keep[0] = wp.where(kk < n_max_iters and not done, 1, 0)


class ConvergenceCheck:
    """Outer-iteration counter and loop flags of a solve, on the device.

    Flags (each a (1,) int32 array, read by conditional graph nodes):

    - ``keep``: the step goes on (not both optimal and feasible, and outer iterations left).
    - ``active``: the current outer iteration runs (``keep`` when it started).
    - ``inner_keep``: the current subproblem's sweep blocks go on.
    - ``subproblem_keep``: the current subproblem's remaining sweeps run (not converged inside it).

    ``k`` counts the outer iterations done, ``penetration`` holds the last feasibility measure.
    :meth:`converged` reads the last gradient test on the host.
    """

    def __init__(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        params: ConvergenceParams | None = None,
    ):
        """
        Args:
            fem (FemElastoDynamics): The problem whose gradient is tested.
            contact (ContactDynamics): Its contact model (multipliers for the gradient, contact sets
                for the feasibility test).
            params (ConvergenceParams | None): Check interval and feasibility tolerance; defaults if None.
        """
        self.params = params or ConvergenceParams()
        self._fem = fem
        self._contact = contact
        self.gradient_norm = GradientNorm(fem, contact, augmented_lagrangian=False)
        # Relative test (params.rtol > 0): sum |g_i|^2 / m_i and sum m_i |d_i|^2 over free nodes
        n_nodes = fem.data.x.shape[0]
        self._terms = wp.zeros(n_nodes, dtype=wp.float32)
        self.residual_m2 = wp.zeros(1, dtype=wp.float32)
        self.displacement_m2 = wp.zeros(1, dtype=wp.float32)
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
        self.penetration = wp.zeros(1, dtype=wp.float32)
        self.k = wp.zeros(1, dtype=wp.int32)
        self.keep = wp.zeros(1, dtype=wp.int32)
        self.active = wp.zeros(1, dtype=wp.int32)
        self.blocks = wp.zeros(1, dtype=wp.int32)
        self.inner_keep = wp.zeros(1, dtype=wp.int32)
        self.subproblem_keep = wp.zeros(1, dtype=wp.int32)

    @property
    def _relative(self) -> bool:
        return float(self.params.rtol) > 0.0

    def _test(self) -> None:
        stream = wp.get_stream()
        self.gradient_norm.compute(stream)
        if self._relative:
            wp.launch(
                _mass_normalized_residual_terms,
                dim=self._terms.shape[0],
                inputs=[self.gradient_norm.g, self._fem.data, self._terms],
            )
            self._reduce_residual(stream)

    def _measure_displacement(self) -> None:
        """The step's sum m_i |x~_i - x^t_i|^2 over free nodes, the relative test's scale."""
        if self._relative:
            wp.launch(
                _free_flight_displacement_terms,
                dim=self._terms.shape[0],
                inputs=[self._fem.data, self._terms],
            )
            self._reduce_displacement(wp.get_stream())

    def _tolerance_inputs(self, gtol: float) -> list:
        return [
            self.gradient_norm.gnorm2,
            float(gtol) ** 2,
            self.residual_m2,
            self.displacement_m2,
            float(self.params.rtol) ** 2 if self._relative else 0.0,
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
        self._measure_displacement()
        self._test()
        self._measure_penetration()
        wp.launch(_start_kernel, dim=1, inputs=self._outer_test_inputs(gtol, n_max_iters))

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

    def end_outer(self, gtol: float, n_max_iters: int) -> None:
        """Test after an outer iteration (C++'s check before the next one, plus feasibility):
        ``k += 1``, update ``keep``.

        Args:
            gtol (float): Gradient-norm tolerance (absolute).
            n_max_iters (int): Maximum outer iterations.
        """
        self._test()
        self._measure_penetration()
        wp.launch(_end_outer_kernel, dim=1, inputs=self._outer_test_inputs(gtol, n_max_iters))

    def converged(self, gtol: float) -> bool:
        """Host-side gradient test from the last test (relative or absolute, see
        :class:`ConvergenceParams`). Synchronizes, so not capturable.

        Args:
            gtol (float): Gradient-norm tolerance (absolute).

        Returns:
            bool: Whether the last gradient test passed (feasibility isn't included).
        """
        dm2 = float(self.displacement_m2.numpy()[0])
        if self._relative and dm2 > 0.0:
            return float(self.residual_m2.numpy()[0]) <= float(self.params.rtol) ** 2 * dm2
        return float(self.gradient_norm.gnorm2.numpy()[0]) <= float(gtol) * float(gtol)

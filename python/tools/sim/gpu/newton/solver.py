"""
Newton solver for FEM elasto-dynamics with contact (augmented Lagrangian).

Outer loop mirrors ``gpu/vbd/solver.py``.  Newton-specific work lives in
``prepare_subproblem`` and ``solve_subproblem``
"""

import cuda.compute
import warp as wp
import warp.sparse
import warp.optim.linear
import cupy as cp

from pbatoolkit import pbat
from ..common.reduce import Reduce
from ..contact.mesh.cd import ContactDetection
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from ..contact.dynamics import (
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
    PenaltyAdaptivity,
)
from ..elasticity.snh import snh_eval, snh_hess
from ..elasticity.chain import hessian_wrt_dofs
from ..gradient import Gradient
from ..contact import halfedges
from .. import types


@wp.struct
class ParamsData:
    """GPU-side Newton solver parameters."""

    # Outer AL iteration control
    n_max_iters: wp.int32
    k: wp.int32

    # Inner Newton iteration control
    n_subproblem_max_iters: wp.int32
    gtol2: wp.float32  # squared gradient-norm convergence threshold
    n_lin_max_iters: wp.int32
    rel_eps_lin: wp.float32
    abs_eps_lin: wp.float32

    # Backtracking line search (Armijo)
    ls_max_iters: wp.int32
    ls_tau: wp.float32  # step shrink factor
    ls_c: wp.float32  # Armijo slope constant
    ls_alpha: wp.float32  # initial step size

    # Vertex-element adjacency (for elastic gradient / Hessian)
    # GVGp: wp.array[wp.int32]
    # GVGadj: wp.array[wp.int32]

    # BSR Hessian 3x3 block triplets
    Hrows: wp.array[wp.int32]
    Hcols: wp.array[wp.int32]
    Hvals: wp.array[wp.mat33f]


@wp.kernel
def _energy_elastic(
    fem: FemElastoDynamicsData,  # type: ignore
    h2: wp.float32,
    energy: wp.array[wp.float32],
):
    e = wp.tid()
    nodes = fem.E[e]
    wge = fem.wg[e]
    GP = fem.GNeg[e]
    mu = fem.mug[e]
    llambda = fem.lambdag[e]
    xe = types.mat3x4f()
    for j in range(4):
        xj = fem.x[nodes[j]]
        for d in range(3):
            xe[d, j] = xj[d]
    F = xe @ GP
    Ue = h2 * wge * snh_eval(F, mu, llambda)
    # TODO: Add into a global array
    # of size (# elems,) without atomics, then
    # compute a global reduction using our Reduce
    # type after this kernel has been invoked.
    wp.atomic_add(energy, 0, Ue)


@wp.kernel
def _energy_inertial(
    fem: FemElastoDynamicsData,  # type: ignore
    energy: wp.array[wp.float32],
):
    i = wp.tid()
    if is_dirichlet_node(fem.dmask, i):
        return
    diff = fem.x[i] - fem.xtilde[i]
    # TODO: The inertial energy can be computed using a single
    # transform+reduce kernel using cuda.compute and our Reduce type.
    wp.atomic_add(energy, 0, wp.float32(0.5) * fem.m[i] * wp.dot(diff, diff))


@wp.kernel
def _energy_vv(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    energy: wp.array[wp.float32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vv.prefix[n_u]:
        return
    u, v = contact.contacts.vv.u[c], contact.contacts.vv.v[c]
    n, t, bvec = (
        contact.contacts.vv_bases.n[c],
        contact.contacts.vv_bases.t[c],
        contact.contacts.vv_bases.b[c],
    )
    i, j = contact.meshes.V[u], contact.meshes.V[v]
    xi, xj, xti, xtj = x[i], x[j], xt[i], xt[j]
    c_n = wp.dot(xi - xj, n) - contact.dmin - contact.cvv.s[c]
    du = (xi - xti) - (xj - xtj)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, bvec))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    E = contact.cvv.gamma[c] * (
        wp.float32(0.5) * sigma_n * c_n * c_n
        - contact.cvv.lambda_n[c] * c_n
        + wp.float32(0.5) * sigma_f * wp.dot(c_f, c_f)
        - wp.dot(contact.cvv.lambda_f[c], c_f)
    )
    wp.atomic_add(energy, 0, E)


@wp.kernel
def _energy_ve(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    contact: ContactDynamicsData,
    n_u: wp.int32,
    energy: wp.array[wp.float32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ve.prefix[n_u]:
        return
    vi = contact.contacts.ve.u[c]
    he = contact.contacts.ve.v[c]
    n, t, bvec = (
        contact.contacts.ve_bases.n[c],
        contact.contacts.ve_bases.t[c],
        contact.contacts.ve_bases.b[c],
    )
    b1 = contact.contacts.ve_bary[c]
    b0 = wp.float32(1) - b1
    i = contact.meshes.V[vi]
    ea = halfedges.incoming_vertex(contact.meshes.F, he)
    eb = halfedges.outgoing_vertex(contact.meshes.F, he)
    xi, xti = x[i], xt[i]
    xcp2 = b0 * x[ea] + b1 * x[eb]
    xtcp2 = b0 * xt[ea] + b1 * xt[eb]
    c_n = wp.dot(xi - xcp2, n) - contact.dmin - contact.cve.s[c]
    du = (xi - xti) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, bvec))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    E = contact.cve.gamma[c] * (
        wp.float32(0.5) * sigma_n * c_n * c_n
        - contact.cve.lambda_n[c] * c_n
        + wp.float32(0.5) * sigma_f * wp.dot(c_f, c_f)
        - wp.dot(contact.cve.lambda_f[c], c_f)
    )
    wp.atomic_add(energy, 0, E)


@wp.kernel
def _energy_vf(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    contact: ContactDynamicsData,
    n_u: wp.int32,
    energy: wp.array[wp.float32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vf.prefix[n_u]:
        return
    vi = contact.contacts.vf.u[c]
    f = contact.contacts.vf.v[c]
    n, t, bvec = (
        contact.contacts.vf_bases.n[c],
        contact.contacts.vf_bases.t[c],
        contact.contacts.vf_bases.b[c],
    )
    uv = contact.contacts.vf_bary[c]
    b1, b2 = uv[0], uv[1]
    b0 = wp.float32(1) - b1 - b2
    i = contact.meshes.V[vi]
    finds = contact.meshes.F[f]
    xi, xti = x[i], xt[i]
    xcp2 = b0 * x[finds[0]] + b1 * x[finds[1]] + b2 * x[finds[2]]
    xtcp2 = b0 * xt[finds[0]] + b1 * xt[finds[1]] + b2 * xt[finds[2]]
    c_n = wp.dot(xi - xcp2, n) - contact.dmin - contact.cvf.s[c]
    du = (xi - xti) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, bvec))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    E = contact.cvf.gamma[c] * (
        wp.float32(0.5) * sigma_n * c_n * c_n
        - contact.cvf.lambda_n[c] * c_n
        + wp.float32(0.5) * sigma_f * wp.dot(c_f, c_f)
        - wp.dot(contact.cvf.lambda_f[c], c_f)
    )
    wp.atomic_add(energy, 0, E)


@wp.kernel
def _energy_ee(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    contact: ContactDynamicsData,
    n_u: wp.int32,
    energy: wp.array[wp.float32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ee.prefix[n_u]:
        return
    he_u = contact.contacts.ee.u[c]
    he_v = contact.contacts.ee.v[c]
    n, t, bvec = (
        contact.contacts.ee_bases.n[c],
        contact.contacts.ee_bases.t[c],
        contact.contacts.ee_bases.b[c],
    )
    st = contact.contacts.ee_bary[c]
    b1, b3 = st[0], st[1]
    b0, b2 = wp.float32(1) - b1, wp.float32(1) - b3
    ia = halfedges.incoming_vertex(contact.meshes.F, he_u)
    ib = halfedges.outgoing_vertex(contact.meshes.F, he_u)
    ic = halfedges.incoming_vertex(contact.meshes.F, he_v)
    id_ = halfedges.outgoing_vertex(contact.meshes.F, he_v)
    xcp1 = b0 * x[ia] + b1 * x[ib]
    xtcp1 = b0 * xt[ia] + b1 * xt[ib]
    xcp2 = b2 * x[ic] + b3 * x[id_]
    xtcp2 = b2 * xt[ic] + b3 * xt[id_]
    c_n = wp.dot(xcp1 - xcp2, n) - contact.dmin - contact.cee.s[c]
    du = (xcp1 - xtcp1) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, bvec))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    E = contact.cee.gamma[c] * (
        wp.float32(0.5) * sigma_n * c_n * c_n
        - contact.cee.lambda_n[c] * c_n
        + wp.float32(0.5) * sigma_f * wp.dot(c_f, c_f)
        - wp.dot(contact.cee.lambda_f[c], c_f)
    )
    wp.atomic_add(energy, 0, E)


@wp.kernel
def _compute_hessian_triplets(
    fem: FemElastoDynamicsData,  # type: ignore
    h2: wp.float32,
    params: ParamsData,  # type: ignore
):
    tid = wp.tid()
    n_nodes = fem.x.shape[0]
    n_elems = fem.E.shape[0]
    identity = wp.identity(n=3, dtype=wp.float32)  # type: ignore
    if tid < n_nodes:
        i = tid
        if is_dirichlet_node(fem.dmask, i):  # type: ignore
            params.Hvals[i] = identity
        else:
            params.Hvals[i] = fem.m[i] * identity
    if tid < n_elems:
        e = tid
        offset = n_nodes
        nodes = fem.E[e]
        wge = fem.wg[e]
        GP = fem.GNeg[e]
        mu = fem.mug[e]
        llambda = fem.lambdag[e]
        xe = types.mat3x4f()
        for j in range(4):
            xj = fem.x[nodes[j]]
            for d in range(3):
                xe[d, j] = xj[d]
        F = xe @ GP
        HF = snh_hess(F, mu, llambda)
        He = hessian_wrt_dofs(HF, GP)
        for ii in range(4):
            for jj in range(4):
                idx = offset + e * 16 + ii * 4 + jj
                if is_dirichlet_node(fem.dmask, nodes[ii]) or is_dirichlet_node(
                    fem.dmask, nodes[jj]
                ):
                    params.Hvals[idx] = wp.mat33f()
                else:
                    Heij = He[(ii) * 3 : (ii + 1) * 3, (jj) * 3 : (jj + 1) * 3]
                    params.Hvals[idx] = h2 * wge * Heij


@wp.kernel
def _axpy(y: wp.array[wp.vec3f], x: wp.array[wp.vec3f], alpha: wp.float32):
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]


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
        self._data.n_lin_max_iters = 300
        self._data.rel_eps_lin = 1e-5
        self._data.abs_eps_lin = 1e-5
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
        self._data.GVGp = wp.zeros(1, dtype=wp.int32)
        self._data.GVGadj = wp.zeros(1, dtype=wp.int32)
        # Create template hessian triplets
        wp.launch(
            kernel=_initialize_hessian_triplets,
            dim=max(n_nodes, n_elems),
            inputs=[fem.data, self._data],
        )
        self._H = warp.sparse.bsr_from_triplets(
            n_nodes,
            n_nodes,
            self._data.Hrows,
            self._data.Hcols,
            self._data.Hvals,
            prune_numerical_zeros=False,
        )
        self._g = wp.zeros(n_nodes, dtype=wp.vec3f)
        self._dx = wp.zeros(n_nodes, dtype=wp.vec3f)
        self._energy_buf = wp.zeros(1, dtype=wp.float32)
        self._gradient = Gradient(fem, contact, self._g)
        g_flat = cp.asarray(self._g).ravel()
        dx_flat = cp.asarray(self._dx).ravel()
        self._gnorm2 = wp.zeros(1, dtype=wp.float32)
        self._slope_buf = wp.zeros(1, dtype=wp.float32)
        self._gnorm2_reduce = Reduce(
            d_in=cuda.compute.TransformIterator(g_flat, lambda x: x * x),
            d_out=cp.asarray(self._gnorm2),
            num_items=3 * n_nodes,
            op=cuda.compute.OpKind.PLUS,
        )
        self._slope_reduce = Reduce(
            d_in=cuda.compute.TransformIterator(
                cuda.compute.ZipIterator(g_flat, dx_flat),
                lambda x: x[0] * x[1],
            ),
            d_out=cp.asarray(self._slope_buf),
            num_items=3 * n_nodes,
            op=cuda.compute.OpKind.PLUS,
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


def _compute_energy(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
    h2: wp.float32,
) -> float:
    params._energy_buf.zero_()
    n_nodes = fem.data.x.shape[0]
    n_elems = fem.data.E.shape[0]
    cd = contact._data
    x, xt = fem.data.x, fem.data.xt
    wp.launch(_energy_elastic, dim=n_elems, inputs=[fem.data, h2, params._energy_buf])
    wp.launch(_energy_inertial, dim=n_nodes, inputs=[fem.data, params._energy_buf])
    for kernel, cs in [
        (_energy_vv, contact.cvv),
        (_energy_ve, contact.cve),
        (_energy_vf, contact.cvf),
        (_energy_ee, contact.cee),
    ]:
        wp.launch(
            kernel, dim=cs.capacity, inputs=[x, xt, cd, cs.n_u, params._energy_buf]
        )
    return float(params._energy_buf.numpy()[0])


def check_convergence(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> bool:
    params._gradient.compute()
    main_stream = wp.get_stream()
    params._gnorm2_reduce(main_stream)
    return float(params._gnorm2.numpy()[0]) <= float(params.data.gtol2)


def prepare_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
) -> None:
    contact.adapt_penalty_parameters()


def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> None:
    n_nodes = fem.data.x.shape[0]
    n_elems = fem.data.E.shape[0]
    h2 = fem.bdf.beta_tilde**2
    params._gradient.compute()
    main_stream = wp.get_stream()
    for _ in range(int(params.data.n_subproblem_max_iters)):
        params._gnorm2_reduce(main_stream)
        if float(params._gnorm2.numpy()[0]) <= float(params.data.gtol2):
            break
        # TODO: Assemble FEM + contact Hessian
        wp.launch(
            _compute_hessian_triplets,
            dim=max(n_nodes, n_elems),
            inputs=[fem.data, h2, params._data],
        )
        warp.sparse.bsr_set_from_triplets(
            dest=params._H,
            rows=params._data.Hrows,
            columns=params._data.Hcols,
            values=params._data.Hvals,
            count=None,
            prune_numerical_zeros=True,
            masked=False,
        )
        M = warp.optim.linear.preconditioner(params._H, "diag")
        # Solve H (-dx) = g
        params._dx.zero_()
        final_iteration, residual_norm, absolute_tolerance = warp.optim.linear.cg(
            params._H,
            params._g,
            x=params._dx,
            tol=params.data.rel_eps_lin,
            atol=params.data.abs_eps_lin,
            maxiter=params.data.n_lin_max_iters,
            M=M,
            use_cuda_graph=True,
        )
        # Check descent direction: slope = dot(g, dx) should be < 0
        params._slope_reduce(main_stream)
        if float(params._slope_buf.numpy()[0]) >= 0.0:
            break
        # Armijo backtracking line search
        E0 = _compute_energy(fem, contact, params, h2)
        alpha = float(params.data.ls_alpha)
        x_backup = wp.clone(fem.data.x)
        accepted = False
        for _ in range(int(params.data.ls_max_iters)):
            wp.copy(fem.data.x, x_backup)
            wp.launch(
                _axpy, dim=n_nodes, inputs=[fem.data.x, params._dx, wp.float32(alpha)]
            )
            E_trial = _compute_energy(fem, contact, params, h2)
            if E_trial <= E0 + float(params.data.ls_c) * alpha * slope:
                accepted = True
                break
            alpha *= float(params.data.ls_tau)
        if not accepted:
            wp.copy(fem.data.x, x_backup)
            break
        params._gradient.compute()


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

    @property
    def supports_graph_capture(self):
        return False

# type: ignore
import warp as wp

from .contact.dynamics import MeshDynamics as ContactDynamics, MeshDynamicsData
from .elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from .elasticity.snh import snh_grad
from .elasticity.chain import gradient_wrt_dofs
from .contact import halfedges
from . import types


@wp.kernel
def _elastic_gradient(
    fem: FemElastoDynamicsData,
    h2: wp.float32,
    g: wp.array[wp.vec3f],
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
    gF = snh_grad(F, mu, llambda)
    ge = h2 * wge * gradient_wrt_dofs(gF, GP)
    node0 = nodes[0]
    if True: # not is_dirichlet_node(fem.dmask, node0):
        gi0 = ge[0:3]
        wp.atomic_add(g, node0, gi0)

    node1 = nodes[1]
    if True: # not is_dirichlet_node(fem.dmask, node1):
        gi1 = ge[3:6]
        wp.atomic_add(g, node1, gi1)

    node2 = nodes[2]
    if True: # not is_dirichlet_node(fem.dmask, node2):
        gi2 = ge[6:9]
        wp.atomic_add(g, node2, gi2)

    node3 = nodes[3]
    if True: # not is_dirichlet_node(fem.dmask, node3):
        gi3 = ge[9:12]
        wp.atomic_add(g, node3, gi3)


@wp.kernel
def _inertial_gradient(
    fem: FemElastoDynamicsData,
    g: wp.array[wp.vec3f],
):
    i = wp.tid()
    if False: # is_dirichlet_node(fem.dmask, i):
        return
    wp.atomic_add(g, i, fem.m[i] * (fem.x[i] - fem.xtilde[i]))


@wp.kernel
def _vv_gradient(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    contact: MeshDynamicsData,
    n_u: wp.int32,
    g: wp.array[wp.vec3f],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vv.prefix[n_u]:
        return
    u = contact.contacts.vv.u[c]
    v = contact.contacts.vv.v[c]
    n = contact.contacts.vv_bases.n[c]
    t = contact.contacts.vv_bases.t[c]
    b = contact.contacts.vv_bases.b[c]
    i = contact.meshes.V[u]
    j = contact.meshes.V[v]
    xi, xj = x[i], x[j]
    xti, xtj = xt[i], xt[j]
    sk = contact.cvv.s[c]
    c_n = wp.dot(xi - xj, n) - contact.dmin - sk
    du = (xi - xti) - (xj - xtj)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    gamma = contact.cvv.gamma[c]
    dEn = sigma_n * c_n - contact.cvv.lambda_n[c]
    dEf = sigma_f * c_f - contact.cvv.lambda_f[c]
    grad = dEn * n + dEf[0] * t + dEf[1] * b
    if True: # not is_dirichlet_node(dmask, i):
        wp.atomic_add(g, i, gamma * grad)
    if True: # not is_dirichlet_node(dmask, j):
        wp.atomic_add(g, j, -gamma * grad)


@wp.kernel
def _ve_gradient(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    contact: MeshDynamicsData,
    n_u: wp.int32,
    g: wp.array[wp.vec3f],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ve.prefix[n_u]:
        return
    vi = contact.contacts.ve.u[c]
    he = contact.contacts.ve.v[c]
    n = contact.contacts.ve_bases.n[c]
    t = contact.contacts.ve_bases.t[c]
    b = contact.contacts.ve_bases.b[c]
    b1 = contact.contacts.ve_bary[c]
    b0 = wp.float32(1) - b1
    i = contact.meshes.V[vi]
    ea = halfedges.incoming_vertex(contact.meshes.F, he)
    eb = halfedges.outgoing_vertex(contact.meshes.F, he)
    xi = x[i]
    xti = xt[i]
    xcp2 = b0 * x[ea] + b1 * x[eb]
    xtcp2 = b0 * xt[ea] + b1 * xt[eb]
    c_n = wp.dot(xi - xcp2, n) - contact.dmin - contact.cve.s[c]
    du = (xi - xti) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    gamma = contact.cve.gamma[c]
    dEn = sigma_n * c_n - contact.cve.lambda_n[c]
    dEf = sigma_f * c_f - contact.cve.lambda_f[c]
    grad = dEn * n + dEf[0] * t + dEf[1] * b
    if True: # not is_dirichlet_node(dmask, i):
        wp.atomic_add(g, i, gamma * grad)
    if True: # not is_dirichlet_node(dmask, ea):
        wp.atomic_add(g, ea, -gamma * b0 * grad)
    if True: # not is_dirichlet_node(dmask, eb):
        wp.atomic_add(g, eb, -gamma * b1 * grad)


@wp.kernel
def _vf_gradient(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    contact: MeshDynamicsData,
    n_u: wp.int32,
    g: wp.array[wp.vec3f],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vf.prefix[n_u]:
        return
    vi = contact.contacts.vf.u[c]
    f = contact.contacts.vf.v[c]
    n = contact.contacts.vf_bases.n[c]
    t = contact.contacts.vf_bases.t[c]
    b = contact.contacts.vf_bases.b[c]
    uv = contact.contacts.vf_bary[c]
    b1 = uv[0]
    b2 = uv[1]
    b0 = wp.float32(1) - b1 - b2
    i = contact.meshes.V[vi]
    finds = contact.meshes.F[f]
    xi = x[i]
    xti = xt[i]
    xcp2 = b0 * x[finds[0]] + b1 * x[finds[1]] + b2 * x[finds[2]]
    xtcp2 = b0 * xt[finds[0]] + b1 * xt[finds[1]] + b2 * xt[finds[2]]
    c_n = wp.dot(xi - xcp2, n) - contact.dmin - contact.cvf.s[c]
    du = (xi - xti) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    gamma = contact.cvf.gamma[c]
    dEn = sigma_n * c_n - contact.cvf.lambda_n[c]
    dEf = sigma_f * c_f - contact.cvf.lambda_f[c]
    grad = dEn * n + dEf[0] * t + dEf[1] * b
    if True: # not is_dirichlet_node(dmask, i):
        wp.atomic_add(g, i, gamma * grad)
    if True: # not is_dirichlet_node(dmask, finds[0]):
        wp.atomic_add(g, finds[0], -gamma * b0 * grad)
    if True: # not is_dirichlet_node(dmask, finds[1]):
        wp.atomic_add(g, finds[1], -gamma * b1 * grad)
    if True: # not is_dirichlet_node(dmask, finds[2]):
        wp.atomic_add(g, finds[2], -gamma * b2 * grad)


@wp.kernel
def _ee_gradient(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    contact: MeshDynamicsData,
    n_u: wp.int32,
    g: wp.array[wp.vec3f],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ee.prefix[n_u]:
        return
    he_u = contact.contacts.ee.u[c]
    he_v = contact.contacts.ee.v[c]
    n = contact.contacts.ee_bases.n[c]
    t = contact.contacts.ee_bases.t[c]
    b = contact.contacts.ee_bases.b[c]
    st = contact.contacts.ee_bary[c]
    b1 = st[0]
    b0 = wp.float32(1) - b1
    b3 = st[1]
    b2 = wp.float32(1) - b3
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
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    gamma = contact.cee.gamma[c]
    dEn = sigma_n * c_n - contact.cee.lambda_n[c]
    dEf = sigma_f * c_f - contact.cee.lambda_f[c]
    grad = dEn * n + dEf[0] * t + dEf[1] * b
    if True: # not is_dirichlet_node(dmask, ia):
        wp.atomic_add(g, ia, gamma * b0 * grad)
    if True: # not is_dirichlet_node(dmask, ib):
        wp.atomic_add(g, ib, gamma * b1 * grad)
    if True: # not is_dirichlet_node(dmask, ic):
        wp.atomic_add(g, ic, -gamma * b2 * grad)
    if True: # not is_dirichlet_node(dmask, id_):
        wp.atomic_add(g, id_, -gamma * b3 * grad)


class Gradient:
    """Computes the full FEM elasto-dynamics + augmented-Lagrangian contact gradient.

    Kernel launches are overlapped across 6 streams (elastic, inertial, vv, ve, vf, ee)
    forked from the active stream, making the computation graph-capturable.
    """

    def __init__(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        g: wp.array,
    ):
        self._fem = fem
        self._contact = contact
        self._g = g
        self._streams = [
            wp.Stream() for _ in range(6)
        ]  # elastic, inertial, vv, ve, vf, ee

    def compute(self, main_stream: wp.Stream) -> None:
        self._g.zero_()
        fem = self._fem
        contact = self._contact
        g = self._g
        h2 = wp.float32(fem.bdf.beta_tilde**2)
        n_nodes = fem.data.x.shape[0]
        n_elems = fem.data.E.shape[0]
        cd = contact._data
        x = fem.data.x
        xt = fem.data.xt
        dmask = fem.data.dmask
        for stream in self._streams:
            stream.wait_stream(main_stream)
        wp.launch(
            _elastic_gradient,
            dim=n_elems,
            inputs=[fem.data, h2, g],
            stream=self._streams[0],
        )
        wp.launch(
            _inertial_gradient,
            dim=n_nodes,
            inputs=[fem.data, g],
            stream=self._streams[1],
        )
        for stream, kernel, cs in zip(
            self._streams[2:],
            [_vv_gradient, _ve_gradient, _vf_gradient, _ee_gradient],
            [contact.cvv, contact.cve, contact.cvf, contact.cee],
        ):
            wp.launch(
                kernel,
                dim=cs.capacity,
                inputs=[x, xt, dmask, cd, cs.n_u, g],
                stream=stream,
            )
        for stream in self._streams:
            main_stream.wait_stream(stream)

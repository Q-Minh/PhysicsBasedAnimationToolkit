"""
Newton solver for FEM elasto-dynamics with contact (augmented Lagrangian).

Outer loop mirrors ``gpu/vbd/solver.py``.  Newton-specific work lives in
``prepare_subproblem`` and ``solve_subproblem``
"""

import unittest
from enum import IntEnum
from types import SimpleNamespace
from typing import NamedTuple

import cuda.compute
import warp as wp
from ..common import profiling
import warp.sparse
import warp.optim.linear
import cupy as cp
import numpy as np

from pbatoolkit import pbat

from ...common.fields import DocField, SerializableMixin
from ..common.reduce import Reduce
from ..contact.mesh.cd import ContactDetection
from ..elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from ..contact.dynamics import (
    CONTACT_EE,
    CONTACT_VE,
    CONTACT_VF,
    CONTACT_VV,
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
    PenaltyAdaptivity,
    augmented_normal_curvature,
    augmented_normal_energy,
    capped_friction_energy,
    capped_friction_scale,
    friction_penalty,
    normal_penalty,
)
from ..contact.constraints import stored_friction_limit
from ..contact.mesh.pairs import ContactBasesData
from ..elasticity.snh import snh_eval, snh_hess
from ..elasticity.spd import filter_eigenvalues9
from ..elasticity.chain import hessian_wrt_dofs
from ..gradient import GradientNorm
from ..vbd.convergence import ratio_to_tolerance
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
    spd_correction: wp.int32  # pbat.fem.HyperElasticSpdCorrection of d^2Psi/dF^2
    n_lin_max_iters: wp.int32
    rel_eps_lin: wp.float32
    abs_eps_lin: wp.float32

    # Backtracking line search (Armijo)
    ls_max_iters: wp.int32
    ls_tau: wp.float32  # step shrink factor
    ls_c: wp.float32  # Armijo slope constant
    ls_alpha: wp.float32  # initial step size

    # Augmented normal contact terms with the slack eliminated (NewtonGpuParams.one_sided_contact)
    one_sided_contact: wp.int32

    # BSR Hessian 3x3 block triplets
    Hrows: wp.array[wp.int32]
    Hcols: wp.array[wp.int32]
    Hvals: wp.array[wp.mat33f]

    # Energy evaluations
    f_obj_partial: wp.array[
        wp.float32
    ]  # (# nodes + # elems + # vv contacts + # ve contacts + # vf contacts + # ee contacts, ) partial energy contributions


@wp.kernel
def _energy_elastic(
    fem: FemElastoDynamicsData,  # type: ignore
    h2: wp.float32,
    f_obj_partial: wp.array[wp.float32],
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
    f_obj_partial[e] = Ue  # type: ignore


@wp.kernel
def _energy_inertial(
    fem: FemElastoDynamicsData,  # type: ignore
    f_obj_partial: wp.array[wp.float32],
):
    i = wp.tid()
    if is_dirichlet_node(fem.dmask, i):  # type: ignore
        return
    if fem.is_static != 0:
        # Static objective Psi(x) - fext^T x
        # TODO: -fext^T x is large next to the energy changes near convergence (~1e3 J, measured on
        # static solves), so its float32 roundoff adds noise to the Armijo test; -fext^T (x - X) (same
        # gradient, constant shift) would reduce it. Only worth it if static Newton line searches stall.
        f_obj_partial[i] = -wp.dot(fem.fext[i], fem.x[i])  # type: ignore
    else:
        diff = fem.x[i] - fem.xtilde[i]
        f_obj_partial[i] = wp.float32(0.5) * fem.m[i] * wp.dot(diff, diff)  # type: ignore


@wp.kernel
def _energy_vv(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    friction_limit: wp.array[wp.float32],
    one_sided: wp.int32,
    f_obj_partial: wp.array[wp.float32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vv.prefix[n_u]:  # type: ignore
        return
    u, v = contact.contacts.vv.u[c], contact.contacts.vv.v[c]
    n, t, b = (
        contact.contacts.vv_bases.n[c],
        contact.contacts.vv_bases.t[c],
        contact.contacts.vv_bases.b[c],
    )
    i, j = contact.meshes.V[u], contact.meshes.V[v]
    xi, xj, xti, xtj = x[i], x[j], xt[i], xt[j]
    g_n = wp.dot(xi - xj, n) - contact.dmin
    du = (xi - xti) - (xj - xtj)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    sigma_n = normal_penalty(contact, CONTACT_VV, c)
    sigma_f = friction_penalty(contact, CONTACT_VV, c)
    E = contact.cvv.gamma[c] * (
        augmented_normal_energy(g_n, contact.cvv.s[c], contact.cvv.lambda_n[c], sigma_n, one_sided)
        + capped_friction_energy(c_f, contact.cvv.lambda_f[c], sigma_f, stored_friction_limit(friction_limit, c))
    )
    f_obj_partial[c] = E  # type: ignore


@wp.kernel
def _energy_ve(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    friction_limit: wp.array[wp.float32],
    one_sided: wp.int32,
    f_obj_partial: wp.array[wp.float32],  # type: ignore
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ve.prefix[n_u]:  # type: ignore
        return
    vi = contact.contacts.ve.u[c]
    he = contact.contacts.ve.v[c]
    n, t, b = (
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
    g_n = wp.dot(xi - xcp2, n) - contact.dmin
    du = (xi - xti) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = normal_penalty(contact, CONTACT_VE, c)
    sigma_f = friction_penalty(contact, CONTACT_VE, c)
    E = contact.cve.gamma[c] * (
        augmented_normal_energy(g_n, contact.cve.s[c], contact.cve.lambda_n[c], sigma_n, one_sided)
        + capped_friction_energy(c_f, contact.cve.lambda_f[c], sigma_f, stored_friction_limit(friction_limit, c))
    )
    f_obj_partial[c] = E  # type: ignore


@wp.kernel
def _energy_vf(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    friction_limit: wp.array[wp.float32],
    one_sided: wp.int32,
    f_obj_partial: wp.array[wp.float32],  # type: ignore
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vf.prefix[n_u]:  # type: ignore
        return
    vi = contact.contacts.vf.u[c]
    f = contact.contacts.vf.v[c]
    n, t, b = (
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
    g_n = wp.dot(xi - xcp2, n) - contact.dmin
    du = (xi - xti) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = normal_penalty(contact, CONTACT_VF, c)
    sigma_f = friction_penalty(contact, CONTACT_VF, c)
    E = contact.cvf.gamma[c] * (
        augmented_normal_energy(g_n, contact.cvf.s[c], contact.cvf.lambda_n[c], sigma_n, one_sided)
        + capped_friction_energy(c_f, contact.cvf.lambda_f[c], sigma_f, stored_friction_limit(friction_limit, c))
    )
    f_obj_partial[c] = E  # type: ignore


@wp.kernel
def _energy_ee(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    friction_limit: wp.array[wp.float32],
    one_sided: wp.int32,
    f_obj_partial: wp.array[wp.float32],  # type: ignore
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ee.prefix[n_u]:  # type: ignore
        return
    he_u = contact.contacts.ee.u[c]
    he_v = contact.contacts.ee.v[c]
    n, t, b = (
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
    id = halfedges.outgoing_vertex(contact.meshes.F, he_v)
    xcp1 = b0 * x[ia] + b1 * x[ib]
    xtcp1 = b0 * xt[ia] + b1 * xt[ib]
    xcp2 = b2 * x[ic] + b3 * x[id]
    xtcp2 = b2 * xt[ic] + b3 * xt[id]
    g_n = wp.dot(xcp1 - xcp2, n) - contact.dmin
    du = (xcp1 - xtcp1) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = normal_penalty(contact, CONTACT_EE, c)
    sigma_f = friction_penalty(contact, CONTACT_EE, c)
    E = contact.cee.gamma[c] * (
        augmented_normal_energy(g_n, contact.cee.s[c], contact.cee.lambda_n[c], sigma_n, one_sided)
        + capped_friction_energy(c_f, contact.cee.lambda_f[c], sigma_f, stored_friction_limit(friction_limit, c))
    )
    f_obj_partial[c] = E  # type: ignore


K_INERTIAL_TRIPLETS = wp.constant(0)
K_ELASTIC_TRIPLETS = wp.constant(1)
K_VV_TRIPLETS = wp.constant(2)
K_VE_TRIPLETS = wp.constant(3)
K_VF_TRIPLETS = wp.constant(4)
K_EE_TRIPLETS = wp.constant(5)


@wp.kernel
def _compute_dynamics_hessian_triplets(
    fem: FemElastoDynamicsData,  # type: ignore
    h2: wp.float32,
    params: ParamsData,  # type: ignore
    offsets: wp.array[wp.int32],
):
    tid = wp.tid()
    n_nodes = fem.x.shape[0]
    n_elems = fem.E.shape[0]
    identity = wp.identity(n=3, dtype=wp.float32)  # type: ignore
    if tid < n_nodes:
        i = tid
        if is_dirichlet_node(fem.dmask, i):  # type: ignore
            params.Hvals[offsets[K_INERTIAL_TRIPLETS] + i] = identity
        elif fem.is_static != 0:
            params.Hvals[offsets[K_INERTIAL_TRIPLETS] + i] = wp.mat33f()
        else:
            params.Hvals[offsets[K_INERTIAL_TRIPLETS] + i] = fem.m[i] * identity
    if tid < n_elems:
        e = tid
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
        # As C++ Newton (HyperElasticPotential's eSpdCorrection): Stable Neo-Hookean's d^2Psi/dF^2 can be
        # indefinite, and without the mass term (static problems) so can the whole Hessian
        HF = filter_eigenvalues9(HF, params.spd_correction)
        He = h2 * wge * hessian_wrt_dofs(HF, GP)
        base = offsets[K_ELASTIC_TRIPLETS] + e * 16
        d0 = is_dirichlet_node(fem.dmask, nodes[0])
        d1 = is_dirichlet_node(fem.dmask, nodes[1])
        d2 = is_dirichlet_node(fem.dmask, nodes[2])
        d3 = is_dirichlet_node(fem.dmask, nodes[3])

        # First block row
        if (not d0) and (not d0):
            params.Hvals[base + 0] = He[0:3, 0:3]
        if (not d0) and (not d1):
            params.Hvals[base + 1] = He[0:3, 3:6]
        if (not d0) and (not d2):
            params.Hvals[base + 2] = He[0:3, 6:9]
        if (not d0) and (not d3):
            params.Hvals[base + 3] = He[0:3, 9:12]
        # Second block row
        if (not d1) and (not d0):
            params.Hvals[base + 4] = He[3:6, 0:3]
        if (not d1) and (not d1):
            params.Hvals[base + 5] = He[3:6, 3:6]
        if (not d1) and (not d2):
            params.Hvals[base + 6] = He[3:6, 6:9]
        if (not d1) and (not d3):
            params.Hvals[base + 7] = He[3:6, 9:12]
        # Third block row
        if (not d2) and (not d0):
            params.Hvals[base + 8] = He[6:9, 0:3]
        if (not d2) and (not d1):
            params.Hvals[base + 9] = He[6:9, 3:6]
        if (not d2) and (not d2):
            params.Hvals[base + 10] = He[6:9, 6:9]
        if (not d2) and (not d3):
            params.Hvals[base + 11] = He[6:9, 9:12]
        # Fourth block row
        if (not d3) and (not d0):
            params.Hvals[base + 12] = He[9:12, 0:3]
        if (not d3) and (not d1):
            params.Hvals[base + 13] = He[9:12, 3:6]
        if (not d3) and (not d2):
            params.Hvals[base + 14] = He[9:12, 6:9]
        if (not d3) and (not d3):
            params.Hvals[base + 15] = He[9:12, 9:12]


@wp.func
def _barycentrically_unscaled_contact_hessian(
    n: wp.vec3f,
    t: wp.vec3f,
    b: wp.vec3f,
    gamma: wp.float32,
    sigma_n: wp.float32,
    sigma_f: wp.float32,
):
    Hi = gamma * (
        sigma_n * wp.outer(n, n) + sigma_f * (wp.outer(t, t) + wp.outer(b, b))  # type: ignore
    )
    return Hi


@wp.kernel
def _compute_vv_contact_hessian_triplets(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    friction_limit: wp.array[wp.float32],
    params: ParamsData,  # type: ignore
    offsets: wp.array[wp.int32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vv.prefix[n_u]:  # type: ignore
        return
    u = contact.contacts.vv.u[c]
    v = contact.contacts.vv.v[c]
    i = contact.meshes.V[u]
    j = contact.meshes.V[v]
    sigma_n = normal_penalty(contact, CONTACT_VV, c)
    sigma_f = friction_penalty(contact, CONTACT_VV, c)
    bases = contact.contacts.vv_bases
    n, t, b = bases.n[c], bases.t[c], bases.b[c]
    gamma = contact.cvv.gamma[c]
    sigma_n = augmented_normal_curvature(
        wp.dot(x[i] - x[j], n) - contact.dmin, contact.cvv.lambda_n[c], sigma_n, params.one_sided_contact
    )
    du = (x[i] - xt[i]) - (x[j] - xt[j])
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    friction_scale = capped_friction_scale(
        c_f, contact.cvv.lambda_f[c], sigma_f, stored_friction_limit(friction_limit, c)
    )
    Hvv_base = _barycentrically_unscaled_contact_hessian(
        n,
        t,
        b,
        gamma,
        sigma_n,
        friction_scale * sigma_f,
    )
    d0 = is_dirichlet_node(dmask, i)
    d1 = is_dirichlet_node(dmask, j)
    offset = offsets[K_VV_TRIPLETS] + c * 4  # 2^2
    # Block row 0
    if (not d0) and (not d0):
        params.Hrows[offset + 0] = i
        params.Hcols[offset + 0] = i
        params.Hvals[offset + 0] = Hvv_base
    if (not d0) and (not d1):
        params.Hrows[offset + 1] = i
        params.Hcols[offset + 1] = j
        params.Hvals[offset + 1] = -Hvv_base
    # Block row 1
    if (not d1) and (not d0):
        params.Hrows[offset + 2] = j
        params.Hcols[offset + 2] = i
        params.Hvals[offset + 2] = -Hvv_base
    if (not d1) and (not d1):
        params.Hrows[offset + 3] = j
        params.Hcols[offset + 3] = j
        params.Hvals[offset + 3] = Hvv_base


@wp.kernel
def _compute_ve_contact_hessian_triplets(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    friction_limit: wp.array[wp.float32],
    params: ParamsData,  # type: ignore
    offsets: wp.array[wp.int32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ve.prefix[n_u]:  # type: ignore
        return
    vi = contact.contacts.ve.u[c]
    he = contact.contacts.ve.v[c]
    n, t, b = (
        contact.contacts.ve_bases.n[c],
        contact.contacts.ve_bases.t[c],
        contact.contacts.ve_bases.b[c],
    )
    gamma = contact.cve.gamma[c]
    b1 = contact.contacts.ve_bary[c]
    b0 = wp.float32(1) - b1
    i = contact.meshes.V[vi]
    ea = halfedges.incoming_vertex(contact.meshes.F, he)
    eb = halfedges.outgoing_vertex(contact.meshes.F, he)
    sigma_n = normal_penalty(contact, CONTACT_VE, c)
    sigma_f = friction_penalty(contact, CONTACT_VE, c)
    sigma_n = augmented_normal_curvature(
        wp.dot(x[i] - (b0 * x[ea] + b1 * x[eb]), n) - contact.dmin, contact.cve.lambda_n[c], sigma_n,
        params.one_sided_contact,
    )
    du = (x[i] - xt[i]) - ((b0 * x[ea] + b1 * x[eb]) - (b0 * xt[ea] + b1 * xt[eb]))
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    friction_scale = capped_friction_scale(
        c_f, contact.cve.lambda_f[c], sigma_f, stored_friction_limit(friction_limit, c)
    )
    Hve_base = _barycentrically_unscaled_contact_hessian(
        n,
        t,
        b,
        gamma,
        sigma_n,
        friction_scale * sigma_f,
    )
    d0 = is_dirichlet_node(dmask, i)
    d1 = is_dirichlet_node(dmask, ea)  # type: ignore
    d2 = is_dirichlet_node(dmask, eb)  # type: ignore
    offset = offsets[K_VE_TRIPLETS] + c * 9  # 3^2
    # Block row 0
    if (not d0) and (not d0):
        params.Hrows[offset + 0] = i
        params.Hcols[offset + 0] = i
        params.Hvals[offset + 0] = Hve_base
    if (not d0) and (not d1):
        params.Hrows[offset + 1] = i
        params.Hcols[offset + 1] = ea
        params.Hvals[offset + 1] = -b0 * Hve_base
    if (not d0) and (not d2):
        params.Hrows[offset + 2] = i
        params.Hcols[offset + 2] = eb
        params.Hvals[offset + 2] = -b1 * Hve_base
    # Block row 1
    if (not d1) and (not d0):
        params.Hrows[offset + 3] = ea
        params.Hcols[offset + 3] = i
        params.Hvals[offset + 3] = -b0 * Hve_base
    if (not d1) and (not d1):
        params.Hrows[offset + 4] = ea
        params.Hcols[offset + 4] = ea
        params.Hvals[offset + 4] = b0 * b0 * Hve_base
    if (not d1) and (not d2):
        params.Hrows[offset + 5] = ea
        params.Hcols[offset + 5] = eb
        params.Hvals[offset + 5] = b0 * b1 * Hve_base
    # Block row 2
    if (not d2) and (not d0):
        params.Hrows[offset + 6] = eb
        params.Hcols[offset + 6] = i
        params.Hvals[offset + 6] = -b1 * Hve_base
    if (not d2) and (not d1):
        params.Hrows[offset + 7] = eb
        params.Hcols[offset + 7] = ea
        params.Hvals[offset + 7] = b1 * b0 * Hve_base
    if (not d2) and (not d2):
        params.Hrows[offset + 8] = eb
        params.Hcols[offset + 8] = eb
        params.Hvals[offset + 8] = b1 * b1 * Hve_base


@wp.kernel
def _compute_vf_contact_hessian_triplets(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    friction_limit: wp.array[wp.float32],
    params: ParamsData,  # type: ignore
    offsets: wp.array[wp.int32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.vf.prefix[n_u]:  # type: ignore
        return
    vi = contact.contacts.vf.u[c]
    f = contact.contacts.vf.v[c]
    n, t, b = (
        contact.contacts.vf_bases.n[c],
        contact.contacts.vf_bases.t[c],
        contact.contacts.vf_bases.b[c],
    )
    gamma = contact.cvf.gamma[c]
    uv = contact.contacts.vf_bary[c]
    b1, b2 = uv[0], uv[1]
    b0 = wp.float32(1) - b1 - b2
    i = contact.meshes.V[vi]
    finds = contact.meshes.F[f]
    j, k, l = finds[0], finds[1], finds[2]
    sigma_n = normal_penalty(contact, CONTACT_VF, c)
    sigma_f = friction_penalty(contact, CONTACT_VF, c)
    sigma_n = augmented_normal_curvature(
        wp.dot(x[i] - (b0 * x[j] + b1 * x[k] + b2 * x[l]), n) - contact.dmin, contact.cvf.lambda_n[c], sigma_n,
        params.one_sided_contact,
    )
    du = (x[i] - xt[i]) - (
        (b0 * x[j] + b1 * x[k] + b2 * x[l]) - (b0 * xt[j] + b1 * xt[k] + b2 * xt[l])
    )
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    friction_scale = capped_friction_scale(
        c_f, contact.cvf.lambda_f[c], sigma_f, stored_friction_limit(friction_limit, c)
    )
    Hvf_base = _barycentrically_unscaled_contact_hessian(
        n,
        t,
        b,
        gamma,
        sigma_n,
        friction_scale * sigma_f,
    )
    d0 = is_dirichlet_node(dmask, i)
    d1 = is_dirichlet_node(dmask, j)
    d2 = is_dirichlet_node(dmask, k)
    d3 = is_dirichlet_node(dmask, l)
    # Block row 0
    offset = offsets[K_VF_TRIPLETS] + c * 16  # 4^2
    if (not d0) and (not d0):
        params.Hrows[offset + 0] = i
        params.Hcols[offset + 0] = i
        params.Hvals[offset + 0] = Hvf_base
    if (not d0) and (not d1):
        params.Hrows[offset + 1] = i
        params.Hcols[offset + 1] = j
        params.Hvals[offset + 1] = -b0 * Hvf_base
    if (not d0) and (not d2):
        params.Hrows[offset + 2] = i
        params.Hcols[offset + 2] = k
        params.Hvals[offset + 2] = -b1 * Hvf_base
    if (not d0) and (not d3):
        params.Hrows[offset + 3] = i
        params.Hcols[offset + 3] = l
        params.Hvals[offset + 3] = -b2 * Hvf_base
    # Block row 1
    if (not d1) and (not d0):
        params.Hrows[offset + 4] = j
        params.Hcols[offset + 4] = i
        params.Hvals[offset + 4] = -b0 * Hvf_base
    if (not d1) and (not d1):
        params.Hrows[offset + 5] = j
        params.Hcols[offset + 5] = j
        params.Hvals[offset + 5] = b0 * b0 * Hvf_base
    if (not d1) and (not d2):
        params.Hrows[offset + 6] = j
        params.Hcols[offset + 6] = k
        params.Hvals[offset + 6] = b0 * b1 * Hvf_base
    if (not d1) and (not d3):
        params.Hrows[offset + 7] = j
        params.Hcols[offset + 7] = l
        params.Hvals[offset + 7] = b0 * b2 * Hvf_base
    # Block row 2
    if (not d2) and (not d0):
        params.Hrows[offset + 8] = k
        params.Hcols[offset + 8] = i
        params.Hvals[offset + 8] = -b1 * Hvf_base
    if (not d2) and (not d1):
        params.Hrows[offset + 9] = k
        params.Hcols[offset + 9] = j
        params.Hvals[offset + 9] = b1 * b0 * Hvf_base
    if (not d2) and (not d2):
        params.Hrows[offset + 10] = k
        params.Hcols[offset + 10] = k
        params.Hvals[offset + 10] = b1 * b1 * Hvf_base
    if (not d2) and (not d3):
        params.Hrows[offset + 11] = k
        params.Hcols[offset + 11] = l
        params.Hvals[offset + 11] = b1 * b2 * Hvf_base
    # Block row 3
    if (not d3) and (not d0):
        params.Hrows[offset + 12] = l
        params.Hcols[offset + 12] = i
        params.Hvals[offset + 12] = -b2 * Hvf_base
    if (not d3) and (not d1):
        params.Hrows[offset + 13] = l
        params.Hcols[offset + 13] = j
        params.Hvals[offset + 13] = b2 * b0 * Hvf_base
    if (not d3) and (not d2):
        params.Hrows[offset + 14] = l
        params.Hcols[offset + 14] = k
        params.Hvals[offset + 14] = b2 * b1 * Hvf_base
    if (not d3) and (not d3):
        params.Hrows[offset + 15] = l
        params.Hcols[offset + 15] = l
        params.Hvals[offset + 15] = b2 * b2 * Hvf_base


@wp.kernel
def _compute_ee_contact_hessian_triplets(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    contact: ContactDynamicsData,  # type: ignore
    n_u: wp.int32,
    friction_limit: wp.array[wp.float32],
    params: ParamsData,  # type: ignore
    offsets: wp.array[wp.int32],
):
    c = wp.tid()
    if wp.uint64(c) >= contact.contacts.ee.prefix[n_u]:  # type: ignore
        return
    he_u = contact.contacts.ee.u[c]
    he_v = contact.contacts.ee.v[c]
    n, t, b = (
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
    id = halfedges.outgoing_vertex(contact.meshes.F, he_v)
    gamma = contact.cee.gamma[c]
    sigma_n = normal_penalty(contact, CONTACT_EE, c)
    sigma_f = friction_penalty(contact, CONTACT_EE, c)
    sigma_n = augmented_normal_curvature(
        wp.dot((b0 * x[ia] + b1 * x[ib]) - (b2 * x[ic] + b3 * x[id]), n) - contact.dmin, contact.cee.lambda_n[c],
        sigma_n, params.one_sided_contact,
    )
    du = ((b0 * x[ia] + b1 * x[ib]) - (b0 * xt[ia] + b1 * xt[ib])) - (
        (b2 * x[ic] + b3 * x[id]) - (b2 * xt[ic] + b3 * xt[id])
    )
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    friction_scale = capped_friction_scale(
        c_f, contact.cee.lambda_f[c], sigma_f, stored_friction_limit(friction_limit, c)
    )
    Hee_base = _barycentrically_unscaled_contact_hessian(
        n,
        t,
        b,
        gamma,
        sigma_n,
        friction_scale * sigma_f,
    )
    d0 = is_dirichlet_node(dmask, ia)  # type: ignore
    d1 = is_dirichlet_node(dmask, ib)  # type: ignore
    d2 = is_dirichlet_node(dmask, ic)  # type: ignore
    d3 = is_dirichlet_node(dmask, id)  # type: ignore
    offset = offsets[K_EE_TRIPLETS] + c * 16  # 4^2
    # Block row 0
    if (not d0) and (not d0):
        params.Hrows[offset + 0] = ia
        params.Hcols[offset + 0] = ia
        params.Hvals[offset + 0] = b0 * b0 * Hee_base
    if (not d0) and (not d1):
        params.Hrows[offset + 1] = ia
        params.Hcols[offset + 1] = ib
        params.Hvals[offset + 1] = b0 * b1 * Hee_base
    if (not d0) and (not d2):
        params.Hrows[offset + 2] = ia
        params.Hcols[offset + 2] = ic
        params.Hvals[offset + 2] = -b0 * b2 * Hee_base
    if (not d0) and (not d3):
        params.Hrows[offset + 3] = ia
        params.Hcols[offset + 3] = id
        params.Hvals[offset + 3] = -b0 * b3 * Hee_base
    # Block row 1
    if (not d1) and (not d0):
        params.Hrows[offset + 4] = ib
        params.Hcols[offset + 4] = ia
        params.Hvals[offset + 4] = b1 * b0 * Hee_base
    if (not d1) and (not d1):
        params.Hrows[offset + 5] = ib
        params.Hcols[offset + 5] = ib
        params.Hvals[offset + 5] = b1 * b1 * Hee_base
    if (not d1) and (not d2):
        params.Hrows[offset + 6] = ib
        params.Hcols[offset + 6] = ic
        params.Hvals[offset + 6] = -b1 * b2 * Hee_base
    if (not d1) and (not d3):
        params.Hrows[offset + 7] = ib
        params.Hcols[offset + 7] = id
        params.Hvals[offset + 7] = -b1 * b3 * Hee_base
    # Block row 2
    if (not d2) and (not d0):
        params.Hrows[offset + 8] = ic
        params.Hcols[offset + 8] = ia
        params.Hvals[offset + 8] = -b2 * b0 * Hee_base
    if (not d2) and (not d1):
        params.Hrows[offset + 9] = ic
        params.Hcols[offset + 9] = ib
        params.Hvals[offset + 9] = -b2 * b1 * Hee_base
    if (not d2) and (not d2):
        params.Hrows[offset + 10] = ic
        params.Hcols[offset + 10] = ic
        params.Hvals[offset + 10] = b2 * b2 * Hee_base
    if (not d2) and (not d3):
        params.Hrows[offset + 11] = ic
        params.Hcols[offset + 11] = id
        params.Hvals[offset + 11] = b2 * b3 * Hee_base
    # Block row 3
    if (not d3) and (not d0):
        params.Hrows[offset + 12] = id
        params.Hcols[offset + 12] = ia
        params.Hvals[offset + 12] = -b3 * b0 * Hee_base
    if (not d3) and (not d1):
        params.Hrows[offset + 13] = id
        params.Hcols[offset + 13] = ib
        params.Hvals[offset + 13] = -b3 * b1 * Hee_base
    if (not d3) and (not d2):
        params.Hrows[offset + 14] = id
        params.Hcols[offset + 14] = ic
        params.Hvals[offset + 14] = b3 * b2 * Hee_base
    if (not d3) and (not d3):
        params.Hrows[offset + 15] = id
        params.Hcols[offset + 15] = id
        params.Hvals[offset + 15] = b3 * b3 * Hee_base


@wp.kernel
def _axpy(y: wp.array[wp.vec3f], x: wp.array[wp.vec3f], alpha: wp.float32):
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]


@wp.kernel
def _block_frobenius_norms(blocks: wp.array[wp.mat33f], norms: wp.array[wp.float32]):
    """|H_ii|_F of each diagonal 3x3 block, as the VBD sweeps store their vertex Hessians' (``Params.data.Hnk``)."""
    i = wp.tid()
    Hi = blocks[i]
    norms[i] = wp.sqrt(wp.ddot(Hi, Hi))


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


class LinearSolverParams(SerializableMixin):
    """GPU-only settings of the Jacobi-preconditioned CG solve of each Newton step.

    The C++ Newton solver uses a direct solver (``ELinearSolver``), so these have no CPU counterpart.
    Static problems lack the mass term and are far worse conditioned, so they typically need many
    more CG iterations than dynamic ones (e.g. 1500-3000; the default cap is 2000).

    Held by ``SimulationState.newton_linear_solver_params`` (kept across resets) and serialized under
    ``Solver/Newton/LinearSolver``: configs set ``Solver.Newton.LinearSolver.n_max_iters``, ``rtol``
    and ``atol``. A change takes effect at the next reset, which rebuilds the GPU Newton params.
    """

    n_max_iters = DocField(2000, "Maximum CG iterations per Newton step")
    rtol = DocField(1e-6, "Relative CG residual tolerance")
    atol = DocField(
        1e-10,
        "Absolute CG residual tolerance, in the units of the h^2-scaled objective's gradient. Keep it "
        "well below |g| at convergence: at small time steps |g| can be ~1e-8, and an atol above |g| makes "
        "CG return a zero step",
    )


class NewtonEarlyExit(IntEnum):
    """Whether passing a convergence test ends a Newton solve or subproblem (``Solver.Newton.early_exit``)."""

    Shared = -1  # as Solver.Convergence.early_exit, the VBD family's switch
    Off = 0  # every solve runs its full budget
    On = 1  # passing a test ends the solve or subproblem


class NewtonGpuParams(SerializableMixin):
    """GPU-only Newton settings, serialized next to the C++ Newton params under ``Solver/Newton``.

    Held by ``SimulationState.newton_gpu_params`` (kept across resets): configs set
    ``Solver.Newton.init_strategy``, ``one_sided_contact`` and ``early_exit``. They take effect at the next time step.
    """

    init_strategy = DocField(
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position,
        "Newton's initial guess for each time step, used instead of Integration.init_strategy whenever Newton is "
        "the active solver (the VBD family keeps the shared one). Position (x_t) by default: Newton's first step "
        "linearizes at the initial guess, and an extrapolated guess (e.g. TrajectoryWithExternalLoad) can start "
        "deep inside a contact partner at an impact, where that linearization is poor (Newton went unstable at the "
        "sphere's impact on the soft-stiff trampoline)",
    )
    one_sided_contact = DocField(
        False,
        "Eliminate the contact slack from Newton's augmented normal terms: E = (max(0, lambda - sigma g)^2 - "
        "lambda^2) / (2 sigma) for the gap past the margin g, so a contact only pushes, and only while pressed, and "
        "its Hessian block is 0 otherwise. Off (as C++), the slack is fixed within each Newton step, and a contact "
        "detected but not pressed is a stiff two-sided spring that holds its gap: on Celebration, confetti stayed on "
        "vertical letter faces with friction off. Takes effect at the next time step",
    )
    early_exit = DocField(
        NewtonEarlyExit.Shared,
        "Whether passing a convergence test ends Newton's solve or subproblem: Shared (the default) follows "
        "Solver.Convergence.early_exit, the VBD family's switch; Off runs every solve's full budget; On exits early "
        "even with Solver.Convergence.early_exit off, so Newton, the reference, can stop at convergence in scenes "
        "that give the VBD family fixed budgets. Takes effect at the next time step",
    )


# Safety factor on the float32 roundoff of the elastic energy, eps * h^2 sum_e w_e mu_e, that the
# Armijo test tolerates (measured noise: about 1x it on the inclined-plane cube)
_ENERGY_ROUNDOFF_FACTOR = 8.0


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
        Squared gradient-norm convergence threshold. Dynamic problems use the shared relative test
        instead when ``Solver.Convergence.rstol`` or ``astol > 0`` (see :func:`check_convergence`).
    linear_solver:
        GPU-only CG settings (:class:`LinearSolverParams`); defaults when omitted.
    spd_correction:
        SPD correction of the elastic Hessians d^2Psi/dF^2, from the CPU params' ``spd_correction``
        (``Solver.Newton.spd_correction``; C++ default Absolute, ``NoCorrection`` to disable). Applied to
        dynamic and static problems, as in C++.
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
    _streams: list[wp.Stream]

    def __init__(
        self,
        params_cpu: pbat.sim.algorithm.newton.Params,
        linear_solver: LinearSolverParams | None = None,
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
        self._data.spd_correction = int(params_cpu.spd_correction.value)
        linear_solver = linear_solver or LinearSolverParams()
        self._data.n_lin_max_iters = int(linear_solver.n_max_iters)
        self._data.rel_eps_lin = float(linear_solver.rtol)
        self._data.abs_eps_lin = float(linear_solver.atol)
        self._data.ls_max_iters = line_search.n_max_iters if line_search else 0
        self._data.ls_tau = line_search.tau if line_search else 0.5
        self._data.ls_c = line_search.c if line_search else 1e-4
        self._data.ls_alpha = line_search.alpha if line_search else 1.0
        self._data.one_sided_contact = 0
        self._streams = [wp.Stream() for _ in range(6)]
        # Shared device-side convergence test (gpu.vbd.convergence.ConvergenceCheck), set by
        # SimulationState. With its relative test on, it replaces gtol2 (see check_convergence).
        self.convergence = None
        # GPU-only settings (NewtonGpuParams), set by SimulationState; read at each time step
        self.gpu: NewtonGpuParams | None = None
        # Host-side solve statuses: the last solve_subproblem's record, and one NewtonSolveRecord per
        # NewtonSolver.solve until a consumer (SimulationState.step) drains them
        self.last_subproblem: NewtonSubproblemRecord | None = None
        self.solve_records: list[NewtonSolveRecord] = []
        # (gradient, feasibility) ratios to their tolerances of the last check_convergence
        self.last_outer_ratios: tuple[float, float] = (float("nan"), float("nan"))

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
        self._ndx = wp.zeros(n_nodes, dtype=wp.vec3f)
        # The stationarity test's float32 floor (_store_hessian_norms): the last Newton step's Hessian diagonal blocks
        # and their norms |H_ii|_F (read from the solve's first Newton step on, see initialize_solve)
        self._hessian_diagonal = wp.zeros(n_nodes, dtype=wp.mat33f)
        self._hessian_norms = wp.zeros(n_nodes, dtype=wp.float32)
        # Energy partial-contributions buffer: one slot per inertial node, elastic element,
        # and each contact pair type.
        f_obj_partial_counts = [
            n_nodes,
            n_elems,
            vv_capacity,
            ve_capacity,
            vf_capacity,
            ee_capacity,
        ]
        self._energy_offsets = [0] * len(f_obj_partial_counts)
        for i in range(1, len(f_obj_partial_counts)):
            self._energy_offsets[i] = (
                self._energy_offsets[i - 1] + f_obj_partial_counts[i - 1]
            )
        n_energy_items = self._energy_offsets[-1] + f_obj_partial_counts[-1]
        self._data.f_obj_partial = wp.zeros(n_energy_items, dtype=wp.float32)
        self._energy_scalar = wp.zeros(1, dtype=wp.float32)
        f_partial_cp = cp.asarray(self._data.f_obj_partial)
        # Armijo's roundoff tolerance (see solve_subproblem): float32 evaluates Stable Neo-Hookean's
        # Psi(F) near rest with cancellation (I_C - 3, J - 1), so the objective's error is about
        # eps * h^2 sum_e w_e mu_e. Without a tolerance, steps near convergence whose decrease is below
        # it fail the line search on noise.
        self._energy_roundoff = (
            _ENERGY_ROUNDOFF_FACTOR
            * float(np.finfo(np.float32).eps)
            * float((fem.data.wg.numpy().astype(np.float64) * fem.data.mug.numpy()).sum())
        )
        self._energy_reduce = Reduce(
            d_in=f_partial_cp,
            d_out=cp.asarray(self._energy_scalar),
            num_items=n_energy_items,
            op=cuda.compute.OpKind.PLUS,
        )
        # Hessian triplet offsets
        # TODO: Contact triplets fill each type's full capacity, and the BSR assembly processes all of them every Newton
        # step, whatever the live contact count: raising CapsulesIntoBowl's capacities from 1513/1513/15137 to
        # 5146/15137/30274 VV/VE/VF slots (and EE from 18127 to 1) made its Newton steps 4.8% slower (116.7 to 122.3 ms
        # from step 0, 2026-10-22; remeasured 115.6 to 123.2 ms), and the baskets' new capacities add ~490k slots. A
        # bit of overhead, negligible for the paper's results (maintainer). Tried and dropped (branch
        # feature/newton-live-contact-triplets, 2026-10-22): the live contacts' triplets packed after the FEM's at
        # device offsets from the pairs' prefix counts. A device count isn't enough: Warp 1.17's CUDA
        # bsr_set_from_triplets masks the triplets past it but still radix-sorts the whole arrays (native/sparse.cu),
        # so the count has to be read on the host and the arrays sliced to it. The packed matrix was bit-identical to
        # this layout's at one capsule assembly, yet Newton's results changed systematically (CapsulesIntoBowl from
        # AAAVBD's step-200 state: 115-123 line-search backtracks and 0 of 60 subproblems converged, against 0 and 10
        # in two dev runs; from step 0, 1.6e-4 m after one step against dev's 5e-7 m run-to-run), cause not found,
        # and step times didn't improve.
        self._hessian_triplet_counts = [
            n_nodes,
            n_elems * (4**2),
            vv_capacity * (2**2),
            ve_capacity * (3**2),
            vf_capacity * (4**2),
            ee_capacity * (4**2),
        ]
        self._hessian_triplet_offsets = [0] * len(self._hessian_triplet_counts)
        for i in range(1, len(self._hessian_triplet_offsets)):
            self._hessian_triplet_offsets[i] = (
                self._hessian_triplet_offsets[i - 1]
                + self._hessian_triplet_counts[i - 1]
            )
        self._hessian_triplet_offsets = wp.array(
            self._hessian_triplet_offsets, dtype=wp.int32
        )
        # Gradient + line search
        # With the per-node force scale, the shared stationarity test's relative term (ConvergenceCheck.relative_test)
        self._gradient_norm = GradientNorm(fem, contact, force_scale=True)
        self._g = self._gradient_norm.g
        self._gnorm2 = self._gradient_norm.gnorm2
        g_flat = cp.asarray(self._g).ravel()
        ndx_flat = cp.asarray(self._ndx).ravel()
        self._nslope = wp.zeros(1, dtype=wp.float32)
        self._nslope_reduce = Reduce(
            d_in=cuda.compute.TransformIterator(
                cuda.compute.ZipIterator(g_flat, ndx_flat),
                lambda x: x[0] * x[1],
            ),
            d_out=cp.asarray(self._nslope),
            num_items=3 * n_nodes,
            op=cuda.compute.OpKind.PLUS,
        )

    @property
    def data(self) -> ParamsData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data


def _relative_check(params: Params):
    """The convergence check whose relative test applies to this solve, or None (absolute gtol2)."""
    check = params.convergence
    return check if check is not None and check.relative else None


def _early_exit(params: Params) -> bool:
    """Whether passing a test ends the solve or subproblem: ``Solver.Newton.early_exit`` when On or Off, else
    (Shared) ``Solver.Convergence.early_exit``, on without the shared convergence check."""
    gpu = getattr(params, "gpu", None)
    mode = NewtonEarlyExit.Shared if gpu is None else NewtonEarlyExit(gpu.early_exit)
    if mode != NewtonEarlyExit.Shared:
        return mode == NewtonEarlyExit.On
    return params.convergence is None or bool(params.convergence.params.early_exit)


@profiling.scoped("solve.initialize")
def initialize_solve(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
) -> None:
    """Detect contacts, initialise the constraint set for the time step, unfreeze the contacts'
    Coulomb limits (each is frozen again at its first positive estimate of the step, see
    :class:`gpu.gradient.Gradient`), adapt the contact penalties once per step (``TIMESTEP``
    adaptivity, as the VBD family) and measure the relative gradient test's scale."""
    one_sided = params.gpu is not None and bool(params.gpu.one_sided_contact)
    params._data.one_sided_contact = int(one_sided)
    params._gradient_norm.one_sided = one_sided
    # Static problems have no inertia: their first detection isn't widened by xtilde
    cd.on_time_step_started(widen_by_inertial_target=not fem.static)
    cd.detect_contacts(from_xt=True)
    contact.update_constraint_set()
    contact.reset_friction_limits()
    cd.filter_step()
    if contact.params.penalty_adaptivity == PenaltyAdaptivity.TIMESTEP:
        adapt_penalty_parameters(fem, contact, params)
    check = _relative_check(params)
    if check is not None:
        # The float32 floor of the stationarity test (ConvergenceParams.floor_local_kappa, floor_kappa), as the VBD
        # family's, from the solve's first Newton step on (solve_subproblem): the vertex Hessian norms |H_ii|_F of the
        # last Newton step's assembled Hessian (_store_hessian_norms), i.e. the previous iterate's, as the VBD family's
        # Hnk is its last sweep's (not the contact-free Hessian adapt_penalty_parameters assembles). The solve's tests
        # before its first Newton step (the initial outer test, the first subproblem's first test) have no floor,
        # where the VBD family's read the previous solve's last sweep: they test Newton's initial guess x^t
        # (Solver.Newton.init_strategy Position), which hasn't moved, and on a stiff body the floor clause passes
        # it. Measured on BendingBeam.100MPa (2026-11-07): RMS_M(phi) = 91 m/s^2, so the floor clause accepts RMS_M(r)
        # up to 18 m/s^2, and x^t's residual (gravity, v^t/h up to 7 m/s^2, the elastic force's float32 noise) is
        # 16-19 m/s^2; with the previous solve's norms, step 2 passed at x^t (v = 0) and the beam froze for good.
        # With smoothing_passes > 0 the smoothed test's floor (the probe's, ConvergenceCheck._measure_smoothed_floor)
        # is measured here, at the solve's initial positions as for the VBD family, and applies from the first Newton
        # step as well; the unsmoothed and smoothed tests must both pass.
        check.set_hessian_norms(params._hessian_norms)
        check.measure_scale()
        check.set_hessian_norms(None)


@profiling.scoped("newton.energy")
def _compute_energy(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
    h2: wp.float32,
) -> float:
    main_stream = wp.get_stream()
    # Zero before forking, so the side streams' writes can't race the memset
    params._data.f_obj_partial.zero_()
    for stream in params._streams[:6]:
        stream.wait_stream(main_stream)
    n_nodes = fem.data.x.shape[0]
    n_elems = fem.data.E.shape[0]
    cd = contact.data
    x, xt = fem.data.x, fem.data.xt
    # Coulomb limits frozen for the time step by the augmented gradient
    limits = params._gradient_norm.friction_limits
    offsets = params._energy_offsets
    f_cp = cp.asarray(params._data.f_obj_partial)
    f_inertial = wp.array(
        data=f_cp[offsets[0] : offsets[1]], copy=False, dtype=wp.float32
    )
    f_elastic = wp.array(
        data=f_cp[offsets[1] : offsets[2]], copy=False, dtype=wp.float32
    )
    f_vv = wp.array(data=f_cp[offsets[2] : offsets[3]], copy=False, dtype=wp.float32)
    f_ve = wp.array(data=f_cp[offsets[3] : offsets[4]], copy=False, dtype=wp.float32)
    f_vf = wp.array(data=f_cp[offsets[4] : offsets[5]], copy=False, dtype=wp.float32)
    f_ee = wp.array(data=f_cp[offsets[5] :], copy=False, dtype=wp.float32)
    wp.launch(
        _energy_inertial,
        dim=n_nodes,
        inputs=[fem.data, f_inertial],
        stream=params._streams[0],
    )
    wp.launch(
        _energy_elastic,
        dim=n_elems,
        inputs=[fem.data, h2, f_elastic],
        stream=params._streams[1],
    )
    wp.launch(
        _energy_vv,
        dim=contact.cvv.capacity,
        inputs=[x, xt, cd, contact.cvv.n_u, limits.vv, params._data.one_sided_contact, f_vv],
        stream=params._streams[2],
    )
    wp.launch(
        _energy_ve,
        dim=contact.cve.capacity,
        inputs=[x, xt, cd, contact.cve.n_u, limits.ve, params._data.one_sided_contact, f_ve],
        stream=params._streams[3],
    )
    wp.launch(
        _energy_vf,
        dim=contact.cvf.capacity,
        inputs=[x, xt, cd, contact.cvf.n_u, limits.vf, params._data.one_sided_contact, f_vf],
        stream=params._streams[4],
    )
    wp.launch(
        _energy_ee,
        dim=contact.cee.capacity,
        inputs=[x, xt, cd, contact.cee.n_u, limits.ee, params._data.one_sided_contact, f_ee],
        stream=params._streams[5],
    )
    for stream in params._streams:
        main_stream.wait_stream(stream)
    params._energy_reduce(main_stream)
    return float(params._energy_scalar.numpy()[0])


def check_convergence(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> bool:
    """Outer test of the current positions on the current contacts, gradient and feasibility at once: the
    test before the first subproblem. After a subproblem, :meth:`NewtonSolver.solve` splits it around the
    re-detection instead (:func:`outer_gradient_test` before it, :func:`outer_feasibility_test` after it).

    With the shared convergence check's stationarity test on (``Solver.Convergence.rstol`` or ``astol > 0``,
    dynamic problems), it's the VBD family's outer test: the Lagrangian gradient,
    ``RMS_M(q) <= 1 and (RMS_M(q0) <= 1 or RMS_M(r) <= kg RMS_M(phi))``, q_i = |r_i| / max(rho_i, alpha_i, kl phi_i)
    with the float32 floor phi_i from the vertex Hessian norms of the last Newton step (:func:`_store_hessian_norms`;
    see ``vbd.convergence.ConvergenceParams``, and with ``Solver.Convergence.node_norm`` MAX the max over nodes),
    without the floor before the solve's first Newton step, so this test before the first subproblem has none
    (:func:`initialize_solve`), and feasibility
    (penetration into the contact margin at most ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``). The Lagrangian gradient has no
    term for a new contact (lambda = 0), so without the feasibility test a penetrating free-flight
    guess passes before any subproblem runs. Otherwise it's ``|g|^2 <= gtol2`` on the
    augmented-Lagrangian gradient, as in C++.

    Sets ``params.last_outer_ratios`` to the test's (gradient, feasibility) measures over their
    tolerances, as the VBD family's status ratios (at most 1 passes). The absolute test doesn't include
    feasibility, but its ratio is still measured when the shared check exists (NaN otherwise).
    """
    gradient = outer_gradient_test(fem, contact, params)
    return outer_feasibility_test(params, gradient)


@profiling.scoped("newton.convergence")
def outer_gradient_test(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> tuple[bool, float]:
    """The gradient part of the outer test (see :func:`check_convergence`) on the current contact set.

    After a subproblem, call it after the full dual update and the feasibility filter
    (:func:`finalize_subproblem`), before re-detection (:func:`update_constraint_set`), and before the
    dual update's pending decay weights are committed: it measures the subproblem's positions on the contact
    set it solved, with the new multipliers and the decay weights it solved with (maintainer, 2026-11-06: the
    AL method solves each subproblem for a fixed constraint set; C++ tests after re-detection).

    Returns:
        tuple[bool, float]: Whether it passes, and its measure over its tolerance.
    """
    check = _relative_check(params)
    if check is not None:
        passed = check.test_gradient()
        # The decisive ratio, at most 1 iff the test passed (with the float32 floor, ||q|| alone isn't)
        return passed, check.decisive_ratio()
    params._gradient_norm.compute(wp.get_stream())
    gnorm2, gtol2 = float(params._gnorm2.numpy()[0]), float(params.data.gtol2)
    return gnorm2 <= gtol2, float(np.sqrt(ratio_to_tolerance(gnorm2, gtol2)))


@profiling.scoped("newton.convergence")
def outer_feasibility_test(params: Params, gradient: tuple[bool, float]) -> bool:
    """Complete the outer test whose gradient part ``gradient`` (:func:`outer_gradient_test`) was measured:
    feasibility of the current positions on the current (after a subproblem: re-detected) contacts. Sets
    ``params.last_outer_ratios``.

    Returns:
        bool: Whether the outer test passes: both parts with the relative test, the gradient alone with
        the absolute one.
    """
    gradient_passes, gradient_ratio = gradient
    check = _relative_check(params)
    if check is not None:
        feasible = check.test_feasibility()
        params.last_outer_ratios = (gradient_ratio, check.feasibility_ratio())
        return gradient_passes and feasible
    feasibility = params.convergence.feasibility_ratio(measure=True) if params.convergence is not None else float("nan")
    params.last_outer_ratios = (gradient_ratio, feasibility)
    return gradient_passes


def prepare_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
) -> None:
    """Adapt the contact penalties every subproblem (``SUBPROBLEM`` adaptivity), as C++
    ``PrepareSubproblem``: see :func:`adapt_penalty_parameters`."""
    if contact.params.penalty_adaptivity == PenaltyAdaptivity.SUBPROBLEM:
        adapt_penalty_parameters(fem, contact, params)


@profiling.scoped("newton.adapt_penalty")
def adapt_penalty_parameters(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> None:
    """C++ ``MeshDynamics::UpdatePenaltyParameter(H)``: assemble the Hessian without contact terms
    at the current positions and set the penalties to the largest contact Rayleigh quotient in it
    (:meth:`MeshDynamics.adapt_penalty_parameters_to_hessian`)."""
    _assemble_hessian(fem, contact, params, wp.float32(fem.h2), with_contacts=False)
    contact.adapt_penalty_parameters_to_hessian(params._H, fem.data.dmask, fem.h2)


class SubproblemStatus(IntEnum):
    """How :func:`solve_subproblem` ended."""

    STEPPED = 0  # took its n_subproblem_max_iters Newton steps
    CONVERGED = 1  # the gradient met gtol2
    NON_DESCENT = 2  # the Newton direction wasn't a descent direction; x is unchanged
    LINE_SEARCH_FAILED = 3  # no step length passed the Armijo test; x is restored


class NewtonSubproblemRecord(NamedTuple):
    """How one :func:`solve_subproblem` call went (``Params.last_subproblem``)."""

    status: SubproblemStatus
    newton_steps: int  # Newton directions computed (one CG solve each)
    max_cg_iters: int  # most CG iterations of one solve
    cg_capped: bool  # a CG solve stopped at its iteration cap (LinearSolverParams.n_max_iters)
    backtracks: int  # line-search step-size reductions, over all steps
    # The last CG solve's residual norm and the absolute tolerance it was tested against (host values cg
    # returns anyway); NaN before any solve
    cg_residual: float = float("nan")
    cg_tolerance: float = float("nan")


class NewtonSolveRecord(NamedTuple):
    """One Newton solve (time step or substep): outer iterations, their subproblems, and the
    (gradient, feasibility) ratios to their tolerances of every outer test, in order: the test before the
    first subproblem (``check_convergence``), then the one after each subproblem (its gradient on the contact
    set the subproblem solved, its feasibility after re-detection), so ``len(outer_tests) ==
    len(subproblems) + 1``. The last entry is the solve's result."""

    outer: int  # outer (augmented-Lagrangian) iterations run
    converged: bool  # the outer test passed within the budget
    subproblems: tuple[NewtonSubproblemRecord, ...]
    outer_tests: tuple[tuple[float, float], ...] = ()


def _assemble_hessian(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
    h2: wp.float32,
    with_contacts: bool = True,
) -> None:
    """Assemble the Newton Hessian ``params._H`` (mass + elastic, plus the contact terms with
    ``with_contacts``) at the current positions. Without contact terms, their triplets stay 0 and
    are pruned."""
    main_stream = wp.get_stream()
    # Zero before forking, so the side streams' writes can't race the memset
    params._data.Hvals.zero_()
    for stream in params._streams[:5]:
        stream.wait_stream(main_stream)
    n_nodes = fem.data.x.shape[0]
    n_elems = fem.data.E.shape[0]
    # Coulomb limits frozen for the time step by the augmented gradient
    limits = params._gradient_norm.friction_limits
    wp.launch(
        kernel=_compute_dynamics_hessian_triplets,
        dim=max(n_nodes, n_elems),
        inputs=[fem.data, h2, params._data, params._hessian_triplet_offsets],
        stream=params._streams[0],
    )
    if with_contacts:
        wp.launch(
            kernel=_compute_vv_contact_hessian_triplets,
            dim=contact.cvv.capacity,
            inputs=[
                fem.data.x,
                fem.data.xt,
                fem.data.dmask,
                contact.data,
                contact.cvv.n_u,
                limits.vv,
                params._data,
                params._hessian_triplet_offsets,
            ],
            stream=params._streams[1],
        )
        wp.launch(
            kernel=_compute_ve_contact_hessian_triplets,
            dim=contact.cve.capacity,
            inputs=[
                fem.data.x,
                fem.data.xt,
                fem.data.dmask,
                contact.data,
                contact.cve.n_u,
                limits.ve,
                params._data,
                params._hessian_triplet_offsets,
            ],
            stream=params._streams[2],
        )
        wp.launch(
            kernel=_compute_vf_contact_hessian_triplets,
            dim=contact.cvf.capacity,
            inputs=[
                fem.data.x,
                fem.data.xt,
                fem.data.dmask,
                contact.data,
                contact.cvf.n_u,
                limits.vf,
                params._data,
                params._hessian_triplet_offsets,
            ],
            stream=params._streams[3],
        )
        wp.launch(
            kernel=_compute_ee_contact_hessian_triplets,
            dim=contact.cee.capacity,
            inputs=[
                fem.data.x,
                fem.data.xt,
                fem.data.dmask,
                contact.data,
                contact.cee.n_u,
                limits.ee,
                params._data,
                params._hessian_triplet_offsets,
            ],
            stream=params._streams[4],
        )
    for stream in params._streams[:5]:
        main_stream.wait_stream(stream)
    warp.sparse.bsr_set_from_triplets(
        dest=params._H,
        rows=params._data.Hrows,
        columns=params._data.Hcols,
        values=params._data.Hvals,
        count=None,
        prune_numerical_zeros=True,
        masked=False,
    )


def _store_hessian_norms(params: Params) -> None:
    """Store the vertex Hessian norms |H_ii|_F of the assembled Newton Hessian ``params._H`` (its diagonal 3x3
    blocks) in ``params._hessian_norms``, the shared stationarity test's float32 floor phi_i = |H_ii|_F eps32 |x_i| /
    (m_i h^2) (``vbd.convergence.ConvergenceParams``), as the VBD sweeps store theirs in ``Params.data.Hnk``.

    Same content as the VBD family's vertex blocks, in the gradient's units: m_i I + h^2 sum_e K_e,ii plus each
    contact's gamma w_i^2 (sigma_n n n^T + friction-scaled sigma_f (t t^T + b b^T)) over the contacts vertex i takes
    part in (as vertex, edge or face node). The elastic blocks differ where Stable Neo-Hookean's d^2Psi/dF^2 is
    indefinite: Newton's are SPD-corrected per element (``spd_correction``; Absolute keeps |d^2Psi/dF^2|_F), the
    VBD family's aren't. With ``Solver.Newton.one_sided_contact`` an unpressed contact's normal block is 0 here,
    while the VBD family keeps gamma sigma_n n n^T for every contact in the set."""
    warp.sparse.bsr_get_diag(params._H, out=params._hessian_diagonal)
    wp.launch(
        _block_frobenius_norms,
        dim=params._hessian_norms.shape[0],
        inputs=[params._hessian_diagonal, params._hessian_norms],
    )


# One-sided (slack-eliminated) AL contact terms: Solver.Newton.one_sided_contact (NewtonGpuParams, off by default
# as C++, which fixes s per step; contact.dynamics.augmented_normal_energy). With s fixed within a Newton step, a
# contact inside the detection band but not pressed is a two-sided spring of stiffness sigma_n (adapted to the
# stiffest partner) that holds its gap for the step: on Celebration (2026-11-01), 7 confetti stayed on vertical
# letter faces at ~0 velocity with friction off (mu_f = 0; 20-step resume from 2 s). Eliminating s gives
# E = (max(0, lambda - sigma g)^2 - lambda^2) / (2 sigma), C^1, with the Hessian block sigma n n^T only on pressed
# contacts (lambda - sigma g > 0). Risk: active-set switching between Newton steps and within the line search
# (energy only C^1, Hessian discontinuous) can cost backtracks or chatter. TODO: also measure it on the soft-stiff
# trampoline + sphere, whose contact-region roughness p95 after the slack refresh is ~0.065 (steps 253-256),
# against AAAVBD's 0.033.
def _subproblem_gradient(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
    main_stream: wp.Stream,
) -> None:
    """The subproblem's augmented-Lagrangian gradient at the current (accepted) iterate, after a
    slack-only dual update there, as C++ ``ToGradient(..., bForSubproblem = true)``: the slack is
    ``max(0, c_n - lambda_n / sigma_n)`` at these positions, then held for the Newton step's Hessian
    and line-search energies. The VBD family refreshes it before every sweep the same way. Without it,
    a subproblem used the slack of the previous subproblem's end, or, for contacts detected since,
    a stale value, which pushed or pulled vertices at random (jagged interfaces) and fooled the decay
    test of :func:`finalize_subproblem`."""
    contact.update_dual(
        fem.data.x,
        fem.xt,
        request_slack_update=True,
        request_decay_update=False,
        request_lagrange_multiplier_update=False,
    )
    params._gradient_norm.compute_gradient(main_stream)


@profiling.scoped("newton.subproblem")
def solve_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> SubproblemStatus:
    """Take up to ``n_subproblem_max_iters`` damped Newton steps on the current subproblem.

    The subproblem converges when its augmented-Lagrangian gradient passes the shared stationarity test
    (``Solver.Convergence.rstol`` or ``astol > 0``, dynamic problems), or else ``|g|^2 <= gtol2``; with
    early exit off (``Solver.Newton.early_exit``, see :func:`_early_exit`), it takes all its steps anyway. Each
    gradient (the subproblem's start and after each accepted step) first refreshes the contact slack
    at the iterate (:func:`_subproblem_gradient`).

    Returns:
        SubproblemStatus: Why it stopped. A rejected step (``NON_DESCENT``, ``LINE_SEARCH_FAILED``)
        leaves ``x`` at the last accepted iterate. ``params.last_subproblem`` holds its
        :class:`NewtonSubproblemRecord`, from values the loop reads anyway (no extra synchronization).
    """
    main_stream = wp.get_stream()
    n_nodes = fem.data.x.shape[0]
    h2 = wp.float32(fem.h2)  # type: ignore
    check = _relative_check(params)
    early_exit = _early_exit(params)
    n_lin_max_iters = int(params.data.n_lin_max_iters)
    steps, max_cg, cg_capped, backtracks = 0, 0, False, 0
    cg_residual, cg_tolerance = float("nan"), float("nan")

    def done(status: SubproblemStatus) -> SubproblemStatus:
        params.last_subproblem = NewtonSubproblemRecord(
            status, steps, max_cg, cg_capped, backtracks, cg_residual, cg_tolerance
        )
        return status

    _subproblem_gradient(fem, contact, params, main_stream)
    for _ in range(int(params.data.n_subproblem_max_iters)):
        if check is not None:
            if check.relative_test(params._gradient_norm) and early_exit:
                return done(SubproblemStatus.CONVERGED)
        else:
            params._gradient_norm.reduce(main_stream)
            gnorm2 = float(params._gnorm2.numpy()[0])
            if gnorm2 <= float(params.data.gtol2) and early_exit:
                return done(SubproblemStatus.CONVERGED)
        # Assemble FEM + contact Hessian
        profiling.push("newton.assemble")
        _assemble_hessian(fem, contact, params, h2)
        if check is not None:
            # The float32 floor of the next tests is this iterate's Hessian: they test the stepped iterate, as the
            # VBD family's test reads the vertex blocks of the sweep that moved the positions (initialize_solve)
            _store_hessian_norms(params)
            check.set_hessian_norms(params._hessian_norms)
        profiling.pop()
        profiling.push("newton.linear_solve")
        M = warp.optim.linear.preconditioner(params._H, "diag")
        # Solve H (-dx) = g
        params._ndx.zero_()
        final_iteration, residual_norm, absolute_tolerance = warp.optim.linear.cg(
            params._H,
            params._g,
            x=params._ndx,
            tol=params.data.rel_eps_lin,
            atol=params.data.abs_eps_lin,
            maxiter=params.data.n_lin_max_iters,
            M=M,
            use_cuda_graph=False,  # True
        )
        profiling.pop()
        # With check_every > 0 (the default), cg returns its iteration count as an int
        steps += 1
        max_cg = max(max_cg, int(final_iteration))
        cg_capped = cg_capped or int(final_iteration) >= n_lin_max_iters
        cg_residual, cg_tolerance = float(residual_norm), float(absolute_tolerance)
        # Check descent direction slope=dot(g, dx) < 0.
        # Since we compute ndx=-dx, we check -dot(g, ndx) < 0.
        params._nslope_reduce(main_stream)
        slope = -params._nslope.numpy()[0]
        if slope >= 0.0:
            return done(SubproblemStatus.NON_DESCENT)
        # Armijo backtracking line search
        profiling.push("newton.line_search")
        x_backup = wp.clone(fem.data.x)
        E0 = _compute_energy(fem, contact, params, h2)
        alpha = float(params.data.ls_alpha)
        wp.launch(_axpy, dim=n_nodes, inputs=[fem.data.x, params._ndx, -alpha])
        c = float(params.data.ls_c)
        # Accept energy changes within float32 roundoff of the objective (Params._energy_roundoff),
        # as in approximate Wolfe line searches: near convergence, the predicted decrease is below what
        # float32 resolves. GPU-only; C++ Newton rejects those steps.
        tol = params._energy_roundoff * float(h2)
        accepted = False
        for _ in range(int(params.data.ls_max_iters)):
            # wp.copy(fem.data.x, x_backup)
            E_trial = _compute_energy(fem, contact, params, h2)
            if E_trial <= E0 + c * alpha * slope + tol:
                accepted = True
                break
            # Move x from x0 - alpha*ndx back to x0 - alpha_next*ndx
            alpha_next = alpha * float(params.data.ls_tau)
            wp.launch(_axpy, dim=n_nodes, inputs=[fem.data.x, params._ndx, alpha - alpha_next])
            alpha = alpha_next
            backtracks += 1
        profiling.pop()
        if not accepted:
            wp.copy(fem.data.x, x_backup)
            return done(SubproblemStatus.LINE_SEARCH_FAILED)
        _subproblem_gradient(fem, contact, params, main_stream)
    return done(SubproblemStatus.STEPPED)


@profiling.scoped("solve.finalize_subproblem")
def finalize_subproblem(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    cd: ContactDetection,
    params: Params,
) -> None:
    """Dual update and feasibility filter -- the first half of ``vbd.solver.finalize_subproblem``; the outer
    test's gradient (:func:`outer_gradient_test`) follows on the contact set the subproblem solved, then
    :func:`update_constraint_set` re-detects. The decay weights stay pending until :meth:`NewtonSolver.solve`
    commits them after the outer test."""
    contact.update_dual(
        fem.data.x,
        fem.xt,
        request_slack_update=True,
        request_decay_update=True,
        request_lagrange_multiplier_update=True,
    )
    cd.filter_step()


@profiling.scoped("solve.finalize_subproblem")
def update_constraint_set(contact: ContactDynamics, cd: ContactDetection, fresh: bool = False) -> None:
    """Re-detect contacts and update the constraint set for the next subproblem (or the next time step's warm
    start) -- the second half of ``vbd.solver.finalize_subproblem``. The dual state, pending decay weights
    included, carries across by the constraint set's warm start. ``fresh`` makes the re-detection full with
    ``freeze_within_solve`` (the solve's last one)."""
    cd.detect_contacts(fresh=fresh)
    contact.update_constraint_set()


def serialize_newton_cpu_params(params: pbat.sim.algorithm.newton.Params, grp) -> None:
    """Serialize a Newton CPU Params object to an h5py group."""
    grp.attrs["n_max_iters"] = params.n_max_iters
    grp.attrs["linear_solver"] = params.linear_solver.value
    grp.attrs["spd_correction"] = params.spd_correction.value
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
    if "spd_correction" in grp.attrs:
        params.spd_correction = pbat.fem.HyperElasticSpdCorrection(
            int(grp.attrs["spd_correction"])
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

    # TODO: GPU Newton runs aren't reproducible: two identical BeamFallingOnBeam runs (0.5 s, 50 steps) differ from
    # the first step, by up to 5.4e-3 m (2026-10-21), so checkpoint resumes are only approximate for Newton (the VBD
    # family's are bit-exact). The cause hasn't been located. Suspects: the atomic triplet/BSR Hessian assembly,
    # CG's dot-product reductions and the penalty adaptation's reductions.
    def solve(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        cd: ContactDetection,
        params: Params,
    ) -> bool:
        early_exit = _early_exit(params)
        subproblems, outer_tests = [], []
        initialize_solve(fem, contact, cd, params)
        converged = check_convergence(fem, contact, params)
        outer_tests.append(params.last_outer_ratios)
        n_max_iters = int(params.data.n_max_iters)
        # With freeze_within_solve, whether the last detection kept frozen stencils: the solve's last detection
        # must be full, since the next solve warm-starts from its keys
        frozen = False
        for k in range(n_max_iters):
            if converged and early_exit:
                break
            prepare_subproblem(fem, contact, cd, params)
            solve_subproblem(fem, contact, params)
            subproblems.append(params.last_subproblem)
            # The outer test (C++ newton::Solve tests before the next subproblem, and after the last one): the
            # gradient on the contact set the subproblem solved, with the new multipliers and the decay weights
            # it solved with (still pending), then feasibility on the re-detected contacts. Only then do the
            # dual update's weights become the next subproblem's
            finalize_subproblem(fem, contact, cd, params)
            gradient = outer_gradient_test(fem, contact, params)
            last = k == n_max_iters - 1
            update_constraint_set(contact, cd, fresh=last)
            frozen = cd.freezes_within_solve and not last
            converged = outer_feasibility_test(params, gradient)
            outer_tests.append(params.last_outer_ratios)
            contact.commit_decay()
        if frozen:
            # An early exit after a frozen re-detection (the VBD family decides this on the device, see
            # vbd.solver.solve_outer_loop)
            update_constraint_set(contact, cd, fresh=True)
        fem.back_substitute_velocities()
        cd.on_time_step_ended()
        # Host-side status of the solve, drained by SimulationState.step
        params.solve_records.append(
            NewtonSolveRecord(len(subproblems), converged, tuple(subproblems), tuple(outer_tests))
        )
        return converged

    @property
    def supports_graph_capture(self):
        return False


class TestHessianNorms(unittest.TestCase):
    """The stationarity test's float32 floor: after a Newton step the shared check reads Newton's vertex Hessian
    norms, the Frobenius norms of the last Newton step's assembled Hessian's diagonal 3x3 blocks, and the outer test
    records its decisive ratio; a solve's tests before its first Newton step have no floor. Two resting cubes
    (``vbd.convergence._two_resting_cubes``, rstol 0, astol 0.1)."""

    def test_norms_and_decisive_ratio(self):
        from ..vbd.convergence import SUM_FLOOR, _two_resting_cubes

        state = _two_resting_cubes("Newton")
        params, check = state.params[state.solver], state.convergence
        self.assertTrue(check.relative)
        self.assertIs(check._hessian_norms, params._hessian_norms)
        # The last assembly of a step is its last Newton step's (penalty adaptation assembles before the subproblems)
        blocks = warp.sparse.bsr_get_diag(params._H).numpy().reshape(-1, 9).astype(np.float64)
        expected = np.linalg.norm(blocks, axis=1)
        np.testing.assert_allclose(params._hessian_norms.numpy(), expected, rtol=1e-6)
        free = state.fem.data.dmask.numpy() == 0
        self.assertTrue(np.all(expected[free] > 0.0))
        # The recorded gradient ratio is the decisive one: at most 1 iff the test passes
        passed = check.test_gradient()
        self.assertEqual(check.decisive_ratio() <= 1.0, passed)
        self.assertGreater(float(check.sums.numpy()[SUM_FLOOR]), 0.0)
        # A solve's tests before its first Newton step (Newton's initial guess x^t) have no floor
        initialize_solve(state.fem, state.contact, state.detector, params)
        self.assertIsNone(check._hessian_norms)


class TestSubproblemGradient(unittest.TestCase):
    """The subproblem gradient refreshes the contact slack (only) at the iterate first."""

    def test_slack_refresh_precedes_gradient(self):
        calls = []
        fem = SimpleNamespace(data=SimpleNamespace(x="x"), xt="xt")
        contact = SimpleNamespace(update_dual=lambda *args, **kwargs: calls.append(("update_dual", args, kwargs)))
        params = SimpleNamespace(
            _gradient_norm=SimpleNamespace(compute_gradient=lambda stream: calls.append(("gradient", stream)))
        )
        _subproblem_gradient(fem, contact, params, "stream")
        self.assertEqual(
            calls,
            [
                (
                    "update_dual",
                    ("x", "xt"),
                    dict(
                        request_slack_update=True,
                        request_decay_update=False,
                        request_lagrange_multiplier_update=False,
                    ),
                ),
                ("gradient", "stream"),
            ],
        )


class TestEarlyExit(unittest.TestCase):
    """Solver.Newton.early_exit overrides Solver.Convergence.early_exit unless Shared (the default)."""

    def test_override(self):
        def early_exit(newton, shared, check=True):
            convergence = SimpleNamespace(params=SimpleNamespace(early_exit=shared)) if check else None
            gpu = NewtonGpuParams()
            if newton is not None:
                gpu.early_exit = newton
            return _early_exit(SimpleNamespace(convergence=convergence, gpu=gpu))

        self.assertEqual(NewtonGpuParams().early_exit, NewtonEarlyExit.Shared)
        for shared in (False, True):
            self.assertEqual(early_exit(None, shared), shared)
            self.assertEqual(early_exit(NewtonEarlyExit.Shared, shared), shared)
            self.assertTrue(early_exit(NewtonEarlyExit.On, shared))
            self.assertFalse(early_exit(NewtonEarlyExit.Off, shared))
        self.assertTrue(early_exit(None, False, check=False))  # no shared check: on
        self.assertTrue(_early_exit(SimpleNamespace(convergence=None)))  # no GPU params: Shared


class TestOuterTestOrder(unittest.TestCase):
    """After each subproblem, the outer test's gradient is measured on the contact set the subproblem solved (after
    the full dual update and the filter, before re-detection), feasibility after re-detection, and the decay
    weights are committed last. Stubbed solver steps, relative test, early exit off."""

    def test_gradient_before_redetection(self):
        from unittest import mock

        calls = []
        detections = [0]  # re-detections so far: the contact set a measure sees

        class Check:
            relative = True
            params = SimpleNamespace(early_exit=False)
            ratio = float("nan")

            def test_gradient(self):
                calls.append("gradient")
                self.ratio = float(detections[0])
                return False

            def decisive_ratio(self):
                return self.ratio

            def test_feasibility(self):
                calls.append("feasibility")
                return True

            def feasibility_ratio(self):
                return 0.0

        def detect_contacts(**kwargs):
            calls.append("detect")
            detections[0] += 1

        fem = SimpleNamespace(data=SimpleNamespace(x="x"), xt="xt", back_substitute_velocities=lambda: None)
        contact = SimpleNamespace(
            params=SimpleNamespace(penalty_adaptivity=PenaltyAdaptivity.TIMESTEP),
            update_dual=lambda *args, **kwargs: calls.append(
                "full dual" if kwargs.get("request_lagrange_multiplier_update") else "slack"
            ),
            update_constraint_set=lambda: calls.append("constraint set"),
            commit_decay=lambda: calls.append("commit"),
        )
        cd = SimpleNamespace(
            filter_step=lambda: calls.append("filter"),
            detect_contacts=detect_contacts,
            on_time_step_ended=lambda: None,
            freezes_within_solve=False,
        )
        params = SimpleNamespace(
            data=SimpleNamespace(n_max_iters=2),
            convergence=Check(),
            last_subproblem=None,
            last_outer_ratios=None,
            solve_records=[],
        )
        stubs = dict(
            initialize_solve=lambda *args: calls.append("initialize"),
            solve_subproblem=lambda *args: calls.append("subproblem"),
        )
        with mock.patch.dict(globals(), stubs):
            NewtonSolver().solve(fem, contact, cd, params)
        after_subproblem = ["subproblem", "full dual", "filter", "gradient", "detect", "constraint set", "feasibility", "commit"]
        self.assertEqual(calls, ["initialize", "gradient", "feasibility"] + 2 * after_subproblem)
        # Each subproblem's gradient ratio is measured on the contact set it solved
        self.assertEqual([r[0] for r in params.solve_records[0].outer_tests], [0.0, 0.0, 1.0])


class TestFreezeWithinSolve(unittest.TestCase):
    """With a detector that freezes linearizations within a solve, the solve's last detection is full: the last
    scheduled outer iteration's, or one more after an early exit. Stubbed solver steps."""

    def _detections(self, n_max_iters: int, converge_after: int | None, freezes: bool = True) -> list:
        from unittest import mock

        detections, tests = [], [0]

        class Check:
            relative = True
            params = SimpleNamespace(early_exit=True)

            def test_gradient(self):
                return True

            def decisive_ratio(self):
                return 0.0

            def test_feasibility(self):
                tests[0] += 1
                # The first test is before any subproblem
                return converge_after is not None and tests[0] > converge_after

            def feasibility_ratio(self):
                return 0.0

        fem = SimpleNamespace(data=SimpleNamespace(x="x"), xt="xt", back_substitute_velocities=lambda: None)
        contact = SimpleNamespace(
            params=SimpleNamespace(penalty_adaptivity=PenaltyAdaptivity.TIMESTEP),
            update_dual=lambda *args, **kwargs: None,
            update_constraint_set=lambda: None,
            commit_decay=lambda: None,
        )
        def detect_contacts(fresh: bool = False):
            detections.append(fresh)

        cd = SimpleNamespace(
            filter_step=lambda: None,
            detect_contacts=detect_contacts,
            on_time_step_ended=lambda: None,
            freezes_within_solve=freezes,
        )
        params = SimpleNamespace(
            data=SimpleNamespace(n_max_iters=n_max_iters),
            convergence=Check(),
            last_subproblem=None,
            last_outer_ratios=None,
            solve_records=[],
        )
        stubs = dict(initialize_solve=lambda *args: None, solve_subproblem=lambda *args: None)
        with mock.patch.dict(globals(), stubs):
            NewtonSolver().solve(fem, contact, cd, params)
        return detections

    def test_last_detection_is_fresh(self):
        # Budget reached: only the last scheduled re-detection is full
        self.assertEqual(self._detections(3, None), [False, False, True])
        # Early exit after the first outer iteration: a full detection follows its frozen one
        self.assertEqual(self._detections(3, 1), [False, True])
        # Converged before any subproblem: no re-detection (the solve's first detection, at xt, is full)
        self.assertEqual(self._detections(3, 0), [])
        # Without freezing, no extra detection
        self.assertEqual(self._detections(3, 1, freezes=False), [False])


class TestEeContactHessianTriplets(unittest.TestCase):
    """An edge-edge contact's Hessian triplets land in the EE slots, scaled by its own decay gamma (Warp's CPU
    device). They used to overwrite the VF slots, scaled by the VF contact's gamma at the same index."""

    def test_one_edge_edge_contact(self):
        from ..contact.constraints import ConstraintSetData
        from ..contact.mesh.pairs import ContactPairsData
        from ..contact.multimesh import MultiMeshData
        from ..contact.pairs import PairsData

        with wp.ScopedDevice("cpu"):
            # Half-edge he of face he // 3 runs from F[he // 3][he % 3] to the next vertex: he 0 is edge (0, 1)
            # and he 3 is edge (3, 4). Vertices 2 and 5 only complete the faces.
            F = [(0, 1, 2), (3, 4, 5)]
            x = np.array(
                [[0, 0, 0], [1, 0, 0], [9, 9, 9], [0.5, -0.5, 0.01], [0.5, 0.5, 0.01], [9, 9, 9]], dtype=np.float32
            )
            n_u = 6
            ee = PairsData()
            ee.prefix = wp.array([0] + [1] * n_u, dtype=wp.uint64)  # one contact, of half-edge u = 0
            ee.u = wp.array([0], dtype=wp.uint32)
            ee.v = wp.array([3], dtype=wp.uint32)
            bases = ContactBasesData()
            n, t, b = np.eye(3, dtype=np.float32)[[2, 0, 1]]
            bases.n, bases.t, bases.b = (wp.array([a], dtype=wp.vec3f) for a in (n, t, b))
            contacts = ContactPairsData()
            contacts.ee = ee
            contacts.ee_bases = bases
            b1, b3 = 0.25, 0.5
            contacts.ee_bary = wp.array([(b1, b3)], dtype=wp.vec2f)
            meshes = MultiMeshData()
            meshes.F = wp.array(F, dtype=wp.vec3i)

            def constraint_set(gamma):
                cs = ConstraintSetData()
                cs.s = wp.zeros(1, dtype=wp.float32)
                cs.gamma = wp.array([gamma], dtype=wp.float32)
                cs.lambda_n = wp.array([1.0], dtype=wp.float32)
                cs.lambda_f = wp.zeros(1, dtype=wp.vec2f)
                return cs

            contact = ContactDynamicsData()
            contact.meshes = meshes
            contact.contacts = contacts
            contact.cvf = constraint_set(0.5)
            gamma_ee = 2.0
            contact.cee = constraint_set(gamma_ee)
            contact.gamma_n = 1.0
            contact.gamma_f = 1.0
            sigma_n, sigma_f = 100.0, 10.0
            contact.sigma_n = wp.array([sigma_n], dtype=wp.float32)
            contact.sigma_f = wp.array([sigma_f], dtype=wp.float32)
            contact.dmin = 0.0
            contact.mu_f = 0.2
            n_slots = 2 * 16  # VF slots 0-15, then EE slots 16-31
            params = ParamsData()
            params.Hrows = wp.full(n_slots, -1, dtype=wp.int32)
            params.Hcols = wp.full(n_slots, -1, dtype=wp.int32)
            params.Hvals = wp.zeros(n_slots, dtype=wp.mat33f)
            offsets = np.zeros(K_EE_TRIPLETS + 1, dtype=np.int32)
            offsets[K_VF_TRIPLETS], offsets[K_EE_TRIPLETS] = 0, 16
            xs = wp.array(x, dtype=wp.vec3f)
            # Inputs by name, so the test holds whatever else the kernel reads
            values = dict(
                x=xs,
                xt=xs,  # no tangential motion: f = -lambda_f = 0, so the friction scale is 1
                dmask=wp.zeros(len(x), dtype=wp.int32),
                contact=contact,
                n_u=n_u,
                friction_limit=wp.full(1, 1.0, dtype=wp.float32),
                params=params,
                offsets=wp.array(offsets, dtype=wp.int32),
            )
            kernel = _compute_ee_contact_hessian_triplets
            wp.launch(kernel, dim=1, inputs=[values[arg.label] for arg in kernel.adj.args])
            rows, cols, vals = params.Hrows.numpy(), params.Hcols.numpy(), params.Hvals.numpy()
        self.assertTrue((rows[:16] == -1).all() and (cols[:16] == -1).all(), "EE triplets wrote VF slots")
        H = gamma_ee * (sigma_n * np.outer(n, n) + sigma_f * (np.outer(t, t) + np.outer(b, b)))
        nodes, coef = (0, 1, 3, 4), (1 - b1, b1, -(1 - b3), -b3)
        for r in range(4):
            for c in range(4):
                k = 16 + 4 * r + c
                self.assertEqual((rows[k], cols[k]), (nodes[r], nodes[c]))
                np.testing.assert_allclose(vals[k], coef[r] * coef[c] * H, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

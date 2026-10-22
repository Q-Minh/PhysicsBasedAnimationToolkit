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
    MeshDynamics as ContactDynamics,
    MeshDynamicsData as ContactDynamicsData,
    PenaltyAdaptivity,
    capped_friction_energy,
    capped_friction_scale,
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
    c_n = wp.dot(xi - xj, n) - contact.dmin - contact.cvv.s[c]  # type: ignore
    du = (xi - xti) - (xj - xtj)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    E = contact.cvv.gamma[c] * (
        wp.float32(0.5) * sigma_n * c_n * c_n  # type: ignore
        - contact.cvv.lambda_n[c] * c_n
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
    c_n = wp.dot(xi - xcp2, n) - contact.dmin - contact.cve.s[c]
    du = (xi - xti) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    E = contact.cve.gamma[c] * (
        wp.float32(0.5) * sigma_n * c_n * c_n  # type: ignore
        - contact.cve.lambda_n[c] * c_n
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
    c_n = wp.dot(xi - xcp2, n) - contact.dmin - contact.cvf.s[c]
    du = (xi - xti) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    E = contact.cvf.gamma[c] * (
        wp.float32(0.5) * sigma_n * c_n * c_n  # type: ignore
        - contact.cvf.lambda_n[c] * c_n
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
    c_n = wp.dot(xcp1 - xcp2, n) - contact.dmin - contact.cee.s[c]
    du = (xcp1 - xtcp1) - (xcp2 - xtcp2)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    E = contact.cee.gamma[c] * (
        wp.float32(0.5) * sigma_n * c_n * c_n  # type: ignore
        - contact.cee.lambda_n[c] * c_n
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
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    bases = contact.contacts.vv_bases
    n, t, b = bases.n[c], bases.t[c], bases.b[c]
    gamma = contact.cvv.gamma[c]
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
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
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
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
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
    gamma = contact.cvf.gamma[c]
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
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
    offset = offsets[K_VF_TRIPLETS] + c * 16  # 4^2
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


class NewtonGpuParams(SerializableMixin):
    """GPU-only Newton settings, serialized next to the C++ Newton params under ``Solver/Newton``.

    Held by ``SimulationState.newton_gpu_params`` (kept across resets): configs set
    ``Solver.Newton.init_strategy``. It takes effect at the next time step.
    """

    init_strategy = DocField(
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position,
        "Newton's initial guess for each time step, used instead of Integration.init_strategy whenever Newton is "
        "the active solver (the VBD family keeps the shared one). Position (x_t) by default: Newton's first step "
        "linearizes at the initial guess, and an extrapolated guess (e.g. TrajectoryWithExternalLoad) can start "
        "deep inside a contact partner at an impact, where that linearization is poor (Newton went unstable at the "
        "sphere's impact on the soft-stiff trampoline)",
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
        self._streams = [wp.Stream() for _ in range(6)]
        # Shared device-side convergence test (gpu.vbd.convergence.ConvergenceCheck), set by
        # SimulationState. With its relative test on, it replaces gtol2 (see check_convergence).
        self.convergence = None
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
        self._gradient_norm = GradientNorm(fem, contact)
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
    cd.on_time_step_started()
    cd.detect_contacts(from_xt=True)
    contact.update_constraint_set()
    contact.reset_friction_limits()
    cd.filter_step()
    if contact.params.penalty_adaptivity == PenaltyAdaptivity.TIMESTEP:
        adapt_penalty_parameters(fem, contact, params)
    check = _relative_check(params)
    if check is not None:
        check.measure_scale()


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
        inputs=[x, xt, cd, contact.cvv.n_u, limits.vv, f_vv],
        stream=params._streams[2],
    )
    wp.launch(
        _energy_ve,
        dim=contact.cve.capacity,
        inputs=[x, xt, cd, contact.cve.n_u, limits.ve, f_ve],
        stream=params._streams[3],
    )
    wp.launch(
        _energy_vf,
        dim=contact.cvf.capacity,
        inputs=[x, xt, cd, contact.cvf.n_u, limits.vf, f_vf],
        stream=params._streams[4],
    )
    wp.launch(
        _energy_ee,
        dim=contact.cee.capacity,
        inputs=[x, xt, cd, contact.cee.n_u, limits.ee, f_ee],
        stream=params._streams[5],
    )
    for stream in params._streams:
        main_stream.wait_stream(stream)
    params._energy_reduce(main_stream)
    return float(params._energy_scalar.numpy()[0])


@profiling.scoped("newton.convergence")
def check_convergence(
    fem: FemElastoDynamics,
    contact: ContactDynamics,
    params: Params,
) -> bool:
    """Outer test between AL iterations.

    With the shared convergence check's stationarity test on (``Solver.Convergence.rstol`` or ``astol > 0``,
    dynamic problems), it's the VBD family's outer test: the Lagrangian gradient,
    ``sum |g_i|^2 / m_i <= max(rstol^2 sum m_i |x~_i - x^t_i|^2, (astol |a| h^2)^2 sum m_i)``, and feasibility
    (penetration into the contact margin at most ``ftol * max(|dmin|, FEASIBILITY_FLOOR)``). The Lagrangian gradient has no
    term for a new contact (lambda = 0), so without the feasibility test a penetrating free-flight
    guess passes before any subproblem runs. Otherwise it's ``|g|^2 <= gtol2`` on the
    augmented-Lagrangian gradient, as in C++.

    Sets ``params.last_outer_ratios`` to the test's (gradient, feasibility) measures over their
    tolerances, as the VBD family's status ratios (at most 1 passes). The absolute test doesn't include
    feasibility, but its ratio is still measured when the shared check exists (NaN otherwise).
    """
    check = _relative_check(params)
    if check is not None:
        passed = check.test_outer()
        params.last_outer_ratios = (check.gradient_ratio(), check.feasibility_ratio())
        return passed
    main_stream = wp.get_stream()
    params._gradient_norm.compute(main_stream)
    gnorm2, gtol2 = float(params._gnorm2.numpy()[0]), float(params.data.gtol2)
    feasibility = params.convergence.feasibility_ratio(measure=True) if params.convergence is not None else float("nan")
    params.last_outer_ratios = (float(np.sqrt(ratio_to_tolerance(gnorm2, gtol2))), feasibility)
    return gnorm2 <= gtol2


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
    contact.adapt_penalty_parameters_to_hessian(params._H, fem.data.dmask)


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
    (gradient, feasibility) ratios to their tolerances of every outer test (``check_convergence``), in
    order: the test before each subproblem, then the final one, so ``len(outer_tests) ==
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


# TODO: One-sided (slack-eliminated) AL contact terms with a semismooth Newton. With s fixed within a Newton
# step, a contact inside the detection band but not pressed is a two-sided spring of stiffness sigma_n
# (adapted to the stiffest partner) that holds its gap for the step. Eliminating s gives
# E = (max(0, lambda - sigma c)^2 - lambda^2) / (2 sigma), C^1, with the Hessian block sigma n n^T only on
# active contacts (lambda - sigma c > 0). Evidence: after the slack refresh, the soft-stiff trampoline +
# sphere (steps 253-256) still has a contact-region roughness p95 of ~0.065, against AAAVBD's 0.033. A
# GPU-only deviation from C++ (which fixes s per step); risk: active-set switching between Newton steps
# and within the line search (energy only C^1, Hessian discontinuous) can cost backtracks or chatter.
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
    (``Solver.Convergence.rstol`` or ``astol > 0``, dynamic problems), or else ``|g|^2 <= gtol2``. Each
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
            if check.relative_test(params._g):
                return done(SubproblemStatus.CONVERGED)
        else:
            params._gradient_norm.reduce(main_stream)
            gnorm2 = float(params._gnorm2.numpy()[0])
            if gnorm2 <= float(params.data.gtol2):
                return done(SubproblemStatus.CONVERGED)
        # Assemble FEM + contact Hessian
        profiling.push("newton.assemble")
        _assemble_hessian(fem, contact, params, h2)
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

    def solve(
        self,
        fem: FemElastoDynamics,
        contact: ContactDynamics,
        cd: ContactDetection,
        params: Params,
    ) -> bool:
        converged = False
        subproblems, outer_tests = [], []
        initialize_solve(fem, contact, cd, params)
        for _ in range(int(params.data.n_max_iters)):
            converged = check_convergence(fem, contact, params)
            outer_tests.append(params.last_outer_ratios)
            if converged:
                break
            prepare_subproblem(fem, contact, cd, params)
            solve_subproblem(fem, contact, params)
            subproblems.append(params.last_subproblem)
            finalize_subproblem(fem, contact, cd, params)
        if not converged:
            # As C++ newton::Solve: test the last subproblem's result too (a test, not a step)
            converged = check_convergence(fem, contact, params)
            outer_tests.append(params.last_outer_ratios)
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


if __name__ == "__main__":
    unittest.main()

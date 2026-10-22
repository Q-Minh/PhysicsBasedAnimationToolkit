import unittest

import warp as wp
from ...common import profiling
import cupy as cp
import numpy as np

from . import pairs
from ..multimesh import MultiMesh, MultiMeshData
from .. import halfedges
from .. import queries
from .cd import ContactDetection
from ...common import reduce, lower_bound
from ....common.fields import DocField, SerializableMixin

# Kinds of meshes, by how a vertex's signed distance to them is computed (see _mesh_kind)
MESH_CLOSED = wp.constant(0)  # ray-parity sign
MESH_OPEN = wp.constant(1)  # sign of the closest face's normal
MESH_PLANE = wp.constant(2)  # planar and open (plane colliders): analytic plane distance, unbounded range

MAX_VV_PER_THREAD = wp.constant(2)
MAX_VE_PER_THREAD = wp.constant(2)
MAX_VF_PER_THREAD = wp.constant(4)
# MAX_EE_PER_THREAD = wp.constant(8)
tvvlist = wp.types.vector(length=MAX_VV_PER_THREAD, dtype=wp.int32)
tvelist = wp.types.vector(length=MAX_VE_PER_THREAD, dtype=wp.int32)
tvflist = wp.types.vector(length=MAX_VF_PER_THREAD, dtype=wp.int32)
# teelist = wp.types.vector(length=MAX_EE_PER_THREAD, dtype=wp.int32)


@wp.kernel
def _detect_vertex_mesh_contacts(
    mesh_ids: wp.array[wp.uint64],
    mesh_kind: wp.array[wp.int32],
    mesh_depth: wp.array[wp.float32],
    max_dist: wp.float32,
    threshold: wp.float32,
    sds: wp.array[wp.float32],
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    contacts: pairs.ContactPairsData,  # type: ignore
):
    tid = wp.tid()
    block_dim = wp.block_dim()
    block_id = tid // block_dim  # pyright: ignore[reportOperatorIssue]
    local_tid = tid % block_dim  # pyright: ignore[reportOperatorIssue]
    v = block_id
    b = lower_bound(meshes.VP, meshes.VP.shape[0], v + 1) - 1  # type: ignore
    i = meshes.V[v]
    xi = x[i]
    n_verts, n_half_edges, n_tris = (
        meshes.V.shape[0],
        3 * meshes.F.shape[0],
        meshes.F.shape[0],
    )
    tnvv = wp.int32(0)
    tnve = wp.int32(0)
    tnvf = wp.int32(0)
    tvv = tvvlist(n_verts)
    tve = tvelist(n_half_edges)
    tvf = tvflist(n_tris)

    # 1. Detect thread local contacts
    tsdmin = max_dist
    # The loop variable m must not be reassigned in the body (k below is a node index).
    for m in range(local_tid, mesh_ids.shape[0], block_dim):
        if m == b:
            continue
        # Closed meshes sign the distance by ray parity. Open ones can't, so they use the closest face's
        # normal. Planes are queried at any distance, so that fast vertices can't tunnel through them.
        kind = mesh_kind[m]
        if kind == MESH_CLOSED:
            qp = wp.mesh_query_point(mesh_ids[m], xi, max_dist)  # type: ignore
        elif kind == MESH_OPEN:
            qp = wp.mesh_query_point_sign_normal(mesh_ids[m], xi, max_dist)  # type: ignore
        else:
            qp = wp.mesh_query_point_no_sign(mesh_ids[m], xi, wp.float32(1e6))  # type: ignore
        if not qp.result:  # type: ignore
            continue
        f = meshes.FP[m] + qp.face  # type: ignore
        finds = meshes.F[f]
        j, k, l = finds[0], finds[1], finds[2]
        xj, xk, xl = x[j], x[k], x[l]
        # Warp's convention: the closest point is u * x_j + v * x_k + (1 - u - v) * x_l
        b0, b1, b2 = qp.u, qp.v, (wp.float32(1) - qp.u - qp.v)  # type: ignore
        xc = b0 * xj + b1 * xk + b2 * xl  # type: ignore
        sd = qp.sign * wp.norm_l2(xi - xc)  # type: ignore
        if kind == MESH_PLANE:
            # Signed distance to the plane, for vertices over the plane's extent only (the closest
            # point is then the vertex's projection)
            n = wp.normalize(wp.cross(xk - xj, xl - xj))
            sd = wp.dot(xi - xc, n)
            # The tolerance scales with the face, since xc's float32 error grows with its vertices'
            # distance (infinite planes are large quads)
            tangential = wp.norm_l2(xi - xc - sd * n)
            tol = wp.float32(1e-6) * (wp.norm_l2(xk - xj) + wp.norm_l2(xl - xj)) + wp.float32(1e-5) * wp.abs(sd)
            if tangential > tol:
                continue
            # Finite planes are plates: vertices deeper than their thickness passed under or beside them
            if sd < -mesh_depth[m]:
                continue
        tsdmin = wp.min(tsdmin, sd)
        # TODO: Runs aren't reproducible when vertices sit exactly over a partner's vertices: two unit cubes
        # stacked 2 mm apart with aligned grids (checkpoint.py's test scene without its xy offset) gave VBD and
        # AAAVBD runs differing from the second step, by up to 1 cm after 10 steps (2026-10-21). Suspected: the
        # closest point is then a shared vertex or edge, so which face the query returns, and so the feature
        # below (VF, VE or VV, from exact zero barycentrics), is a tie. Offsetting the cube made runs
        # bit-identical. Not the feature: since it comes from the closest point's region (the same whichever
        # face shares the closest edge or vertex), one of two repeat runs still differed from the first step
        # (VBD 7.8 mm, AAAVBD 11.5 mm after 10 steps, 2026-10-22). Left to check: the signed distance's last
        # bits (from the returned face's barycentrics), and the order-dependent work after detection.
        if sd < threshold:
            # The closest feature (face interior, edge or vertex) comes from the closest point's Voronoi region
            # of the face, whose barycentrics are exactly zero off the interior, as C++ OGC's
            # (UvwPointInTriangle) and the second pass's (pairs._compute_vf_contact_data). Warp's barycentrics
            # aren't: w = 1 - u - v is ~1e-8 on the edge jk, which made such contacts VF, with the face normal.
            # Behind a sharp edge (faces' normals more than 90 deg apart) the vertex is then behind that face's
            # plane though outside the body, a false penetration of up to the distance (ArmadilloSpaghetti.Fall).
            # The feature no longer depends on which of the faces sharing the closest edge or vertex the query
            # returns.
            uvw = queries.closest_point_triangle(xi, xj, xk, xl)
            is_j_zero = wp.int32(uvw[0] == wp.float32(0))
            is_k_zero = wp.int32(uvw[1] == wp.float32(0))
            is_l_zero = wp.int32(uvw[2] == wp.float32(0))
            nz = is_j_zero + is_k_zero + is_l_zero
            if nz == 0:
                # Triangle contact
                assert tnvf < MAX_VF_PER_THREAD
                tvf[tnvf] = f
                tnvf += 1
            elif nz == 1:
                # Edge contact
                assert tnve < MAX_VE_PER_THREAD
                helocal = is_k_zero * wp.int32(1) + is_l_zero * wp.int32(2)  # type: ignore
                he_candidate = halfedges.next_half_edge(
                    halfedges.half_edge_of_face(f, helocal)
                )
                he = wp.max(
                    he_candidate,
                    halfedges.opposite_half_edge(meshes.F, he_candidate, meshes.GHEF),
                )
                tve[tnve] = he
                tnve += 1
            else:
                # Vertex contact
                assert tnvv < MAX_VV_PER_THREAD
                jnode = wp.int32(not is_j_zero) * j + wp.int32(not is_k_zero) * k + wp.int32(not is_l_zero) * l  # type: ignore
                tvv[tnvv] = meshes.GXV[jnode]  # type: ignore
                tnvv += 1

    # 2. Use block-level primitives to avoid atomic contention on global memory when writing contact pairs.
    # Contacts that don't fit the pair storage's capacity are dropped (raise the n_*_contact_capacity params).
    # NOTE:
    # 1. All contact pairs should be unique here, because this block's vertex only detects 1 contact
    # per every other mesh.
    # 2. We still need to sort the contacts by their partner indices, this is a precondition to ContactPairs.assemble_contacts.
    tnvv, tnve, tnvf = (
        wp.tile_extract(wp.tile_sum(wp.tile(tnvv)), 0),  # type: ignore
        wp.tile_extract(wp.tile_sum(wp.tile(tnve)), 0),  # type: ignore
        wp.tile_extract(wp.tile_sum(wp.tile(tnvf)), 0),  # type: ignore
    )
    has_vv_contacts, has_ve_contacts, has_vf_contacts = tnvv > 0, tnve > 0, tnvf > 0
    if has_vv_contacts:
        bvv = wp.tile(tvv)  # type: ignore
        wp.tile_sort(keys=bvv, values=bvv)
        vv_offset = wp.uint64(0)
        if local_tid == 0:
            vv_offset = wp.atomic_add(contacts.vv.prefix, n_verts, wp.uint64(tnvv))
        vv_offset = wp.tile_extract(wp.tile_from_thread(shape=1, value=vv_offset, thread_idx=0), 0)  # type: ignore
        for block_row in range(MAX_VV_PER_THREAD):
            c = vv_offset + wp.uint64(block_row * block_dim + local_tid)
            if bvv[block_row, local_tid] < n_verts and c < wp.uint64(contacts.vv.u.shape[0]):
                contacts.vv.u[c] = wp.uint32(v)
                contacts.vv.v[c] = wp.uint32(bvv[block_row, local_tid])
    if has_ve_contacts:
        bve = wp.tile(tve)  # type: ignore
        wp.tile_sort(keys=bve, values=bve)
        ve_offset = wp.uint64(0)
        if local_tid == 0:
            ve_offset = wp.atomic_add(contacts.ve.prefix, n_verts, wp.uint64(tnve))
        ve_offset = wp.tile_extract(wp.tile_from_thread(shape=1, value=ve_offset, thread_idx=0), 0)  # type: ignore
        for block_row in range(MAX_VE_PER_THREAD):
            c = ve_offset + wp.uint64(block_row * block_dim + local_tid)
            if bve[block_row, local_tid] < n_half_edges and c < wp.uint64(contacts.ve.u.shape[0]):
                contacts.ve.u[c] = wp.uint32(v)
                contacts.ve.v[c] = wp.uint32(bve[block_row, local_tid])
    if has_vf_contacts:
        bvf = wp.tile(tvf)  # type: ignore
        wp.tile_sort(keys=bvf, values=bvf)
        vf_offset = wp.uint64(0)
        if local_tid == 0:
            vf_offset = wp.atomic_add(contacts.vf.prefix, n_verts, wp.uint64(tnvf))
        vf_offset = wp.tile_extract(wp.tile_from_thread(shape=1, value=vf_offset, thread_idx=0), 0)  # type: ignore
        for block_row in range(MAX_VF_PER_THREAD):
            c = vf_offset + wp.uint64(block_row * block_dim + local_tid)
            if bvf[block_row, local_tid] < n_tris and c < wp.uint64(contacts.vf.u.shape[0]):
                contacts.vf.u[c] = wp.uint32(v)
                contacts.vf.v[c] = wp.uint32(bvf[block_row, local_tid])

    # 3. Update signed distances
    bsdmin = wp.tile(tsdmin)  # type: ignore
    sdmin = wp.tile_extract(wp.tile_min(bsdmin), 0)  # type: ignore
    sds[v] = sdmin  # type: ignore


@wp.func
def _face_normal(x: wp.array[wp.vec3f], F: wp.array[wp.vec3i], f: wp.int32) -> wp.vec3f:
    """Unnormalized normal of face f, of length twice its area."""
    finds = F[f]
    xj = x[finds[0]]
    return wp.cross(x[finds[1]] - xj, x[finds[2]] - xj)


@wp.func
def _unit_face_normal(x: wp.array[wp.vec3f], F: wp.array[wp.vec3i], f: wp.int32) -> wp.vec3f:
    return wp.normalize(_face_normal(x, F, f))


@wp.func
def _oriented_normal(d: wp.vec3f, pseudonormal: wp.vec3f) -> wp.vec3f:
    """The contact normal: d (from the partner's closest point to the vertex) normalized and signed by the
    partner's pseudonormal, which tells the outside of its surface. For a vertex on the surface (d ~ 0), the
    pseudonormal itself."""
    if wp.length(d) <= wp.float32(1e-7) * (wp.float32(1) + wp.length(pseudonormal)):
        return wp.normalize(pseudonormal)
    n = wp.normalize(d)
    if wp.dot(d, pseudonormal) < wp.float32(0):
        n = -n
    return n


@wp.kernel
def _orient_vv_contact_normals(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    cpairs: pairs.PairsData,  # type: ignore
    bases: pairs.ContactBasesData,  # type: ignore
    n_u: wp.int32,
):
    k = wp.tid()
    if wp.uint64(k) >= cpairs.prefix[n_u]:  # type: ignore
        return
    i, j = meshes.V[wp.int32(cpairs.u[k])], meshes.V[wp.int32(cpairs.v[k])]
    # Angle-weighted vertex pseudonormal of j: the sign of d against it is exact
    pn = wp.vec3f(0.0)
    for a in range(meshes.GVHEp[j], meshes.GVHEp[j + 1]):
        f = halfedges.face_of_half_edge(meshes.GVHEadj[a])
        finds = meshes.F[f]
        p, q = finds[1], finds[2]
        if finds[1] == j:
            p, q = finds[2], finds[0]
        elif finds[2] == j:
            p, q = finds[0], finds[1]
        e1, e2 = wp.normalize(x[p] - x[j]), wp.normalize(x[q] - x[j])
        angle = wp.acos(wp.clamp(wp.dot(e1, e2), wp.float32(-1), wp.float32(1)))
        pn += angle * _unit_face_normal(x, meshes.F, f)
    n = _oriented_normal(x[i] - x[j], pn)
    t, b = pairs._build_contact_basis(n)
    bases.n[k] = n
    bases.t[k] = t
    bases.b[k] = b


@wp.kernel
def _orient_ve_contact_normals(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    cpairs: pairs.PairsData,  # type: ignore
    bases: pairs.ContactBasesData,  # type: ignore
    bary: wp.array[wp.float32],
    n_u: wp.int32,
):
    k = wp.tid()
    if wp.uint64(k) >= cpairs.prefix[n_u]:  # type: ignore
        return
    i = meshes.V[wp.int32(cpairs.u[k])]
    he = wp.int32(cpairs.v[k])
    xa = x[halfedges.incoming_vertex(meshes.F, he)]
    xb = x[halfedges.outgoing_vertex(meshes.F, he)]
    xc = (wp.float32(1) - bary[k]) * xa + bary[k] * xb
    # Edge pseudonormal: the sum of its faces' unit normals
    pn = _unit_face_normal(x, meshes.F, halfedges.face_of_half_edge(he))
    fo = meshes.GHEF[he][1]
    if fo >= 0:
        pn += _unit_face_normal(x, meshes.F, fo)
    n = _oriented_normal(x[i] - xc, pn)
    t, b = pairs._build_contact_basis(n)
    bases.n[k] = n
    bases.t[k] = t
    bases.b[k] = b


@wp.kernel
def _orient_vf_contact_normals(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # type: ignore
    cpairs: pairs.PairsData,  # type: ignore
    bases: pairs.ContactBasesData,  # type: ignore
    n_u: wp.int32,
):
    k = wp.tid()
    if wp.uint64(k) >= cpairs.prefix[n_u]:  # type: ignore
        return
    # The closest point is inside the face, so the face's (outward) normal is the contact normal on both
    # sides of the surface
    n = _unit_face_normal(x, meshes.F, wp.int32(cpairs.v[k]))
    t, b = pairs._build_contact_basis(n)
    bases.n[k] = n
    bases.t[k] = t
    bases.b[k] = b


@wp.kernel
def _filter_initial_step(
    sds: wp.array[wp.float32],
    meshes: MultiMeshData,  # type: ignore
    xt: wp.array[wp.vec3f],
    dmask: wp.array[wp.int32],
    x: wp.array[wp.vec3f],
):
    tid = wp.tid()
    v = tid
    i = meshes.V[v]
    if dmask[i] != 0:
        return
    if sds[v] <= wp.float32(0):
        x[i] = xt[i]  # type: ignore


def _mesh_kind(F: np.ndarray, X: np.ndarray) -> int:
    """Kind of the triangle mesh F (|# triangles| x 3) over points X (|# points| x 3).

    MESH_CLOSED if every edge is shared by exactly 2 triangles, else MESH_PLANE if all its vertices lie in
    the plane of its first triangle (plane colliders), else MESH_OPEN.
    """
    E = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(E, axis=0, return_counts=True)
    if np.all(counts == 2):
        return int(MESH_CLOSED)
    P = X[np.unique(F)]
    n = np.cross(X[F[0, 1]] - X[F[0, 0]], X[F[0, 2]] - X[F[0, 0]])
    n /= np.linalg.norm(n)
    diameter = np.linalg.norm(P.max(axis=0) - P.min(axis=0))
    planar = np.all(np.abs((P - X[F[0, 0]]) @ n) <= 1e-6 * diameter)
    return int(MESH_PLANE) if planar else int(MESH_OPEN)


class Params(SerializableMixin):
    max_dist = DocField(
        0.05,
        "Maximum distance for closest point computations. It must exceed the detection threshold "
        "Contact.ContactDynamics.dmin + band, and bounds the signed distances that are tracked.",
    )
    band = DocField(
        1e-3,
        "Contacts are created for vertices closer than Contact.ContactDynamics.dmin + band to another mesh "
        "(or inside it). The band above the AL margin dmin keeps resting contacts, which the AL pushes to dmin, "
        "detected, so their multipliers persist. (Replaces the creation threshold `dmin`, an absolute signed "
        "distance whose default -1e-4 detected penetrating vertices only.)",
    )
    use_step_filter = DocField(
        False,
        "Whether to apply initial step filter.",
    )
    deactivate = DocField(False, "Whether to deactivate contact detection.")


class Sd(ContactDetection):
    """Vertex-(Mesh)SDF based contact detection."""

    # TODO: CubeFallingOnCube's cubes pass through each other, before and after the detection band (2026-10-20).
    # Detection is vertex-only (no edge-edge pairs), which misses edges crossing with no vertex inside; edge-edge
    # pairs are the likely fix.

    _wp_meshes: list[wp.Mesh]
    _mesh_ids: wp.array[wp.uint64]
    _mesh_kind: wp.array[wp.int32]  # (# meshes,) MESH_CLOSED, MESH_OPEN or MESH_PLANE
    _mesh_depth: wp.array[wp.float32]  # (# meshes,) contact depth of plane meshes (inf: half-space)
    _sds: wp.array[wp.float32]  # (# verts,) signed distances
    _mesh_streams: list[wp.Stream]
    _normal_streams: list[wp.Stream]

    def __init__(self, params: Params | None):
        self.params = params or Params()
        self._contact_dmin = 0.0

    def set_contact_margin(self, dmin: float):
        """Set the AL contact margin (Contact.ContactDynamics.dmin): contacts are detected below dmin + band."""
        self._contact_dmin = float(dmin)

    def register_handles(
        self,
        xt: wp.array[wp.vec3f],
        xk: wp.array[wp.vec3f],
        x: wp.array[wp.vec3f],
        xtilde: wp.array[wp.vec3f],
        meshes: MultiMesh,
        contacts: pairs.ContactPairs,
        dmask: wp.array[wp.int32] | None = None,
    ):
        super().register_handles(xt, xk, x, xtilde, meshes, contacts, dmask)

        FP = self._meshes.data.FP.numpy()
        Fb = [
            cp.asarray(self._meshes.data.F)[FP[b] : FP[b + 1], :].ravel()
            for b in range(FP.shape[0] - 1)
        ]
        self._wp_meshes = [
            wp.Mesh(
                self._x,
                wp.array(
                    data=Fb[b],
                    dtype=wp.int32,
                ),
            )
            for b in range(FP.shape[0] - 1)
        ]
        self._mesh_ids = wp.array([m.id for m in self._wp_meshes], dtype=wp.uint64)
        X = self._x.numpy()
        self._mesh_kind = wp.array(
            [_mesh_kind(Fb[b].get().reshape(-1, 3), X) for b in range(len(Fb))],
            dtype=wp.int32,
        )
        self._mesh_depth = wp.full(len(Fb), float("inf"), dtype=wp.float32)
        self._sds = wp.full(
            (self._meshes.n_verts,), self.params.max_dist, dtype=wp.float32
        )
        self._mesh_streams = [wp.Stream() for _ in range(len(self._wp_meshes))]
        self._normal_streams = [wp.Stream() for _ in range(3)]  # (vv, ve, vf)

    def set_plane_depths(self, depths: np.ndarray):
        """Set how far below each plane mesh a vertex is still in contact with it.

        Args:
            depths (np.ndarray): (# meshes,) depth per contact mesh component, used for MESH_PLANE meshes
                only. inf (the default) makes a plane a half-space, as for infinite plane colliders.
                Finite plane colliders use their thickness.
        """
        self._mesh_depth.assign(np.asarray(depths, dtype=np.float32))

    contact_free = False
    """Set for contact surfaces with a single component (one body, no colliders): vertex SDF detection
    has no self-contact, so such scenes can't have contacts and detection is skipped."""

    @property
    def inactive(self) -> bool:
        """Whether detection, refit and the step filter are skipped (``deactivate`` or contact-free)."""
        return bool(self.params.deactivate) or self.contact_free

    def on_time_step_started(self):
        self.request_step_filter = True

    @profiling.scoped("contact.detect")
    def prepare_detection(self):
        """Refit the meshes' BVHs. wp.Mesh.refit allocates scan temporaries, so it can't run in a
        conditional graph body."""
        if self.inactive:
            return
        self._refit()

    def _refit(self):
        main_stream = wp.get_stream()
        for mesh, stream in zip(self._wp_meshes, self._mesh_streams):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                mesh.refit()
            main_stream.wait_stream(stream)

    @profiling.scoped("contact.detect")
    def detect_contacts(self, from_xt: bool = False, prepared: bool = False):
        if self.inactive:
            return
        if not prepared:
            self._refit()  # unscoped, so the refit doesn't nest as contact.detect/contact.detect
        main_stream = wp.get_stream()
        self._contacts.clear()
        n_verts = self._meshes.data.V.shape[0]
        block_dim = 256
        wp.launch(
            _detect_vertex_mesh_contacts,
            dim=n_verts * block_dim,
            inputs=[
                self._mesh_ids,
                self._mesh_kind,
                self._mesh_depth,
                self.params.max_dist,  # max_dist
                self._contact_dmin + self.params.band,  # detection threshold
                self._sds,
                self._x,
                self._meshes.data,  # type: ignore
                self._contacts.write_data,  # type: ignore
            ],
            block_dim=block_dim,
            stream=main_stream,
        )
        self._contacts.assemble_contacts(self._x, with_reverse_contacts=True)
        # Normals point from the partner surface to the vertex, on both sides of the surface: assemble_contacts
        # computes them as from outside (OGC's convention)
        contact_data = self._contacts.read_data[0]
        for kernel, capacity, extra, stream in zip(
            (_orient_vv_contact_normals, _orient_ve_contact_normals, _orient_vf_contact_normals),
            self._contacts.capacity[:3],
            (
                [contact_data.vv, contact_data.vv_bases],
                [contact_data.ve, contact_data.ve_bases, contact_data.ve_bary],
                [contact_data.vf, contact_data.vf_bases],
            ),
            self._normal_streams,
        ):  # (vv, ve, vf)
            stream.wait_stream(main_stream)
            wp.launch(
                kernel=kernel,
                dim=capacity,
                inputs=[self._x, self._meshes.data] + extra + [self._meshes.n_verts],
                stream=stream,
            )
            main_stream.wait_stream(stream)

    @profiling.scoped("contact.filter")
    def filter_step(self):
        if self.inactive:
            return
        if self.request_step_filter and self.params.use_step_filter:
            n_verts = self._meshes.n_verts
            wp.launch(
                kernel=_filter_initial_step,
                dim=n_verts,
                inputs=[self._sds, self._meshes.data, self._xt, self._dmask, self._x],
            )
        self.request_step_filter = False

    def on_time_step_ended(self):
        pass

    @property
    def supports_conditional_capture(self) -> bool:
        # Only the refit allocates, and it lives in prepare_detection
        return True


@wp.kernel
def _edge_region_barycentrics(
    mesh: wp.uint64,
    x: wp.array[wp.vec3f],
    P: wp.array[wp.vec3f],
    warp_w: wp.array[wp.float32],
    uvw: wp.array[wp.vec3f],
):
    t = wp.tid()
    qp = wp.mesh_query_point_no_sign(mesh, x[t], wp.float32(10))
    warp_w[t] = wp.float32(1) - qp.u - qp.v  # type: ignore
    uvw[t] = queries.closest_point_triangle(x[t], P[0], P[1], P[2])


class TestFeatureClassification(unittest.TestCase):
    """The closest feature of a detected vertex (VF, VE or VV) comes from exact zero barycentrics of the
    closest point's region, not from Warp's mesh query."""

    def test_edge_region_is_exact(self):
        """Points beyond edge jk of a triangle: Warp's w = 1 - u - v is often not exactly 0 there (298 of
        these 1000 points, |w| up to 3e-8, Warp 1.17), the region-based barycentrics are."""
        rng = np.random.default_rng(0)
        P = np.array([[0.1, 0.0, 0.0], [0.9, 0.2, 0.0], [0.3, 0.8, 0.05]], dtype=np.float32)
        e = P[1] - P[0]
        normal = np.cross(e, P[2] - P[0])
        normal /= np.linalg.norm(normal)
        out = np.cross(e, normal)
        out *= -np.sign(np.dot(out, P[2] - P[0])) / np.linalg.norm(out)  # in plane, away from P[2]
        n = 1000
        X = (
            P[0]
            + rng.uniform(0.05, 0.95, (n, 1)) * e
            + rng.uniform(1e-3, 1e-2, (n, 1)) * out
            + rng.uniform(-1e-2, 1e-2, (n, 1)) * normal
        ).astype(np.float32)
        mesh = wp.Mesh(wp.array(P, dtype=wp.vec3f), wp.array([0, 1, 2], dtype=wp.int32))
        warp_w = wp.zeros(n, dtype=wp.float32)
        uvw = wp.zeros(n, dtype=wp.vec3f)
        wp.launch(
            _edge_region_barycentrics,
            dim=n,
            inputs=[mesh.id, wp.array(X, dtype=wp.vec3f), wp.array(P, dtype=wp.vec3f), warp_w, uvw],
        )
        warp_w, uvw = warp_w.numpy(), uvw.numpy()
        self.assertGreater(int(np.count_nonzero(warp_w)), 0, "no inexact Warp barycentric: the case isn't covered")
        np.testing.assert_array_equal(uvw[:, 2], 0)
        self.assertTrue(np.all((uvw[:, 0] > 0) & (uvw[:, 1] > 0)))

    def test_sharp_edge(self):
        """Vertices 2.5 mm outside a sharp wedge's apex edge (22.6 deg), in the edge's region and behind one
        face's plane: each is a VE contact at its true distance, with no penetration. With Warp's
        barycentrics, one of them was a VF contact with the face normal, at gap -2.3 mm."""
        from pbatoolkit import pbat, pypbat
        from ....state import SimulationState, load_params

        # The wedge: cross-section (x, z) apex (0, 0), (-1, 0.2), (-1, -0.2), extruded along y over [0, 1]
        W = np.array([[x, y, z] for y in (0.0, 1.0) for x, z in ((0, 0), (-1, 0.2), (-1, -0.2))])
        C = [np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])]
        n_top = np.array([0.2, 0.0, 1.0]) / np.hypot(0.2, 1.0)
        n_bottom = np.array([0.2, 0.0, -1.0]) / np.hypot(0.2, 1.0)
        distance = 2.5e-3
        X, probes = [W], []
        for q, y in enumerate(np.linspace(0.15, 0.9, 6)):
            # Mostly along one face's normal: behind the other face's plane
            d = 0.1 * n_top + n_bottom if q % 2 == 0 else n_top + 0.1 * n_bottom
            p = np.array([0.0, y, 0.0]) + distance * d / np.linalg.norm(d)
            probes.append(len(W) + 4 * q)
            X.append([p, p + [0.05, 0, 0.02], p + [0.05, 0.02, -0.02], p + [0.05, -0.02, -0.02]])
            C.append(np.array([[0, 1, 2, 3]]) + probes[-1])
        X, C = np.vstack(X), np.vstack(C)
        for e in C:  # positive volumes
            a, b, c, d = X[e]
            if np.dot(np.cross(b - a, c - a), d - a) < 0:
                e[[2, 3]] = e[[3, 2]]
        fem = pbat.sim.dynamics.FemElastoDynamics(X.T, C.T)
        fem.set_mass_matrix(1e3)
        fem.set_elastic_energy(*pypbat.fem.lame_coefficients(1e6, 0.45))
        fem.set_time_integration_scheme(dt=1e-2, s=1)
        state = SimulationState(fem, load_params(fem), None)
        for k in ("vv", "ve", "vf"):  # the default multipliers of 30 vertices leave one VE slot
            setattr(state.contact_storage_params, f"n_{k}_contact_capacity", 1.0)
        state.reset()
        dmin = float(state.contact.params.dmin)
        self.assertLess(dmin, distance)
        state.detector.detect_contacts()
        state.contact.update_constraint_set()
        contacts = state.detector._contacts
        n_vv, n_ve, n_vf, _ = (int(c) for c in contacts.num_contacts)
        self.assertEqual((n_vv, n_ve, n_vf), (0, len(probes), 0))
        V, F = state.multimesh.data.V.numpy(), state.multimesh.data.F.numpy()
        x = state.fem.data.x.numpy()
        u, he = contacts.ve_contacts
        data = contacts.read_data[0]
        t, normals = data.ve_bary.numpy(), data.ve_bases.n.numpy()
        self.assertEqual(sorted(int(V[i]) for i in u[:n_ve]), probes)
        for c in range(n_ve):
            f, h = divmod(int(he[c]), 3)
            xc = (1 - t[c]) * x[F[f][h]] + t[c] * x[F[f][(h + 1) % 3]]
            self.assertAlmostEqual(float(np.dot(x[V[u[c]]] - xc, normals[c])), distance, delta=1e-6)
        state.convergence._measure_penetration()
        self.assertEqual(float(state.convergence.penetration.numpy()[0]), 0.0)


if __name__ == "__main__":
    wp.init()
    unittest.main()

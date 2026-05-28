import warp as wp
import cupy as cp
import numpy as np

from . import pairs
from ..multimesh import MultiMesh, MultiMeshData
from .. import halfedges
from .cd import ContactDetection
from ...common import reduce, lower_bound
from ....common.fields import DocField

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
    max_dist: wp.float32,
    dmin: wp.float32,
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
    for k in range(local_tid, mesh_ids.shape[0], block_dim):
        if k == b:
            continue
        qp = wp.mesh_query_point(mesh_ids[k], xi, max_dist)  # type: ignore
        if not qp.result:  # type: ignore
            continue
        f = meshes.FP[k] + qp.face  # type: ignore
        finds = meshes.F[f]
        j, k, l = finds[0], finds[1], finds[2]
        xj, xk, xl = x[j], x[k], x[l]
        b0, b1, b2 = (wp.float32(1) - qp.u - qp.v), qp.u, qp.v  # type: ignore
        xc = b0 * xj + b1 * xk + b2 * xl  # type: ignore
        sd = qp.sign * wp.norm_l2(xi - xc)  # type: ignore
        tsdmin = wp.min(tsdmin, sd)
        if sd < dmin:
            is_j_zero = wp.int32(b0 == wp.float32(0))
            is_k_zero = wp.int32(b1 == wp.float32(0))
            is_l_zero = wp.int32(b2 == wp.float32(0))
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
            if bvv[block_row, local_tid] < n_verts:
                c = vv_offset + wp.uint64(block_row * block_dim + local_tid)
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
            if bve[block_row, local_tid] < n_half_edges:
                c = ve_offset + wp.uint64(block_row * block_dim + local_tid)
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
            if bvf[block_row, local_tid] < n_tris:
                c = vf_offset + wp.uint64(block_row * block_dim + local_tid)
                contacts.vf.u[c] = wp.uint32(v)
                contacts.vf.v[c] = wp.uint32(bvf[block_row, local_tid])

    # 3. Update signed distances
    bsdmin = wp.tile(tsdmin)  # type: ignore
    sdmin = wp.tile_extract(wp.tile_min(bsdmin), 0)  # type: ignore
    sds[v] = sdmin  # type: ignore


@wp.kernel
def _flip_contact_normals(
    cpairs: pairs.PairsData,  # type: ignore
    bases: pairs.ContactBasesData,  # type: ignore
    n_u: wp.int32,
):
    k = wp.tid()
    n_uv = cpairs.prefix[n_u]
    if wp.uint64(k) >= n_uv:  # type: ignore
        return
    bases.n[k] = -bases.n[k]


@wp.kernel
def _filter_initial_step(
    sds: wp.array[wp.float32],
    meshes: MultiMeshData,  # type: ignore
    xt: wp.array[wp.vec3f],
    x: wp.array[wp.vec3f],
):
    tid = wp.tid()
    v = tid
    i = meshes.V[v]
    if sds[v] <= wp.float32(0):
        x[i] = xt[i]  # type: ignore


class Params:
    max_dist = DocField(0.01, "Maximum distance for closest point computations.")
    dmin = DocField(-0.0001, "Minimum distance threshold for contacts to be created.")
    use_step_filter = DocField(
        False,
        "Whether to apply initial step filter.",
    )
    deactivate = DocField(False, "Whether to deactivate contact detection.")


class Sd(ContactDetection):
    """Vertex-(Mesh)SDF based contact detection."""

    _wp_meshes: list[wp.Mesh]
    _mesh_ids: wp.array[wp.uint64]
    _sds: wp.array[wp.float32]  # (# verts,) signed distances
    _mesh_streams: list[wp.Stream]
    _normal_flip_streams: list[wp.Stream]

    def __init__(self, params: Params | None):
        self.params = params or Params()

    def register_handles(
        self,
        xt: wp.array[wp.vec3f],
        xk: wp.array[wp.vec3f],
        x: wp.array[wp.vec3f],
        xtilde: wp.array[wp.vec3f],
        meshes: MultiMesh,
        contacts: pairs.ContactPairs,
    ):
        super().register_handles(xt, xk, x, xtilde, meshes, contacts)

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
        self._sds = wp.full(
            (self._meshes.n_verts,), self.params.max_dist, dtype=wp.float32
        )
        self._mesh_streams = [wp.Stream() for _ in range(len(self._wp_meshes))]
        self._normal_flip_streams = [wp.Stream() for _ in range(3)]  # (vv, ve, vf)

    def on_time_step_started(self):
        self.request_step_filter = True

    def detect_contacts(self, from_xt: bool = False):
        if self.params.deactivate:
            return
        main_stream = wp.get_stream()
        for mesh, stream in zip(self._wp_meshes, self._mesh_streams):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                mesh.refit()
            main_stream.wait_stream(stream)
        self._contacts.clear()
        n_verts = self._meshes.data.V.shape[0]
        block_dim = 256
        wp.launch(
            _detect_vertex_mesh_contacts,
            dim=n_verts * block_dim,
            inputs=[
                self._mesh_ids,
                self.params.max_dist,  # max_dist
                self.params.dmin,  # dmin
                self._sds,
                self._x,
                self._meshes.data,  # type: ignore
                self._contacts.write_data,  # type: ignore
            ],
            block_dim=block_dim,
            stream=main_stream,
        )
        self._contacts.assemble_contacts(self._x, with_reverse_contacts=True)
        contact_data = self._contacts.read_data[0]
        for capacity, cpairs, bases, stream in zip(
            self._contacts.capacity[:3],
            [contact_data.vv, contact_data.ve, contact_data.vf],
            [contact_data.vv_bases, contact_data.ve_bases, contact_data.vf_bases],
            self._normal_flip_streams,
        ):  # (vv, ve, vf)
            stream.wait_stream(main_stream)
            wp.launch(
                kernel=_flip_contact_normals,
                dim=capacity,
                inputs=[cpairs, bases, self._meshes.n_verts],
                stream=stream,
            )
            main_stream.wait_stream(stream)

    def filter_step(self):
        if self.params.deactivate:
            return
        if self.request_step_filter and self.params.use_step_filter:
            n_verts = self._meshes.n_verts
            wp.launch(
                kernel=_filter_initial_step,
                dim=n_verts,
                inputs=[self._sds, self._meshes.data, self._xt, self._x],
            )
        self.request_step_filter = False

    def on_time_step_ended(self):
        pass

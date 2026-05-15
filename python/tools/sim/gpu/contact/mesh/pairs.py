from typing import Tuple
import warp as wp
import cupy as cp
import numpy as np
import cuda.compute

from ..pairs import Pairs, PairsData
from ..multimesh import MultiMesh, MultiMeshData
from ....common.fields import DocField
from ...common import sort, search
from ... import common
from .. import halfedges
from .. import queries


class Params:
    n_vv_contact_capacity = DocField(
        1.0, "Vertex-vertex capacity multiplier (x num_vertices)."
    )
    n_ve_contact_capacity = DocField(
        1.0, "Vertex-edge capacity multiplier (x num_vertices)."
    )
    n_vf_contact_capacity = DocField(
        1.0, "Vertex-face capacity multiplier (x num_vertices)."
    )
    n_ee_contact_capacity = DocField(
        1.0, "Edge-edge capacity multiplier (x num_edges)."
    )


@wp.struct
class ContactBasesData:
    """Data structure for contact bases."""

    n: wp.array[wp.vec3f]  # (# capacity,) contact normals
    t: wp.array[wp.vec3f]  # (# capacity,) contact tangents
    b: wp.array[wp.vec3f]  # (# capacity,) contact bitangents


class ContactBases:
    data: ContactBasesData  # Contact bases data # type: ignore

    def __init__(self, capacity: int):
        self.data = ContactBasesData()
        self.data.n = wp.zeros(shape=(capacity,), dtype=wp.vec3f)
        self.data.t = wp.zeros(shape=(capacity,), dtype=wp.vec3f)
        self.data.b = wp.zeros(shape=(capacity,), dtype=wp.vec3f)


@wp.struct
class ContactPairsData:
    """Data structure for contact pairs."""

    vv: PairsData  # vertex-vertex contact pairs # type: ignore
    ve: PairsData  # vertex-edge contact pairs # type: ignore
    vf: PairsData  # vertex-face contact pairs # type: ignore
    ee: PairsData  # edge-edge contact pairs # type: ignore

    # Contact bases
    vv_bases: ContactBasesData  # (vv_capacity,) orthonormal contact frame per VV pair # type: ignore
    ve_bases: ContactBasesData  # (ve_capacity,) orthonormal contact frame per VE pair # type: ignore
    vf_bases: ContactBasesData  # (vf_capacity,) orthonormal contact frame per VF pair # type: ignore
    ee_bases: ContactBasesData  # (ee_capacity,) orthonormal contact frame per EE pair # type: ignore

    # Closest-point coordinates (computed in a second pass)
    ve_bary: wp.array[
        wp.float32
    ]  # (ve_capacity,) parameter t of closest point on the edge
    vf_bary: wp.array[
        wp.vec2f
    ]  # (vf_capacity,) barycentric uv (with w = 1-u-v) of closest point on triangle
    ee_bary: wp.array[wp.vec2f]  # (ee_capacity,) parameters (s,t) on edge1 and edge2


@wp.struct
class ReverseContactPairsData:
    """Data structure for reverse contact pairs."""

    rvv: PairsData  # vertex-vertex reverse contact pairs # type: ignore
    rve: PairsData  # vertex-edge reverse contact pairs # type: ignore
    rvf: PairsData  # vertex-face reverse contact pairs # type: ignore
    ree: PairsData  # edge-edge reverse contact pairs # type: ignore

    rvv2vv: wp.array[
        wp.int32
    ]  # (vv_capacity,) reverse-vv index -> forward-vv index, of size # vertex-vertex contacts
    rve2ve: wp.array[
        wp.int32
    ]  # (ve_capacity,) reverse-ve index -> forward-ve index, of size # vertex-edge contacts
    rvf2vf: wp.array[
        wp.int32
    ]  # (vf_capacity,) reverse-vf index -> forward-vf index, of size # vertex-face contacts
    ree2ee: wp.array[
        wp.int32
    ]  # (ee_capacity,) reverse-ee index -> forward-ee index, of size # edge-edge contacts


@wp.kernel
def _map_reverse_to_forward(
    fwd: PairsData,  # pyright: ignore[reportGeneralTypeIssues]
    rev: PairsData,  # pyright: ignore[reportGeneralTypeIssues]
    n_fwd: wp.int32,  # last index into fwd.prefix for the total forward count
    rx2x: wp.array[wp.int32],
):
    """Map each reverse contact index k to its corresponding forward contact index.

    For the k-th reverse contact (rev.u[k], rev.v[k]) = (r_u, r_v), the matching
    forward contact is the pair (r_v, r_u) in the sorted forward list. It is located
    via a binary search using the pair-keyed lower_bound.
    """
    k = wp.tid()
    n = fwd.prefix[n_fwd]
    if wp.uint64(k) >= n:  # type: ignore
        return
    ru = rev.u[k]
    rv = rev.v[k]
    l = common.lower_bound(fwd.u, fwd.v, n, rv, ru)  # type: ignore
    rx2x[k] = wp.int32(l)  # type: ignore


@wp.func
def _build_contact_basis(n: wp.vec3f) -> Tuple[wp.vec3f, wp.vec3f]:
    """Build an orthonormal contact frame from a unit normal vector.

    The returned mat33f has the three basis vectors as rows:
      row 0 = n  (contact normal)
      row 1 = t1 (first tangent)
      row 2 = t2 (second tangent)
    """
    right = wp.vec3f(wp.float32(1), wp.float32(0), wp.float32(0))
    if wp.abs(n[0]) > wp.float32(0.9):  # type: ignore
        right = wp.vec3f(wp.float32(0), wp.float32(1), wp.float32(0))
    t1 = wp.normalize(wp.cross(n, right))  # type: ignore
    t2 = wp.cross(n, t1)  # type: ignore
    return t1, t2  # type: ignore


@wp.kernel
def _compute_vv_contact_data(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: ContactPairsData,  # type: ignore
):
    """Second pass: compute contact basis for each vertex-vertex contact pair."""
    k = wp.tid()
    n_verts = meshes.V.shape[0]
    n_vv = contacts.vv.prefix[n_verts]
    if wp.uint64(k) >= n_vv:  # type: ignore
        return
    u, v = contacts.vv.u[k], contacts.vv.v[k]
    xi = x[meshes.V[u]]
    xj = x[meshes.V[v]]
    # assert wp.norm_l2(xi - xj) > wp.float32(1e-10)  # type: ignore
    n = wp.normalize(xi - xj)  # type: ignore
    t, b = _build_contact_basis(n)
    contacts.vv_bases.n[k] = n
    contacts.vv_bases.t[k] = t
    contacts.vv_bases.b[k] = b


@wp.kernel
def _compute_ve_contact_data(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: ContactPairsData,  # type: ignore
):
    """Second pass: compute contact basis and edge parameter t for each vertex-edge contact pair."""
    k = wp.tid()
    n_verts = meshes.V.shape[0]
    n_ve = contacts.ve.prefix[n_verts]
    if wp.uint64(k) >= n_ve:  # type: ignore
        return
    v, he = contacts.ve.u[k], contacts.ve.v[k]
    xi = x[meshes.V[v]]
    xa = x[halfedges.incoming_vertex(meshes.F, he)]  # type: ignore
    xb = x[halfedges.outgoing_vertex(meshes.F, he)]  # type: ignore
    uv = queries.closest_point_on_line_segment(xi, xa, xb)  # type: ignore
    # assert wp.norm_l2(xi - (uv[0] * xa + uv[1] * xb)) > wp.float32(1e-10)  # type: ignore
    n = wp.normalize(xi - (uv[0] * xa + uv[1] * xb))  # type: ignore
    t, b = _build_contact_basis(n)
    contacts.ve_bases.n[k] = n
    contacts.ve_bases.t[k] = t
    contacts.ve_bases.b[k] = b
    contacts.ve_bary[k] = uv[1]  # type: ignore


@wp.kernel
def _compute_vf_contact_data(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: ContactPairsData,  # type: ignore
):
    """Second pass: compute contact basis and barycentric uvw for each vertex-face contact pair."""
    k = wp.tid()
    n_verts = meshes.V.shape[0]
    n_vf = contacts.vf.prefix[n_verts]
    if wp.uint64(k) >= n_vf:  # type: ignore
        return
    v, f = contacts.vf.u[k], contacts.vf.v[k]
    xi = x[meshes.V[v]]
    finds = meshes.F[f]
    xa = x[finds[0]]
    xb = x[finds[1]]
    xc = x[finds[2]]
    uvw = queries.closest_point_triangle(xi, xa, xb, xc)  # type: ignore
    xc = uvw[0] * xa + uvw[1] * xb + uvw[2] * xc  # type: ignore
    assert wp.norm_l2(xi - xc) > wp.float32(1e-10)  # type: ignore
    n = wp.normalize(xi - xc)
    t, b = _build_contact_basis(n)
    contacts.vf_bases.n[k] = n
    contacts.vf_bases.t[k] = t
    contacts.vf_bases.b[k] = b
    contacts.vf_bary[k] = wp.vec2f(uvw[1], uvw[2])  # type: ignore


@wp.kernel
def _compute_ee_contact_data(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: ContactPairsData,  # type: ignore
):
    """Second pass: compute contact basis and parameters (s,t) for each edge-edge contact pair."""
    k = wp.tid()
    n_half_edges = contacts.ee.prefix.shape[0] - wp.int32(1)
    n_ee = contacts.ee.prefix[n_half_edges]
    if wp.uint64(k) >= n_ee:  # type: ignore
        return
    he1 = contacts.ee.u[k]
    he2 = contacts.ee.v[k]
    xi1 = x[halfedges.incoming_vertex(meshes.F, he1)]
    xj1 = x[halfedges.outgoing_vertex(meshes.F, he1)]
    xi2 = x[halfedges.incoming_vertex(meshes.F, he2)]
    xj2 = x[halfedges.outgoing_vertex(meshes.F, he2)]
    std = wp.closest_point_edge_edge(xi1, xj2, xi2, xj2, epsilon=wp.float32(1e-3))  # type: ignore
    s0, s1 = std[0], std[1]  # type: ignore
    xc1 = (wp.float32(1.0) - s0) * xi1 + s0 * xj1  # type: ignore
    xc2 = (wp.float32(1.0) - s1) * xi2 + s1 * xj2  # type: ignore
    # assert wp.norm_l2(xc1 - xc2) > wp.float32(1e-10)  # type: ignore
    n = wp.normalize(xc1 - xc2)
    t, b = _build_contact_basis(n)
    contacts.ee_bases.n[k] = n
    contacts.ee_bases.t[k] = t
    contacts.ee_bases.b[k] = b
    contacts.ee_bary[k] = wp.vec2f(s0, s1)  # type: ignore


class ContactPairs:
    """Mesh contact pair (CSR) data structure."""

    # Pair storage
    _vv: Pairs  # Vertex-Vertex contact pair list
    _ve: Pairs  # Vertex-Edge contact pair list
    _vf: Pairs  # Vertex-Face contact pair list
    _ee: Pairs  # Edge-Edge contact pair list

    _rvv: Pairs  # Vertex-Vertex reverse contact pair list
    _rve: Pairs  # Vertex-Edge reverse contact pair list
    _rvf: Pairs  # Vertex-Face reverse contact pair list
    _ree: Pairs  # Edge-Edge reverse contact pair list

    # Sorting storage.
    # NOTE: The _*_u_* and _*_v_* buffers are used for the user to write into
    # transparently through the .write_data property.
    _vv_u_buffer: wp.array[wp.uint32]
    _vv_v_buffer: wp.array[wp.uint32]
    _vv_sort: sort.Sort
    _rvv_sort: sort.Sort

    _ve_u_buffer: wp.array[wp.uint32]
    _ve_v_buffer: wp.array[wp.uint32]
    _ve_sort: sort.Sort
    _rve_sort: sort.Sort

    _vf_u_buffer: wp.array[wp.uint32]
    _vf_v_buffer: wp.array[wp.uint32]
    _vf_sort: sort.Sort
    _rvf_sort: sort.Sort

    _ee_u_buffer: wp.array[wp.uint32]
    _ee_v_buffer: wp.array[wp.uint32]
    _ee_sort: sort.Sort
    _ree_sort: sort.Sort

    # CSR compression (i.e. prefix sum)
    _vv_lower_bound: search.LowerBound
    _ve_lower_bound: search.LowerBound
    _vf_lower_bound: search.LowerBound
    _ee_lower_bound: search.LowerBound

    _rvv_lower_bound: search.LowerBound
    _rve_lower_bound: search.LowerBound
    _rvf_lower_bound: search.LowerBound
    _ree_lower_bound: search.LowerBound

    # Parallel execution
    _streams: list[wp.Stream]  # Stream list

    # Kernel inputs
    _data: ContactPairsData  # Contact pairs data # type: ignore
    _rdata: ReverseContactPairsData  # Reverse contact pairs data # type: ignore
    meshes: MultiMesh

    def __init__(self, meshes: MultiMesh, params: Params):
        self.meshes = meshes
        self._data = ContactPairsData()
        self._rdata = ReverseContactPairsData()

        vv_capacity = int(params.n_vv_contact_capacity * meshes.n_verts)  # type: ignore
        ve_capacity = int(params.n_ve_contact_capacity * meshes.n_verts)  # type: ignore
        vf_capacity = int(params.n_vf_contact_capacity * meshes.n_verts)  # type: ignore
        ee_capacity = int(params.n_ee_contact_capacity * meshes.n_edges)  # type: ignore

        # Pair storage
        self._vv, self._rvv = Pairs(meshes.n_verts, meshes.n_verts, vv_capacity), Pairs(
            meshes.n_verts, meshes.n_verts, vv_capacity
        )
        self._data.vv, self._rdata.rvv = self._vv.data, self._rvv.data

        self._ve, self._rve = Pairs(
            meshes.n_verts, meshes.n_half_edges, ve_capacity
        ), Pairs(meshes.n_half_edges, meshes.n_verts, ve_capacity)
        self._data.ve, self._rdata.rve = self._ve.data, self._rve.data

        self._vf, self._rvf = Pairs(
            meshes.n_verts, meshes.n_triangles, vf_capacity
        ), Pairs(meshes.n_triangles, meshes.n_verts, vf_capacity)
        self._data.vf, self._rdata.rvf = self._vf.data, self._rvf.data

        self._ee, self._ree = Pairs(
            meshes.n_half_edges, meshes.n_half_edges, ee_capacity
        ), Pairs(meshes.n_half_edges, meshes.n_half_edges, ee_capacity)
        self._data.ee, self._rdata.ree = self._ee.data, self._ree.data

        self._rdata.rvv2vv = wp.empty(shape=(vv_capacity,), dtype=wp.int32)  # type: ignore
        self._rdata.rve2ve = wp.empty(shape=(ve_capacity,), dtype=wp.int32)  # type: ignore
        self._rdata.rvf2vf = wp.empty(shape=(vf_capacity,), dtype=wp.int32)  # type: ignore
        self._rdata.ree2ee = wp.empty(shape=(ee_capacity,), dtype=wp.int32)  # type: ignore

        self._data.vv_bases = ContactBases(vv_capacity).data
        self._data.ve_bases = ContactBases(ve_capacity).data
        self._data.vf_bases = ContactBases(vf_capacity).data
        self._data.ee_bases = ContactBases(ee_capacity).data
        self._data.ve_bary = wp.empty((ve_capacity,), dtype=wp.float32)  # type: ignore
        self._data.vf_bary = wp.empty((vf_capacity,), dtype=wp.vec2f)  # type: ignore
        self._data.ee_bary = wp.empty((ee_capacity,), dtype=wp.vec2f)  # type: ignore

        # Sorting storage
        self._vv_u_buffer = wp.empty_like(self._vv.data.u)
        self._vv_v_buffer = wp.empty_like(self._vv.data.v)
        self._vv_sort = sort.Sort(
            d_in_keys=cp.asarray(self._vv_u_buffer),
            d_in_values=cp.asarray(self._vv_v_buffer),
            d_out_keys=cp.asarray(self._vv.data.u),
            d_out_values=cp.asarray(self._vv.data.v),
            n_items=vv_capacity,
        )
        self._rvv_sort = sort.Sort(
            d_in_keys=cp.asarray(self._vv.data.v),
            d_in_values=cp.asarray(self._vv.data.u),
            d_out_keys=cp.asarray(self._rvv.data.u),
            d_out_values=cp.asarray(self._rvv.data.v),
            n_items=vv_capacity,
        )

        self._ve_u_buffer = wp.empty_like(self._ve.data.u)
        self._ve_v_buffer = wp.empty_like(self._ve.data.v)
        self._ve_sort = sort.Sort(
            d_in_keys=cp.asarray(self._ve_u_buffer),
            d_in_values=cp.asarray(self._ve_v_buffer),
            d_out_keys=cp.asarray(self._ve.data.u),
            d_out_values=cp.asarray(self._ve.data.v),
            n_items=ve_capacity,
        )
        self._rve_sort = sort.Sort(
            d_in_keys=cp.asarray(self._ve.data.v),
            d_in_values=cp.asarray(self._ve.data.u),
            d_out_keys=cp.asarray(self._rve.data.u),
            d_out_values=cp.asarray(self._rve.data.v),
            n_items=ve_capacity,
        )

        self._vf_u_buffer = wp.empty_like(self._vf.data.u)
        self._vf_v_buffer = wp.empty_like(self._vf.data.v)
        self._vf_sort = sort.Sort(
            d_in_keys=cp.asarray(self._vf_u_buffer),
            d_in_values=cp.asarray(self._vf_v_buffer),
            d_out_keys=cp.asarray(self._vf.data.u),
            d_out_values=cp.asarray(self._vf.data.v),
            n_items=vf_capacity,
        )
        self._rvf_sort = sort.Sort(
            d_in_keys=cp.asarray(self._vf.data.v),
            d_in_values=cp.asarray(self._vf.data.u),
            d_out_keys=cp.asarray(self._rvf.data.u),
            d_out_values=cp.asarray(self._rvf.data.v),
            n_items=vf_capacity,
        )

        self._ee_u_buffer = wp.empty_like(self._ee.data.u)
        self._ee_v_buffer = wp.empty_like(self._ee.data.v)
        self._ee_sort = sort.Sort(
            d_in_keys=cp.asarray(self._ee_u_buffer),
            d_in_values=cp.asarray(self._ee_v_buffer),
            d_out_keys=cp.asarray(self._ee.data.u),
            d_out_values=cp.asarray(self._ee.data.v),
            n_items=ee_capacity,
        )
        self._ree_sort = sort.Sort(
            d_in_keys=cp.asarray(self._ee.data.v),
            d_in_values=cp.asarray(self._ee.data.u),
            d_out_keys=cp.asarray(self._ree.data.u),
            d_out_values=cp.asarray(self._ree.data.v),
            n_items=ee_capacity,
        )

        # CSR compression (i.e. prefix sum) algorithms.
        # NOTE: For the LowerBoun algorithms, we directly use the CuPy
        # arrays exposed through Pairs.u, Pairs.v, Pairs.prefix for 
        # cuda.compute interoperability.
        self._vv_lower_bound = search.LowerBound(
            d_data=self._vv.u,
            num_items=vv_capacity,
            d_values=cuda.compute.CountingIterator(np.uint32(0)),
            num_values=meshes.n_verts + 1,
            d_out=self._vv.prefix,
        )
        self._ve_lower_bound = search.LowerBound(
            d_data=self._ve.u,
            num_items=ve_capacity,
            d_values=cuda.compute.CountingIterator(np.uint32(0)),
            num_values=meshes.n_verts + 1,
            d_out=self._ve.prefix,
        )
        self._vf_lower_bound = search.LowerBound(
            d_data=self._vf.u,
            num_items=vf_capacity,
            d_values=cuda.compute.CountingIterator(np.uint32(0)),
            num_values=meshes.n_verts + 1,
            d_out=self._vf.prefix,
        )
        self._ee_lower_bound = search.LowerBound(
            d_data=self._ee.u,
            num_items=ee_capacity,
            d_values=cuda.compute.CountingIterator(np.uint32(0)),
            num_values=meshes.n_half_edges + 1,
            d_out=self._ee.prefix,
        )

        self._rvv_lower_bound = search.LowerBound(
            d_data=self._rvv.u,
            num_items=vv_capacity,
            d_values=cuda.compute.CountingIterator(np.uint32(0)),
            num_values=meshes.n_verts + 1,
            d_out=self._rvv.prefix,
        )
        self._rve_lower_bound = search.LowerBound(
            d_data=self._rve.u,
            num_items=ve_capacity,
            d_values=cuda.compute.CountingIterator(np.uint32(0)),
            num_values=meshes.n_half_edges + 1,
            d_out=self._rve.prefix,
        )
        self._rvf_lower_bound = search.LowerBound(
            d_data=self._rvf.u,
            num_items=vf_capacity,
            d_values=cuda.compute.CountingIterator(np.uint32(0)),
            num_values=meshes.n_triangles + 1,
            d_out=self._rvf.prefix,
        )
        self._ree_lower_bound = search.LowerBound(
            d_data=self._ree.u,
            num_items=ee_capacity,
            d_values=cuda.compute.CountingIterator(np.uint32(0)),
            num_values=meshes.n_half_edges + 1,
            d_out=self._ree.prefix,
        )

        # Stream parallelism
        self._streams = [wp.Stream() for _ in range(24)]

    def clear(self):
        """Clear all contact pair data. Resets all empty contacts to their sentinel values."""
        main_stream = wp.get_stream()
        for pairs, stream in zip(
            [
                self._vv,
                self._ve,
                self._vf,
                self._ee,
                self._rvv,
                self._rve,
                self._rvf,
                self._ree,
            ],
            self._streams[:8],
        ):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream):
                pairs.clear()
        # Also reset write buffers to sentinel values so that the Sort
        # produces an all-sentinel read array when there are no contacts.
        write_buffer_fills = [
            (self._vv_u_buffer, self._vv.nu),
            (self._vv_v_buffer, self._vv.nv),
            (self._ve_u_buffer, self._ve.nu),
            (self._ve_v_buffer, self._ve.nv),
            (self._vf_u_buffer, self._vf.nu),
            (self._vf_v_buffer, self._vf.nv),
            (self._ee_u_buffer, self._ee.nu),
            (self._ee_v_buffer, self._ee.nv),
        ]
        for (buf, sentinel), stream in zip(write_buffer_fills, self._streams[8:16]):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream):
                buf.fill_(sentinel)

    def assemble_contacts(
        self, x: wp.array[wp.vec3f], with_reverse_contacts: bool = False
    ):
        """Prepare read data after write data (i.e. forward u, v and prefix[-1]) has
        been processed by user.

        Preconditions:
            - Write data (i.e. forward u, v and prefix[-1]) has been processed by user s.t.
            v is sorted w.r.t. u for any given u (but u is not necessarily sorted).
            - Write data for contacts involving edges stores only the largest half-edge index
            of 2 opposite half-edges, i.e. he=wp.max(hei, hej), where hei, hej are opposite
            half-edges, is the stored index in forward v of vertex-edge contacts, and forward u,v
            of edge-edge contacts. Boundary edges have hej == -1, which is implicitly handled by
            max(hei, hej).
            - Empty contacts u, v need to store their sentinel values nu, nv, for
            which u < nu, v < nv for all u,v.
        Postconditions:
            - Forward and reverse contacts are assembled into CSR format.
            - Contact bases and barycentric coordinates on edges and faces are computed.

        Args:
            x (wp.array[wp.vec3f]): Vertex positions
        """
        # 0. Fork all streams so that they are register in a CUDA graph
        main_stream = wp.get_stream()
        for stream in self._streams:
            stream.wait_stream(main_stream)

        # The dependency graph is as follows, where each line can be further
        # decoupled by the 4 contact types (vv,ve,vf,ee):
        #
        # |-> Sort forward contacts
        #     |-> Compute contact bases and coords
        #     |-> Compute forward contact prefix sums
        #     |-> Sort reverse contacts
        #         |-> Compute reverse contact prefix sums
        #         |-> Compute reverse to forward map
        #
        # Let's use 6 (streams) x 4 (contact types) = 24 streams for each workload.
        forward_sort_streams = self._streams[:4]
        contact_bases_streams = self._streams[4:8]
        forward_prefix_streams = self._streams[8:12]
        reverse_sort_streams = self._streams[12:16]
        reverse_prefix_streams = self._streams[16:20]
        reverse_to_forward_map_streams = self._streams[20:24]
        capacities = [
            self._vv.capacity,
            self._ve.capacity,
            self._vf.capacity,
            self._ee.capacity,
        ]

        # 1a. Sort forward contacts
        for sort, stream in zip(
            [self._vv_sort, self._ve_sort, self._vf_sort, self._ee_sort],
            forward_sort_streams,
        ):
            sort(stream)
        # 2a. Compute bases and coords
        compute_bases_kernels = [
            _compute_vv_contact_data,
            _compute_ve_contact_data,
            _compute_vf_contact_data,
            _compute_ee_contact_data,
        ]
        for kernel, capacity, parent_stream, child_stream in zip(
            compute_bases_kernels,
            capacities,
            forward_sort_streams,
            contact_bases_streams,
        ):
            child_stream.wait_stream(parent_stream)
            wp.launch(
                kernel=kernel,
                dim=capacity,
                inputs=[x, self.meshes.data, self._data],
                stream=child_stream,
            )
        # 2b. Compute forward contact prefix sums
        forward_contact_prefix_computers = [
            self._vv_lower_bound,
            self._ve_lower_bound,
            self._vf_lower_bound,
            self._ee_lower_bound,
        ]
        for compute_forward_contact_prefix, parent_stream, child_stream in zip(
            forward_contact_prefix_computers,
            forward_sort_streams,
            forward_prefix_streams,
        ):
            child_stream.wait_stream(parent_stream)
            compute_forward_contact_prefix(child_stream)

        if with_reverse_contacts:
            # 2c. Sort reverse contacts
            reverse_contact_sorts = [
                self._rvv_sort,
                self._rve_sort,
                self._rvf_sort,
                self._ree_sort,
            ]
            for sort_reverse_contacts, parent_stream, child_stream in zip(
                reverse_contact_sorts, forward_sort_streams, reverse_sort_streams
            ):
                child_stream.wait_stream(parent_stream)
                sort_reverse_contacts(child_stream)
            # 3a. Compute reverse contact prefix
            reverse_contact_prefix_computers = [
                self._rvv_lower_bound,
                self._rve_lower_bound,
                self._rvf_lower_bound,
                self._ree_lower_bound,
            ]
            for compute_reverse_contact_prefix, parent_stream, child_stream in zip(
                reverse_contact_prefix_computers,
                reverse_sort_streams,
                reverse_prefix_streams,
            ):
                child_stream.wait_stream(parent_stream)
                compute_reverse_contact_prefix(child_stream)
            # 3b. Build reverse to forward contact map
            forward_pairs = [self._vv, self._ve, self._vf, self._ee]
            reverse_pairs = [self._rvv, self._rve, self._rvf, self._ree]
            r2f_maps = [
                self._rdata.rvv2vv,
                self._rdata.rve2ve,
                self._rdata.rvf2vf,
                self._rdata.ree2ee,
            ]
            for fwd, rev, rx2x, capacity, child_stream, parent_stream in zip(
                forward_pairs,
                reverse_pairs,
                r2f_maps,
                capacities,
                reverse_to_forward_map_streams,
                reverse_sort_streams,
            ):
                child_stream.wait_stream(parent_stream)
                wp.launch(
                    kernel=_map_reverse_to_forward,
                    dim=capacity,
                    inputs=[fwd.data, rev.data, fwd.nu, rx2x],
                    stream=child_stream,
                )

        # Join
        for stream in self._streams[:12]:
            stream.wait_stream(main_stream)
        if with_reverse_contacts:
            for stream in self._streams[12:]:
                stream.wait_stream(main_stream)

    @property
    def write_data(self) -> ContactPairsData:  # type: ignore
        """Return write data for contact pairs, in which it is safe to overwrite prefix[-1], u and v."""
        write_data = ContactPairsData()

        write_data.vv = PairsData()
        write_data.ve = PairsData()
        write_data.vf = PairsData()
        write_data.ee = PairsData()

        write_data.vv.prefix = self._data.vv.prefix
        write_data.vv.u = self._vv_u_buffer
        write_data.vv.v = self._vv_v_buffer

        write_data.ve.prefix = self._data.ve.prefix
        write_data.ve.u = self._ve_u_buffer
        write_data.ve.v = self._ve_v_buffer

        write_data.vf.prefix = self._data.vf.prefix
        write_data.vf.u = self._vf_u_buffer
        write_data.vf.v = self._vf_v_buffer

        write_data.ee.prefix = self._data.ee.prefix
        write_data.ee.u = self._ee_u_buffer
        write_data.ee.v = self._ee_v_buffer

        return write_data

    @property
    def read_data(self) -> Tuple[ContactPairsData, ReverseContactPairsData]:  # type: ignore
        """Return read data for contact pairs, where it is safe to access
        prefix, u, v for forward and reverse contacts."""
        return self._data, self._rdata

    @property
    def capacity(self) -> Tuple[int, int, int, int]:
        """Return the capacity of each contact pair type.

        Returns:
            Tuple[int, int, int, int]: The capacity of each contact pair type.
        """
        return (
            self._vv.capacity,
            self._ve.capacity,
            self._vf.capacity,
            self._ee.capacity,
        )

    @property
    def num_contacts(self) -> Tuple[int, int, int, int]:
        """Return the number of contacts for each contact pair type.

        Returns:
            Tuple[int, int, int, int]: The number of contacts for each contact pair type.
        """
        return (
            self._vv.size(),
            self._ve.size(),
            self._vf.size(),
            self._ee.size(),
        )

    @property
    def vv_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the vertex-vertex contact pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The vertex-vertex contact pairs (u, v).
        """
        return self._vv.uv()

    @property
    def ve_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the vertex-edge contact pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The vertex-edge contact pairs (u, v).
        """
        return self._ve.uv()

    @property
    def vf_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the vertex-face contact pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The vertex-face contact pairs (u, v).
        """
        return self._vf.uv()

    @property
    def ee_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the edge-edge contact pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The edge-edge contact pairs (u, v).
        """
        return self._ee.uv()

    @property
    def rvv_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the reverse vertex-vertex contact pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The reverse vertex-vertex contact pairs (u, v).
        """
        return self._rvv.uv()

    @property
    def rve_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the reverse vertex-edge contact pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The reverse vertex-edge contact pairs (u, v).
        """
        return self._rve.uv()

    @property
    def rvf_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the reverse vertex-face contact pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The reverse vertex-face contact pairs (u, v).
        """
        return self._rvf.uv()

    @property
    def ree_contacts(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the reverse edge-edge contact pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The reverse edge-edge contact pairs (u, v).
        """
        return self._ree.uv()


# TODO:
# First simple contact detection:
# 0. Have a wp.Mesh (has BVH) for every connected component of our global mesh.
# 1. For every mesh vertex, compute its signed distance to every OTHER mesh.
# 2. If signed distance < offset,
#       do not move the vertex due to solver iterate initialization.
#    Else
#       ray cast from vertex at time t, to initialized solver iterate,
#       and move vertex until first impact if t_{impact} < 1.
# 3. If signed distance < offset,
#       compute closest point to vertex on every OTHER mesh, and create vv,ve or vf contacts for each
#    Else
#       Use an oracle to pre-emptively create contact constraints
#           Options (recall initial iterate is often x = xt + h*vt + h^2 at (at = gravity)):
#           1 - Use ray cast's first time of impact < 1 and create constraint with ray's hit for every OTHER mesh
#           2 - Query all nearby faces on every OTHER mesh and their closest points, and if they are in the offset
#               geometry (i.e. OGC's feasibility test), we pre-emptively create that contact.
#       .
# _____
#     \
#     |->
# ____/
#  |
#  ^
#
#  .
#

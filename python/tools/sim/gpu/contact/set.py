from typing import Tuple

import cupy as cp
import warp as wp


@wp.struct
class ContactSetData:
    """Data structure for a contact set, which can be vertex-vertex, vertex-edge, vertex-face, or edge-edge."""

    nvv: wp.array[wp.int32]  # (# verts + 1,) prefix sum of vertex-vertex contacts
    nve: wp.array[wp.int32]  # (# verts + 1,) prefix sum of vertex-edge contacts
    nvf: wp.array[wp.int32]  # (# verts + 1,) prefix sum of vertex-face contacts
    nee: wp.array[
        wp.int32
    ]  # (# half-edges + 1,) prefix sum of (half-)edge-(half-)edge contacts

    vv: wp.array[
        wp.int32
    ]  # (# vertex-vertex contacts capacity,) pairs (i, j) stored as j
    ve: wp.array[
        wp.int32
    ]  # (# vertex-(half-)edge contacts capacity,) pairs (i, he) stored as he
    vf: wp.array[
        wp.int32
    ]  # (# vertex-face contacts capacity,) pairs (i, f) stored as f
    ee: wp.array[
        wp.int32
    ]  # (# (half-)edge-(half-)edge contacts capacity,) pairs (hei, hej) stored as hej


class ContactSet:

    _data: ContactSetData  # pyright: ignore[reportGeneralTypeIssues]

    def __init__(
        self,
        n_verts: int,
        n_faces: int,
        max_contact_pairs: Tuple[int, int, int, int],
    ):
        n_half_edges = 3 * n_faces
        self._data = ContactSetData()
        self._data.nvv = wp.zeros((n_verts + 1,), dtype=wp.int32)
        self._data.nve = wp.zeros((n_verts + 1,), dtype=wp.int32)
        self._data.nvf = wp.zeros((n_verts + 1,), dtype=wp.int32)
        self._data.nee = wp.zeros((n_half_edges + 1,), dtype=wp.int32)
        self._data.vv = wp.zeros((max_contact_pairs[0],), dtype=wp.int32)
        self._data.ve = wp.zeros((max_contact_pairs[1],), dtype=wp.int32)
        self._data.vf = wp.zeros((max_contact_pairs[2],), dtype=wp.int32)
        self._data.ee = wp.zeros((max_contact_pairs[3],), dtype=wp.int32)

    def clear(self, reset_pairs: bool = False):
        """Reset prefix/count buffers and optionally contact pair buffers."""
        self._data.nvv.fill_(0)
        self._data.nve.fill_(0)
        self._data.nvf.fill_(0)
        self._data.nee.fill_(0)
        if reset_pairs:
            self._data.vv.fill_(0)
            self._data.ve.fill_(0)
            self._data.vf.fill_(0)
            self._data.ee.fill_(0)


    @property
    def num_contacts(self) -> Tuple[int, int, int, int]:
        """Return total contact counts (VV, VE, VF, EE) from prefix-sum tails."""
        return (
            int(cp.asarray(self._data.nvv)[-1].get()),
            int(cp.asarray(self._data.nve)[-1].get()),
            int(cp.asarray(self._data.nvf)[-1].get()),
            int(cp.asarray(self._data.nee)[-1].get()),
        )

    @property
    def data(self) -> ContactSetData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data

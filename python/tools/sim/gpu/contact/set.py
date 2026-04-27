from typing import Tuple

import warp as wp


@wp.struct
class ContactSetData:
    """Data structure for a contact set, which can be point-point, point-edge, point-face, or edge-edge."""

    nxx: wp.array[wp.int32]  # (# points + 1,) prefix sum of point-point contacts
    nxe: wp.array[wp.int32]  # (# points + 1,) prefix sum of point-edge contacts
    nxf: wp.array[wp.int32]  # (# points + 1,) prefix sum of point-face contacts
    nee: wp.array[
        wp.int32
    ]  # (# half-edges + 1,) prefix sum of (half-)edge-(half-)edge contacts

    xx: wp.array[
        wp.int32
    ]  # (# point-point contacts capacity,) pairs (i, j) stored as j
    xe: wp.array[
        wp.int32
    ]  # (# point-(half-)edge contacts capacity,) pairs (i, he) stored as he
    xf: wp.array[wp.int32]  # (# point-face contacts capacity,) pairs (i, f) stored as f
    ee: wp.array[
        wp.int32
    ]  # (# (half-)edge-(half-)edge contacts capacity,) pairs (hei, hej) stored as hej


class ContactSet:

    _data: ContactSetData  # pyright: ignore[reportGeneralTypeIssues]

    def __init__(
        self,
        n_points: int,
        n_faces: int,
        max_contact_pairs: Tuple[int, int, int, int],
    ):
        n_half_edges = 3 * n_faces
        self._data = ContactSetData()
        self._data.nxx = wp.zeros((n_points + 1,), dtype=wp.int32)
        self._data.nxe = wp.zeros((n_points + 1,), dtype=wp.int32)
        self._data.nxf = wp.zeros((n_points + 1,), dtype=wp.int32)
        self._data.nee = wp.zeros((n_half_edges + 1,), dtype=wp.int32)
        self._data.xx = wp.zeros((max_contact_pairs[0],), dtype=wp.int32)
        self._data.xe = wp.zeros((max_contact_pairs[1],), dtype=wp.int32)
        self._data.xf = wp.zeros((max_contact_pairs[2],), dtype=wp.int32)
        self._data.ee = wp.zeros((max_contact_pairs[3],), dtype=wp.int32)

    def clear(self):
        self._data.nxx.fill_(0)
        self._data.nxe.fill_(0)
        self._data.nxf.fill_(0)
        self._data.nee.fill_(0)

    @property
    def data(self) -> ContactSetData:  # pyright: ignore[reportGeneralTypeIssues]
        return self._data

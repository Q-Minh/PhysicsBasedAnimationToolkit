import warp as wp


@wp.func
def face_of_half_edge(he: wp.int32) -> wp.int32:  # type: ignore
    """Return the face index adjacent to half-edge `he`."""
    return wp.int32(he // wp.int32(3))  # type: ignore


@wp.func
def face_of_half_edge(he: wp.uint32) -> wp.uint32:  # type: ignore
    """Return the face index adjacent to half-edge `he`."""
    return wp.uint32(he // wp.uint32(3))  # type: ignore


@wp.func
def incoming_vertex(F: wp.array[wp.vec3i], he: wp.int32) -> wp.int32:  # type: ignore
    """Source vertex of half-edge he of triangle mesh F

    Args:
        F (wp.array[wp.vec3i]): Triangle vertex indices
        he (wp.int32): Half-edge index

    Returns:
        wp.int32: Source vertex index of half-edge `he`
    """
    return F[face_of_half_edge(he)][he % wp.int32(3)]  # type: ignore


@wp.func
def incoming_vertex(F: wp.array[wp.vec3i], he: wp.uint32) -> wp.int32:  # type: ignore
    """Source vertex of half-edge he of triangle mesh F (uint32 overload)."""
    return F[face_of_half_edge(he)][he % wp.uint32(3)]  # type: ignore


@wp.func
def outgoing_vertex(F: wp.array[wp.vec3i], he: wp.int32) -> wp.int32:  # type: ignore
    """Return the outgoing vertex index of half-edge `he`."""
    return F[face_of_half_edge(he)][(he + wp.int32(1)) % wp.int32(3)]  # type: ignore


@wp.func
def outgoing_vertex(F: wp.array[wp.vec3i], he: wp.uint32) -> wp.int32:  # type: ignore
    """Return the outgoing vertex index of half-edge `he` (uint32 overload)."""
    return F[face_of_half_edge(he)][(he + wp.uint32(1)) % wp.uint32(3)]  # type: ignore


@wp.func
def next_vertex(F: wp.array[wp.vec3i], he: wp.int32, step: wp.int32) -> wp.int32:
    """Return the next vertex `step` steps away from the source vertex of half-edge `he`."""
    return F[face_of_half_edge(he)][(he + step) % wp.int32(3)]  # type: ignore


@wp.func
def next_half_edge(he: wp.int32) -> wp.int32:
    """Return the next half-edge index of half-edge `he`."""
    three = wp.int32(3)
    one = wp.int32(1)
    return (he // three) * three + (  # pyright: ignore[reportOperatorIssue]
        he + one
    ) % three


@wp.func
def first_half_edge_of_face(f: wp.int32) -> wp.int32:
    """Return the first half-edge index of face `f`."""
    return f * wp.int32(3)  # pyright: ignore[reportOperatorIssue]


@wp.func
def half_edge_of_face(f: wp.int32, helocal: wp.int32) -> wp.int32:
    """Return the half-edge index of face `f` and local half-edge index `helocal`."""
    three = wp.int32(3)
    return f * three + helocal  # pyright: ignore[reportOperatorIssue]


@wp.func
def are_opposite_half_edges(
    F: wp.array[wp.vec3i], hei: wp.int32, hej: wp.int32
) -> bool:
    fi = face_of_half_edge(hei)  # type: ignore
    fj = face_of_half_edge(hej)  # type: ignore
    three = wp.int32(3)
    one = wp.int32(1)
    via, vib = F[hei % three, fi], F[(hei + one) % three, fi]
    vja, vjb = F[hej % three, fj], F[(hej + one) % three, fj]
    return (via == vjb) and (vib == vja)


@wp.func
def opposite_half_edge(
    F: wp.array[wp.vec3i], he: wp.int32, GHEF: wp.array[wp.vec2i]
) -> wp.int32:
    """Get opposite half-edge index, or -1 if none

    Args:
        F (wp.array[wp.vec3i]): Triangle vertex indices
        he (wp.int32): Half-edge index
        GHEF (wp.array[wp.vec2i]): 2 x |# half edges| half-edge to face adjacency, where GHEF[1, he] == -1 if boundary half-edge

    Returns:
        wp.int32: Opposite half-edge index, or -1 if none
    """
    fj = GHEF[he][1]  # pyright: ignore[reportIndexIssue]
    if fj == -1:
        return wp.int32(-1)
    vib = outgoing_vertex(F, he)  # type: ignore
    one, two, three = wp.int32(1), wp.int32(2), wp.int32(3)
    ej = (
        wp.int32(F[fj][1] == vib) * one  # pyright: ignore
        + wp.int32(F[fj][2] == vib) * two  # pyright: ignore
    )
    hej = fj * three + ej
    return hej

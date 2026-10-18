"""
Static colliders of CPU editor scenes (mesh and plane colliders), as pinned contact geometry for the GPU
simulators.

GPU contact only knows the FEM nodes. A scene's static colliders therefore join the problem as
element-less FEM nodes appended after the bodies' nodes (see `scenario.make_fem_dynamics_object`'s
`static_points`), fixed forever by a kinematic constraint, and their triangles join the contact surface.
Contact detection and resolution then need no static-geometry special case. Plane colliders come in as
quads (`tools/vbd/scenario.py::static_meshes`), which vertex-SDF detection recognizes as planar and tests
analytically. The only collider-specific code is in this module, plus the open-mesh and plane queries in
`gpu/contact/mesh/sd.py`.
"""

import typing
import numpy as np
from pbatoolkit import pbat
from .gpu.kinematics.constraints import KinematicConstraint, KinematicConstraintKind


class ContactGeometry(typing.NamedTuple):
    """Contact surface of a scene with colliders, for state.SimulationState."""

    mesh: pbat.sim.contact.MultiMesh  # one component per body, then per collider
    plane_depths: np.ndarray  # (# components,) how deep a vertex may be under a plane component and
    # still touch it: inf except for finite planes (their thickness)


def collider_points(colliders: list) -> np.ndarray | None:
    """`|# points| x 3` vertices of all colliders (`scenario.StaticMesh`es), in order, or None without
    colliders."""
    return np.vstack([c.V for c in colliders]) if len(colliders) > 0 else None


def add_colliders(
    fem: pbat.sim.dynamics.FemElastoDynamics,
    body_offsets: np.ndarray,
    colliders: list,
) -> tuple[KinematicConstraint, ContactGeometry]:
    """Turn the element-less nodes after the bodies into static colliders.

    Gives them the bodies' mean nodal mass, which only scales the contact penalty through the Rayleigh
    quotients of the contacts they take part in, and builds the scene's contact surface.

    Args:
        fem: FEM problem whose nodes after `body_offsets[-1]` are the colliders' vertices, in order.
        body_offsets: `|# bodies| + 1` prefix offsets of the bodies' nodes.
        colliders: The scene's static colliders as `scenario.StaticMesh`es (see `scenario.static_meshes`).

    Returns:
        The kinematic constraint that fixes the collider nodes forever, and the contact geometry: each
        body's boundary, then each collider's triangles, one component per body and per collider, with
        each component's plane depth.
    """
    n_bodies = len(body_offsets) - 1
    n_body_nodes = int(body_offsets[-1])
    m = fem.m
    m[n_body_nodes:] = m[:n_body_nodes].mean()
    fem.m = m
    constraint = KinematicConstraint(
        name="Static colliders",
        kind=KinematicConstraintKind.Fixed,
        begin=0.0,
        duration=np.inf,
        nodes=np.arange(n_body_nodes, fem.X.shape[1]),
    )
    XCC = [np.repeat(np.arange(n_bodies), np.diff(body_offsets))]
    tets = pbat.sim.contact.MultiMesh()
    tets.construct_from_tetrahedral_mesh(fem.E, XCC[0], n_components=n_bodies)
    F = [tets.F]
    start = n_body_nodes
    for c, collider in enumerate(colliders):
        F.append((collider.F + start).T)
        XCC.append(np.full(collider.V.shape[0], n_bodies + c))
        start += collider.V.shape[0]
    contact_mesh = pbat.sim.contact.MultiMesh()
    contact_mesh.construct_from_triangle_mesh(
        np.hstack(F), np.concatenate(XCC), n_components=n_bodies + len(colliders)
    )
    depths = np.concatenate([np.full(n_bodies, np.inf), [c.depth for c in colliders]])
    return constraint, ContactGeometry(contact_mesh, depths)


def collider_faces(E: np.ndarray, F: np.ndarray, n_nodes: int) -> np.ndarray:
    """Contact triangles of static colliders, i.e. those over nodes that belong to no element.

    Args:
        E: `4 x |# tetrahedra|` FEM elements.
        F: `|# triangles| x 3` contact surface triangles.
        n_nodes: Number of FEM nodes.

    Returns:
        `|# collider triangles| x 3` triangles.
    """
    in_elements = np.zeros(n_nodes, dtype=bool)
    in_elements[E.ravel()] = True
    return F[~in_elements[F].any(axis=1)]

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


def scene_extents(
    X: np.ndarray, E: np.ndarray, contact_geometry: ContactGeometry | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """The scene's size: bounding-box extents of its elastic bodies, and of the bodies with its static mesh
    colliders.

    Plane colliders, infinite or finite, are left out: they enter as flat quads (`scenario.static_meshes`), told
    apart from static meshes by their zero thickness.

    Args:
        X: `3 x |# nodes|` node positions.
        E: `4 x |# tetrahedra|` FEM elements: the bodies' nodes.
        contact_geometry: The scene's contact geometry, one component per body, then per collider; None without
            colliders.

    Returns:
        `(3,)` extents of the bodies, and `(3,)` extents of the bodies with the static mesh colliders (None
        without static mesh colliders).
    """
    bodies = X[:, np.unique(E)]
    extents = np.ptp(bodies, axis=1)
    if contact_geometry is None:
        return extents, None
    in_elements = np.zeros(X.shape[1], dtype=bool)
    in_elements[E.ravel()] = True
    V, VP = np.asarray(contact_geometry.mesh.V), np.asarray(contact_geometry.mesh.VP)
    meshes = []
    for c in range(len(VP) - 1):
        nodes = V[VP[c] : VP[c + 1]]
        if nodes.size == 0 or in_elements[nodes].any():  # a body
            continue
        P = X[:, nodes]
        s = np.linalg.svd(P - P.mean(axis=1, keepdims=True), compute_uv=False)
        if s[-1] > 1e-9 * max(s[0], 1e-30):  # not flat: a static mesh, not a plane's quad
            meshes.append(P)
    if not meshes:
        return extents, None
    allP = np.hstack([bodies, *meshes])
    return extents, np.ptp(allP, axis=1)


import unittest


class TestSceneExtents(unittest.TestCase):
    def test_bodies_meshes_and_planes(self):
        # A unit tetrahedron (nodes 0-3), a static tetrahedron's surface 2 m away (4-7), a plane's quad (8-11)
        body = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        X = np.hstack([body, body + np.array([[2.0], [0.0], [0.0]]),
                       np.array([[-50, 50, 50, -50], [-50, -50, 50, 50], [-1, -1, -1, -1]], dtype=float)])
        E = np.arange(4).reshape(4, 1)
        mesh = typing.NamedTuple("Mesh", [("V", np.ndarray), ("VP", np.ndarray)])(np.arange(12), np.array([0, 4, 8, 12]))
        bodies, with_meshes = scene_extents(X, E, ContactGeometry(mesh, np.full(3, np.inf)))
        np.testing.assert_allclose(bodies, [1, 1, 1])
        np.testing.assert_allclose(with_meshes, [3, 1, 1])  # the plane is left out
        self.assertIsNone(scene_extents(X[:, :4], E, None)[1])


if __name__ == "__main__":
    unittest.main()

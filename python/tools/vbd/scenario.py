# type: ignore
"""
Simulation scenarios built from editor scene/session files.

Shared by the editor (`editor.py`), the CPU offline simulator (`simulate.py`) and the GPU simulators
(`python/tools/sim`). A scene file holds the tetrahedral elastodynamic bodies, the static mesh colliders and
the transform library. Nothing here initializes Polyscope; the body classes only import it.
"""

import itertools
import typing
import math
import h5py as h5
import numpy as np
from pbatoolkit import pbat, pypbat
from .ui.tetrahedral_elastodynamics_body import TetrahedralElastodynamicsBody
from .ui.static_mesh_collider import StaticMeshCollider
from .ui.plane_collider import PlaneCollider, deserialize_plane_colliders
from .ui.utils.transform_library import TransformLibrary

_archive_tlib_group = "transform_library"


def make_fem_dynamics_object(
    tet_elastic_bodies, get_external_info=False, static_points: np.ndarray | None = None
):
    """Build the FEM elastodynamics problem of the scene's tetrahedral bodies, concatenated in order.

    Args:
        tet_elastic_bodies: The scene's tetrahedral elastodynamic bodies.
        get_external_info (bool): Also return the per-body vertex counts and vertex offsets.
        static_points (np.ndarray | None): `|# points| x 3` extra nodes appended after the bodies' nodes,
            belonging to no element (zero mass and load, zero initial velocity). The GPU simulators use
            them for static mesh colliders.
    """
    vert_counts = [body.VT.shape[0] for body in tet_elastic_bodies]
    tet_counts = [body.T.shape[0] for body in tet_elastic_bodies]
    # Per-body offsets (prefix) array for vertices
    VP = np.array([0] + list(itertools.accumulate(vert_counts)))
    # Per-body offsets (prefix) array for tets
    TP = np.array([0] + list(itertools.accumulate(tet_counts)))
    V = np.vstack([body.VT for body in tet_elastic_bodies])
    n_static = 0 if static_points is None else static_points.shape[0]
    if n_static > 0:
        V = np.vstack([V, static_points])
    T = np.vstack([body.T + VP[b] for b, body in enumerate(tet_elastic_bodies)])
    # Construct the fem dynamics
    fem_dynamics = pbat.sim.dynamics.FemElastoDynamics(V.T, T.T)
    dims = fem_dynamics.X.shape[0]
    element = pbat.fem.Element.Tetrahedron
    element_order = 1
    mass_quadrature_order = 2
    elasticity_quadrature_order = 1
    load_quadrature_order = 1
    n_elems = fem_dynamics.E.shape[1]
    # Mass
    wgM = pbat.fem.mesh_quadrature_weights(
        fem_dynamics.E,
        fem_dynamics.X,
        element,
        order=element_order,
        quadrature_order=mass_quadrature_order,
    )
    egM = pbat.fem.mesh_quadrature_elements(fem_dynamics.E, wgM)
    XigM = pbat.fem.mesh_reference_quadrature_points(
        n_elems,
        element=element,
        order=element_order,
        quadrature_order=mass_quadrature_order,
    )
    rhog = np.zeros_like(wgM)
    for start, end, body in zip(TP[:-1], TP[1:], tet_elastic_bodies):
        rhog[:, start:end] = body.rhoe[np.newaxis, :]
    fem_dynamics.set_mass_matrix(
        eg=np.ravel(egM, order="F"),
        wg=np.ravel(wgM, order="F"),
        Xig=XigM,
        rhog=np.ravel(rhog, order="F"),
    )
    # Elasticity
    wgU = pbat.fem.mesh_quadrature_weights(
        fem_dynamics.E,
        fem_dynamics.X,
        element,
        order=element_order,
        quadrature_order=elasticity_quadrature_order,
    )
    egU = pbat.fem.mesh_quadrature_elements(fem_dynamics.E, wgU)
    XigU = pbat.fem.mesh_reference_quadrature_points(
        n_elems,
        element=element,
        order=element_order,
        quadrature_order=elasticity_quadrature_order,
    )
    mug, lambdag = np.zeros_like(wgU), np.zeros_like(wgU)
    for start, end, body in zip(TP[:-1], TP[1:], tet_elastic_bodies):
        mue, llambdae = pypbat.fem.lame_coefficients(body.Ye, body.nue)
        mug[:, start:end] = mue
        lambdag[:, start:end] = llambdae
    fem_dynamics.set_elastic_energy(
        eg=np.ravel(egU, order="F"),
        wg=np.ravel(wgU, order="F"),
        Xig=XigU,
        mug=np.ravel(mug, order="F"),
        lambdag=np.ravel(lambdag, order="F"),
    )
    # Load
    wgB = pbat.fem.mesh_quadrature_weights(
        fem_dynamics.E,
        fem_dynamics.X,
        element,
        order=element_order,
        quadrature_order=load_quadrature_order,
    )
    egB = pbat.fem.mesh_quadrature_elements(fem_dynamics.E, wgB)
    XigB = pbat.fem.mesh_reference_quadrature_points(
        n_elems,
        element=element,
        order=element_order,
        quadrature_order=load_quadrature_order,
    )
    bg = np.zeros((dims, math.prod(wgB.shape)))
    for start, end, body in zip(TP[:-1], TP[1:], tet_elastic_bodies):
        aext = body.aext.reshape(dims, 1)
        fext = (body.rhoe * aext) + body.bext.T
        bg[:, start:end] = fext
    fem_dynamics.set_external_load(
        eg=np.ravel(egB, order="F"), wg=np.ravel(wgB, order="F"), Xig=XigB, bg=bg
    )
    # Initial velocity
    x0 = fem_dynamics.X
    v0 = np.vstack(
        [body.v0 for body in tet_elastic_bodies] + [np.zeros((n_static, dims))]
    ).T
    fem_dynamics.x = x0
    fem_dynamics.v = v0

    if get_external_info:
        return fem_dynamics, vert_counts, VP
    return fem_dynamics


class StaticMesh(typing.NamedTuple):
    """Static collision geometry: a triangle mesh, and how deep below it a vertex is still in contact."""

    V: np.ndarray  # |# vertices| x 3
    F: np.ndarray  # |# triangles| x 3
    depth: float = np.inf  # finite planes: their thickness (see PlaneCollider.depth)


def static_meshes(
    tet_elastic_bodies: list[TetrahedralElastodynamicsBody],
    static_mesh_colliders: list[StaticMeshCollider],
    plane_colliders: list[PlaneCollider],
    tessellate_planes: bool = False,
) -> list[StaticMesh]:
    """The scene's static collision geometry: the static mesh colliders, then each plane collider's
    rectangle (infinite planes sized by the scene, finite planes with their thickness as depth).

    Args:
        tessellate_planes (bool): Split planes into cells about the scene's diameter wide, for the CPU's
            OGC contact. The GPU uses single quads (2 triangles), which it detects analytically.
    """
    meshes = [StaticMesh(smc.VT, smc.F) for smc in static_mesh_colliders]
    if len(plane_colliders) > 0:
        P = np.vstack([b.VT for b in tet_elastic_bodies] + [m.V for m in meshes])
        diameter = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
        cell_size = diameter if tessellate_planes else None
        meshes += [
            StaticMesh(*plane.mesh(diameter, cell_size=cell_size), plane.depth)
            for plane in plane_colliders
        ]
    return meshes


def make_contact_dynamics_object(
    fem_dynamics: pbat.sim.dynamics.FemElastoDynamics,
    n_bodies: int,
    vert_counts: list[int],
    static_meshes: list[StaticMesh],
):
    """Build the CPU contact problem: each body's boundary as dynamic geometry (one component per body),
    and the static meshes (see `static_meshes`) as static geometry."""
    contact_dynamics = pbat.sim.contact.MeshDynamics()
    XCC = np.concatenate([np.full(nverts, b) for b, nverts in enumerate(vert_counts)])
    contact_meshes = pbat.sim.contact.MultiMesh()
    contact_meshes.construct_from_tetrahedral_mesh(
        fem_dynamics.E, XCC, n_components=n_bodies
    )
    contact_dynamics.set_dynamic_geometry(fem_dynamics.x, contact_meshes)
    if len(static_meshes) > 0:
        static_vert_counts = [m.V.shape[0] for m in static_meshes]
        Vstatic = np.vstack([m.V for m in static_meshes])
        VPstatic = [0] + list(itertools.accumulate(static_vert_counts))
        Fstatic = np.vstack([m.F + VPstatic[b] for b, m in enumerate(static_meshes)])
        XCCstatic = np.concatenate(
            [np.full(nverts, b) for b, nverts in enumerate(static_vert_counts)]
        )
        static_contact_meshes = pbat.sim.contact.MultiMesh()
        static_contact_meshes.construct_from_triangle_mesh(
            Fstatic.T, XCCstatic, n_components=len(static_meshes)
        )
        contact_dynamics.set_static_geometry(Vstatic.T, static_contact_meshes)
    return contact_dynamics


def read_scene(path: str, tessellate_planes: bool = False) -> tuple[
    list[TetrahedralElastodynamicsBody],
    list[StaticMesh],
    TransformLibrary,
    list[str],
]:
    """Read the bodies, static collision geometry and transform library of a scene file.

    Returns:
        The tetrahedral elastodynamic bodies, the static meshes (mesh colliders and plane colliders' quads,
        see `static_meshes`, which `tessellate_planes` is passed to), the transform library (synced to the
        body names) and the body names, in body order.
    """
    tet_elastic_bodies = []
    static_mesh_colliders = []
    with h5.File(path, "r") as f:
        grp = f["fem_tet_elastic_bodies"]
        num_bodies = grp.attrs["num_tet_elastic_bodies"]
        for b in range(num_bodies):
            body_grp = grp[f"{b}"]
            body = TetrahedralElastodynamicsBody()
            body.deserialize(body_grp, headless=True)
            tet_elastic_bodies.append(body)

        # Scenes saved without colliders may lack the group altogether
        grp = f.get("static_mesh_colliders", None)
        num_static_meshes = 0 if grp is None else grp.attrs["num_static_mesh_colliders"]
        for b in range(num_static_meshes):
            body_grp = grp[f"{b}"]
            body = StaticMeshCollider()
            body.deserialize(body_grp, headless=True)
            static_mesh_colliders.append(body)

        plane_colliders = deserialize_plane_colliders(f.get("plane_colliders", None))
        transform_library = TransformLibrary()
        tlib_h5grp = f[_archive_tlib_group]
        transform_library.deserialize(tlib_h5grp)
    # Constraint nodes are mapped onto the FEM mesh by body, in body order. The library's own
    # mesh_names may list other meshes or another order (e.g. ArmadilloFallingInBasket).
    tet_elastic_body_names = [str(body.name) for body in tet_elastic_bodies]
    transform_library.sync_mesh_names(tet_elastic_body_names)
    meshes = static_meshes(
        tet_elastic_bodies, static_mesh_colliders, plane_colliders, tessellate_planes
    )
    return tet_elastic_bodies, meshes, transform_library, tet_elastic_body_names


def load_simulation_scenario_from_scene(path: str):
    """Load a scene file into the CPU simulation objects.

    Returns:
        The FEM problem, the CPU contact problem, the transform library (synced to the body names), the
        body names and the `|# bodies| + 1` node offsets of the bodies.
    """
    try:
        tet_elastic_bodies, meshes, transform_library, tet_elastic_body_names = read_scene(
            path, tessellate_planes=True
        )
        fem, vert_counts, XP = make_fem_dynamics_object(
            tet_elastic_bodies, get_external_info=True
        )
        contact = make_contact_dynamics_object(
            fem, len(tet_elastic_bodies), vert_counts, meshes
        )
        return fem, contact, transform_library, tet_elastic_body_names, XP
    except Exception as e:
        raise RuntimeError(
            f"Failed to load simulation scenario from '{path}': {e}"
        ) from e

# type: ignore

import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import h5py as h5

# An infinite plane is simulated as a quad of half-size INFINITE_SCALE times the scene's diameter
INFINITE_SCALE = 20.0
# Infinite planes are drawn as quads of half-size VISUAL_SCALE times the scene's diameter
VISUAL_SCALE = 2.0


def _perpendicular(n: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to the unit vector n."""
    a = np.eye(3)[np.argmin(np.abs(n))]
    t = np.cross(n, a)
    return t / np.linalg.norm(t)


class PlaneCollider:
    """Static plane collider: everything on the back side of a plane collides with it.

    The plane goes through `point` with unit `normal`, pointing to the free side. Its extent is the
    rectangle of half-sizes `half_extents` along the in-plane axes `tangent` and normal x tangent, or
    infinite (inf half-extents, the default). Simulators collide with it as a quad: the GPU detects
    contacts with it analytically, and the CPU treats it as a static triangle mesh.
    """

    name: str
    point: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    half_extents: np.ndarray
    _sm: ps.SurfaceMesh

    def __init__(
        self,
        name: str = "Plane",
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        tangent=None,
        half_extents=(np.inf, np.inf),
    ):
        self.name = name
        self.point = np.asarray(point, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.tangent = None if tangent is None else np.asarray(tangent, dtype=float)
        self.half_extents = np.asarray(half_extents, dtype=float)
        self._sm = None
        self.adjust()

    @property
    def infinite(self) -> bool:
        return not np.all(np.isfinite(self.half_extents))

    def adjust(self):
        """Normalize the normal and make the tangent a unit vector perpendicular to it."""
        self.normal = self.normal / np.linalg.norm(self.normal)
        t = _perpendicular(self.normal) if self.tangent is None else self.tangent
        t = t - np.dot(t, self.normal) * self.normal
        self.tangent = t / np.linalg.norm(t) if np.linalg.norm(t) > 1e-8 else _perpendicular(self.normal)

    def mesh(
        self,
        scene_diameter: float,
        scale: float = INFINITE_SCALE,
        cell_size: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The plane's rectangle as triangles facing along `normal` (in-plane axes tangent and
        normal x tangent).

        Args:
            scene_diameter (float): Diameter of the scene, which sizes infinite planes.
            scale (float): Half-size of an infinite plane, in scene diameters.
            cell_size (float | None): If given, split the rectangle into a grid of cells at most this
                wide. The CPU's OGC contact misses fast impacts on triangles much larger than the
                scene. The GPU detects planes analytically and uses the single quad.

        Returns:
            `|# vertices| x 3` vertices and `|# triangles| x 3` triangles.
        """
        a, b = np.where(np.isfinite(self.half_extents), self.half_extents, scale * scene_diameter)
        na, nb = (
            (1, 1)
            if cell_size is None
            else (max(1, int(np.ceil(2 * a / cell_size))), max(1, int(np.ceil(2 * b / cell_size))))
        )
        u, w = self.tangent, np.cross(self.normal, self.tangent)
        I, J = np.meshgrid(np.arange(na + 1), np.arange(nb + 1), indexing="ij")
        sa, sb = np.linspace(-a, a, na + 1)[I.ravel()], np.linspace(-b, b, nb + 1)[J.ravel()]
        V = self.point + sa[:, np.newaxis] * u + sb[:, np.newaxis] * w
        i, j = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
        v00, v10 = (i * (nb + 1) + j).ravel(), ((i + 1) * (nb + 1) + j).ravel()
        v11, v01 = v10 + 1, v00 + 1
        F = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
        return V, F

    def draw(self, scene_diameter: float):
        """Editor widgets. Returns True if the plane changed."""
        imgui.PushID(self.name)
        changed = False
        c, point = imgui.InputFloat3("Point", self.point.tolist())
        changed |= c
        c, normal = imgui.InputFloat3("Normal", self.normal.tolist())
        changed |= c
        c, tangent = imgui.InputFloat3("Tangent", self.tangent.tolist())
        changed |= c
        c, infinite = imgui.Checkbox("Infinite", self.infinite)
        changed |= c
        half_extents = self.half_extents
        if c:
            half_extents = (
                np.full(2, np.inf) if infinite else np.full(2, 0.5 * max(scene_diameter, 1e-3))
            )
        elif not infinite:
            c, he = imgui.InputFloat2("Half extents", self.half_extents.tolist())
            changed |= c
            half_extents = np.maximum(np.asarray(he, dtype=float), 1e-6)
        if changed:
            self.point, self.normal = np.asarray(point, dtype=float), np.asarray(normal, dtype=float)
            if np.linalg.norm(self.normal) < 1e-8:
                self.normal = np.array([0.0, 0.0, 1.0])
            self.tangent, self.half_extents = np.asarray(tangent, dtype=float), half_extents
            self.adjust()
            self.update_visual(scene_diameter)
        imgui.PopID()
        return changed

    def update_visual(self, scene_diameter: float):
        V, F = self.mesh(max(scene_diameter, 1e-3), scale=VISUAL_SCALE)
        if self._sm is not None:
            ps.remove_surface_mesh(self._sm.get_name())
        self._sm = ps.register_surface_mesh(self.name, V, F)

    def set_visible(self, visible: bool):
        if self._sm is not None:
            self._sm.set_enabled(visible)

    def on_removed(self):
        if self._sm is not None:
            ps.remove_surface_mesh(self._sm.get_name())
            self._sm = None

    def serialize(self, grp: h5.Group):
        grp = grp.create_group("tools.vbd.ui.PlaneCollider")
        grp.attrs["name"] = self.name
        grp.attrs["point"] = self.point
        grp.attrs["normal"] = self.normal
        grp.attrs["tangent"] = self.tangent
        grp.attrs["half_extents"] = self.half_extents

    def deserialize(self, grp: h5.Group):
        grp = grp["tools.vbd.ui.PlaneCollider"]
        self.name = str(grp.attrs["name"])
        self.point = np.asarray(grp.attrs["point"], dtype=float)
        self.normal = np.asarray(grp.attrs["normal"], dtype=float)
        self.tangent = np.asarray(grp.attrs["tangent"], dtype=float)
        self.half_extents = np.asarray(grp.attrs["half_extents"], dtype=float)
        self.adjust()


def serialize_plane_colliders(planes: list[PlaneCollider], grp: h5.Group):
    """Write the scene's `plane_colliders` group."""
    grp.attrs["num_plane_colliders"] = len(planes)
    for i, plane in enumerate(planes):
        plane.serialize(grp.create_group(f"{i}"))


def deserialize_plane_colliders(grp: h5.Group | None) -> list[PlaneCollider]:
    """Read the scene's `plane_colliders` group (None or absent: no planes)."""
    if grp is None:
        return []
    planes = []
    for i in range(int(grp.attrs["num_plane_colliders"])):
        plane = PlaneCollider()
        plane.deserialize(grp[f"{i}"])
        planes.append(plane)
    return planes


import unittest


class TestPlaneCollider(unittest.TestCase):
    def test_mesh_faces_along_normal(self):
        plane = PlaneCollider(point=(1, 2, 3), normal=(0, 1, 1), half_extents=(2, 0.5))
        V, F = plane.mesh(scene_diameter=10.0)
        for f in F:
            n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
            np.testing.assert_allclose(n / np.linalg.norm(n), plane.normal, atol=1e-12)
        np.testing.assert_allclose(V.mean(axis=0), plane.point, atol=1e-12)
        np.testing.assert_allclose((V - plane.point) @ plane.normal, 0, atol=1e-12)
        np.testing.assert_allclose(np.abs((V - plane.point) @ plane.tangent), 2, atol=1e-12)

    def test_infinite_plane_size(self):
        V, F = PlaneCollider().mesh(scene_diameter=3.0)
        np.testing.assert_allclose(np.abs(V[:, :2]).max(), INFINITE_SCALE * 3.0)
        self.assertEqual(F.shape, (2, 3))

    def test_tessellation(self):
        plane = PlaneCollider(normal=(1, 1, 0), half_extents=(2, 1))
        V, F = plane.mesh(scene_diameter=1.0, cell_size=0.5)
        self.assertEqual(F.shape[0], 2 * 8 * 4)
        area = sum(0.5 * np.linalg.norm(np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])) for f in F)
        self.assertAlmostEqual(area, 4 * 2 * 1)
        for f in F:
            n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
            np.testing.assert_allclose(n / np.linalg.norm(n), plane.normal, atol=1e-12)

    def test_serialization(self):
        planes = [PlaneCollider("ground"), PlaneCollider("wall", (1, 0, 0), (-1, 0, 0), half_extents=(1, 2))]
        with h5.File("planes.h5", "w", driver="core", backing_store=False) as f:
            serialize_plane_colliders(planes, f.create_group("plane_colliders"))
            loaded = deserialize_plane_colliders(f["plane_colliders"])
        for a, b in zip(planes, loaded):
            self.assertEqual(a.name, b.name)
            for attr in ("point", "normal", "tangent", "half_extents"):
                np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))


if __name__ == "__main__":
    unittest.main()

import math
import polyscope as ps
import polyscope.imgui as imgui
from ..ogc import *


class OgcContactBrowser:
    """Simple Polyscope/imgui browser for OGC contact pairs."""

    _CONTACT_KINDS = ["VV", "VE", "VF", "EE"]

    def __init__(self, ogc: Ogc, screen_fraction: float = 0.25):
        self._ogc = ogc
        self._x = ogc._points.numpy()
        self._V = ogc._meshes.data.V.numpy()
        self._F = ogc._meshes.data.F.numpy()
        self._screen_fraction: float = screen_fraction
        self._kind_idx: int = 0
        self._contact_idx: int = 0
        self._show_reverse: bool = False
        self._stencil_pc = None
        self._stencil_cn = None
        self._stencil_sm = None
        self._last_visualized: tuple[int, int, int, bool] | None = None

    def clear(self):
        if self._stencil_pc is not None:
            ps.remove_point_cloud(self._stencil_pc.get_name())
            self._stencil_pc = None
        if self._stencil_cn is not None:
            ps.remove_curve_network(self._stencil_cn.get_name())
            self._stencil_cn = None
        if self._stencil_sm is not None:
            ps.remove_surface_mesh(self._stencil_sm.get_name())
            self._stencil_sm = None

    def update(self, ogc: Ogc):
        """Refresh contact data from a new or updated Ogc instance."""
        self.clear()
        self._ogc = ogc
        self._x = ogc._points.numpy()
        self._last_visualized = None

    def draw(self):
        imgui.PushID("OgcContactBrowser")  # type: ignore

        changed, self._screen_fraction = imgui.SliderFloat(  # type: ignore
            "Zoom level", self._screen_fraction, 0.05, 1.0
        )
        if changed:
            self._focus_camera_on_contact()

        changed, self._show_reverse = imgui.Checkbox("Show Reverse", self._show_reverse)  # type: ignore
        if changed:
            self._contact_idx = 0

        contact_kinds = [
            f"{kind} ({count})"
            for kind, count in zip(self._CONTACT_KINDS, self._ogc.num_contacts)
        ]
        changed, self._kind_idx = imgui.Combo("Kind", self._kind_idx, contact_kinds)  # type: ignore
        if changed:
            self._contact_idx = 0

        contacts = self._get_selected_contacts()
        n = len(contacts[0])
        if n == 0:
            self._contact_idx = 0
            self.clear()
            self._last_visualized = None
            imgui.Text("No contacts of selected kind.")  # type: ignore
            imgui.PopID()  # type: ignore
            return

        self._contact_idx = max(0, min(self._contact_idx, n - 1))

        if imgui.Button("<##ogc_prev"):  # type: ignore
            self._contact_idx = max(0, self._contact_idx - 1)
        imgui.SameLine()  # type: ignore
        imgui.SetNextItemWidth(80)  # type: ignore
        _, self._contact_idx = imgui.InputInt("Index", self._contact_idx)  # type: ignore
        self._contact_idx = max(0, min(self._contact_idx, n - 1))
        imgui.SameLine()  # type: ignore
        if imgui.Button(">##ogc_next"):  # type: ignore
            self._contact_idx = min(n - 1, self._contact_idx + 1)
        imgui.SameLine()  # type: ignore
        imgui.Text(f"/ {n - 1}")  # type: ignore

        signature = (self._kind_idx, self._contact_idx, n, self._show_reverse)
        if signature != self._last_visualized:
            self._visualize_current_contact()
            self._last_visualized = signature

        u = int(contacts[0][self._contact_idx])
        v = int(contacts[1][self._contact_idx])
        imgui.Text(f"pair = ({u}, {v})")  # type: ignore
        imgui.Text(self._contact_stencil_str())  # type: ignore

        imgui.PopID()  # type: ignore

    def _contact_stencil_str(self) -> str:
        """Return a human-readable string of the actual point indices for the currently selected contact pair."""
        contacts = self._get_selected_contacts()
        if len(contacts[0]) == 0:
            return ""
        k = self._contact_idx
        V = self._V
        F = self._F
        u_raw = int(contacts[0][k])
        v_raw = int(contacts[1][k])

        def he_verts(he: int) -> tuple[int, int]:
            f, e_local = he // 3, he % 3
            return int(F[f, e_local]), int(F[f, (e_local + 1) % 3])

        def tri_verts(f: int) -> tuple[int, int, int]:
            return int(F[f, 0]), int(F[f, 1]), int(F[f, 2])

        if self._kind_idx == 0:  # VV — both u and v are vertex indices
            iu, iv = int(V[u_raw]), int(V[v_raw])
            return f"vv stencil = ({iu}, {iv})"
        elif self._kind_idx == 1:  # VE
            if not self._show_reverse:  # forward: u=vertex, v=half-edge
                iu = int(V[u_raw])
                i, j = he_verts(v_raw)
                return f"ve stencil = ({iu}, ({i}, {j}))"
            else:  # reverse: u=half-edge, v=vertex
                i, j = he_verts(u_raw)
                iv = int(V[v_raw])
                return f"ev stencil = (({i}, {j}), {iv})"
        elif self._kind_idx == 2:  # VF
            if not self._show_reverse:  # forward: u=vertex, v=face
                iu = int(V[u_raw])
                i, j, kk = tri_verts(v_raw)
                return f"vf stencil = ({iu}, ({i}, {j}, {kk}))"
            else:  # reverse: u=face, v=vertex
                i, j, kk = tri_verts(u_raw)
                iv = int(V[v_raw])
                return f"fv stencil = (({i}, {j}, {kk}), {iv})"
        else:  # EE — both u and v are half-edges regardless of direction
            i, j = he_verts(u_raw)
            kk, ll = he_verts(v_raw)
            return f"ee stencil = (({i}, {j}), ({kk}, {ll}))"

    def _get_selected_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        if self._show_reverse:
            if self._kind_idx == 0:
                return self._ogc.rvv_contacts
            if self._kind_idx == 1:
                return self._ogc.rve_contacts
            if self._kind_idx == 2:
                return self._ogc.rvf_contacts
            return self._ogc.ree_contacts
        else:
            if self._kind_idx == 0:
                return self._ogc.vv_contacts
            if self._kind_idx == 1:
                return self._ogc.ve_contacts
            if self._kind_idx == 2:
                return self._ogc.vf_contacts
            return self._ogc.ee_contacts

    def _visualize_current_contact(self):
        self.clear()

        contacts = self._get_selected_contacts()
        if len(contacts[0]) == 0:
            return

        x = self._x
        V = self._V
        F = self._F
        k = self._contact_idx

        if self._kind_idx == 0:
            u, v = int(contacts[0][k]), int(contacts[1][k])
            iu = int(V[u])
            iv = int(V[v])
            self._stencil_pc = ps.register_point_cloud("OGC VV Contact", x[[iu, iv], :])

        elif self._kind_idx == 1:
            v, he = (
                (int(contacts[0][k]), int(contacts[1][k]))
                if not self._show_reverse
                else (int(contacts[1][k]), int(contacts[0][k]))
            )
            iv = int(V[v])
            f = he // 3
            e_local = he % 3
            i = int(F[f, e_local])
            j = int(F[f, (e_local + 1) % 3])
            self._stencil_pc = ps.register_point_cloud("OGC VE Vertex", x[[iv], :])
            self._stencil_cn = ps.register_curve_network(
                "OGC VE Edge",
                x[[i, j], :],
                np.array([[0, 1]], dtype=np.int32),
            )

        elif self._kind_idx == 2:
            v, f = (
                (int(contacts[0][k]), int(contacts[1][k]))
                if not self._show_reverse
                else (int(contacts[1][k]), int(contacts[0][k]))
            )
            iv = int(V[v])
            tri = F[f, :]
            self._stencil_pc = ps.register_point_cloud("OGC VF Vertex", x[[iv], :])
            self._stencil_sm = ps.register_surface_mesh(
                "OGC VF Triangle",
                x[tri, :],
                np.array([[0, 1, 2]], dtype=np.int32),
            )

        elif self._kind_idx == 3:
            he0, he1 = int(contacts[0][k]), int(contacts[1][k])
            f0, e0 = he0 // 3, he0 % 3
            f1, e1 = he1 // 3, he1 % 3
            i0 = int(F[f0, e0])
            i1 = int(F[f0, (e0 + 1) % 3])
            j0 = int(F[f1, e1])
            j1 = int(F[f1, (e1 + 1) % 3])
            self._stencil_cn = ps.register_curve_network(
                "OGC EE Edges",
                x[[i0, i1, j0, j1], :],
                np.array([[0, 1], [2, 3]], dtype=np.int32),
            )

        else:
            raise NotImplementedError(f"Contact kind {self._kind_idx} not supported")

        self._focus_camera_on_contact()

    def _focus_camera_on_contact(self):
        """Reposition the camera so the contact stencil fills `screen_fraction` of screen height."""
        contacts = self._get_selected_contacts()
        if len(contacts[0]) == 0:
            return
        x = self._x
        V = self._V
        F = self._F
        k = self._contact_idx

        if self._kind_idx == 0:  # VV
            u, v = int(contacts[0][k]), int(contacts[1][k])
            pts = x[[int(V[u]), int(V[v])], :]
        elif self._kind_idx == 1:  # VE
            v_idx, he = (
                (int(contacts[0][k]), int(contacts[1][k]))
                if not self._show_reverse
                else (int(contacts[1][k]), int(contacts[0][k]))
            )
            f, e_local = he // 3, he % 3
            pts = x[
                [int(V[v_idx]), int(F[f, e_local]), int(F[f, (e_local + 1) % 3])], :
            ]
        elif self._kind_idx == 2:  # VF
            v_idx, f = (
                (int(contacts[0][k]), int(contacts[1][k]))
                if not self._show_reverse
                else (int(contacts[1][k]), int(contacts[0][k]))
            )
            pts = x[[int(V[v_idx]), int(F[f, 0]), int(F[f, 1]), int(F[f, 2])], :]
        else:  # EE
            he0, he1 = int(contacts[0][k]), int(contacts[1][k])
            f0, e0 = he0 // 3, he0 % 3
            f1, e1 = he1 // 3, he1 % 3
            pts = x[
                [
                    int(F[f0, e0]),
                    int(F[f0, (e0 + 1) % 3]),
                    int(F[f1, e1]),
                    int(F[f1, (e1 + 1) % 3]),
                ],
                :,
            ]

        centroid = pts.mean(axis=0)
        # Bounding-sphere radius of the stencil points around the centroid
        radius = float(np.linalg.norm(pts - centroid, axis=1).max())
        # Ensure a minimum radius so we don't fly into a single degenerate point
        radius = max(radius, 1e-4)

        # Compute the camera distance so the stencil subtends `screen_fraction` of
        # screen height: screen_fraction = radius / (dist * tan(fov_half))
        cam_params = ps.get_view_camera_parameters()
        fov_half_rad = math.radians(cam_params.get_fov_vertical_deg()) * 0.5
        dist = radius / (self._screen_fraction * math.tan(fov_half_rad))

        # Keep the current view direction, just reposition along it
        look_dir = np.array(cam_params.get_look_dir(), dtype=float)
        cam_pos = centroid - look_dir * dist
        ps.look_at(cam_pos, centroid, fly_to=True)

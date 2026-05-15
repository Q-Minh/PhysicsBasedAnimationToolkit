import math
import numpy as np
import warp as wp
import cupy as cp
import polyscope as ps
import polyscope.imgui as imgui

from ..mesh.pairs import ContactPairs
from ..multimesh import MultiMesh


class ContactBrowser:
    """Simple Polyscope/imgui browser for mesh contact pairs.

    Displays one contact stencil at a time, browsable by kind (VV, VE, VF, EE)
    and index, with optional contact-frame basis vector overlays.
    """

    _CONTACT_KINDS = ["VV", "VE", "VF", "EE"]
    _BASIS_COLORS: dict[str, tuple[float, float, float]] = {
        "Normal": (0.90, 0.20, 0.20),
        "Tangent": (0.20, 0.75, 0.20),
        "Bitangent": (0.20, 0.45, 0.90),
    }

    def __init__(
        self,
        x: wp.array,
        meshes: MultiMesh,
        contacts: ContactPairs,
        screen_fraction: float = 0.25,
    ):
        self._contacts = contacts
        self._x = x.numpy()
        self._V = meshes.data.V.numpy()
        self._F = meshes.data.F.numpy()
        self._screen_fraction: float = screen_fraction
        self._kind_idx: int = 0
        self._contact_idx: int = 0
        self._show_reverse: bool = False
        self._stencil_pc = None
        self._stencil_cn = None
        self._stencil_sm = None
        self._last_visualized: tuple | None = None
        self._show_normal: bool = True
        self._show_tangent: bool = False
        self._show_bitangent: bool = False
        self._point_radius: float = 0.001
        self._edge_radius: float = 0.001
        self._vector_length: float = 0.05
        self._vector_radius: float = 0.001

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

    def update(self, x: wp.array, meshes: MultiMesh, contacts: ContactPairs):
        """Refresh visualization data from updated positions and contact pairs."""
        self.clear()
        self._contacts = contacts
        self._x = x.numpy()
        self._V = meshes.data.V.numpy()
        self._F = meshes.data.F.numpy()
        self._last_visualized = None

    def draw(self):
        imgui.PushID("ContactBrowser")  # type: ignore

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
            for kind, count in zip(self._CONTACT_KINDS, self._contacts.num_contacts)
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

        if imgui.Button("<##cb_prev"):  # type: ignore
            self._contact_idx = max(0, self._contact_idx - 1)
        imgui.SameLine()  # type: ignore
        imgui.SetNextItemWidth(80)  # type: ignore
        _, self._contact_idx = imgui.InputInt("Index", self._contact_idx)  # type: ignore
        self._contact_idx = max(0, min(self._contact_idx, n - 1))
        imgui.SameLine()  # type: ignore
        if imgui.Button(">##cb_next"):  # type: ignore
            self._contact_idx = min(n - 1, self._contact_idx + 1)
        imgui.SameLine()  # type: ignore
        imgui.Text(f"/ {n - 1}")  # type: ignore

        imgui.Separator()  # type: ignore
        _, self._show_normal = imgui.Checkbox("Normal##basis", self._show_normal)  # type: ignore
        imgui.SameLine()  # type: ignore
        _, self._show_tangent = imgui.Checkbox("Tangent##basis", self._show_tangent)  # type: ignore
        imgui.SameLine()  # type: ignore
        _, self._show_bitangent = imgui.Checkbox("Bitangent##basis", self._show_bitangent)  # type: ignore

        imgui.Separator()  # type: ignore
        imgui.Text("Scale:")  # type: ignore
        imgui.PushItemWidth(120)  # type: ignore
        _, self._point_radius = imgui.SliderFloat("Point radius##size", self._point_radius, 0.001, 0.02)  # type: ignore
        imgui.SameLine()  # type: ignore
        _, self._edge_radius = imgui.SliderFloat("Edge radius##size", self._edge_radius, 0.001, 0.02)  # type: ignore
        _, self._vector_length = imgui.SliderFloat("Vector length##size", self._vector_length, 0.001, 0.5)  # type: ignore
        imgui.SameLine()  # type: ignore
        _, self._vector_radius = imgui.SliderFloat("Vector radius##size", self._vector_radius, 0.0005, 0.01)  # type: ignore
        imgui.PopItemWidth()  # type: ignore

        signature = (
            self._kind_idx,
            self._contact_idx,
            n,
            self._show_reverse,
            self._show_normal,
            self._show_tangent,
            self._show_bitangent,
            self._point_radius,
            self._edge_radius,
            self._vector_length,
            self._vector_radius,
        )
        if signature != self._last_visualized:
            self._visualize_current_contact()
            self._last_visualized = signature  # type: ignore

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
                return self._contacts.rvv_contacts
            if self._kind_idx == 1:
                return self._contacts.rve_contacts
            if self._kind_idx == 2:
                return self._contacts.rvf_contacts
            return self._contacts.ree_contacts
        else:
            if self._kind_idx == 0:
                return self._contacts.vv_contacts
            if self._kind_idx == 1:
                return self._contacts.ve_contacts
            if self._kind_idx == 2:
                return self._contacts.vf_contacts
            return self._contacts.ee_contacts

    def _get_basis(self, kind: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (n, t, b) unit vectors for contact index k of the given kind."""
        fwd_data, _ = self._contacts.read_data
        if kind == "VV":
            bases = fwd_data.vv_bases
        elif kind == "VE":
            bases = fwd_data.ve_bases
        elif kind == "VF":
            bases = fwd_data.vf_bases
        else:  # EE
            bases = fwd_data.ee_bases
        n = cp.asarray(bases.n)[k].get()  # (3,)
        t = cp.asarray(bases.t)[k].get()  # (3,)
        b = cp.asarray(bases.b)[k].get()  # (3,)
        return n, t, b

    def _visualize_current_contact(self):
        self.clear()

        contacts = self._get_selected_contacts()
        if len(contacts[0]) == 0:
            return

        x = self._x
        V = self._V
        F = self._F
        k = self._contact_idx
        fwd_data, _ = self._contacts.read_data

        _basis_enabled = {
            "Normal": self._show_normal,
            "Tangent": self._show_tangent,
            "Bitangent": self._show_bitangent,
        }

        if self._kind_idx == 0:  # VV
            u, v = int(contacts[0][k]), int(contacts[1][k])
            iu = int(V[u])
            iv = int(V[v])
            self._stencil_pc = ps.register_point_cloud("Contact VV", x[[iu, iv], :])
            self._stencil_pc.set_radius(self._point_radius, relative=True)
            if not self._show_reverse:
                n_v, t_v, b_v = self._get_basis("VV", k)
                for name, vec in (("Normal", n_v), ("Tangent", t_v), ("Bitangent", b_v)):
                    self._stencil_pc.add_vector_quantity(
                        name,
                        np.array([vec, -vec]),
                        enabled=_basis_enabled[name],
                        color=self._BASIS_COLORS[name],
                        length=self._vector_length,
                        radius=self._vector_radius,
                    )

        elif self._kind_idx == 1:  # VE
            v_prim, he = (
                (int(contacts[0][k]), int(contacts[1][k]))
                if not self._show_reverse
                else (int(contacts[1][k]), int(contacts[0][k]))
            )
            iv = int(V[v_prim])
            f = he // 3
            e_local = he % 3
            i = int(F[f, e_local])
            j = int(F[f, (e_local + 1) % 3])
            self._stencil_pc = ps.register_point_cloud("Contact VE Vertex", x[[iv], :])
            self._stencil_pc.set_radius(self._point_radius, relative=True)
            self._stencil_cn = ps.register_curve_network(
                "Contact VE Edge",
                x[[i, j], :],
                np.array([[0, 1]], dtype=np.int32),
            )
            self._stencil_cn.set_radius(self._edge_radius, relative=True)
            if not self._show_reverse:
                n_v, t_v, b_v = self._get_basis("VE", k)
                t_bary = float(cp.asarray(fwd_data.ve_bary)[k].get())
                for name, vec in (("Normal", n_v), ("Tangent", t_v), ("Bitangent", b_v)):
                    self._stencil_pc.add_vector_quantity(
                        name,
                        np.array([vec]),
                        enabled=_basis_enabled[name],
                        color=self._BASIS_COLORS[name],
                        length=self._vector_length,
                        radius=self._vector_radius,
                    )
                    self._stencil_cn.add_vector_quantity(
                        name,
                        -np.array([(1.0 - t_bary) * vec, t_bary * vec]),
                        enabled=_basis_enabled[name],
                        color=self._BASIS_COLORS[name],
                        length=self._vector_length,
                        radius=self._vector_radius,
                    )

        elif self._kind_idx == 2:  # VF
            v_prim, f = (
                (int(contacts[0][k]), int(contacts[1][k]))
                if not self._show_reverse
                else (int(contacts[1][k]), int(contacts[0][k]))
            )
            iv = int(V[v_prim])
            tri = F[f, :]
            self._stencil_pc = ps.register_point_cloud("Contact VF Vertex", x[[iv], :])
            self._stencil_pc.set_radius(self._point_radius, relative=True)
            self._stencil_sm = ps.register_surface_mesh(
                "Contact VF Triangle",
                x[tri, :],
                np.array([[0, 1, 2]], dtype=np.int32),
            )
            if not self._show_reverse:
                n_v, t_v, b_v = self._get_basis("VF", k)
                vw = cp.asarray(fwd_data.vf_bary)[k].get()  # (v, w)
                v_b, w_b = float(vw[0]), float(vw[1])
                u_b = 1.0 - v_b - w_b
                for name, vec in (("Normal", n_v), ("Tangent", t_v), ("Bitangent", b_v)):
                    self._stencil_pc.add_vector_quantity(
                        name,
                        np.array([vec]),
                        enabled=_basis_enabled[name],
                        color=self._BASIS_COLORS[name],
                        length=self._vector_length,
                        radius=self._vector_radius,
                    )
                    self._stencil_sm.add_vector_quantity(
                        name,
                        -np.array([u_b * vec, v_b * vec, w_b * vec]),
                        enabled=_basis_enabled[name],
                        color=self._BASIS_COLORS[name],
                        length=self._vector_length,
                        radius=self._vector_radius,
                    )

        elif self._kind_idx == 3:  # EE
            he0, he1 = int(contacts[0][k]), int(contacts[1][k])
            f0, e0 = he0 // 3, he0 % 3
            f1, e1 = he1 // 3, he1 % 3
            i0 = int(F[f0, e0])
            i1 = int(F[f0, (e0 + 1) % 3])
            j0 = int(F[f1, e1])
            j1 = int(F[f1, (e1 + 1) % 3])
            self._stencil_cn = ps.register_curve_network(
                "Contact EE Edges",
                x[[i0, i1, j0, j1], :],
                np.array([[0, 1], [2, 3]], dtype=np.int32),
            )
            self._stencil_cn.set_radius(self._edge_radius, relative=True)
            if not self._show_reverse:
                n_v, t_v, b_v = self._get_basis("EE", k)
                st = cp.asarray(fwd_data.ee_bary)[k].get()  # (s, t)
                s, t_bary = float(st[0]), float(st[1])
                w_nodes = np.array([1.0 - s, s, 1.0 - t_bary, t_bary])
                for name, vec in (("Normal", n_v), ("Tangent", t_v), ("Bitangent", b_v)):
                    self._stencil_cn.add_vector_quantity(
                        name,
                        np.array([
                            w_nodes[0] * vec,
                            w_nodes[1] * vec,
                            -w_nodes[2] * vec,
                            -w_nodes[3] * vec,
                        ]),
                        enabled=_basis_enabled[name],
                        color=self._BASIS_COLORS[name],
                        length=self._vector_length,
                        radius=self._vector_radius,
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
            pts = x[[int(V[v_idx]), int(F[f, e_local]), int(F[f, (e_local + 1) % 3])], :]
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
            pts = x[[int(F[f0, e0]), int(F[f0, (e0 + 1) % 3]), int(F[f1, e1]), int(F[f1, (e1 + 1) % 3])], :]

        centroid = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - centroid, axis=1).max())
        radius = max(radius, 1e-4)
        cam_params = ps.get_view_camera_parameters()
        fov_half_rad = math.radians(cam_params.get_fov_vertical_deg()) * 0.5
        screen_fraction = max(self._screen_fraction, 1e-4)
        dist = radius / (screen_fraction * math.tan(fov_half_rad))
        look_dir = np.array(cam_params.get_look_dir(), dtype=float)
        cam_pos = centroid - look_dir * dist
        ps.look_at(cam_pos, centroid, fly_to=True)


class ContactOverview:
    """Polyscope overlay that displays *all* contacts of every type simultaneously.

    Each contact type uses a consistent two-color coding:
      - first  side of a pair → orange
      - second side of a pair → blue

    Structures registered:
      1. PointCloud   - VV vertex A  (all first vertices)
      2. PointCloud   - VV vertex B  (all second vertices)
      3. PointCloud   - VE vertex    (query vertices)
      4. CurveNetwork - VE edge      (candidate edges)
      5. PointCloud   - VF vertex    (query vertices)
      6. SurfaceMesh  - VF triangle  (candidate triangles)
      7. CurveNetwork - EE edge 0    (first edges)
      8. CurveNetwork - EE edge 1    (second edges)
    """

    _COLOR_FIRST = (0.95, 0.45, 0.10)   # orange
    _COLOR_SECOND = (0.20, 0.55, 0.90)  # blue

    _PS_NAMES = {
        "vv_u":  "Contact Overview - VV vertex A",
        "vv_v":  "Contact Overview - VV vertex B",
        "ve_v":  "Contact Overview - VE vertex",
        "ve_e":  "Contact Overview - VE edge",
        "vf_v":  "Contact Overview - VF vertex",
        "vf_f":  "Contact Overview - VF triangle",
        "ee_e0": "Contact Overview - EE edge 0",
        "ee_e1": "Contact Overview - EE edge 1",
    }

    _TYPE_KEYS = {
        "VV": ("vv_u", "vv_v"),
        "VE": ("ve_v", "ve_e"),
        "VF": ("vf_v", "vf_f"),
        "EE": ("ee_e0", "ee_e1"),
    }

    def __init__(
        self,
        x: wp.array,
        meshes: MultiMesh,
        contacts: ContactPairs,
        point_radius: float = 0.005,
        edge_radius: float = 0.003,
    ):
        self._point_radius = point_radius
        self._edge_radius = edge_radius
        self._structures: dict = {}
        self._type_visible: dict[str, bool] = {t: True for t in self._TYPE_KEYS}
        self._dirty = False
        self._x_np: np.ndarray | None = None
        self._V_np: np.ndarray | None = None
        self._F_np: np.ndarray | None = None
        self._cached_contacts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.update(x, meshes, contacts)

    def update(self, x: wp.array, meshes: MultiMesh, contacts: ContactPairs):
        """Cache contact data for the next draw() call. Does not register any polyscope structures."""
        self._remove_all()
        self._x_np = x.numpy()
        self._V_np = meshes.data.V.numpy()
        self._F_np = meshes.data.F.numpy()
        self._cached_contacts = {
            "VV": contacts.vv_contacts,
            "VE": contacts.ve_contacts,
            "VF": contacts.vf_contacts,
            "EE": contacts.ee_contacts,
        }
        self._dirty = True

    def remove(self):
        """Remove all polyscope structures owned by this overview."""
        self._remove_all()

    def set_visible(self, visible: bool):
        for t in self._TYPE_KEYS:
            self._type_visible[t] = visible
        for s in self._structures.values():
            s.set_enabled(visible)

    def set_type_visible(self, contact_type: str, visible: bool):
        """Show or hide structures for a single contact type (``"VV"``, ``"VE"``, ``"VF"``, ``"EE"``)."""
        self._type_visible[contact_type] = visible
        for key in self._TYPE_KEYS[contact_type]:
            if key in self._structures:
                self._structures[key].set_enabled(visible)

    def draw(self):
        """Draw imgui controls for the overview. Builds polyscope structures on first call after update()."""
        imgui.PushID("ContactOverview")  # type: ignore

        if self._dirty:
            self._build()
            self._dirty = False

        imgui.Text("Show:")  # type: ignore
        first = True
        for t in self._TYPE_KEYS:
            if not first:
                imgui.SameLine()  # type: ignore
            first = False
            changed, self._type_visible[t] = imgui.Checkbox(f"{t}##ov_vis", self._type_visible[t])  # type: ignore
            if changed:
                self.set_type_visible(t, self._type_visible[t])

        imgui.PushItemWidth(150)  # type: ignore
        changed_pr, self._point_radius = imgui.SliderFloat(  # type: ignore
            "Point radius##ov", self._point_radius, 0.001, 0.05
        )
        imgui.SameLine()  # type: ignore
        changed_er, self._edge_radius = imgui.SliderFloat(  # type: ignore
            "Edge radius##ov", self._edge_radius, 0.001, 0.05
        )
        imgui.PopItemWidth()  # type: ignore
        if changed_pr or changed_er:
            for key, s in self._structures.items():
                if key in ("vv_u", "vv_v", "ve_v", "vf_v"):
                    s.set_radius(self._point_radius, relative=True)
                elif key in ("ve_e", "ee_e0", "ee_e1"):
                    s.set_radius(self._edge_radius, relative=True)

        imgui.PopID()  # type: ignore

    def _remove_all(self):
        for key, s in self._structures.items():
            name = self._PS_NAMES[key]
            if key in ("vv_u", "vv_v", "ve_v", "vf_v"):
                ps.remove_point_cloud(name)
            elif key == "vf_f":
                ps.remove_surface_mesh(name)
            else:
                ps.remove_curve_network(name)
        self._structures = {}

    def _build(self):
        assert self._x_np is not None and self._V_np is not None and self._F_np is not None
        x = self._x_np
        V = self._V_np
        F = self._F_np

        # Vertex-Vertex
        vv_u, vv_v = self._cached_contacts["VV"]
        if len(vv_u) > 0:
            s = ps.register_point_cloud(self._PS_NAMES["vv_u"], x[V[vv_u.astype(int)], :])
            s.set_radius(self._point_radius, relative=True)
            s.set_color(self._COLOR_FIRST)
            self._structures["vv_u"] = s
            s = ps.register_point_cloud(self._PS_NAMES["vv_v"], x[V[vv_v.astype(int)], :])
            s.set_radius(self._point_radius, relative=True)
            s.set_color(self._COLOR_SECOND)
            self._structures["vv_v"] = s

        # Vertex-Edge
        ve_v, ve_he = self._cached_contacts["VE"]
        if len(ve_v) > 0:
            s = ps.register_point_cloud(self._PS_NAMES["ve_v"], x[V[ve_v.astype(int)], :])
            s.set_radius(self._point_radius, relative=True)
            s.set_color(self._COLOR_FIRST)
            self._structures["ve_v"] = s

            he_arr = ve_he.astype(int)
            f_arr = he_arr // 3
            e_local_arr = he_arr % 3
            i_arr = F[f_arr, e_local_arr]
            j_arr = F[f_arr, (e_local_arr + 1) % 3]
            n_ve = len(he_arr)
            verts_e = np.empty((2 * n_ve, 3), dtype=float)
            verts_e[0::2] = x[i_arr, :]
            verts_e[1::2] = x[j_arr, :]
            edges_e = np.stack(
                [np.arange(0, 2 * n_ve, 2), np.arange(1, 2 * n_ve, 2)], axis=1
            ).astype(np.int32)
            s = ps.register_curve_network(self._PS_NAMES["ve_e"], verts_e, edges_e)
            s.set_radius(self._edge_radius, relative=True)
            s.set_color(self._COLOR_SECOND)
            self._structures["ve_e"] = s

        # Vertex-Face
        vf_v, vf_f = self._cached_contacts["VF"]
        if len(vf_v) > 0:
            s = ps.register_point_cloud(self._PS_NAMES["vf_v"], x[V[vf_v.astype(int)], :])
            s.set_radius(self._point_radius, relative=True)
            s.set_color(self._COLOR_FIRST)
            self._structures["vf_v"] = s

            f_arr = vf_f.astype(int)
            tri_inds = F[f_arr, :]
            n_vf = len(f_arr)
            verts_f = x[tri_inds.reshape(-1), :]
            faces_f = np.arange(3 * n_vf, dtype=np.int32).reshape(n_vf, 3)
            s = ps.register_surface_mesh(self._PS_NAMES["vf_f"], verts_f, faces_f)
            s.set_color(self._COLOR_SECOND)
            self._structures["vf_f"] = s

        # Edge-Edge
        ee_he0, ee_he1 = self._cached_contacts["EE"]
        if len(ee_he0) > 0:
            for key, he_arr_raw in (("ee_e0", ee_he0), ("ee_e1", ee_he1)):
                color = self._COLOR_FIRST if key == "ee_e0" else self._COLOR_SECOND
                he_arr = he_arr_raw.astype(int)
                f_arr = he_arr // 3
                e_local_arr = he_arr % 3
                i_arr = F[f_arr, e_local_arr]
                j_arr = F[f_arr, (e_local_arr + 1) % 3]
                n_ee = len(he_arr)
                verts_e = np.empty((2 * n_ee, 3), dtype=float)
                verts_e[0::2] = x[i_arr, :]
                verts_e[1::2] = x[j_arr, :]
                edges_e = np.stack(
                    [np.arange(0, 2 * n_ee, 2), np.arange(1, 2 * n_ee, 2)], axis=1
                ).astype(np.int32)
                s = ps.register_curve_network(self._PS_NAMES[key], verts_e, edges_e)
                s.set_radius(self._edge_radius, relative=True)
                s.set_color(color)
                self._structures[key] = s

        # Apply deferred visibility
        for t, keys in self._TYPE_KEYS.items():
            if not self._type_visible.get(t, True):
                for key in keys:
                    if key in self._structures:
                        self._structures[key].set_enabled(False)

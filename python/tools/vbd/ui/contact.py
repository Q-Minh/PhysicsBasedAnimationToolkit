# type: ignore

from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
from .params import ParameterObject


class Contact:
    _contact_dynamics: pbat.sim.contact.MeshDynamics
    _contact_params: ParameterObject
    _environment_mesh: ps.SurfaceMesh
    _show_debug: bool
    _debug_data: pbat.sim.contact.DebugMeshDynamics
    _selected_type: int
    _selected_idx: dict

    # Polyscope structures for the currently visualized contact stencil
    _stencil_pc: ps.PointCloud
    _stencil_cn: ps.CurveNetwork
    _stencil_sm: ps.SurfaceMesh

    _CONTACT_TYPE_NAMES = [
        "Point-Point",
        "Point-Edge",
        "Point-Triangle",
        "Edge-Edge",
    ]

    _SORT_KEYS = [
        ("None", None),
        ("c(x)", lambda c: c.cx),
        ("lambda", lambda c: c.lam),
        ("slack", lambda c: c.slack),
        ("friction constraint", lambda c: np.linalg.norm(c.cf)),
    ]

    def __init__(self):
        self._contact_dynamics = None
        self._contact_params = None
        self._environment_mesh = None
        self._show_debug = False
        self._debug_data = None
        self._selected_type = 0
        self._selected_idx = {0: 0, 1: 0, 2: 0, 3: 0}
        self._stencil_pc = None
        self._stencil_cn = None
        self._stencil_sm = None
        self._normalize_gradients = False
        # Sort order state: index into _SORT_KEYS and cached permutation per type
        self._sort_key_idx = 0
        self._sort_orders: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        self.on_new_contact_dynamics(
            pbat.sim.contact.MeshDynamics(),
        )

    def draw(self):
        imgui.PushID("Contact")
        if imgui.TreeNode("Parameters"):
            self._contact_params.draw()
            params: pbat.sim.contact.MeshDynamicsParams = self._contact_params.params
            params.with_normal_contact(params.kc).with_frictional_contact(
                params.mu, params.epsv
            )
            imgui.TreePop()
        changed, self._show_debug = imgui.Checkbox("Debug Contacts", self._show_debug)
        if changed and not self._show_debug:
            self._clear_stencil_visualization()
        if self._show_debug:
            norm_changed, self._normalize_gradients = imgui.Checkbox(
                "Normalize Gradients", self._normalize_gradients
            )
            if self._debug_data is not None:
                if norm_changed:
                    self._visualize_current_contact()
                self._draw_debug_browser()
        imgui.PopID()

    def _draw_debug_browser(self):
        dd = self._debug_data
        counts = [
            len(dd.point_point_contacts),
            len(dd.point_edge_contacts),
            len(dd.point_triangle_contacts),
            len(dd.edge_edge_contacts),
        ]
        total = sum(counts)
        imgui.Text(f"Total contacts: {total}")

        # Sort order selector
        sort_names = [name for name, _ in self._SORT_KEYS]
        imgui.SetNextItemWidth(200)
        sort_changed, self._sort_key_idx = imgui.Combo(
            "Sort by##contact_sort", self._sort_key_idx, sort_names
        )
        if sort_changed:
            self._rebuild_all_sort_orders()
            # Reset browsing indices when sort changes
            for k in self._selected_idx:
                self._selected_idx[k] = 0
            self._visualize_current_contact()

        imgui.Separator()

        # Contact type selector
        for i, (name, count) in enumerate(zip(self._CONTACT_TYPE_NAMES, counts)):
            selected = self._selected_type == i
            if imgui.Selectable(f"{name} ({count})##type{i}", selected)[0]:
                self._selected_type = i
                self._visualize_current_contact()

        imgui.Separator()

        contact_list = self._get_contact_list(self._selected_type)
        order = self._sort_orders.get(self._selected_type, [])
        n = len(order)
        if n == 0:
            imgui.Text("No contacts of this type.")
            return

        # Index browser with arrow buttons
        idx = self._selected_idx.get(self._selected_type, 0)
        idx = max(0, min(idx, n - 1))

        if imgui.Button("<##prev"):
            idx = max(0, idx - 1)
        imgui.SameLine()
        imgui.SetNextItemWidth(50)
        _, idx = imgui.InputInt(f"##idx{self._selected_type}", idx)
        idx = max(0, min(idx, n - 1))
        imgui.SameLine()
        if imgui.Button(">##next"):
            idx = min(n - 1, idx + 1)
        imgui.SameLine()
        imgui.Text(f"/ {n - 1}")

        old_idx = self._selected_idx.get(self._selected_type, -1)
        self._selected_idx[self._selected_type] = idx
        if idx != old_idx:
            self._visualize_current_contact()

        # Show scalar data for current contact (via sorted indirection)
        c = contact_list[order[idx]]
        if imgui.TreeNode("Details##contact_details"):
            imgui.Text(f"c(x) = {c.cx:.6g}")
            imgui.Text(f"lambda = {c.lam:.6g}")
            imgui.Text(f"slack = {c.slack:.6g}")
            imgui.Text(f"decay = {c.decay:.6g}")
            imgui.Text(f"c = {c.c:.6g}")
            imgui.Text(f"cf = [{c.cf[0]:.6g}, {c.cf[1]:.6g}]")
            imgui.Text(f"lambdaf = [{c.lambdaf[0]:.6g}, {c.lambdaf[1]:.6g}]")
            imgui.Text(f"u={c.u}, v={c.v} (gu={c.gu}, gv={c.gv})")
            imgui.Text(f"nodes = {list(c.nodes)}")
            imgui.TreePop()

    def _get_contact_list(self, type_idx: int):
        if self._debug_data is None:
            return []
        lists = [
            self._debug_data.point_point_contacts,
            self._debug_data.point_edge_contacts,
            self._debug_data.point_triangle_contacts,
            self._debug_data.edge_edge_contacts,
        ]
        return lists[type_idx]

    def _build_sort_order(self, type_idx: int) -> list[int]:
        """Return a permutation (list of original indices) for the given contact type."""
        contact_list = self._get_contact_list(type_idx)
        n = len(contact_list)
        if n == 0:
            return []
        _, key_fn = self._SORT_KEYS[self._sort_key_idx]
        indices = list(range(n))
        if key_fn is not None:
            indices.sort(key=lambda i: key_fn(contact_list[i]))
        return indices

    def _rebuild_all_sort_orders(self):
        """Recompute cached sort permutations for every contact type."""
        for t in range(4):
            self._sort_orders[t] = self._build_sort_order(t)

    def _visualize_current_contact(self):
        self._clear_stencil_visualization()
        contact_list = self._get_contact_list(self._selected_type)
        order = self._sort_orders.get(self._selected_type, [])
        if len(order) == 0:
            return
        idx = self._selected_idx.get(self._selected_type, 0)
        idx = max(0, min(idx, len(order) - 1))
        c = contact_list[order[idx]]

        # Xc is kDims x kStencil (3 x N), grad/gradx same shape
        Xc = np.array(c.Xc)  # 3 x kStencil
        grad = np.array(c.grad)  # 3 x kStencil
        gradx = np.array(c.gradx)  # 3 x kStencil
        nfgrad = np.array(c.ngradf)  # 3 x kStencil
        if self._normalize_gradients:
            grad = self._normalized(grad)
            gradx = self._normalized(gradx)
            nfgrad = self._normalized(nfgrad)
        pts = Xc.T  # kStencil x 3

        if self._selected_type == 0:
            # Point-Point: 2 points
            self._stencil_pc = ps.register_point_cloud("Contact Stencil", pts)
            self._stencil_pc.add_vector_quantity(
                "grad (cached)", grad.T, vectortype="standard"
            )
            self._stencil_pc.add_vector_quantity(
                "grad (at x)", gradx.T, vectortype="standard"
            )
            self._stencil_pc.add_vector_quantity(
                "friction force", nfgrad.T, vectortype="standard"
            )
            # Also show the edge connecting them
            self._stencil_cn = ps.register_curve_network(
                "Contact Edge", pts, np.array([[0, 1]])
            )

        elif self._selected_type == 1:
            # Point-Edge: point (0) + edge (1,2)
            self._stencil_pc = ps.register_point_cloud("Contact Point", pts[0:1])
            self._stencil_pc.add_vector_quantity(
                "grad (cached)", grad[:, 0:1].T, vectortype="standard"
            )
            self._stencil_pc.add_vector_quantity(
                "grad (at x)", gradx[:, 0:1].T, vectortype="standard"
            )
            self._stencil_pc.add_vector_quantity(
                "friction force", nfgrad[:, 0:1].T, vectortype="standard"
            )
            self._stencil_cn = ps.register_curve_network(
                "Contact Edge", pts[1:3], np.array([[0, 1]])
            )
            self._stencil_cn.add_vector_quantity(
                "grad (cached)", grad[:, 1:3].T, vectortype="standard"
            )
            self._stencil_cn.add_vector_quantity(
                "grad (at x)", gradx[:, 1:3].T, vectortype="standard"
            )
            self._stencil_cn.add_vector_quantity(
                "friction force", nfgrad[:, 1:3].T, vectortype="standard"
            )

        elif self._selected_type == 2:
            # Point-Triangle: point (0) + triangle (1,2,3)
            self._stencil_pc = ps.register_point_cloud("Contact Point", pts[0:1])
            self._stencil_pc.add_vector_quantity(
                "grad (cached)", grad[:, 0:1].T, vectortype="standard"
            )
            self._stencil_pc.add_vector_quantity(
                "grad (at x)", gradx[:, 0:1].T, vectortype="standard"
            )
            self._stencil_pc.add_vector_quantity(
                "friction force", nfgrad[:, 0:1].T, vectortype="standard"
            )
            self._stencil_sm = ps.register_surface_mesh(
                "Contact Triangle", pts[1:4], np.array([[0, 1, 2]])
            )
            self._stencil_sm.add_vector_quantity(
                "grad (cached)",
                grad[:, 1:4].T,
                vectortype="standard",
                defined_on="vertices",
            )
            self._stencil_sm.add_vector_quantity(
                "grad (at x)",
                gradx[:, 1:4].T,
                vectortype="standard",
                defined_on="vertices",
            )
            self._stencil_sm.add_vector_quantity(
                "friction force",
                nfgrad[:, 1:4].T,
                vectortype="standard",
                defined_on="vertices",
            )

        elif self._selected_type == 3:
            # Edge-Edge: edge (0,1) + edge (2,3)
            edge_pts = pts  # 4 x 3
            edges = np.array([[0, 1], [2, 3]])
            self._stencil_cn = ps.register_curve_network(
                "Contact Edges", edge_pts, edges
            )
            self._stencil_cn.add_vector_quantity(
                "grad (cached)", grad.T, vectortype="standard"
            )
            self._stencil_cn.add_vector_quantity(
                "grad (at x)", gradx.T, vectortype="standard"
            )
            self._stencil_cn.add_vector_quantity(
                "friction force", nfgrad.T, vectortype="standard"
            )

    @staticmethod
    def _normalized(g: np.ndarray) -> np.ndarray:
        """Column-wise normalize a 3 x N gradient matrix."""
        norms = np.linalg.norm(g, axis=0, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return g / norms

    def _clear_stencil_visualization(self):
        if self._stencil_pc is not None:
            ps.remove_point_cloud(self._stencil_pc.get_name())
            self._stencil_pc = None
        if self._stencil_cn is not None:
            ps.remove_curve_network(self._stencil_cn.get_name())
            self._stencil_cn = None
        if self._stencil_sm is not None:
            ps.remove_surface_mesh(self._stencil_sm.get_name())
            self._stencil_sm = None

    def set_visible(self, visible: bool):
        if self._environment_mesh is not None:
            self._environment_mesh.set_enabled(visible)
        if not visible:
            self._show_debug = False
            self._clear_stencil_visualization()

    def on_new_contact_dynamics(self, contact_dynamics: pbat.sim.contact.MeshDynamics):
        self._contact_dynamics = contact_dynamics
        self._contact_params = ParameterObject(
            self._contact_dynamics.params,
            {
                "ogc_params": None,
            },
        )
        if self._contact_dynamics.ogc_input.has_static_geometry:
            self._environment_mesh = ps.register_surface_mesh(
                "Contact Environment",
                self._contact_dynamics.Xstatic.T,
                self._contact_dynamics.static_meshes.F.T,
                color=(0.72, 0.72, 0.72),
            )
        else:
            if self._environment_mesh is not None:
                ps.remove_surface_mesh(self._environment_mesh.get_name())
                self._environment_mesh = None

    def serialize(self, archive: pbat.io.Archive):
        self._contact_dynamics.params.serialize(archive)

    def deserialize(self, archive: pbat.io.Archive):
        self._contact_dynamics.params.deserialize(archive)

    def on_debug_display_requested(self, x: np.ndarray):
        """Update debug contact data from current positions.

        Args:
            x: Current vertex positions (3 x |# points|) or (3*|# points| x 1).
        """
        if self._contact_dynamics is None:
            return
        self._debug_data = pbat.sim.contact.DebugMeshDynamics(self._contact_dynamics, x)
        self._rebuild_all_sort_orders()
        self._visualize_current_contact()

    @property
    def contact_dynamics(self):
        return self._contact_dynamics

    @property
    def requires_debug_display(self) -> bool:
        return self._show_debug

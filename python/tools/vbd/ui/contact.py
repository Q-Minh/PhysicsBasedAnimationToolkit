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
    _contact_forces_pc: ps.PointCloud
    _show_forces: bool
    _show_stencils: bool
    # Individual visibility flags for each contact type
    _show_vv: bool  # Vertex-vertex
    _show_ve: bool  # Vertex-edge
    _show_vt: bool  # Vertex-triangle
    _show_ee: bool  # Edge-edge
    _show_ven: bool  # Vertex-environment
    _show_een: bool  # Edge-environment
    _show_ten: bool  # Triangle-environment
    # Stencil point clouds for each contact type
    _vv_stencil_pc: ps.PointCloud  # Vertex-vertex (2-node stencil)
    _ve_stencil_pc: ps.PointCloud  # Vertex-edge (3-node stencil)
    _vt_stencil_pc: ps.PointCloud  # Vertex-triangle (4-node stencil)
    _ee_stencil_pc: ps.PointCloud  # Edge-edge (4-node stencil)
    _ven_stencil_pc: ps.PointCloud  # Vertex-environment (1-node stencil)
    _een_stencil_pc: ps.PointCloud  # Edge-environment (2-node stencil)
    _ten_stencil_pc: ps.PointCloud  # Triangle-environment (3-node stencil)

    def __init__(self):
        self._contact_dynamics = None
        self._contact_params = None
        self._environment_mesh = None
        self._contact_forces_pc = None
        self._show_forces = False
        self._show_stencils = False
        # Initialize individual visibility flags (all enabled by default when stencils are shown)
        self._show_vv = True
        self._show_ve = True
        self._show_vt = True
        self._show_ee = True
        self._show_ven = True
        self._show_een = True
        self._show_ten = True
        # Initialize stencil point clouds
        self._vv_stencil_pc = None
        self._ve_stencil_pc = None
        self._vt_stencil_pc = None
        self._ee_stencil_pc = None
        self._ven_stencil_pc = None
        self._een_stencil_pc = None
        self._ten_stencil_pc = None
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
        changed, self._show_forces = imgui.Checkbox(
            "Show Contact Forces", self._show_forces
        )
        if changed and self._contact_forces_pc is not None:
            self._contact_forces_pc.set_enabled(self._show_forces)
        stencil_changed, self._show_stencils = imgui.Checkbox(
            "Show Contact Stencils", self._show_stencils
        )
        if stencil_changed:
            self._update_all_stencil_visibility()
        # Individual contact type checkboxes (only shown when stencils are enabled)
        if self._show_stencils:
            imgui.Indent()
            if imgui.TreeNode("Contact Types"):
                # Mesh-mesh contacts
                changed_vv, self._show_vv = imgui.Checkbox(
                    "Vertex-Vertex", self._show_vv
                )
                if changed_vv and self._vv_stencil_pc is not None:
                    self._vv_stencil_pc.set_enabled(self._show_vv)
                changed_ve, self._show_ve = imgui.Checkbox("Vertex-Edge", self._show_ve)
                if changed_ve and self._ve_stencil_pc is not None:
                    self._ve_stencil_pc.set_enabled(self._show_ve)
                changed_vt, self._show_vt = imgui.Checkbox(
                    "Vertex-Triangle", self._show_vt
                )
                if changed_vt and self._vt_stencil_pc is not None:
                    self._vt_stencil_pc.set_enabled(self._show_vt)
                changed_ee, self._show_ee = imgui.Checkbox("Edge-Edge", self._show_ee)
                if changed_ee and self._ee_stencil_pc is not None:
                    self._ee_stencil_pc.set_enabled(self._show_ee)
                # Mesh-environment contacts
                imgui.Separator()
                changed_ven, self._show_ven = imgui.Checkbox(
                    "Vertex-Environment", self._show_ven
                )
                if changed_ven and self._ven_stencil_pc is not None:
                    self._ven_stencil_pc.set_enabled(self._show_ven)
                changed_een, self._show_een = imgui.Checkbox(
                    "Edge-Environment", self._show_een
                )
                if changed_een and self._een_stencil_pc is not None:
                    self._een_stencil_pc.set_enabled(self._show_een)
                changed_ten, self._show_ten = imgui.Checkbox(
                    "Triangle-Environment", self._show_ten
                )
                if changed_ten and self._ten_stencil_pc is not None:
                    self._ten_stencil_pc.set_enabled(self._show_ten)
                imgui.TreePop()
            imgui.Unindent()
        imgui.PopID()

    def set_visible(self, visible: bool):
        if self._environment_mesh is not None:
            self._environment_mesh.set_enabled(visible)
        if self._contact_forces_pc is not None:
            self._show_forces = self._show_forces and visible
            self._contact_forces_pc.set_enabled(self._show_forces)
        if not visible:
            self._show_stencils = False
        self._update_all_stencil_visibility()

    def _update_all_stencil_visibility(self):
        """Update visibility for all stencil point clouds based on individual flags."""
        stencil_data = [
            (self._vv_stencil_pc, self._show_vv),
            (self._ve_stencil_pc, self._show_ve),
            (self._vt_stencil_pc, self._show_vt),
            (self._ee_stencil_pc, self._show_ee),
            (self._ven_stencil_pc, self._show_ven),
            (self._een_stencil_pc, self._show_een),
            (self._ten_stencil_pc, self._show_ten),
        ]
        for pc, show_type in stencil_data:
            if pc is not None:
                pc.set_enabled(self._show_stencils and show_type)

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

    def on_contact_force_display_requested(
        self, x: np.ndarray, xt: np.ndarray, h: float
    ):
        if self._contact_dynamics is None:
            return

        # Compute contact energies
        flags = pbat.sim.contact.EMeshEnergyComputationFlags.Gradient
        self._contact_dynamics.update_constraint_set(x)
        self._contact_dynamics.compute_energies(x, xt, h, flags)

        # Get the gradients (3*|# points| x 1) and reshape to (|# points| x 3)
        n_points = x.shape[1]
        normal_gradient = self._contact_dynamics.normal_gradient
        normal_forces = -normal_gradient.reshape((3, n_points), order="F").T
        frictional_gradient = self._contact_dynamics.frictional_gradient
        frictional_forces = -frictional_gradient.reshape((3, n_points), order="F").T

        # Register or update the point cloud for contact forces visualization
        if self._contact_forces_pc is None:
            self._contact_forces_pc = ps.register_point_cloud("Contact dynamics", x.T)
        else:
            self._contact_forces_pc.update_point_positions(x.T)

        # Add/update the vector quantities for contact forces (negative gradient = force)
        self._contact_forces_pc.add_vector_quantity(
            "normal",
            normal_forces,
            enabled=self._show_forces,
            vectortype="standard",
        )
        self._contact_forces_pc.add_vector_quantity(
            "frictional",
            frictional_forces,
            enabled=self._show_forces,
            vectortype="standard",
        )

    @property
    def contact_dynamics(self):
        return self._contact_dynamics

    @property
    def requires_force_display(self) -> bool:
        return self._show_forces

    @property
    def requires_stencil_display(self) -> bool:
        return self._show_stencils

    def _update_stencil_point_cloud(
        self,
        name: str,
        energies: list,
        x: np.ndarray,
        stencil_size: int,
        show_type: bool,
    ) -> ps.PointCloud:
        """
        Update or create a point cloud for contact stencils.

        Args:
            name: Name for the point cloud
            energies: List of MeshContactEnergy objects
            x: Current vertex positions (3 x |# points|)
            stencil_size: Number of nodes in the stencil (1, 2, 3, or 4)
            show_type: Whether this contact type should be visible

        Returns:
            The updated or created point cloud, or None if no contacts
        """
        if len(energies) == 0:
            return ps.register_point_cloud(name, np.empty((0, 3), dtype=x.dtype))

        n_contacts = len(energies)
        n_stencil_points = n_contacts * stencil_size

        # Collect stencil positions and forces
        positions = np.zeros((n_stencil_points, 3), dtype=x.dtype)
        normal_forces = np.zeros((n_stencil_points, 3), dtype=x.dtype)
        frictional_forces = np.zeros((n_stencil_points, 3), dtype=x.dtype)

        for i, energy in enumerate(energies):
            stencil = energy.stencil
            grad_n = -energy.gradEn  # Negative gradient = force
            grad_f = -energy.gradEf
            positions[i * stencil_size : (i + 1) * stencil_size, :] = x[:, stencil].T
            normal_forces[i * stencil_size : (i + 1) * stencil_size, :] = (
                grad_n.reshape((3, stencil_size), order="F").T
            )
            frictional_forces[i * stencil_size : (i + 1) * stencil_size, :] = (
                grad_f.reshape((3, stencil_size), order="F").T
            )

        # Determine effective visibility
        is_visible = self._show_stencils and show_type

        # Register or update point cloud
        pc = ps.register_point_cloud(name, positions)
        pc.add_vector_quantity(
            "normal",
            normal_forces,
            enabled=is_visible,
            vectortype="standard",
        )
        pc.add_vector_quantity(
            "frictional",
            frictional_forces,
            enabled=is_visible,
            vectortype="standard",
        )
        pc.set_enabled(is_visible)
        return pc

    def on_stencil_display_requested(self, x: np.ndarray, xt: np.ndarray, h: float):
        """
        Update all contact stencil point clouds.

        Args:
            x: Current vertex positions (3 x |# points|)
            xt: Reference vertex positions (3 x |# points|)
            h: Time step size
        """
        if self._contact_dynamics is None:
            return

        # Compute contact energies
        flags = pbat.sim.contact.EMeshEnergyComputationFlags.Gradient
        self._contact_dynamics.update_constraint_set(x)
        self._contact_dynamics.compute_energies(x, xt, h, flags)

        # Mesh-mesh contacts
        self._vv_stencil_pc = self._update_stencil_point_cloud(
            "VV Stencils",
            self._contact_dynamics.vertex_vertex_energies,
            x,
            stencil_size=2,
            show_type=self._show_vv,
        )
        self._ve_stencil_pc = self._update_stencil_point_cloud(
            "VE Stencils",
            self._contact_dynamics.vertex_edge_energies,
            x,
            stencil_size=3,
            show_type=self._show_ve,
        )
        self._vt_stencil_pc = self._update_stencil_point_cloud(
            "VT Stencils",
            self._contact_dynamics.vertex_triangle_energies,
            x,
            stencil_size=4,
            show_type=self._show_vt,
        )
        self._ee_stencil_pc = self._update_stencil_point_cloud(
            "EE Stencils",
            self._contact_dynamics.edge_edge_energies,
            x,
            stencil_size=4,
            show_type=self._show_ee,
        )

        # Mesh-environment contacts
        self._ven_stencil_pc = self._update_stencil_point_cloud(
            "V-Env Stencils",
            self._contact_dynamics.vertex_environment_energies,
            x,
            stencil_size=1,
            show_type=self._show_ven,
        )
        self._een_stencil_pc = self._update_stencil_point_cloud(
            "E-Env Stencils",
            self._contact_dynamics.edge_environment_energies,
            x,
            stencil_size=2,
            show_type=self._show_een,
        )
        self._ten_stencil_pc = self._update_stencil_point_cloud(
            "T-Env Stencils",
            self._contact_dynamics.triangle_environment_energies,
            x,
            stencil_size=3,
            show_type=self._show_ten,
        )

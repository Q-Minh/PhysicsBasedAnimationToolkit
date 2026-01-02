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

    def __init__(self):
        self._contact_dynamics = None
        self._contact_params = None
        self._environment_mesh = None
        self._contact_forces_pc = None
        self._show_forces = False
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
        imgui.PopID()

    def set_visible(self, visible: bool):
        if self._environment_mesh is not None:
            self._environment_mesh.set_enabled(visible)
        if self._contact_forces_pc is not None:
            self._show_forces = self._show_forces and visible
            self._contact_forces_pc.set_enabled(self._show_forces)

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

    def on_contact_force_display_requested(self, x: np.ndarray, xt: np.ndarray, h: float):
        if self._contact_dynamics is None:
            return

        # Compute contact energies
        flags = pbat.sim.contact.EMeshEnergyComputationFlags.Gradient
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

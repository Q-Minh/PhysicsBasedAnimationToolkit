# type: ignore

from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
from .params import ParameterObject


class Contact:
    _contact_dynamics: pbat.sim.contact.MeshDynamics
    _contact_params: ParameterObject
    _environment_mesh: ps.SurfaceMesh

    def __init__(self):
        self._contact_dynamics = None
        self._contact_params = None
        self._environment_mesh = None
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
        imgui.PopID()

    def set_visible(self, visible: bool):
        if self._environment_mesh is not None:
            self._environment_mesh.set_enabled(visible)

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

    @property
    def contact_dynamics(self):
        return self._contact_dynamics

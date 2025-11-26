# type: ignore

import enum
import typing
from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
from .draw_parameter_object import ParameterObject


class Contact:
    _contact_dynamics: pbat.sim.contact.MeshDynamics
    _contact_params: ParameterObject

    def __init__(self):
        self._contact_dynamics = pbat.sim.contact.MeshDynamics()
        self._contact_params = ParameterObject(
            self._contact_dynamics,
            {
                "env_contact_dynamics_params": None,
                "offset_geometry_contact": {"params": None},
                "mesh_sdf_contact": {"params": None},
            },
        )

    def draw(self):
        imgui.PushID("Contact")
        if imgui.TreeNode("Parameters"):
            self._contact_params.draw()
            imgui.TreePop()
        imgui.PopID()

    def on_new_fem_elasto_dynamics(
        self, fem_dynamics: pbat.sim.dynamics.FemElastoDynamics
    ):
        # TODO: Set up contact dynamics with the new fem_dynamics
        pass

    @property
    def contact_dynamics(self):
        return self._contact_dynamics

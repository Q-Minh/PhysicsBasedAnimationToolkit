# type: ignore
import polyscope as ps
import polyscope.imgui as imgui
from pbatoolkit import pbat, pypbat
import numpy as np
from .ui.scene import Scene
from .ui.simulation import Simulation
from .scenario import make_fem_dynamics_object, make_contact_dynamics_object, static_meshes
from statemachine import State, StateMachine, Event


class ModeStateMachine(StateMachine):
    editing = State("Editing", initial=True)
    simulating = State("Simulating")
    simulate = editing.to(simulating) | simulating.to.itself()
    edit = simulating.to(editing) | editing.to.itself()

    scene: Scene
    simulation: Simulation

    def __init__(self, scene: Scene, simulation: Simulation):
        super().__init__()
        self.scene = scene
        self.simulation = simulation

    def before_simulate(self, event: Event, source: State, target: State):
        if source == self.editing:
            self._convert_scene_to_simulation()
        if source != self.simulating:
            self.simulation.set_visible(True)

    def on_simulate(self):
        self.simulation.draw()

    def on_exit_simulating(self, event: Event, source: State, target: State):
        if target != self.simulating:
            self.simulation.set_visible(False)

    def before_edit(self, event: Event, source: State, target: State):
        if source != self.editing:
            self.scene.set_visible(True)

    def on_edit(self):
        self.scene.draw()

    def on_exit_editing(self, event: Event, source: State, target: State):
        if target != self.editing:
            self.scene.set_visible(False)

    def _convert_scene_to_simulation(self):
        tet_elastic_bodies = self.scene.tet_elastic_bodies
        n_bodies = len(tet_elastic_bodies)
        if n_bodies == 0:
            return
        # FEM dynamics object
        fem_dynamics, vert_counts, VP = make_fem_dynamics_object(
            tet_elastic_bodies, get_external_info=True
        )
        # Contact
        contact_dynamics = make_contact_dynamics_object(
            fem_dynamics,
            n_bodies,
            vert_counts,
            static_meshes(
                tet_elastic_bodies,
                self.scene.static_mesh_colliders,
                self.scene.plane_colliders,
                tessellate_planes=True,
            ),
        )
        # Pass simulation scenario to simulation UI
        tet_elastic_body_names = [body.name for body in tet_elastic_bodies]
        self.scene.transform_library.sync_mesh_names(tet_elastic_body_names)
        self.simulation.on_simulation_scenario_created(
            tet_elastic_body_names,
            VP,
            fem_dynamics,
            contact_dynamics,
            self.scene.transform_library,
        )


def main():
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Editor")
    ps.init()
    mode_state_machine = ModeStateMachine(Scene(), Simulation())

    def callback():
        nonlocal mode_state_machine
        if imgui.BeginTabBar("Mode bar"):
            if imgui.BeginTabItem("Scene", True)[0]:
                mode_state_machine.edit()
                imgui.EndTabItem()
            if imgui.BeginTabItem("Simulation", True)[0]:
                mode_state_machine.simulate()
                imgui.EndTabItem()
            imgui.EndTabBar()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()

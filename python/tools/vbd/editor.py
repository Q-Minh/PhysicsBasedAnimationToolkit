# type: ignore
import polyscope as ps
import polyscope.imgui as imgui
from .ui.scene import Scene
from .ui.simulation import Simulation
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
            # TODO
            # fem_dynamics, contact_dynamics = self.scene.build_simulation_scenario()
            # self.simulation.on_simulation_scenario_created(
            #     fem_dynamics, contact_dynamics
            # )
            pass
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

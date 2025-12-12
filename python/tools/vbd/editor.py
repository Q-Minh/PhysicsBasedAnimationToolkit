# type: ignore
import polyscope as ps
import polyscope.imgui as imgui
from pbatoolkit import pbat, pypbat
import numpy as np
from .ui.scene import Scene
from .ui.simulation import Simulation
from statemachine import State, StateMachine, Event
import itertools
import math


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
        if len(tet_elastic_bodies) == 0:
            return
        vert_counts = [body.VT.shape[0] for body in tet_elastic_bodies]
        tet_counts = [body.T.shape[0] for body in tet_elastic_bodies]
        # Per-body offsets (prefix) array for vertices
        VP = np.array([0] + list(itertools.accumulate(vert_counts)))
        # Per-body offsets (prefix) array for tets
        TP = np.array([0] + list(itertools.accumulate(tet_counts)))
        V = np.vstack([body.VT for body in tet_elastic_bodies])
        T = np.vstack([body.T + VP[b] for b, body in enumerate(tet_elastic_bodies)])
        # Construct the fem dynamics
        fem_dynamics = pbat.sim.dynamics.FemElastoDynamics(V.T, T.T)
        dims = fem_dynamics.X.shape[0]
        element = pbat.fem.Element.Tetrahedron
        element_order = 1
        mass_quadrature_order = 2
        elasticity_quadrature_order = 1
        load_quadrature_order = 1
        n_elems = fem_dynamics.E.shape[1]
        # Mass
        wgM = pbat.fem.mesh_quadrature_weights(
            fem_dynamics.E,
            fem_dynamics.X,
            element,
            order=element_order,
            quadrature_order=mass_quadrature_order,
        )
        egM = pbat.fem.mesh_quadrature_elements(fem_dynamics.E, wgM)
        XigM = pbat.fem.mesh_reference_quadrature_points(
            n_elems,
            element=element,
            order=element_order,
            quadrature_order=mass_quadrature_order,
        )
        rhog = np.zeros_like(wgM)
        for start, end, body in zip(TP[:-1], TP[1:], tet_elastic_bodies):
            rhog[:, start:end] = body.rhoe[np.newaxis, :]
        fem_dynamics.set_mass_matrix(
            eg=np.ravel(egM, order="F"),
            wg=np.ravel(wgM, order="F"),
            Xig=XigM,
            rhog=np.ravel(rhog, order="F"),
        )
        # Elasticity
        wgU = pbat.fem.mesh_quadrature_weights(
            fem_dynamics.E,
            fem_dynamics.X,
            element,
            order=element_order,
            quadrature_order=elasticity_quadrature_order,
        )
        egU = pbat.fem.mesh_quadrature_elements(fem_dynamics.E, wgU)
        XigU = pbat.fem.mesh_reference_quadrature_points(
            n_elems,
            element=element,
            order=element_order,
            quadrature_order=elasticity_quadrature_order,
        )
        mug, lambdag = np.zeros_like(wgU), np.zeros_like(wgU)
        for start, end, body in zip(TP[:-1], TP[1:], tet_elastic_bodies):
            mue, llambdae = pypbat.fem.lame_coefficients(body.Ye, body.nue)
            mug[:, start:end] = mue
            lambdag[:, start:end] = llambdae
        fem_dynamics.set_elastic_energy(
            eg=np.ravel(egU, order="F"),
            wg=np.ravel(wgU, order="F"),
            Xig=XigU,
            mug=np.ravel(mug, order="F"),
            lambdag=np.ravel(lambdag, order="F"),
        )
        # Load
        wgB = pbat.fem.mesh_quadrature_weights(
            fem_dynamics.E,
            fem_dynamics.X,
            element,
            order=element_order,
            quadrature_order=load_quadrature_order,
        )
        egB = pbat.fem.mesh_quadrature_elements(fem_dynamics.E, wgB)
        XigB = pbat.fem.mesh_reference_quadrature_points(
            n_elems,
            element=element,
            order=element_order,
            quadrature_order=load_quadrature_order,
        )
        bg = np.zeros((dims, math.prod(wgB.shape)))
        for start, end, body in zip(TP[:-1], TP[1:], tet_elastic_bodies):
            aext = body.aext.reshape(dims, 1)
            fext = (body.rhoe * aext) + body.bext.T
            bg[:, start:end] = fext
        fem_dynamics.set_external_load(
            eg=np.ravel(egB, order="F"), wg=np.ravel(wgB, order="F"), Xig=XigB, bg=bg
        )
        # Initial velocity
        x0 = fem_dynamics.X
        v0 = np.vstack([body.v0 for body in tet_elastic_bodies]).T
        fem_dynamics.x = x0
        fem_dynamics.v = v0
        # Contact
        contact_dynamics = pbat.sim.contact.MeshDynamics()
        n_bodies = len(tet_elastic_bodies)
        XCC = np.concatenate(
            [np.full(nverts, b) for b, nverts in enumerate(vert_counts)]
        )
        contact_meshes = pbat.sim.contact.MultiMesh(
            fem_dynamics.E, XCC, n_components=n_bodies
        )
        contact_dynamics.set_dynamic_geometry(fem_dynamics.x, contact_meshes)
        # Pass simulation scenario to simulation UI
        tet_elastic_body_names = [body.name for body in tet_elastic_bodies]
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

# type: ignore
import polyscope as ps
import polyscope.imgui as imgui
from .ui.scene import Scene
from .ui.simulation import Simulation


def main():
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Editor")
    ps.init()
    scene = Scene()
    simulation = Simulation()

    def callback():
        nonlocal scene, simulation
        if imgui.BeginTabBar("Mode bar"):
            if imgui.BeginTabItem("Scene", True)[0]:
                scene.draw()
                imgui.EndTabItem()
            if imgui.BeginTabItem("Simulation", True)[0]:
                simulation.draw()
                imgui.EndTabItem()
            imgui.EndTabBar()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()

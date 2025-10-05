from pbatoolkit import pbat, pypbat
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import time
import meshio
import argparse
import math


def signal(w: float, v: np.ndarray, t: float, c: float, k: float):
    u = c*np.sin(k*w*t)*v
    return u


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Elastic stiffness modal decomposition demo",
    )
    parser.add_argument("-i", "--input", help="Path to input tetrahedral mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    parser.add_argument("-k", "--num-modes", help="Number of modes to compute", type=int,
                        dest="modes", default=30)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    element = pbat.fem.Element.Tetrahedron
    X, E = pbat.fem.mesh(
        V.T, C.T, element=element)
    w, Veigs = pypbat.fem.rest_pose_hyper_elastic_modes(
        E, X, element, Y=args.Y, nu=args.nu, rho=args.rho, modes=args.modes)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vm = ps.register_volume_mesh("model", X.T, E.T)
    mode = 6
    t0 = time.time()
    t = 0
    c = 0.15
    k = 0.05

    def callback():
        global mode, c, k
        changed, mode = imgui.InputInt("Mode", mode)
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)

        t = time.time() - t0
        V = X.T + signal(w[mode], Veigs[:, mode],
                              t, c, k).reshape(X.shape[1], 3)
        vm.update_vertex_positions(V)

    ps.set_user_callback(callback)
    ps.show()

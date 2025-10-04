from pbatoolkit import pbat, pypbat
import igl
import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import scipy as sp
import argparse
import meshio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Diffusion mesh smoothing demo",
    )
    parser.add_argument("-i", "--input", help="Path to input triangle mesh", type=str,
                        dest="input", required=True)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, F = imesh.points, imesh.cells_dict["triangle"]

    # Construct Galerkin laplacian, mass and gradient operators
    element = pbat.fem.Element.Triangle
    order = 1
    X, E = pbat.fem.mesh(
        V.T, F.T, element=element, order=1)
    L = pbat.fem.laplacian_matrix(E, X, element=element, order=order)
    M = pbat.fem.mass_matrix(E, X, rho=1, dims=1, element=element, order=order)
    # Setup heat diffusion
    dt = 0.016
    c = 1
    A = M - c*dt*L
    # Precompute linear solvers
    Ainv = pbat.math.linalg.ldlt(A)
    Ainv.compute(A)

    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    vmm = ps.register_surface_mesh("model", V, F)
    vmm.set_smooth_shade(True)
    vms = ps.register_surface_mesh("smoothed", V, F)
    vms.set_smooth_shade(True)
    smooth = False

    def callback():
        global dt, Ainv, M, L, smooth, V, c
        dtchanged, dt = imgui.InputFloat("dt", dt)
        cchanged, c = imgui.SliderFloat("c", c, v_min=0, v_max=100)
        if dtchanged or cchanged:
            A = M - c*dt*L
            Ainv.factorize(A)
        _, smooth = imgui.Checkbox("smooth", smooth)
        if smooth:
            V = Ainv.solve(M @ V)
            vms.update_vertex_positions(V)

    ps.set_user_callback(callback)
    ps.show()

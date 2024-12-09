import pbatoolkit as pbat
import polyscope as ps
import polyscope.imgui as imgui
import meshio
import argparse
import time
import numpy as np


def signal(w: float, v: np.ndarray, t: float, c: float, k: float):
    u = c*np.sin(k*w*t)*v
    return u


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Shape function interpolation prolongation",
    )
    parser.add_argument("-i", "--input", help="Ordered paths to input tetrahedral meshes", nargs="+",
                        dest="input", required=True)
    args = parser.parse_args()

    # Load input meshes
    imeshes = [meshio.read(input) for input in args.input]
    meshes = [
        pbat.fem.Mesh(
            imesh.points.T,
            imesh.cells_dict["tetra"].T, element=pbat.fem.Element.Tetrahedron)
        for imesh in imeshes
    ]

    # Compute levels and prolongation operators
    levels = [
        pbat.sim.vbd.multigrid.Level(mesh) for mesh in meshes
    ]
    prolongators = [
        pbat.sim.vbd.multigrid.Prolongation(fine_mesh, coarse_mesh) for
        (fine_mesh, coarse_mesh) in zip(meshes[:-1], meshes[1:])
    ]
    prolongators.reverse()

    # Compute modes on coarsest mesh
    w, U = pbat.fem.rest_pose_hyper_elastic_modes(meshes[-1])

    # Create visual meshes
    V, F = zip(*[
        pbat.geometry.simplex_mesh_boundary(mesh.E, n=mesh.X.shape[1])
        for mesh in meshes
    ])
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()

    min_transparency, max_transparency = 0.25, 1.
    n_levels = len(levels)
    lsms = [None]*n_levels
    for l, level in enumerate(levels):
        lsms[l] = ps.register_surface_mesh(f"Level {l}", level.x.T, F[l].T)
        transparency = min_transparency + (n_levels-l-1) * \
            (max_transparency - min_transparency) / (n_levels-1)
        lsms[l].set_transparency(transparency)

    mode = 6
    t0 = time.time()
    t = 0
    c = 10
    k = 0.1

    n_levels = len(levels)

    def callback():
        global mode, c, k
        changed, mode = imgui.InputInt("Mode", mode)
        changed, c = imgui.InputFloat("Wave amplitude", c)
        changed, k = imgui.InputFloat("Wave frequency", k)

        t = time.time() - t0
        X = levels[-1].X
        u = signal(w[mode], U[:, mode], t, c, k).reshape(X.shape, order="F")
        levels[-1].x = X + u
        for l, prolongator in enumerate(prolongators):
            prolongator.apply(levels[-1-l], levels[-2-l])
        for l, level in enumerate(levels):
            lsms[l].update_vertex_positions(level.x.T)
        

    ps.set_user_callback(callback)
    ps.show()

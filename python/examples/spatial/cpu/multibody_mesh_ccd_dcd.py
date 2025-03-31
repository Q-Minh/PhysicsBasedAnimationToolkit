import pbatoolkit as pbat
import argparse
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D multibody mesh CCD and DCD contact detection algorithm",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input mesh", dest="input", required=True
    )
    parser.add_argument(
        "--rows", help="Number of rows in the grid", dest="rows", type=int, default=4
    )
    parser.add_argument(
        "--cols", help="Number of columns in the grid", dest="cols", type=int, default=4
    )
    parser.add_argument(
        "--height", help="Height of the grid", dest="height", type=int, default=2
    )
    parser.add_argument(
        "--separation",
        help="Separation between meshes as percentage of input mesh bounding box extents",
        dest="separation",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()

    # Load input mesh
    imesh = meshio.read(args.input)
    V, T = imesh.points.astype(np.float64), imesh.cells_dict["tetra"].astype(np.int64)

    # Duplicate mesh into grid of meshes
    grows = args.rows
    gcols = args.cols
    gheight = args.height
    aabb = pbat.geometry.aabb(V.T)
    extents = aabb.max - aabb.min
    separation = args.separation * extents
    meshes = [
        V
        + np.array(
            [
                i * (extents[0] + separation[0]),
                j * (extents[1] + separation[1]),
                k * (extents[2] + separation[2]),
            ]
        )
        for k in range(gheight)
        for j in range(gcols)
        for i in range(grows)
    ]

    ps.init()
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("3D multibody mesh CCD and DCD contact detection algorithm")

    vm = [ps.register_volume_mesh(f"Mesh {m}", VI, T) for m, VI in enumerate(meshes)]

    t = 0

    def callback():
        global t
        for m, VI in enumerate(meshes):
            r = m % 3
            # Update mesh position
            offset = args.separation * extents
            dir = np.ones(3)
            dir[r] = -1
            u = 1.2 * dir * offset * np.sin(t * 2)
            vm[m].update_vertex_positions(VI + u)
        t = t + 1 / 60

    ps.set_user_callback(callback)
    ps.show()
